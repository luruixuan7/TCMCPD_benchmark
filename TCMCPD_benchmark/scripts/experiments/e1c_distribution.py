#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R3: discrete-distribution fits for the three vocabularies (AIC + Vuong test).

Fixes three methodological problems in the original run:
  A. R1 distribution: the original fitted CONTINUOUS densities (and used KS) on DISCRETE
     counts. Here we fit discrete models (Poisson, Geometric, Negative-Binomial, discrete
     power law / zeta) by MLE, compare by AIC, and run a Vuong test PL vs NB.
     Counts are de-duplicated (visits containing the item), with raw-edge counts as a
     sensitivity analysis.
  B. R5 entropy bound: plug-in MI is positively biased. We build a patient-level label
     permutation null (sparse matmul) and report observed vs null, plus the excess.
  C. R4 transitions: replaced by a set-level definition (Jaccard of syndrome sets between
     consecutive labelled visits), with the old cross-product kept as sensitivity.
  D. R8 clustering: silhouette for k=2..6 + bootstrap stability (ARI).
  E. Bootstrap CIs for Gini and entropy of the three vocabularies.

Writes results/e1_e2_distribution.json
"""
import os, json, math, itertools, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats, optimize, special
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)
_REPO = os.environ.get(
    "TCMCPD_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = _REPO  # set TCMCPD_ROOT to override the checkout location
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
A = os.path.join(ROOT, "results")
R = {}

pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
v2p = dict(zip(pv.visit_id, pv.patient_id))


def counts_dedup(edge_df, val_col):
    """number of distinct visits containing each item"""
    return edge_df.groupby(val_col).visit_id.nunique()


def counts_raw(edge_df, val_col):
    return edge_df.groupby(val_col).size()


# ============================================================ A. discrete distributions
print("=" * 80)
print("A  DISCRETE DISTRIBUTION COMPARISON (dedup counts as primary)")

ZETA = lambda a: special.zeta(a)          # sum_{k>=1} k^-a


def ll_poisson(x):
    lam = x.mean()
    return np.sum(stats.poisson.logpmf(x, lam)), 1, {"lambda": lam}


def ll_geom(x):
    p = 1.0 / x.mean()
    return np.sum(stats.geom.logpmf(x, p)), 1, {"p": p}


def ll_nbinom(x):
    """NB parametrised by (mu, phi): var = mu + phi*mu^2 ; bounded L-BFGS-B."""
    m, v = float(x.mean()), float(x.var())
    phi0 = max((v - m) / (m * m), 1e-3)
    def nll(th):
        mu, phi = th
        if mu <= 0 or phi <= 0:
            return 1e15
        n = 1.0 / phi
        p = n / (n + mu)
        return -np.sum(stats.nbinom.logpmf(x, n, p))
    res = optimize.minimize(nll, [m, phi0], method="L-BFGS-B",
                            bounds=[(1e-3, 1e4), (1e-6, 1e3)])
    mu, phi = float(res.x[0]), float(res.x[1])
    ll = -float(res.fun)
    return ll, 2, {"mu": mu, "phi": phi, "r": 1.0 / phi, "p": (1.0 / phi) / (1.0 / phi + mu)}


def ll_pl(x):
    x = x.astype(float)
    def nll(a):
        if a <= 1.0:
            return 1e12
        return -(np.sum(-a * np.log(x)) - len(x) * math.log(ZETA(a)))
    res = optimize.minimize_scalar(nll, bounds=(1.01, 6.0), method="bounded")
    a = float(res.x)
    ll = -res.fun
    return ll, 1, {"alpha": a}


MODELS = {"poisson": ll_poisson, "geometric": ll_geom,
          "negative_binomial": ll_nbinom, "discrete_power_law": ll_pl}


def compare(x):
    out = {}
    for name, fn in MODELS.items():
        try:
            ll, k, params = fn(x)
            out[name] = {"loglik": float(ll), "k": int(k), "aic": float(2 * k - 2 * ll),
                         "params": {kk: float(vv) for kk, vv in params.items()}}
        except Exception as e:
            out[name] = {"error": type(e).__name__ + ": " + str(e)[:80]}
    best = min((v["aic"], n) for n, v in out.items() if "aic" in v)
    return out, best[1]


def per_obs_ll(x, name):
    if name == "discrete_power_law":
        a = R["A"][CUR][name]["params"]["alpha"]
        return -a * np.log(x) - math.log(ZETA(a))
    if name == "negative_binomial":
        q = R["A"][CUR][name]["params"]
        return stats.nbinom.logpmf(x, q["r"], q["p"])
    raise ValueError(name)


def vuong(x, m1, m2):
    l1, l2 = per_obs_ll(x, m1), per_obs_ll(x, m2)
    d = l1 - l2
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return {"R": None, "p": None}
    Rv = d.sum() / (math.sqrt(n) * sd)
    p = float(math.erfc(abs(Rv) / math.sqrt(2)))
    return {"R": float(Rv), "p": p,
            "favours": m1 if Rv > 0 else m2}


R["A"] = {}
for tag, dfall, col in (("symptom", vsym, "TCM_symptom_id"),
                        ("syndrome", vsyn, "Syndrome_id"),
                        ("herb", vher, "Herb_id")):
    CUR = tag
    xd = counts_dedup(dfall, col).values.astype(int)
    xr = counts_raw(dfall, col).values.astype(int)
    fits_d, best_d = compare(xd)
    R["A"][tag] = fits_d
    R["A"][tag + "_best"] = best_d
    R["A"][tag + "_n"] = int(len(xd))
    fits_r, best_r = compare(xr)
    v = vuong(xd, "discrete_power_law", "negative_binomial")
    R["A"][tag + "_vuong_PL_vs_NB"] = v
    R["A"][tag + "_best_raw"] = best_r
    print("\n  [%s] n=%d (dedup)" % (tag, len(xd)))
    for n_, v_ in sorted(fits_d.items(), key=lambda kv: kv[1]["aic"]):
        print("     %-22s AIC=%10.1f  LL=%10.1f  k=%d" % (n_, v_["aic"], v_["loglik"], v_["k"]))
    print("     best(dedup)=%s | best(raw)=%s" % (best_d, best_r))
    print("     Vuong PL vs NB: R=%.3f p=%.3g -> favours %s"
          % (v["R"], v["p"], v["favours"]) if v["R"] is not None else "     Vuong: undefined")

# ============================================================ B. entropy bound + null
print("\n" + "=" * 80)
print("B  SINGLE-SYMPTOM ENTROPY BOUND WITH PATIENT-LEVEL PERMUTATION NULL")
both = sorted(set(vsym.visit_id) & set(vsyn.visit_id))
M = defaultdict(set); Y = defaultdict(set)
for v in both:
    M[v2p[v]] |= set(vsym.loc[vsym.visit_id == v, "TCM_symptom_id"])
    Y[v2p[v]] |= set(vsyn.loc[vsyn.visit_id == v, "Syndrome_id"])
pats = sorted(set(M) & set(Y))
pindex = {p: i for i, p in enumerate(pats)}
syms = sorted({s for p in pats for s in M[p]})
syns = sorted({y for p in pats for y in Y[p]})
si = {s: i for i, s in enumerate(syms)}; yi = {y: i for i, y in enumerate(syns)}
rows_, cols_ = [], []
for p in pats:
    i = pindex[p]
    for s in M[p]: rows_.append(i); cols_.append(si[s])
Msp = sparse.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(pats), len(syms)))
rows_, cols_ = [], []
for p in pats:
    i = pindex[p]
    for y in Y[p]: rows_.append(i); cols_.append(yi[y])
Ysp = sparse.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(pats), len(syns)))
N = len(pats)
print("  patients=%d symptoms=%d syndromes=%d" % (N, len(syms), len(syns)))

sc = np.asarray(Msp.sum(0)).ravel()          # symptom counts
yc = np.asarray(Ysp.sum(0)).ravel()          # syndrome counts
keep_s = sc >= 5                              # documented threshold
sy_ok = yc >= 30

def bound_from(Ym):
    C = (Ym.T @ Msp).toarray()               # syndromes x symptoms co-occurrence
    out = []
    for j in np.where(sy_ok)[0]:
        ny = yc[j]
        py = ny / N
        hy = -py * math.log2(py) - (1 - py) * math.log2(1 - py)
        if hy <= 0:
            continue
        c = C[j]
        ok = keep_s & (c >= 3)
        if not ok.any():
            continue
        a = c[ok]; ns = sc[ok]
        b = ns - a; cc = ny - a; dd = N - a - b - cc
        valid = (np.minimum(np.minimum(a, b), np.minimum(cc, dd)) > 0)
        if not valid.any():
            continue
        a, b, cc, dd = a[valid], b[valid], cc[valid], dd[valid]
        ns = a + b; nyv = a + cc
        ps = ns / N; pyv = nyv / N
        p11 = a / N; p10 = b / N; p01 = cc / N; p00 = dd / N
        mi = (p11 * np.log2(p11 / (ps * pyv)) + p10 * np.log2(p10 / (ps * (1 - pyv)))
              + p01 * np.log2(p01 / ((1 - ps) * pyv)) + p00 * np.log2(p00 / ((1 - ps) * (1 - pyv))))
        out.append(float(np.maximum(mi, 0).sum() / hy))
    return np.array(out)

obs = bound_from(Ysp)
rng = np.random.RandomState(42)
nulls = []
n_perm = int(os.environ.get("N_PERM", "60"))
for it in range(n_perm):
    idx = rng.permutation(N)
    nulls.append(float(np.median(bound_from(Ysp[idx]))))
obs_med = float(np.median(obs))
R["B"] = {"patients": N, "syndromes_evaluated": int(len(obs)),
          "observed_median_bound": obs_med,
          "null_median_mean": float(np.mean(nulls)),
          "null_median_p95": float(np.percentile(nulls, 95)),
          "excess_over_null": obs_med - float(np.mean(nulls)),
          "p_value": float((np.sum(np.array(nulls) >= obs_med) + 1) / (n_perm + 1)),
          "n_perm": n_perm,
          "thr_symptom_support": 5, "min_cell": 3}
print("  observed median bound = %.4f" % obs_med)
print("  null (labels permuted): mean=%.4f  p95=%.4f  p=%s"
      % (R["B"]["null_median_mean"], R["B"]["null_median_p95"], R["B"]["p_value"]))

# ============================================================ C. set-level transitions
print("\n" + "=" * 80)
print("C  SYNDROME-SET TRANSITIONS (Jaccard, consecutive labelled visits)")
pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int("".join(ch for ch in str(x) if ch.isdigit()) or -1))
seq = pvo.sort_values(["patient_id", "o"]).groupby("patient_id").visit_id.apply(list).to_dict()
jac, ident, n_tr = [], 0, 0
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        sa = set(vsyn.loc[vsyn.visit_id == a, "Syndrome_id"])
        sb = set(vsyn.loc[vsyn.visit_id == b, "Syndrome_id"])
        if not sa or not sb:
            continue
        n_tr += 1
        j = len(sa & sb) / len(sa | sb)
        jac.append(j)
        if sa == sb:
            ident += 1
jac = np.array(jac)
R["C"] = {"transitions": int(n_tr), "mean_jaccard": float(jac.mean()),
          "median_jaccard": float(np.median(jac)),
          "frac_identical_sets": float(ident / n_tr),
          "frac_jaccard_zero": float((jac == 0).mean()),
          "frac_jaccard_ge_0.5": float((jac >= .5).mean())}
print("  transitions=%d mean Jaccard=%.3f median=%.3f | identical sets %.1f%% | disjoint %.1f%%"
      % (n_tr, jac.mean(), np.median(jac), 100 * ident / n_tr, 100 * (jac == 0).mean()))

# ============================================================ D. clustering stability
print("\n" + "=" * 80)
print("D  PATIENT CLUSTERING: silhouette k=2..6 + bootstrap stability")
rows = []
for p, vs in seq.items():
    if len(vs) < 2:
        continue
    rows.append((float(np.mean([vher[vher.visit_id == v].Herb_id.nunique() for v in vs])),
                 float(np.mean([vsym[vsym.visit_id == v].TCM_symptom_id.nunique() for v in vs]))))
X = StandardScaler().fit_transform(np.array(rows))
sil = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    sil[k] = float(silhouette_score(X, km.labels_))
best_k = max(sil, key=sil.get)
km2 = KMeans(n_clusters=2, n_init=20, random_state=42).fit(X)
rng2 = np.random.RandomState(0)
aris = []
for _ in range(20):
    idx = rng2.choice(len(X), size=int(.8 * len(X)), replace=False)
    a = KMeans(n_clusters=2, n_init=10, random_state=1).fit(X[idx]).labels_
    b = KMeans(n_clusters=2, n_init=10, random_state=2).fit(X[idx]).labels_
    aris.append(adjusted_rand_score(a, b))
R["D"] = {"patients": int(len(X)), "silhouette_by_k": {str(k): round(v, 4) for k, v in sil.items()},
          "best_k": int(best_k), "best_silhouette": round(sil[best_k], 4),
          "k2_bootstrap_ari_mean": float(np.mean(aris)),
          "k2_bootstrap_ari_min": float(np.min(aris)),
          "cluster_sizes_k2": [int((km2.labels_ == c).sum()) for c in (0, 1)]}
print("  silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print("  best k=%d (%.3f) | k=2 bootstrap ARI mean=%.3f min=%.3f"
      % (best_k, sil[best_k], np.mean(aris), np.min(aris)))

# ============================================================ E. Gini / entropy: no CI
print("\n" + "=" * 80)
print("E  GINI / ENTROPY (deterministic; bootstrap CI is NOT reported)")
print("   reason: resampling visits with replacement reduces the number of distinct visits")
print("   to ~0.63n, which systematically thins the tail and biases Gini downward.")
def gini(c):
    x = np.sort(np.asarray(c, float)); n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def ent(c):
    pr = np.asarray(c, float); pr = pr / pr.sum()
    return float(-(pr * np.log2(pr)).sum())


R["E"] = {"note": "point statistics are deterministic functions of the full data; "
                  "visit-level bootstrap is invalid here because distinct-visit count shrinks",
          "values": {}}
for tag, df_, col in (("symptom", vsym, "TCM_symptom_id"),
                      ("syndrome", vsyn, "Syndrome_id"),
                      ("herb", vher, "Herb_id")):
    cd = counts_dedup(df_, col).values.astype(float)
    cr = df_.groupby(col).size().values.astype(float)
    R["E"]["values"][tag] = {
        "gini_dedup": gini(cd), "gini_raw": gini(cr),
        "entropy_bits_dedup": ent(cd), "entropy_bits_raw": ent(cr),
        "n_items_dedup": int(len(cd)), "n_items_raw": int(len(cr))}
    print("   %-9s Gini dedup %.4f / raw %.4f | H dedup %.3f / raw %.3f"
          % (tag, gini(cd), gini(cr), ent(cd), ent(cr)))

json.dump(R, open(os.path.join(A, "e1_e2_distribution.json"), "w"), ensure_ascii=False, indent=1)
print("\nwrote e1_e2_distribution.json")
