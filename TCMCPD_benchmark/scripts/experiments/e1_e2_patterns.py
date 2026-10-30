#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three pattern-discovery analyses to lift Part A from description to regularity.

P1 LABEL-PRESENCE LAW: how predictable is "this visit carries a syndrome label"?
   logistic model on completeness/context features -> AUC, propensity overlap, and the
   practical consequence (the unlabelled visits are not a random sample).

P2 LEARNABILITY LAW: what determines whether a syndrome is predictable at all?
   per-label AP ~ log(train frequency) + graph degree + co-occurrence entropy;
   report R^2 of frequency-only vs full model.

P3 BACKGROUND-VS-SPECIFIC STRUCTURE: is the enrichment distribution bimodal?
   1- vs 2-component Gaussian mixture on log enrichment, BIC comparison.

Writes analysis/pattern_discovery.json
"""
import os, json, math, re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
A = os.path.join(ROOT, "analysis")
B = os.path.join(ROOT, "bench")
R = {}

vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
herb = pd.read_csv(os.path.join(D, "entities/herb.csv"))
syn = pd.read_csv(os.path.join(D, "entities/syndrome.csv"))
v2p = dict(zip(pv.visit_id, pv.patient_id))

V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)

# ============================================================ P1 label presence
print("=" * 80); print("P1  LABEL-PRESENCE LAW")
YEAR = re.compile(r"(\d{4})\s*年")
rows = []
pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
pvo = pvo.sort_values(["patient_id", "o"])
seq = pvo.groupby("patient_id").visit_id.apply(list).to_dict()
v2idx = {}
for p, vs in seq.items():
    for i, v in enumerate(vs):
        v2idx[v] = (i, len(vs))
for _, r in vis.iterrows():
    v = r.visitid
    txt = r.content if isinstance(r.content, str) else ""
    m = YEAR.search(txt)
    yr = int(m.group(1)) if m else np.nan
    idx, tot = v2idx.get(v, (0, 1))
    rows.append({
        "v": v, "has_syn": int(v in V2Y),
        "tongue": int(isinstance(r.tongue, str) and r.tongue.strip() != ""),
        "pulse": int(isinstance(r.pulse, str) and r.pulse.strip() != ""),
        "treat": int(isinstance(r.treatment, str) and r.treatment.strip() != ""),
        "loglen": math.log1p(len(txt)), "n_sym": len(V2M.get(v, ())),
        "n_herb": len(V2H.get(v, ())), "year_known": int(not np.isnan(yr)),
        "year": 0 if np.isnan(yr) else yr, "visit_idx": idx, "pat_n": tot,
        "pat": v2p.get(v)})
DF = pd.DataFrame(rows)
FEATS = ["tongue", "pulse", "treat", "loglen", "n_sym", "n_herb", "year_known", "visit_idx", "pat_n"]
X = DF[FEATS].values; y = DF["has_syn"].values; g = DF["pat"].values
gkf = GroupKFold(n_splits=5)
aucs, prop = [], np.zeros(len(DF))
for tr, te in gkf.split(X, y, g):
    m = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    prop[te] = p
    aucs.append(roc_auc_score(y[te], p))
AUC = float(np.mean(aucs))
m_full = LogisticRegression(max_iter=1000).fit(X, y)
coef = {f: float(c) for f, c in zip(FEATS, m_full.coef_[0])}
pl, pu = prop[y == 1], prop[y == 0]
lo, hi = np.percentile(pl, 5), np.percentile(pl, 95)
inside = float(np.mean((pu >= lo) & (pu <= hi)))
R["P1"] = {"n_visits": int(len(DF)), "n_labelled": int(y.sum()),
           "auc_cv": AUC, "auc_folds": [float(a) for a in aucs],
           "coef": coef,
           "propensity_labelled_mean": float(pl.mean()),
           "propensity_unlabelled_mean": float(pu.mean()),
           "labelled_p5_p95": [float(lo), float(hi)],
           "frac_unlabelled_inside_labelled_range": inside,
           "mannwhitney_U": float(stats.mannwhitneyu(pl, pu, alternative="two-sided").statistic),
           "mannwhitney_p": float(stats.mannwhitneyu(pl, pu, alternative="two-sided").pvalue),
           "prob_labelled_gt_unlabelled": float(stats.mannwhitneyu(pl, pu).statistic / (len(pl) * len(pu)))}
# --- figure payload: propensity distributions on a shared bin grid
edges = np.linspace(0.0, 1.0, 41)
hl, _ = np.histogram(pl, bins=edges, density=True)
hu, _ = np.histogram(pu, bins=edges, density=True)
R["P1"]["figure"] = {"bin_edges": [float(e) for e in edges],
                     "hist_labelled": [float(v) for v in hl],
                     "hist_unlabelled": [float(v) for v in hu],
                     "n_labelled": int(len(pl)), "n_unlabelled": int(len(pu))}
print("  AUC (5-fold by patient) = %.3f  [folds %s]" % (AUC, [round(a, 3) for a in aucs]))
print("  propensity: labelled %.3f vs unlabelled %.3f" % (pl.mean(), pu.mean()))
print("  unlabelled visits inside labelled P5-P95 range: %.1f%%" % (100 * inside))
print("  top coefficients:", {k: round(v, 3) for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:5]})

# ============================================================ P2 learnability law
print("=" * 80); print("P2  LEARNABILITY LAW")


def label_stats(task):
    c = np.load(os.path.join(B, "cache_%s.npz" % task))
    preds = {}
    for f in os.listdir(B):
        if f.startswith("preds_%s_" % task) and f.endswith(".npz"):
            z = np.load(os.path.join(B, f))
            for k in z.files:
                if k.startswith("te_"):
                    preds[k[3:]] = z[k]
    return c, preds


def per_label_ap(S, Y, k=10):
    order = np.argsort(-S, axis=1)[:, :k]
    aps = []
    for j in range(Y.shape[1]):
        npos = Y[:, j].sum()
        if npos == 0:
            aps.append(np.nan); continue
        hit = 0
        for i in range(len(Y)):
            if Y[i, j] > 0 and j in order[i]:
                hit += 1
        aps.append(hit / npos)
    return np.array(aps)


R["P2"] = {}
for task, name in (("B1", "label-graph+ASL"), ("B2", "label-graph(ML-GCN)")):
    c, preds = label_stats(task)
    Ytr = c["Y"][c["itr"]]; Yte = c["Y"][c["ite"]]
    S = preds[name]
    ap = per_label_ap(S, Yte)
    freq_tr = Ytr.sum(0).astype(float)
    # graph degree from training co-occurrence
    C = Ytr.T @ Ytr; d = np.diag(C).copy(); d[d == 0] = 1
    A_ = C / np.sqrt(np.outer(d, d)); np.fill_diagonal(A_, 0)
    deg = (A_ > 0.05).sum(0).astype(float)
    # co-occurrence entropy
    pent = []
    for j in range(Ytr.shape[1]):
        row = A_[j] / max(A_[j].sum(), 1e-9)
        pent.append(-(row[row > 0] * np.log2(row[row > 0])).sum())
    pent = np.array(pent)
    ok = ~np.isnan(ap) & (freq_tr > 0)
    yv = ap[ok]
    X1 = np.log1p(freq_tr[ok])
    Xfull = np.column_stack([np.log1p(freq_tr[ok]), deg[ok], pent[ok]])
    def r2(Xm):
        Xc = np.column_stack([np.ones(len(Xm)), Xm])
        beta, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
        pred = Xc @ beta
        ss_res = float(((yv - pred) ** 2).sum()); ss_tot = float(((yv - yv.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot, beta
    r2_freq, beta_freq = r2(X1.reshape(-1, 1))
    r2_full, beta = r2(Xfull)
    R["P2"][task] = {"n_labels": int(ok.sum()),
                     "r2_log_frequency_only": float(r2_freq),
                     "r2_full": float(r2_full),
                     "beta": [float(b) for b in beta],
                     "spearman_ap_freq": float(stats.spearmanr(yv, freq_tr[ok]).statistic),
                     "spearman_ap_degree": float(stats.spearmanr(yv, deg[ok]).statistic)}
    # --- figure payload: per-label scatter (log training frequency vs AP@10 proxy)
    # the plotted line is the FREQUENCY-ONLY fit (that is the R^2 quoted next to it)
    R["P2"][task]["figure"] = {
        "log_freq": [float(v) for v in np.log10(freq_tr[ok])],
        "log1p_freq": [float(v) for v in np.log1p(freq_tr[ok])],
        "ap": [float(v) for v in yv],
        "slope_freq_only": float(beta_freq[1]),
        "intercept_freq_only": float(beta_freq[0]),
        "slope_full_model": float(beta[1]),
        "intercept_full_model": float(beta[0])}
    print("  [%s] n=%d | R2(logfreq)=%.3f | R2(logfreq+degree+entropy)=%.3f | rho(freq)=%.3f"
          % (task, ok.sum(), r2_freq, r2_full, R["P2"][task]["spearman_ap_freq"]))

# ============================================================ P3 background vs specific
print("=" * 80); print("P3  BACKGROUND-VS-SPECIFIC STRUCTURE")
n_rx = len([v for v in V2H if V2H[v]])
herb_visits = Counter()
for v in V2H:
    for h in set(V2H[v]): herb_visits[h] += 1
base_p = {h: c / n_rx for h, c in herb_visits.items()}
sy_visits = Counter()
for v, ys in V2Y.items():
    for y in ys: sy_visits[y] += 1
cells = []
for y, cy in sy_visits.most_common(40):
    vset = {v for v, ys in V2Y.items() if y in ys}
    tot = len([v for v in vset if V2H.get(v)])
    if tot < 50: continue
    hc = Counter()
    for v in vset:
        for h in set(V2H.get(v, [])): hc[h] += 1
    for h, n in hc.items():
        if n < 10: continue
        pb = base_p.get(h, 0)
        if pb <= 0: continue
        cells.append({"syndrome": int(y), "herb": int(h), "n": int(n), "tot": tot,
                      "p_cond": n / tot, "p_base": pb, "enrich": (n / tot) / pb})
CD = pd.DataFrame(cells)
# ---- test A: ALL cells (proper test of background vs specific)
allc = CD[CD.n >= 10].copy()
le = np.log(allc.enrich.clip(lower=1e-3).values).reshape(-1, 1)
gm1 = GaussianMixture(1, random_state=0).fit(le)
gm2 = GaussianMixture(2, random_state=0, n_init=20).fit(le)
bic1, bic2 = gm1.bic(le), gm2.bic(le)
order = np.argsort(gm2.means_.ravel())
means = np.exp(gm2.means_.ravel()[order]); weights = gm2.weights_[order]
# ---- descriptive B: best herb per syndrome (selection view, NOT used for the test)
best = CD.sort_values("enrich", ascending=False).groupby("syndrome").head(1)
R["P3"] = {"test_unit": "all (syndrome,herb) cells with co-occurrence>=10",
           "n_cells": int(len(allc)),
           "bic_1comp": float(bic1), "bic_2comp": float(bic2),
           "delta_bic": float(bic1 - bic2),
           "component_means_enrichment": [float(m) for m in means],
           "component_weights": [float(w) for w in weights],
           "share_cells_enrichment_lt_1_5": float((allc.enrich < 1.5).mean()),
           "share_cells_enrichment_ge_2": float((allc.enrich >= 2).mean()),
           "best_per_syndrome_n": int(len(best)),
           "best_per_syndrome_median": float(best.enrich.median()),
           "best_per_syndrome_p90": float(np.percentile(best.enrich, 90))}
print("  ALL cells: n=%d | GMM BIC 1-comp %.1f vs 2-comp %.1f (delta %.1f) -> %s"
      % (len(allc), bic1, bic2, bic1 - bic2, "two groups" if bic2 < bic1 else "single group"))
print("  2-component means (enrichment): %.2f / %.2f ; weights %.2f / %.2f"
      % (means[0], means[1], weights[0], weights[1]))
print("  share of cells with enrichment <1.5: %.1f%% ; >=2: %.1f%%"
      % (100 * (allc.enrich < 1.5).mean(), 100 * (allc.enrich >= 2).mean()))
print("  [descriptive] best herb per syndrome: n=%d median %.2f p90 %.2f"
      % (len(best), best.enrich.median(), np.percentile(best.enrich, 90)))

json.dump(R, open(os.path.join(A, "pattern_discovery.json"), "w"), ensure_ascii=False, indent=1)
print("\nwrote pattern_discovery.json")
