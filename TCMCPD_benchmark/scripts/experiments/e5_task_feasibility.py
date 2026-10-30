#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TASK AUDIT: do B1/B2/B4/B6 survive the analysis findings?
Checks:
  B4a consecutive-prescription similarity (is there temporal persistence?)
  B4b does HISTORY add value over current symptoms?  (else B4 == B2)
  B6  explicit-date-only feasibility + is dT predictable at all?
  B1/B2 sample counts under patient-level split
"""
import os, sys, json, math, re, datetime, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)
ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
T = os.path.join(ROOT, "analysis", "tables")
R = {}

vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
v2p = dict(zip(pv.visit_id, pv.patient_id))

# ordered visits per patient
pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
pvo = pvo.sort_values(["patient_id", "o"])
seq = pvo.groupby("patient_id").visit_id.apply(list).to_dict()

V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)

# ---------- B4a: consecutive prescription similarity ----------
print("=" * 78)
print("B4a  CONSECUTIVE PRESCRIPTION SIMILARITY (temporal persistence?)")
jac = []
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        if a in V2H and b in V2H:
            A, B = V2H[a], V2H[b]
            jac.append(len(A & B) / len(A | B))
jac = np.array(jac)
print("  pairs=%d  Jaccard(t,t+1): mean=%.3f median=%.3f P25=%.3f P75=%.3f"
      % (len(jac), jac.mean(), np.median(jac), np.percentile(jac, 25), np.percentile(jac, 75)))
print("  fraction with Jaccard>0.5: %.1f%% | >0.8: %.1f%%"
      % (100 * (jac > .5).mean(), 100 * (jac > .8).mean()))
# null: random pairs from different patients
rnd = []
allh = [h for h in V2H.values() if h]
for _ in range(5000):
    A, B = allh[np.random.randint(len(allh))], allh[np.random.randint(len(allh))]
    if A and B:
        rnd.append(len(A & B) / len(A | B))
print("  null (random visit pairs): mean Jaccard=%.3f" % np.mean(rnd))
print("  -> temporal persistence is %s" % ("STRONG" if jac.mean() > 2 * np.mean(rnd) else "WEAK"))
R["B4a"] = {"n_pairs": int(len(jac)), "mean_jaccard": float(jac.mean()),
            "median": float(np.median(jac)), "null_mean": float(np.mean(rnd)),
            "frac_gt_0.5": float((jac > .5).mean())}

# ---------- B4b: does history beat current symptoms? ----------
print("\n" + "=" * 78)
print("B4b  DOES HISTORY ADD VALUE BEYOND CURRENT SYMPTOMS?")


def vec(sets, vocab):
    idx = {x: i for i, x in enumerate(vocab)}
    M = np.zeros((len(sets), len(vocab)), np.float32)
    for r, st in enumerate(sets):
        for x in st:
            j = idx.get(x)
            if j is not None:
                M[r, j] = 1.0
    return M, idx


pairs = []
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        if b in V2H and (a in V2M or a in V2H):
            pairs.append((p, a, b))
print("  usable pairs:", len(pairs))

# patient-level split 80/20
pats = sorted({p for p, _, _ in pairs})
np.random.shuffle(pats)
tr_p = set(pats[:int(.8 * len(pats))])
tr = [x for x in pairs if x[0] in tr_p]
te = [x for x in pairs if x[0] not in tr_p]
print("  train pairs=%d (%d patients) | test pairs=%d (%d patients)"
      % (len(tr), len(tr_p), len(te), len(pats) - len(tr_p)))

sympt_vocab = [s for s, c in Counter(s for v in V2M for s in V2M[v]).most_common() if c >= 5]
herb_vocab = [h for h, c in Counter(h for v in V2H for h in V2H[v]).most_common() if c >= 5]
print("  symptom vocab=%d herb vocab=%d" % (len(sympt_vocab), len(herb_vocab)))

Xtr_sym, _ = vec([V2M.get(a, set()) for _, a, _ in tr], sympt_vocab)
Xte_sym, _ = vec([V2M.get(a, set()) for _, a, _ in te], sympt_vocab)
Xtr_herb, _ = vec([V2H.get(a, set()) for _, a, _ in tr], herb_vocab)
Xte_herb, _ = vec([V2H.get(a, set()) for _, a, _ in te], herb_vocab)


def eval_sets(pred_sets, true_sets, ks=(5, 10)):
    out = {}
    for k in ks:
        rec = [len(set(p[:k]) & t) / len(t) if t else 0 for p, t in zip(pred_sets, true_sets)]
        out["R@%d" % k] = float(np.mean(rec))
    out["Jaccard@10"] = float(np.mean([
        len(set(p[:10]) & t) / len(set(p[:10]) | t) if (t or p) else 0
        for p, t in zip(pred_sets, true_sets)]))
    return out


true_te = [V2H[b] for _, _, b in te]

# (1) popularity
pop = [h for h, _ in Counter(h for v in V2H for h in V2H[v]).most_common()]
print("\n  [popularity]           ", eval_sets([pop] * len(te), true_te))

# (2) last basket copy
last_pred = []
for _, a, b in te:
    prev = list(V2H.get(a, set()))
    prev.sort(key=lambda h: -Counter(h for v in V2H for h in V2H[v]).get(h, 0))
    last_pred.append(prev)
print("  [last-basket copy]     ", eval_sets(last_pred, true_te))


def knn_pred(Xtr, Xte, k=25):
    if Xtr.sum() == 0:
        return [pop[:10]] * len(Xte)
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr)), metric="cosine").fit(Xtr)
    dist, ind = nn.kneighbors(Xte)
    out = []
    for row in ind:
        cnt = Counter()
        for j in row:
            for h in V2H.get(tr[j][2], set()):
                cnt[h] += 1
        out.append([h for h, _ in cnt.most_common(10)])
    return out


print("  [kNN symptoms->next]   ", eval_sets(knn_pred(Xtr_sym, Xte_sym), true_te))
print("  [kNN cur-herbs->next]  ", eval_sets(knn_pred(Xtr_herb, Xte_herb), true_te))
Xtr_b = np.hstack([Xtr_sym, Xtr_herb]); Xte_b = np.hstack([Xte_sym, Xte_herb])
print("  [kNN symptoms+herbs]   ", eval_sets(knn_pred(Xtr_b, Xte_b), true_te))

R["B4b"] = {"n_train": len(tr), "n_test": len(te),
            "popularity": eval_sets([pop] * len(te), true_te),
            "last_basket": eval_sets(last_pred, true_te),
            "knn_symptom": eval_sets(knn_pred(Xtr_sym, Xte_sym), true_te),
            "knn_herb": eval_sets(knn_pred(Xtr_herb, Xte_herb), true_te),
            "knn_both": eval_sets(knn_pred(Xtr_b, Xte_b), true_te)}

# ---------- B6: explicit-date feasibility + predictability ----------
print("\n" + "=" * 78)
print("B6  INTERVAL PREDICTION: EXPLICIT-DATE SUBSET")
FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
vmap = vis.set_index("visitid")
rows = []
for p, vs in seq.items():
    prev, prev_ex = None, None
    for vid in vs:
        c = vmap.at[vid, "content"] if vid in vmap.index else ""
        c = c if isinstance(c, str) else ""
        m = FULL.search(c)
        d = None
        if m:
            try:
                d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                d = None
        if d and prev and prev_ex:
            dt = (d - prev).days
            if 0 < dt < 3650:
                s_ = len(V2M.get(prev_v, set())); h_ = len(V2H.get(prev_v, set()))
                y_ = len(V2Y.get(prev_v, set()))
                rows.append({"dt": dt, "n_sym": s_, "n_herb": h_, "n_syn": y_,
                             "patient": p, "order": len(rows)})
        if d:
            prev, prev_ex, prev_v = d, True, vid
DT = pd.DataFrame(rows)
print("  strict explicit-both-dates intervals: %d (patients %d)"
      % (len(DT), DT.patient.nunique()))
print("  dt: median=%.0f mean=%.1f log1p mean=%.2f" % (DT.dt.median(), DT.dt.mean(),
                                                       np.log1p(DT.dt).mean()))
# predictability
y = np.log1p(DT.dt.values)
X = DT[["n_sym", "n_herb", "n_syn"]].values
base = np.expm1(np.full(len(y), y.mean()))
print("  baseline (global mean) MAE = %.2f days" % np.mean(np.abs(np.expm1(y) - base)))
gkf = GroupKFold(n_splits=5)
maes, maes0 = [], []
for tr_i, te_i in gkf.split(X, y, groups=DT.patient.values):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=31,
                          verbose=-1, random_state=42)
    m.fit(X[tr_i], y[tr_i])
    pred = np.expm1(m.predict(X[te_i]))
    maes.append(np.mean(np.abs(pred - np.expm1(y[te_i]))))
    med = np.expm1(np.median(y[tr_i]))
    maes0.append(np.mean(np.abs(med - np.expm1(y[te_i]))))
print("  median baseline MAE = %.2f | LightGBM MAE = %.2f (5-fold by patient)"
      % (np.mean(maes0), np.mean(maes)))
print("  improvement: %.1f%%" % (100 * (1 - np.mean(maes) / np.mean(maes0))))
for c in ("n_sym", "n_herb", "n_syn"):
    rho = stats.spearmanr(DT[c], DT.dt)
    print("  spearman(%s, dt) = %+.3f (p=%.2g)" % (c, rho.statistic, rho.pvalue))
R["B6"] = {"n_intervals": int(len(DT)), "n_patients": int(DT.patient.nunique()),
           "median_dt": float(DT.dt.median()),
           "mae_median_baseline": float(np.mean(maes0)),
           "mae_lgbm": float(np.mean(maes)),
           "improvement": float(1 - np.mean(maes) / np.mean(maes0))}

json.dump(R, open(os.path.join(ROOT, "analysis", "results_task_audit.json"), "w"),
          ensure_ascii=False, indent=1, default=str)
print("\ntask audit done")
