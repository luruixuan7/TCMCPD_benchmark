#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B4-delta: reframed next-prescription task.
Absolute next-basket prediction is dominated by prescription persistence (last-basket
mAP 0.654 vs 0.51-0.56 for all trained models), so we predict the CHANGE instead:
  target = herbs ADDED at t+1 relative to t
Baselines: empty (floor), popularity-of-additions, kNN, LightGBM.
"""
import os, sys, json, re, time, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)
ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
OUT = os.path.join(ROOT, "bench")
os.makedirs(OUT, exist_ok=True)

vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))

V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)
v2p = dict(zip(pv.visit_id, pv.patient_id))

pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
seq = pvo.sort_values(["patient_id", "o"]).groupby("patient_id").visit_id.apply(list).to_dict()

pairs = []
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        if a in V2H and b in V2H:
            pairs.append((p, a, b))
print("pairs:", len(pairs))
n_add = [len(V2H[b] - V2H[a]) for _, a, b in pairs]
n_rem = [len(V2H[a] - V2H[b]) for _, a, b in pairs]
inter = [len(V2H[a] & V2H[b]) for _, a, b in pairs]
print("mean |added|=%.2f |removed|=%.2f |kept|=%.2f" % (np.mean(n_add), np.mean(n_rem), np.mean(inter)))
print("pairs with >=1 addition: %.1f%%" % (100 * np.mean([x > 0 for x in n_add])))

# patient-level split 70/15/15
pats = sorted({p for p, _, _ in pairs}); rng = np.random.RandomState(42); rng.shuffle(pats)
n = len(pats)
trp, vap, tep = set(pats[:int(.7 * n)]), set(pats[int(.7 * n):int(.85 * n)]), set(pats[int(.85 * n):])
def sp(p): return "train" if p in trp else "val" if p in vap else "test"
tr = [x for x in pairs if sp(x[0]) == "train"]
va = [x for x in pairs if sp(x[0]) == "val"]
te = [x for x in pairs if sp(x[0]) == "test"]
print("train/val/test = %d/%d/%d" % (len(tr), len(va), len(te)))

added = {i: (V2H[b] - V2H[a]) for i, (_, a, b) in enumerate(pairs)}
# vocabulary: herbs appearing as an ADDITION in train at least 5 times
ac = Counter(h for i, (_, a, b) in enumerate(tr) for h in (V2H[b] - V2H[a]))
vocab = [h for h, c in ac.most_common() if c >= 5]
hv = {h: i for i, h in enumerate(vocab)}
print("addition-vocab (>=5 in train): %d" % len(vocab))

xc = Counter(s for _, a, _ in tr for s in V2M.get(a, ()))
sym_vocab = [s for s, c in xc.most_common() if c >= 5]
hc = Counter(h for _, a, _ in tr for h in V2H.get(a, ()))
herb_vocab = [h for h, c in hc.most_common() if c >= 5]
print("symptom vocab=%d | current-herb vocab=%d" % (len(sym_vocab), len(herb_vocab)))

def build_x(prs):
    xi = {s: i for i, s in enumerate(sym_vocab)}
    hi = {h: i for i, h in enumerate(herb_vocab)}
    M = np.zeros((len(prs), len(sym_vocab) + len(herb_vocab)), np.float32)
    for r, (_, a, _) in enumerate(prs):
        for s in V2M.get(a, ()):
            j = xi.get(s)
            if j is not None: M[r, j] = 1.0
        for h in V2H.get(a, ()):
            j = hi.get(h)
            if j is not None: M[r, len(sym_vocab) + j] = 1.0
    return M

def build_y(prs):
    M = np.zeros((len(prs), len(vocab)), np.float32)
    for r, (_, a, b) in enumerate(prs):
        for h in (V2H.get(b, set()) - V2H.get(a, set())):
            j = hv.get(h)
            if j is not None: M[r, j] = 1.0
    return M

Xtr, Xva, Xte = build_x(tr), build_x(va), build_x(te)
Ytr, Yva, Yte = build_y(tr), build_y(va), build_y(te)
print("X shape %s | Y shape %s | positives/sample (test) %.2f"
      % (Xtr.shape, Ytr.shape, Yte.sum(axis=1).mean()))

def eval_at(S, Y, k):
    order = np.argsort(-S, axis=1)[:, :k]
    hit = np.take_along_axis(Y, order, axis=1).sum(axis=1)
    prec = hit / k
    rec = np.array([hit[i] / max(1.0, Y[i].sum()) for i in range(len(Y))])
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    any_add = Y.sum(axis=1) > 0
    return {"P@%d" % k: float(prec.mean()), "R@%d" % k: float(rec[any_add].mean()),
            "F1@%d" % k: float(f1.mean())}

res = {}
print("\nbaselines (target = ADDED herbs)")
# 1. empty prediction (floor)
for k in (5, 10):
    res["empty"] = {"P": 0.0, "R": 0.0, "F1": 0.0}
print("   %-22s P@5=0.000 R@5=0.000 F1@5=0.000  (floor: predicts nothing)")

# 2. popularity of additions
pop = np.array([ac.get(h, 0) for h in vocab], float)
S_pop = np.tile(pop, (len(Yte), 1))
for k in (5, 10):
    r = eval_at(S_pop, Yte, k); res.setdefault("popularity", {}).update(r)
print("   %-22s P@5=%.3f R@5=%.3f F1@5=%.3f | P@10=%.3f R@10=%.3f F1@10=%.3f"
      % ("popularity(additions)", res["popularity"]["P@5"], res["popularity"]["R@5"],
         res["popularity"]["F1@5"], res["popularity"]["P@10"], res["popularity"]["R@10"],
         res["popularity"]["F1@10"]))

# 3. kNN
for k_nb in (10, 25):
    nn = NearestNeighbors(n_neighbors=k_nb, metric="cosine").fit(Xtr)
    dist, ind = nn.kneighbors(Xte)
    S = np.zeros((len(Xte), len(vocab)), np.float32)
    for r in range(len(Xte)):
        w = 1.0 / (dist[r] + 1e-6)
        for jj, ww in zip(ind[r], w):
            S[r] += ww * Ytr[jj]
    rr = {}
    for k in (5, 10):
        rr.update(eval_at(S, Yte, k))
    res["knn%d" % k_nb] = rr
    print("   %-22s P@5=%.3f R@5=%.3f F1@5=%.3f | P@10=%.3f R@10=%.3f F1@10=%.3f"
          % ("knn%d" % k_nb, rr["P@5"], rr["R@5"], rr["F1@5"], rr["P@10"], rr["R@10"], rr["F1@10"]))

# 4. LightGBM per label
t0 = time.time()
Sva = np.zeros_like(Yva, dtype=np.float32); Ste = np.zeros_like(Yte, dtype=np.float32)
for j in range(len(vocab)):
    yj = Ytr[:, j]
    if yj.sum() < 3:
        Sva[:, j] = yj.mean(); Ste[:, j] = yj.mean(); continue
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=.08, num_leaves=31,
                           subsample=.9, colsample_bytree=.8, verbose=-1,
                           n_jobs=8, random_state=42)
    m.fit(Xtr, yj)
    Sva[:, j] = m.predict_proba(Xva)[:, 1]
    Ste[:, j] = m.predict_proba(Xte)[:, 1]
rr = {}
for k in (5, 10):
    rr.update(eval_at(Ste, Yte, k))
res["lightgbm"] = rr
print("   %-22s P@5=%.3f R@5=%.3f F1@5=%.3f | P@10=%.3f R@10=%.3f F1@10=%.3f  (%.0fs)"
      % ("lightgbm", rr["P@5"], rr["R@5"], rr["F1@5"], rr["P@10"], rr["R@10"], rr["F1@10"],
         time.time() - t0))

out = {"task": "B4-delta",
       "definition": "predict herbs ADDED at t+1 relative to t (not the absolute basket)",
       "why": "absolute next-basket is dominated by last-basket persistence "
              "(mAP 0.654 vs <=0.563 for trained models)",
       "data": {"pairs": len(pairs), "train": len(tr), "val": len(va), "test": len(te),
                "mean_added": float(np.mean(n_add)), "mean_removed": float(np.mean(n_rem)),
                "frac_with_addition": float(np.mean([x > 0 for x in n_add])),
                "addition_vocab": len(vocab)},
       "results": res,
       "protocol": "patient-level 70/15/15 seed=42; target=added-herb set"}
json.dump(out, open(os.path.join(OUT, "results_B4delta.json"), "w"),
          ensure_ascii=False, indent=1)
print("\nsaved results_B4delta.json")
