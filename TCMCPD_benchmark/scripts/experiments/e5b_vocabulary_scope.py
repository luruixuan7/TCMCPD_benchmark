#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5b: what a larger herb label space costs.

Check C1 of the paper claims that benchmarking on the full herb dictionary would spend
capacity on labels that never appear in practice. This script tests that claim: the same
visits, features and model are used, and only the minimum training frequency that defines the
label space is varied (1, 5, 20). For each space we report macro-F1 over the whole space and
macro-F1 restricted to the labels that all three spaces share, so the comparison is fair.

Writes results/appendix/vocabulary_scope.json
"""
import sys
sys.dont_write_bytecode = True

import json, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "TCMCPD_TCM_Case_Records_v1.0"
OUT = ROOT / "results" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

pv = pd.read_csv(D / "relationships/patient_visit.csv")
vsym = pd.read_csv(D / "relationships/visit_symptom.csv")
vherb = pd.read_csv(D / "relationships/visit_herb.csv")
v2p = dict(zip(pv.visit_id, pv.patient_id))
V2M, V2H = defaultdict(set), defaultdict(set)
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values:
    V2M[v].add(s)
for v, h in vherb[["visit_id", "Herb_id"]].values:
    V2H[v].add(h)
rows = [v for v in V2M if v in V2H]
idx = {v: i for i, v in enumerate(rows)}

xv = [s for s, c in Counter(s for v in rows for s in V2M[v]).most_common() if c >= 5]
xi = {s: i for i, s in enumerate(xv)}
X = np.zeros((len(rows), len(xv)), np.float32)
for r, v in enumerate(rows):
    for s in V2M[v]:
        j = xi.get(s)
        if j is not None:
            X[r, j] = 1.0
# episode index as one extra context column (comparable across label spaces)
order = {}
for r, v in enumerate(rows):
    order.setdefault(v2p[v], []).append(r)
eps = np.zeros(len(rows), np.float32)
for p, ii in order.items():
    for k, r in enumerate(sorted(ii)):
        eps[r] = k
X = np.hstack([X, (eps / max(1.0, eps.max()))[:, None]])

pats = sorted({v2p[v] for v in rows})
pl = pats[:]
np.random.RandomState(42).shuffle(pl)
n = len(pl)
trp, tep = set(pl[:int(.7 * n)]), set(pl[int(.85 * n):])
itr = np.array([idx[v] for v in rows if v2p[v] in trp])
ite = np.array([idx[v] for v in rows if v2p[v] in tep])
print("visits %d (train %d, test %d) | symptom features %d"
      % (len(rows), len(itr), len(ite), X.shape[1]), flush=True)


def ranks(S, Yt, ks=(1, 5, 10)):
    order_ = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(Yt, order_[:, :k], axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Yt)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Yt.sum()))
    rel = np.take_along_axis(Yt, order_[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1)
    ok = den > 0
    out["mAP@10"] = float(((prec * rel).sum(1)[ok] / den[ok]).mean())
    return out


train_counts = Counter(h for i in itr for h in V2H[rows[i]])
spaces = {}
for mc in (1, 5, 20):
    labels = [h for h, c in train_counts.most_common() if c >= mc]
    li = {h: i for i, h in enumerate(labels)}
    Y = np.zeros((len(rows), len(labels)), np.float32)
    for r, v in enumerate(rows):
        for h in V2H[v]:
            j = li.get(h)
            if j is not None:
                Y[r, j] = 1.0
    spaces[mc] = {"labels": labels, "li": li, "Y": Y}

common = set(spaces[20]["labels"])
res = {"data": {"visits": len(rows), "train": int(len(itr)), "test": int(len(ite)),
                "features": int(X.shape[1])}, "spaces": {}
       , "common_reference_labels": len(common)}

for mc in (1, 5, 20):
    labels, Y = spaces[mc]["labels"], spaces[mc]["Y"]
    model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True,
                          n_iter_no_change=10, validation_fraction=0.12, random_state=42)
    model.fit(X[itr], Y[itr])
    S = model.predict_proba(X[ite])
    Yt = Y[ite]
    pred = (S >= 0.5).astype(np.int8)
    # metrics over the whole space of this threshold
    r = {"n_labels": len(labels),
         "n_labels_never_used_in_test": int((Yt.sum(0) == 0).sum()),
         "share_never_used_in_test": float((Yt.sum(0) == 0).mean()),
         "macro_f1_full": float(f1_score(Yt, pred, average="macro", zero_division=0)),
         "micro_f1_full": float(f1_score(Yt, pred, average="micro", zero_division=0)),
         }
    r.update(ranks(S, Yt))
    # fair comparison: restrict scores AND truth to the labels every space shares, then rank
    cols = [labels.index(h) for h in labels if h in common]
    Sc, Yc = S[:, cols], Yt[:, cols]
    r["common_space"] = ranks(Sc, Yc)
    # how many labels never receive a positive prediction at 0.5
    r["labels_never_predicted"] = int((pred.sum(0) == 0).sum())
    r["n_common"] = len(cols)
    # aggregated over visits: how many of the top-10 slots are filled by rare labels
    rare = np.array([h not in common for h in labels])
    if rare.any():
        top = np.argsort(-S, axis=1)[:, :10]
        r["share_top10_rare"] = float(rare[top].mean())
    else:
        r["share_top10_rare"] = 0.0
    res["spaces"][str(mc)] = r
    print("min_count=%-3d labels=%-4d never-in-test=%-4d common-R@10=%.4f "
          "common-mAP@10=%.4f  top10-rare=%.1f%%  (%.0fs)"
          % (mc, r["n_labels"], r["n_labels_never_used_in_test"],
             r["common_space"]["R@10"], r["common_space"]["mAP@10"],
             100 * r["share_top10_rare"], time.time() - t0), flush=True)

a, b = res["spaces"]["1"], res["spaces"]["20"]
res["delta_full_vs_working"] = {
    "extra_labels": a["n_labels"] - b["n_labels"],
    "common_R10_change": a["common_space"]["R@10"] - b["common_space"]["R@10"],
    "common_mAP10_change": a["common_space"]["mAP@10"] - b["common_space"]["mAP@10"],
    "common_R10_change_5_vs_20": res["spaces"]["5"]["common_space"]["R@10"]
                                 - b["common_space"]["R@10"],
    "share_labels_never_in_test_min1": a["share_never_used_in_test"],
    "share_top10_rare_min1": a["share_top10_rare"],
}
res["elapsed_seconds"] = round(time.time() - t0, 1)
json.dump(res, open(OUT / "vocabulary_scope.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwrote results/appendix/vocabulary_scope.json (%.0fs)" % (time.time() - t0))
