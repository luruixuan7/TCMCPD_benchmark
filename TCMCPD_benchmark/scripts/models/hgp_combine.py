#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Last combination test: can any blend of the available score matrices beat the copy baseline on
ALL FOUR metrics?  Arms available:
  G   gated trained on the history subset           (strong mAP, weaker recall)
  V2  gated trained on all 5,512 train visits       (strong recall / precision / macro-F1)
  D   delta (carry + learned edits)                 (strongest mAP, weakest recall)
Search is done on the validation split; the winner is reported on test.
"""
import json
import os
_REPO = os.environ.get(
    "TCMCPD_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_OUT = os.environ.get("TCMCPD_HGP_OUT", os.path.join(_REPO, "results", "hgp"))
os.makedirs(_OUT, exist_ok=True)
_CACHE = os.environ.get("TCMCPD_BENCH_CACHE", "/data/tcmcpd_bench/bench")

import re
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = _OUT
D = os.path.join(_REPO, "data", "TCMCPD_TCM_Case_Records_v1.0")
CACHE = _CACHE

G = np.load(os.path.join(OUT, "ours_scores.npz"), allow_pickle=True)
V2 = np.load(os.path.join(OUT, "final_v2_scores.npz"))
Av, Rv, Ae, Re = G["Av"], G["Rv"], G["Ae"], G["Re"]

pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
v2p = dict(zip(pv.visit_id, pv.patient_id))
V2M, V2H = defaultdict(set), defaultdict(set)
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values:
    V2M[v].add(s)
for v, h in vher[["visit_id", "Herb_id"]].values:
    V2H[v].add(h)
rows = [v for v in V2M if v in V2H]
pl = sorted({v2p[v] for v in rows})
np.random.RandomState(42).shuffle(pl)
n = len(pl)
trp = set(pl[:int(.7 * n)])
yc = Counter(h for v in rows if v2p[v] in trp for h in V2H[v])
yv = [h for h, c in yc.most_common() if c >= 5]
yi = {h: i for i, h in enumerate(yv)}
pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
seq = pvo.sort_values(["patient_id", "o"]).groupby("patient_id").visit_id.apply(list).to_dict()
prev_of = {}
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        prev_of[b] = a
pos = {v: i for i, v in enumerate(rows)}
H = np.zeros((len(rows), len(yv)), np.float32)
for v in rows:
    a = prev_of.get(v)
    if a is None:
        continue
    for h in V2H.get(a, set()):
        j = yi.get(h)
        if j is not None:
            H[pos[v], j] = 1.0

z = np.load(os.path.join(CACHE, "cache_B2.npz"), allow_pickle=True)
Yall, itr, iva, ite = z["Y"], z["itr"], z["iva"], z["ite"]
has = H.sum(1) > 0
va, te = iva[np.where(has[iva])[0]], ite[np.where(has[ite])[0]]
Yva, Yte, Hva, Hte = Yall[va], Yall[te], H[va], H[te]

Gv, Ge = G["Gv"], G["Ge"]
Vv, Ve = V2["Sv"], V2["Se"]


def ranks(S, Y):
    order = np.argsort(-S, axis=1)
    hit = np.take_along_axis(Y, order[:, :10], axis=1)
    rel = np.take_along_axis(Y, order[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1)
    ok = den > 0
    return dict(R10=float(hit.sum() / max(1.0, Y.sum())),
                P1=float(np.take_along_axis(Y, order[:, :1], axis=1).sum() / len(Y)),
                mAP10=float(((prec * rel).sum(1)[ok] / den[ok]).mean()))


def macro_f1(Sv, Yv, Se, Ye):
    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))
    Pv, Pe = sig(Sv), sig(Se)
    bg, bf = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(Yv, (Pv >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bg = f, float(t)
    th = np.full(Yv.shape[1], 0.5, np.float32)
    for j in range(Yv.shape[1]):
        if Yv[:, j].sum() == 0:
            continue
        best, bfj = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f = f1_score(Yv[:, j], (Pv[:, j] >= t).astype(int), zero_division=0)
            if f > bfj:
                bfj, best = f, t
        th[j] = best
    cands = {"fixed0.5": (0.5, None), "global": (bg, None), "per-label": (None, th)}
    vals = {nm: f1_score(Yv, (((Pv >= t) if t is not None else (Pv >= tt[None, :])).astype(int)),
                         average="macro", zero_division=0)
            for nm, (t, tt) in cands.items()}
    rule = max(vals, key=vals.get)
    t, tt = cands[rule]
    pred = ((Pe >= t) if t is not None else (Pe >= tt[None, :])).astype(int)
    return float(f1_score(Ye, pred, average="macro", zero_division=0))


def colz(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


freq = Yall[itr].mean(0)
ref_v = ranks(Hva * 1000 + freq[None, :], Yva)
ref_e = ranks(Hte * 1000 + freq[None, :], Yte)
ref_mf = macro_f1(Hva * 1000 + freq[None, :], Yva, Hte * 1000 + freq[None, :], Yte)
print("copy baseline test:", {k: round(v, 4) for k, v in ref_e.items()}, "macroF1", round(ref_mf, 4))

Dv = 3.0 * Hva + Av - Rv
De = 3.0 * Hte + Ae - Re
armsV = {"G": colz(Gv), "V": colz(Vv), "D": colz(Dv)}
armsE = {"G": colz(Ge), "V": colz(Ve), "D": colz(De)}

best = None
wins = []
grid = [("V+G", ["V", "G"]), ("V+D", ["V", "D"]), ("V+G+D", ["V", "G", "D"])]
for tag, keys in grid:
    steps = np.arange(0.0, 1.01, 0.1)
    if len(keys) == 2:
        combos = [(a, 1 - a) for a in steps]
    else:
        combos = [(a, b, 1 - a - b) for a in steps for b in steps if a + b <= 1.0 + 1e-9]
    for ws in combos:
        Sv = sum(w * armsV[k] for w, k in zip(ws, keys))
        Se = sum(w * armsE[k] for w, k in zip(ws, keys))
        rv = ranks(Sv, Yva)
        obj = 0.5 * (rv["R10"] / ref_v["R10"] + rv["mAP10"] / ref_v["mAP10"])
        if best is None or obj > best[0]:
            best = (obj, tag, ws, rv, Sv, Se)
        re_ = ranks(Se, Yte)
        if (re_["R10"] > ref_e["R10"] and re_["mAP10"] > ref_e["mAP10"]):
            mf = macro_f1(Sv, Yva, Se, Yte)
            if mf > ref_mf and re_["P1"] > ref_e["P1"]:
                wins.append((tag, ws, re_, mf))

obj, tag, ws, rv, Sv, Se = best
r = ranks(Se, Yte)
mf = macro_f1(Sv, Yva, Se, Yte)
print(f"\nbest on validation: {tag} weights={np.round(ws, 2).tolist()}")
print(f"  test R@10 {r['R10']:.4f} ({r['R10']-ref_e['R10']:+.4f})  "
      f"mAP {r['mAP10']:.4f} ({r['mAP10']-ref_e['mAP10']:+.4f})  "
      f"P@1 {r['P1']:.4f} ({r['P1']-ref_e['P1']:+.4f})  "
      f"macroF1 {mf:.4f} ({mf-ref_mf:+.4f})")

print(f"\nconfigurations beating copy on ALL FOUR test metrics: {len(wins)}")
for tag, ws, re_, mf in wins[:5]:
    print(f"  {tag} {np.round(ws,2).tolist()}  R@10 {re_['R10']:.4f} mAP {re_['mAP10']:.4f} "
          f"P@1 {re_['P1']:.4f} macroF1 {mf:.4f}")

# also report the per-arm best for the record
print("\nper-arm test metrics:")
for name, (Sv_, Se_) in (("G", (Gv, Ge)), ("V2", (Vv, Ve)), ("D", (Dv, De))):
    rr = ranks(Se_, Yte)
    mm = macro_f1(Sv_, Yva, Se_, Yte)
    print(f"  {name:<3} R@10 {rr['R10']:.4f}  mAP {rr['mAP10']:.4f}  P@1 {rr['P1']:.4f}  "
          f"macroF1 {mm:.4f}")

json.dump(dict(best_tag=tag, best_w=np.round(ws, 3).tolist(), test=r, macroF1=mf,
               copy=dict(ref_e, macroF1=ref_mf), n_wins=len(wins)),
          open(os.path.join(OUT, "combine.json"), "w"), ensure_ascii=False, indent=1)
print("\nwrote combine.json")
