#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final attempt: train the gated model on ALL B2 training visits (5,512) instead of only the
2,374 that have a previous prescription, with the history vector left empty where absent.
Evaluation stays on the same 520-visit test subset, so the comparison is unchanged.

Also sweeps an extra carry term on validation (the only hyper-parameter) and reports whether the
copy baseline can be beaten on BOTH recall@10 and mAP@10.
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
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

D = os.path.join(_REPO, "data", "TCMCPD_TCM_Case_Records_v1.0")
CACHE = _CACHE
OUT = _OUT
torch.set_num_threads(8)
SEEDS = (42, 7, 13)
t0 = time.time()


def log(*a):
    print(*a, flush=True)


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
X, Y, itr, iva, ite = z["X"], z["Y"], z["itr"], z["iva"], z["ite"]
has = H.sum(1) > 0
pos_va = np.where(has[iva])[0]
pos_te = np.where(has[ite])[0]
va, te = iva[pos_va], ite[pos_te]
tr = itr              # <-- ALL training visits
log(f"train (all)={len(tr)}  val subset={len(va)}  test subset={len(te)}")


def ranks(S, Yarr):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in (1, 10):
        hit = np.take_along_axis(Yarr, order[:, :k], axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Yarr)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Yarr.sum()))
    rel = np.take_along_axis(Yarr, order[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1)
    ok = den > 0
    out["mAP@10"] = float(((prec * rel).sum(1)[ok] / den[ok]).mean())
    return out


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
    return float(f1_score(Ye, pred, average="macro", zero_division=0)), rule


def label_adj(Ya):
    C = Ya.T @ Ya
    d = np.diag(C).copy()
    d[d == 0] = 1
    P = C / np.sqrt(np.outer(d, d))
    np.fill_diagonal(P, 1.0)
    return P.astype(np.float32)


class ASL(nn.Module):
    def forward(self, z, y):
        xs = torch.sigmoid(z)
        xn = torch.clamp(xs - 0.05, min=0.)
        los = y * torch.log(xs.clamp(min=1e-8))
        lne = (1 - y) * torch.log((1 - xn).clamp(min=1e-8)) * (xn ** 4.0)
        return -(los + lne).sum() / max(1.0, y.sum().item())


class Gated(nn.Module):
    def __init__(self, d_in, adj, d=160):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj))
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(.2), nn.Linear(d, d), nn.ReLU())
        self.emb = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.bias = nn.Parameter(torch.zeros(adj.shape[0]))
        self.gate = nn.Parameter(torch.zeros(adj.shape[0]))
        self.hscale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, h):
        e = self.enc(x)
        W = torch.tanh(self.A @ self.emb)
        g = F.softplus(self.gate) * torch.exp(self.hscale)
        return e @ W.t() + self.bias + g[None, :] * h


P = label_adj(Y[tr])
Xtr, Htr, Ytr = X[tr], H[tr], Y[tr]
Xva, Hva, Yva = X[va], H[va], Y[va]

Sv, Se = [], []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Gated(X.shape[1], P)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = ASL()
    Xt, Ht, Yt = torch.tensor(Xtr), torch.tensor(Htr), torch.tensor(Ytr)
    Xv, Hv, Yv_ = torch.tensor(Xva), torch.tensor(Hva), torch.tensor(Yva)
    best = (1e9, None)
    for ep in range(45):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lf(m(Xt[idx], Ht[idx]), Yt[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lf(m(Xv, Hv), Yv_).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad():
        Sv.append(m(Xv, Hv).numpy())
        Se.append(m(torch.tensor(X[te]), torch.tensor(H[te])).numpy())
    log(f"  seed {seed} done ({(time.time()-t0)/60:.1f} min)")
Sv, Se = np.mean(Sv, 0), np.mean(Se, 0)

freq = Ytr.mean(0)
lb_v, lb_e = Hva * 1000 + freq[None, :], H[te] * 1000 + freq[None, :]
rfv, rfe = ranks(lb_v, Yva), ranks(lb_e, Y[te])
log(f"\ncopy baseline: val R@10 {rfv['R@10']:.4f} mAP {rfv['mAP@10']:.4f} | "
    f"test R@10 {rfe['R@10']:.4f} mAP {rfe['mAP@10']:.4f}")

best = None
for extra in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
    sv = Sv + extra * Hva
    rv = ranks(sv, Yva)
    obj = 0.5 * (rv["R@10"] / rfv["R@10"] + rv["mAP@10"] / rfv["mAP@10"])
    if best is None or obj > best[0]:
        best = (obj, extra, rv)
obj, extra, rv = best
log(f"selected extra carry={extra} on val (val R@10 {rv['R@10']:.4f} mAP {rv['mAP@10']:.4f})")

Sv2, Se2 = Sv + extra * Hva, Se + extra * H[te]
r = ranks(Se2, Y[te])
mf, rule = macro_f1(Sv2, Yva, Se2, Y[te])
log(f"\nHGP-set v2 (all-train)       R@10 {r['R@10']:.4f} ({r['R@10']-rfe['R@10']:+.4f})  "
    f"mAP@10 {r['mAP@10']:.4f} ({r['mAP@10']-rfe['mAP@10']:+.4f})  "
    f"P@1 {r['P@1']:.4f}  macro-F1 {mf:.4f}")

print(f"{'extra':>6}{'val R@10':>10}{'val mAP':>10}{'test R@10':>11}{'test mAP':>10}")
for e_ in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
    t_ = ranks(Se + e_ * H[te], Y[te])
    v_ = ranks(Sv + e_ * Hva, Yva)
    print(f"{e_:>6.1f}{v_['R@10']:>10.4f}{v_['mAP@10']:>10.4f}{t_['R@10']:>11.4f}{t_['mAP@10']:>10.4f}")

json.dump(dict(extra=extra, R10=r["R@10"], mAP10=r["mAP@10"], P1=r["P@1"], macroF1=mf,
               copy=dict(R10=rfe["R@10"], mAP10=rfe["mAP@10"], P1=rfe["P@1"]),
               elapsed_min=round((time.time() - t0) / 60, 1)),
          open(os.path.join(OUT, "final_v2.json"), "w"), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(OUT, "final_v2_scores.npz"), Sv=Sv, Se=Se)
log(f"\nwrote final_v2.json ({(time.time()-t0)/60:.1f} min)")
