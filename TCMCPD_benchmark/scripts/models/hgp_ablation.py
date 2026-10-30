#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Component ablation for the PRG scorer on TCMCPD.

Faithful re-implementation of scripts/models/prg.py (same split, same backbone, same ASL
recipe, same three seeds, same metric definitions), extended to

  * cache the averaged val/test logits, and
  * evaluate a fixed set of component configurations, including macro-F1 (threshold rule
    chosen on validation, exactly as prg.py does for its two reported operating points).

Writes /data/tcmcpd_ablation/ablation_<task>.json.  Nothing is written into the project.
"""
import json
import os
_REPO = os.environ.get(
    "TCMCPD_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_OUT = os.environ.get("TCMCPD_HGP_OUT", os.path.join(_REPO, "results", "hgp"))
os.makedirs(_OUT, exist_ok=True)
_CACHE = os.environ.get("TCMCPD_BENCH_CACHE", "/data/tcmcpd_bench/bench")

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(_REPO)
D = ROOT / "data" / "TCMCPD_TCM_Case_Records_v1.0"
OUT = Path(_OUT)
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)
SEEDS = (42, 7, 13)
KSEED = 5


# ----------------------------------------------------------------- data (copied from prg.py)
def build(task):
    pv = pd.read_csv(D / "relationships/patient_visit.csv")
    vsym = pd.read_csv(D / "relationships/visit_symptom.csv")
    vsyn = pd.read_csv(D / "relationships/visit_syndrome.csv")
    vher = pd.read_csv(D / "relationships/visit_herb.csv")
    v2p = dict(zip(pv.visit_id, pv.patient_id))
    V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
    for v, s in vsym[["visit_id", "TCM_symptom_id"]].values:
        V2M[v].add(s)
    for v, y in vsyn[["visit_id", "Syndrome_id"]].values:
        V2Y[v].add(y)
    for v, h in vher[["visit_id", "Herb_id"]].values:
        V2H[v].add(h)
    Xsrc, Ysrc = (V2M, V2Y) if task == "B1" else (V2M, V2H)
    rows = [v for v in Xsrc if v in Ysrc]
    pl = sorted({v2p[v] for v in rows})
    np.random.RandomState(42).shuffle(pl)
    n = len(pl)
    trp, vap, tep = set(pl[:int(.7 * n)]), set(pl[int(.7 * n):int(.85 * n)]), set(pl[int(.85 * n):])
    tr = [v for v in rows if v2p[v] in trp]
    va = [v for v in rows if v2p[v] in vap]
    te = [v for v in rows if v2p[v] in tep]
    xc = Counter(s for v in tr for s in Xsrc[v])
    yc = Counter(y for v in tr for y in Ysrc[v])
    min_lab = 30 if task == "B1" else 5
    xv = [s for s, c in xc.most_common() if c >= 5]
    yv = [y for y, c in yc.most_common() if c >= min_lab]
    xi = {s: i for i, s in enumerate(xv)}
    yi = {y: i for i, y in enumerate(yv)}
    pos = {v: i for i, v in enumerate(rows)}
    X = np.zeros((len(rows), len(xv)), np.float32)
    Y = np.zeros((len(rows), len(yv)), np.float32)
    for r, v in enumerate(rows):
        for s in Xsrc[v]:
            j = xi.get(s)
            if j is not None:
                X[r, j] = 1.0
        for y in Ysrc[v]:
            j = yi.get(y)
            if j is not None:
                Y[r, j] = 1.0
    f = lambda vs: np.array([pos[v] for v in vs])
    return X, Y, f(tr), f(va), f(te)


def label_adj(Ytr):
    C = Ytr.T @ Ytr
    d = np.diag(C).copy()
    d[d == 0] = 1
    P = C / np.sqrt(np.outer(d, d))
    np.fill_diagonal(P, 1.0)
    return P.astype(np.float32)


class ASL(nn.Module):
    def __init__(self, gn=4.0, gp=0.0, clip=0.05):
        super().__init__()
        self.gn, self.gp, self.clip = gn, gp, clip

    def forward(self, logits, y):
        xs = torch.sigmoid(logits)
        xn = torch.clamp(xs - self.clip, min=0.)
        los = y * torch.log(xs.clamp(min=1e-8)) * ((1 - xs) ** self.gp)
        lne = (1 - y) * torch.log((1 - xn).clamp(min=1e-8)) * (xn ** self.gn)
        return -(los + lne).sum() / max(1.0, y.sum().item())


class LabelGraph(nn.Module):
    def __init__(self, d_in, adj, d=128):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.proj = nn.Linear(d_in, d)
        self.emb = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.bias = nn.Parameter(torch.zeros(adj.shape[0]))

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        W = torch.tanh(self.A @ self.emb)
        return h @ W.t() + self.bias


def train_torch(model, Xtr, Ytr, Xva, Yva, epochs=30, bs=128, lr=1e-3, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = ASL()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    Xv, Yv = torch.tensor(Xva), torch.tensor(Yva)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xt[idx]), Yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xv), Yv).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


def logits_of(model, X, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs])).numpy())
    return np.vstack(out)


# ----------------------------------------------------------------- metrics (copied from prg.py)
def ranks(S, Yt, ks=(1, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(Yt, order[:, :k], axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Yt)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Yt.sum()))
    rel = np.take_along_axis(Yt, order[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1)
    ok = den > 0
    out["mAP@10"] = float(((prec * rel).sum(1)[ok] / den[ok]).mean())
    return out


def tune_thresholds(Pv, Yv):
    th = np.full(Yv.shape[1], 0.5, np.float32)
    for j in range(Yv.shape[1]):
        if Yv[:, j].sum() == 0:
            continue
        best, bf = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f = f1_score(Yv[:, j], (Pv[:, j] >= t).astype(int), zero_division=0)
            if f > bf:
                bf, best = f, t
        th[j] = best
    return th


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def macro_f1(Sv, Yv, Se, Ye):
    """Threshold rule chosen on validation (fixed0.5 / global / per-label), reported on test."""
    Pv, Pe = sig(Sv), sig(Se)
    cand = {"fixed0.5": (0.5, None)}
    bg, bf = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(Yv, (Pv >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bg = f, float(t)
    cand["global"] = (bg, None)
    cand["per-label"] = (None, tune_thresholds(Pv, Yv))
    vals = {}
    for name, (t, tt) in cand.items():
        pred = ((Pv >= t) if t is not None else (Pv >= tt[None, :])).astype(int)
        vals[name] = f1_score(Yv, pred, average="macro", zero_division=0)
    rule = max(vals, key=vals.get)
    t, tt = cand[rule]
    pred = ((Pe >= t) if t is not None else (Pe >= tt[None, :])).astype(int)
    return float(f1_score(Ye, pred, average="macro", zero_division=0)), rule


def set_rerank(S, Ytr, eta, kseed):
    if eta <= 0:
        return S
    C = (Ytr.T @ Ytr).astype(np.float32)
    np.fill_diagonal(C, 0.0)
    L = np.log1p(C)
    top = np.argsort(-S, axis=1)[:, :kseed]
    out = S.copy()
    for i in range(len(S)):
        out[i] = S[i] + eta * L[top[i]].sum(0)
    return out


def fuse(Z, Ytr, P, prior_logit, p=0.0, b=0.0, eta=0.0, kseed=KSEED):
    S = Z.copy()
    if p:
        S = S + p * prior_logit[None, :]
    if b > 0:
        S = (1 - b) * S + b * (S @ P)
    return set_rerank(S, Ytr, eta, kseed)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "B2"
    t0 = time.time()
    X, Y, itr, iva, ite = build(task)
    P = label_adj(Y[itr])
    prev = Y[itr].mean(0).clip(1e-4, 1 - 1e-4)
    prior_logit = np.log(prev / (1 - prev)).astype(np.float32)
    print(f"[{task}] visits={len(X)} features={X.shape[1]} labels={Y.shape[1]} "
          f"train={len(itr)} val={len(iva)} test={len(ite)}", flush=True)

    Zv, Ze = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = LabelGraph(X.shape[1], P)
        m = train_torch(m, X[itr], Y[itr], X[iva], Y[iva])
        Zv.append(logits_of(m, X[iva]))
        Ze.append(logits_of(m, X[ite]))
        print(f"  seed {seed} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    Zv, Ze = np.mean(Zv, 0), np.mean(Ze, 0)
    np.savez_compressed(OUT / f"logits_{task}.npz", Zv=Zv, Ze=Ze)

    Yva, Yte, Ytr = Y[iva], Y[ite], Y[itr]
    base = ranks(Ze, Yte)["R@10"]

    CONFIGS = [
        ("R0 backbone (no post-processing)",      dict()),
        ("R1 + frequency prior (p=+0.5)",         dict(p=0.5)),
        ("R2 + frequency prior (p=-1.0)",         dict(p=-1.0)),
        ("R3 + graph smoothing (beta=0.25)",      dict(b=0.25)),
        ("R4 + set re-ranking (eta=1.0,k=5)",     dict(eta=1.0)),
        ("R5 + prior(-1.0) + re-ranking",         dict(p=-1.0, eta=1.0)),
        ("R6 + prior(+0.5) + smoothing",          dict(p=0.5, b=0.25)),
    ]
    rows = []
    for name, kw in CONFIGS:
        Sv = fuse(Zv, Ytr, P, prior_logit, **kw)
        Se = fuse(Ze, Ytr, P, prior_logit, **kw)
        r = ranks(Se, Yte)
        mf, rule = macro_f1(Sv, Yva, Se, Yte)
        rows.append(dict(config=name, **kw, R10=r["R@10"], mAP10=r["mAP@10"], P1=r["P@1"],
                         macroF1=mf, threshold_rule=rule))
        print(f"  {name:<38} R@10 {r['R@10']:.4f}  mAP {r['mAP@10']:.4f}  "
              f"macroF1 {mf:.4f}  ({rule})", flush=True)

    ref = json.load(open(ROOT / "results" / f"e3_{task.lower()}_all_models.json"))
    pt = {k: v for k, v in ref["point"].items() if "prior" not in k}
    best = {m: max(v[m] for v in pt.values()) for m in ("R@10", "mAP@10", "macroF1")}

    out = dict(task=task, n_visits=int(len(X)), n_features=int(X.shape[1]),
               n_labels=int(Y.shape[1]), seeds=list(SEEDS), grid=rows,
               best_of_18=best, elapsed_min=round((time.time() - t0) / 60, 1))
    with open(OUT / f"ablation_{task}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== summary ===")
    print(f"backbone R@10 {base:.4f} (stored grid_test value 0.2971 for B2)")
    print("best of 18 :", {k: round(v, 4) for k, v in best.items()})
    print("wrote", OUT / f"ablation_{task}.json", f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
