#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main comparison on the ORIGINAL fixed B2 split — no baseline is retrained.

Protocol
  split   : patient-level 70/15/15, seed 42, exactly as bench_data.py
  subset  : test (and train/val) visits whose previous visit carries a prescription
  baselines: the 19 stored prediction files (18 models + prevalence prior), restricted to the subset
  ours    : gated (content + per-label history gate) and delta (carry + learned edits),
            trained on the SAME train split, same label space

Writes /data/tcmcpd_ablation/main.json
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


# ------------------------------------------------------------------ rebuild the fixed split
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
pats = {v2p[v] for v in rows}
pl = sorted(pats)
np.random.RandomState(42).shuffle(pl)
n = len(pl)
trp, vap, tep = set(pl[:int(.7 * n)]), set(pl[int(.7 * n):int(.85 * n)]), set(pl[int(.85 * n):])
spl = {v: (0 if v2p[v] in trp else 1 if v2p[v] in vap else 2) for v in rows}

yc = Counter(h for v in rows if spl[v] == 0 for h in V2H[v])
yv = [h for h, c in yc.most_common() if c >= 5]
yi = {h: i for i, h in enumerate(yv)}

z = np.load(os.path.join(CACHE, "cache_B2.npz"), allow_pickle=True)
Xall, Yall, itr, iva, ite = z["X"], z["Y"], z["itr"], z["iva"], z["ite"]
assert len(yv) == Yall.shape[1]

pvo = pv.copy()
pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
pvo = pvo.sort_values(["patient_id", "o"])
seq = pvo.groupby("patient_id").visit_id.apply(list).to_dict()
prev_of = {}
for p, vs in seq.items():
    for a, b in zip(vs, vs[1:]):
        prev_of[b] = a

# history matrix over all rows: herbs of the previous visit, in the B2 label space
pos = {v: i for i, v in enumerate(rows)}
H = np.zeros((len(rows), Yall.shape[1]), np.float32)
for v in rows:
    a = prev_of.get(v)
    if a is None:
        continue
    for h in V2H.get(a, set()):
        j = yi.get(h)
        if j is not None:
            H[pos[v], j] = 1.0
has_hist = H.sum(1) > 0
log(f"rows={len(rows)}  train={len(itr)} val={len(iva)} test={len(ite)}")
log(f"rows with a usable history: {int(has_hist.sum())}/{len(rows)} "
    f"(train {int(has_hist[itr].sum())}, val {int(has_hist[iva].sum())}, test {int(has_hist[ite].sum())})")

pos_tr = np.where(has_hist[itr])[0]
pos_va = np.where(has_hist[iva])[0]
pos_te = np.where(has_hist[ite])[0]
tr = itr[pos_tr]
va = iva[pos_va]
te = ite[pos_te]
log(f"subset used: train={len(tr)} val={len(va)} test={len(te)}")


# ------------------------------------------------------------------ metrics
def ranks(S, Y, ks=(1, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(Y, order[:, :k], axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Y)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Y.sum()))
    rel = np.take_along_axis(Y, order[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1)
    ok = den > 0
    out["mAP@10"] = float(((prec * rel).sum(1)[ok] / den[ok]).mean())
    return out


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def macro_f1(Sv, Yv, Se, Ye):
    Pv, Pe = _sig(Sv), _sig(Se)
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


# ------------------------------------------------------------------ the 19 stored baselines
Y_te_sub = Yall[te]
bstore = {}
for f in sorted(os.listdir(CACHE)):
    if not (f.startswith("preds_B2") and f.endswith(".npz")):
        continue
    zz = np.load(os.path.join(CACHE, f), allow_pickle=True)
    for k in zz.files:
        if not k.startswith("te_"):
            continue
        name = k[3:]
        vk = "va_" + name
        if vk not in zz.files:
            continue
        pre = zz[k]
        if pre.shape[0] != len(ite):
            log(f"  skip {name}: shape {pre.shape}")
            continue
        r = ranks(pre[pos_te], Y_te_sub)
        bstore[name] = dict(R10=r["R@10"], mAP10=r["mAP@10"], P1=r["P@1"],
                            S_te=pre[pos_te], S_va=zz[vk][pos_va])
log(f"\nreused baselines: {len(bstore)} (no retraining)")
for k in sorted(bstore, key=lambda x: -bstore[x]["mAP10"])[:6]:
    b = bstore[k]
    log(f"  {k:<26} R@10 {b['R10']:.4f}  mAP@10 {b['mAP10']:.4f}")

# persistence baseline on the same subset
freq = Yall[tr].mean(0)
lb_te = H[te] * 1000 + freq[None, :]
lb_va = H[va] * 1000 + freq[None, :]
r = ranks(lb_te, Y_te_sub)
mf, rule = macro_f1(lb_va, Yall[va], lb_te, Y_te_sub)
bstore["last-basket"] = dict(R10=r["R@10"], mAP10=r["mAP@10"], P1=r["P@1"],
                             S_te=lb_te, S_va=lb_va, macroF1=mf, rule=rule)
log(f"  {'last-basket':<26} R@10 {r['R@10']:.4f}  mAP@10 {r['mAP@10']:.4f}  macro-F1 {mf:.4f}")

# ------------------------------------------------------------------ our models (same split)
def label_adj(Y):
    C = Y.T @ Y
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
    def __init__(self, d_in, adj, d=128):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj))
        self.proj = nn.Linear(d_in, d)
        self.emb = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.bias = nn.Parameter(torch.zeros(adj.shape[0]))
        self.gate = nn.Parameter(torch.zeros(adj.shape[0]))
        self.hscale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, h):
        hh = torch.tanh(self.proj(x))
        W = torch.tanh(self.A @ self.emb)
        return hh @ W.t() + self.bias + (F.softplus(self.gate) * torch.exp(self.hscale))[None, :] * h


class Delta(nn.Module):
    def __init__(self, d_in, adj, d=128):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj))
        self.proj = nn.Linear(d_in, d)
        self.emb_a = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.emb_r = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.bias_a = nn.Parameter(torch.zeros(adj.shape[0]))
        self.bias_r = nn.Parameter(torch.zeros(adj.shape[0]))

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        Wa = torch.tanh(self.A @ self.emb_a)
        Wr = torch.tanh(self.A @ self.emb_r)
        return h @ Wa.t() + self.bias_a, h @ Wr.t() + self.bias_r


P = label_adj(Yall[tr])
Xs = Xall
Y = Yall
ADD_tr = Y[tr] * (1 - H[tr])
REM_tr = H[tr] * (1 - Y[tr])


def fit_gated(seed):
    torch.manual_seed(seed)
    m = Gated(Xs.shape[1], P)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = ASL()
    Xt, Ht, Yt = torch.tensor(Xs[tr]), torch.tensor(H[tr]), torch.tensor(Y[tr])
    Xv, Hv, Yv = torch.tensor(Xs[va]), torch.tensor(H[va]), torch.tensor(Y[va])
    best = (1e9, None)
    for _ in range(40):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lf(m(Xt[idx], Ht[idx]), Yt[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lf(m(Xv, Hv), Yv).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad():
        return m(torch.tensor(Xs[va]), torch.tensor(H[va])).numpy(), \
               m(torch.tensor(Xs[te]), torch.tensor(H[te])).numpy()


def fit_delta(seed):
    torch.manual_seed(seed)
    m = Delta(Xs.shape[1], P)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = ASL()
    Xt = torch.tensor(Xs[tr])
    At, Rt = torch.tensor(ADD_tr), torch.tensor(REM_tr)
    Xv = torch.tensor(Xs[va])
    Av, Rv = torch.tensor(Y[va] * (1 - H[va])), torch.tensor(H[va] * (1 - Y[va]))
    best = (1e9, None)
    for _ in range(40):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            za, zr = m(Xt[idx])
            (lf(za, At[idx]) + lf(zr, Rt[idx])).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            za, zr = m(Xv)
            v = (lf(za, Av) + lf(zr, Rv)).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad():
        av, rv = m(torch.tensor(Xs[va]))
        ae, re = m(torch.tensor(Xs[te]))
    return av.numpy(), rv.numpy(), ae.numpy(), re.numpy()


Gv, Ge, Dv = [], [], []
for seed in SEEDS:
    a, b = fit_gated(seed)
    Gv.append(a)
    Ge.append(b)
    av, rv, ae, re = fit_delta(seed)
    Dv.append((av, rv, ae, re))
    log(f"  seed {seed} done ({(time.time()-t0)/60:.1f} min)")
Gv, Ge = np.mean(Gv, 0), np.mean(Ge, 0)
Av, Rv = np.mean([d[0] for d in Dv], 0), np.mean([d[1] for d in Dv], 0)
Ae, Re = np.mean([d[2] for d in Dv], 0), np.mean([d[3] for d in Dv], 0)

best_carry = None
for carry in (1, 2, 3, 4, 6, 8):
    S = carry * H[va] + Av - Rv
    rv = ranks(S, Y[va])["mAP@10"]
    if best_carry is None or rv > best_carry[1]:
        best_carry = (carry, rv)
carry = best_carry[0]
log(f"  delta carry selected on val: {carry} (val mAP {best_carry[1]:.4f})")

Ours = {}
def add_ours(name, Sv, Se):
    r = ranks(Se, Y_te_sub)
    mf, rule = macro_f1(Sv, Yall[va], Se, Y_te_sub)
    Ours[name] = dict(R10=r["R@10"], mAP10=r["mAP@10"], P1=r["P@1"], macroF1=mf, rule=rule)
    log(f"  {name:<26} R@10 {r['R@10']:.4f}  mAP@10 {r['mAP@10']:.4f}  P@1 {r['P@1']:.4f}  "
        f"macro-F1 {mf:.4f}")


add_ours("gated (ours)", Gv, Ge)
add_ours(f"delta (ours, carry={carry})", carry * H[va] + Av - Rv, carry * H[te] + Ae - Re)

json.dump(dict(n_train=len(tr), n_val=len(va), n_test=len(te), carry=carry,
               baselines={k: {kk: vv for kk, vv in v.items() if not kk.startswith("S_")}
                          for k, v in bstore.items()},
               ours=Ours, elapsed_min=round((time.time() - t0) / 60, 1)),
          open(os.path.join(OUT, "main.json"), "w"), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(OUT, "ours_scores.npz"), Gv=Gv, Ge=Ge,
                    Av=Av, Rv=Rv, Ae=Ae, Re=Re, te=te, va=va, tr=tr)
log(f"\nwrote main.json ({(time.time()-t0)/60:.1f} min)")
