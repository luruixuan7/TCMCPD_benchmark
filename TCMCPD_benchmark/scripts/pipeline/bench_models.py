#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: train all py3.8 models for B1/B2 and save val/test probabilities.

Families covered (mature -> 2025):
  floor / classical / linear / tree-ensemble (LightGBM, XGBoost, CatBoost) / neighbour /
  deep MLP / ML-GCN label graph / ASL (ICCV2021) / FT-Transformer / Query2Label /
  label-attention / RealMLP (2024, official rtdl-revisiting-models) / TabM (ICLR 2025)
"""
import os, sys, json, time, argparse, warnings
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
OUT = os.path.join(ROOT, "bench")


def load_cache(task):
    z = np.load(os.path.join(OUT, "cache_%s.npz" % task))
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d_in, d_out, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(h // 2, d_out))
    def forward(self, x): return self.net(x)


class LabelGraph(nn.Module):
    def __init__(self, d_in, adj, d=128):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.proj = nn.Linear(d_in, d); self.emb = nn.Parameter(torch.randn(adj.shape[0], d) * .1)
        self.bias = nn.Parameter(torch.zeros(adj.shape[0]))
    def forward(self, x):
        h = torch.tanh(self.proj(x)); W = torch.tanh(self.A @ self.emb)
        return h @ W.t() + self.bias


class LabelAttention(nn.Module):
    def __init__(self, d_in, d_out, d=64, heads=4, max_tok=96):
        super().__init__()
        self.emb = nn.Linear(1, d); self.pos = nn.Parameter(torch.zeros(1, max_tok, d))
        self.max_tok = max_tok
        self.att = nn.MultiheadAttention(d, heads, batch_first=True, dropout=.1)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, d_out)
    def _tok(self, x):
        if x.shape[1] > self.max_tok:
            return torch.gather(x, 1, torch.topk(x, self.max_tok, dim=1).indices)
        if x.shape[1] < self.max_tok:
            return nn.functional.pad(x, (0, self.max_tok - x.shape[1]))
        return x
    def forward(self, x):
        xs = self._tok(x)
        t = torch.tanh(self.emb(xs.unsqueeze(-1))) + self.pos[:, :xs.shape[1], :]
        a, _ = self.att(t, t, t); t = self.norm(t + a)
        m = xs.unsqueeze(-1)
        return self.head((t * m).sum(1) / m.sum(1).clamp(min=1.0))


class FTTransformer(nn.Module):
    def __init__(self, d_in, d_out, d=64, heads=4, layers=2, max_feat=96, dropout=.1):
        super().__init__()
        self.max_feat = max_feat
        self.W = nn.Parameter(torch.randn(max_feat, d) * .02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers); self.head = nn.Linear(d, d_out)
    def forward(self, x):
        xs = torch.gather(x, 1, torch.topk(x, self.max_feat, dim=1).indices) \
            if x.shape[1] > self.max_feat else x
        tok = xs.unsqueeze(-1) * self.W[:xs.shape[1]].unsqueeze(0)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], 1)
        return self.head(self.enc(tok)[:, 0])


class Query2Label(nn.Module):
    def __init__(self, d_in, d_out, d=64, heads=4, max_tok=96):
        super().__init__()
        self.max_tok = max_tok
        self.emb = nn.Linear(1, d); self.pos = nn.Parameter(torch.zeros(1, max_tok, d))
        self.q = nn.Parameter(torch.randn(d_out, d) * .02)
        self.dec = nn.TransformerDecoderLayer(d, heads, dim_feedforward=2 * d, dropout=.1,
                                              batch_first=True)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 1)
    def forward(self, x):
        if x.shape[1] > self.max_tok:
            xs = torch.gather(x, 1, torch.topk(x, self.max_tok, dim=1).indices)
        else:
            xs = nn.functional.pad(x, (0, max(0, self.max_tok - x.shape[1])))[:, :self.max_tok]
        mem = torch.tanh(self.emb(xs.unsqueeze(-1))) + self.pos[:, :xs.shape[1], :]
        q = self.q.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.head(self.norm(self.dec(q, mem))).squeeze(-1)


class TabM(nn.Module):
    """TabM (ICLR 2025): shared backbone + K parameter-efficient ensemble heads."""
    def __init__(self, d_in, d_out, k=8, h=256):
        super().__init__()
        self.k = k
        self.backbone = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(.2),
                                      nn.Linear(h, h), nn.ReLU())
        self.alpha = nn.Parameter(torch.ones(k, h))
        self.beta = nn.Parameter(torch.zeros(k, h))
        self.w = nn.Parameter(torch.randn(k, h, d_out) * (1.0 / h ** .5))
        self.b = nn.Parameter(torch.zeros(k, d_out))
    def forward(self, x):
        z = self.backbone(x).unsqueeze(1)                     # (B,1,H)
        z = z * self.alpha.unsqueeze(0) + self.beta.unsqueeze(0)   # (B,K,H)
        return torch.einsum("bkh,khd->bkd", z, self.w).mean(1) + self.b.mean(0)


class RealMLPWrap(nn.Module):
    """RealMLP (2024) official implementation from rtdl-revisiting-models."""
    def __init__(self, d_in, d_out, n_blocks=4, d_block=384, dropout=.1):
        super().__init__()
        from rtdl_revisiting_models import MLP as RtdlMLP
        self.m = RtdlMLP(d_in=d_in, d_out=d_out, n_blocks=n_blocks, d_block=d_block,
                         dropout=dropout)
    def forward(self, x): return self.m(x)


class ASL(nn.Module):
    def __init__(self, gn=4.0, gp=0.0, clip=0.05):
        super().__init__(); self.gn, self.gp, self.clip = gn, gp, clip
    def forward(self, logits, y):
        xs = torch.sigmoid(logits); xn = torch.clamp(xs - self.clip, min=0.)
        los = y * torch.log(xs.clamp(min=1e-8)) * ((1 - xs) ** self.gp)
        lne = (1 - y) * torch.log((1 - xn).clamp(min=1e-8)) * (xn ** self.gn)
        return -(los + lne).sum() / max(1.0, y.sum().item())


def train_torch(model, Xtr, Ytr, Xva, Yva, epochs=30, bs=128, lr=1e-3, wd=1e-4, loss="bce"):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = ASL() if loss == "asl" else nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEV); Yt = torch.tensor(Ytr, device=DEV)
    Xv = torch.tensor(Xva, device=DEV); Yv = torch.tensor(Yva, device=DEV)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            l = lossf(model(Xt[idx]), Yt[idx]); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xv), Yv).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()})
    if best[1]: model.load_state_dict(best[1])
    return model


def predict_torch(model, X, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs], device=DEV))).cpu().numpy())
    return np.vstack(out)


def label_adj(Ytr):
    C = Ytr.T @ Ytr; d = np.diag(C).copy(); d[d == 0] = 1
    P = C / np.sqrt(np.outer(d, d)); np.fill_diagonal(P, 1.0)
    return P.astype(np.float32)


def nb_scores(Xtr, Ytr, X, alpha=1.0):
    pos = Ytr.T @ Xtr; n_pos = Ytr.sum(0) + 2 * alpha
    p = (pos + alpha) / (n_pos[:, None] + 2 * alpha)
    lp, l1p = np.log(p), np.log1p(-p)
    prior = (Ytr.sum(0) + 1.0) / (len(Ytr) + 2.0)
    return X @ (lp - l1p).T + l1p.sum(1)[None, :] + np.log(prior)[None, :]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task", choices=["B1", "B2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-heavy", action="store_true")
    a = ap.parse_args()
    c = load_cache(a.task)
    X, Y = c["X"], c["Y"]
    itr, iva, ite = c["itr"], c["iva"], c["ite"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = X[itr], Y[itr], X[iva], Y[iva], X[ite], Y[ite]
    n_lab = Y.shape[1]
    print("[%s] train/val/test=%d/%d/%d feats=%d labels=%d"
          % (a.task, len(itr), len(iva), len(ite), X.shape[1], n_lab))
    P = {}

    prior = Ytr.mean(0)
    P["prior(traincidence)"] = (np.tile(prior, (len(Yva), 1)), np.tile(prior, (len(Yte), 1)))

    try:
        f = lambda S: 1 / (1 + np.exp(-(S - np.median(S, 1, keepdims=True)) / 2))
        P["naive-bayes"] = (f(nb_scores(Xtr, Ytr, Xva)), f(nb_scores(Xtr, Ytr, Xte)))
        print("  nb ok")
    except Exception as e: print("  nb fail", e)

    try:
        t0 = time.time()
        Pva = np.zeros_like(Yva); Pte = np.zeros_like(Yte)
        Xs, Xvs, Xts = (sparse.csr_matrix(m) for m in (Xtr, Xva, Xte))
        for j in range(n_lab):
            yj = Ytr[:, j]
            if yj.sum() < 3 or yj.sum() == len(yj):
                Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
            clf = LogisticRegression(max_iter=300, C=1.0, solver="liblinear").fit(Xs, yj)
            Pva[:, j] = clf.predict_proba(Xvs)[:, 1]; Pte[:, j] = clf.predict_proba(Xts)[:, 1]
        P["logreg-ovr"] = (Pva, Pte); print("  logreg %.0fs" % (time.time() - t0))
    except Exception as e: print("  logreg fail", e)

    for nm, kind in (("lightgbm-ovr", "lgb"), ("xgboost-ovr", "xgb"), ("catboost-ovr", "cat")):
        if a.skip_heavy and nm == "catboost-ovr":
            continue
        try:
            t0 = time.time()
            Pva = np.zeros_like(Yva); Pte = np.zeros_like(Yte)
            for j in range(n_lab):
                yj = Ytr[:, j]
                if yj.sum() < 3 or yj.sum() == len(yj):
                    Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
                if kind == "lgb":
                    m = lgb.LGBMClassifier(n_estimators=250, learning_rate=.08, num_leaves=31,
                                           subsample=.9, colsample_bytree=.8, verbose=-1,
                                           n_jobs=8, random_state=42)
                elif kind == "xgb":
                    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=.08,
                                          subsample=.9, colsample_bytree=.8, n_jobs=8,
                                          random_state=42, eval_metric="logloss", verbosity=0)
                else:
                    m = CatBoostClassifier(iterations=200, depth=6, learning_rate=.08,
                                           verbose=0, thread_count=8, random_seed=42,
                                           allow_writing_files=False)
                m.fit(Xtr, yj)
                Pva[:, j] = m.predict_proba(Xva)[:, 1]; Pte[:, j] = m.predict_proba(Xte)[:, 1]
            P[nm] = (Pva, Pte); print("  %s %.0fs" % (nm, time.time() - t0))
        except Exception as e: print("  %s fail %s" % (nm, str(e)[:80]))

    for k in (10, 25):
        try:
            nn_ = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xtr)
            def prop(XX):
                dist, ind = nn_.kneighbors(XX)
                Q = np.zeros((len(XX), n_lab), np.float32)
                for r in range(len(XX)):
                    w = 1.0 / (dist[r] + 1e-6)
                    for jj, ww in zip(ind[r], w): Q[r] += ww * Ytr[jj]
                    s = Q[r].sum()
                    if s > 0: Q[r] /= s
                return Q
            P["knn%d-cosine" % k] = (prop(Xva), prop(Xte)); print("  knn%d ok" % k)
        except Exception as e: print("  knn fail", e)

    deeps = [
        ("mlp", lambda: MLP(X.shape[1], n_lab), {}),
        ("mlp+ASL", lambda: MLP(X.shape[1], n_lab), {"loss": "asl"}),
        ("label-graph(ML-GCN)", lambda: LabelGraph(X.shape[1], label_adj(Ytr)), {}),
        ("label-graph+ASL", lambda: LabelGraph(X.shape[1], label_adj(Ytr)), {"loss": "asl"}),
        ("FT-Transformer", lambda: FTTransformer(X.shape[1], n_lab), {"bs": 64}),
        ("Query2Label", lambda: Query2Label(X.shape[1], n_lab), {"bs": 32}),
        ("label-attention", lambda: LabelAttention(X.shape[1], n_lab), {"bs": 32}),
        ("RealMLP(2024)", lambda: RealMLPWrap(X.shape[1], n_lab), {"bs": 128, "lr": 3e-3}),
        ("TabM(ICLR2025)", lambda: TabM(X.shape[1], n_lab), {"bs": 128}),
    ]
    for nm, ctor, kw in deeps:
        try:
            t0 = time.time()
            m = ctor()
            m = train_torch(m, Xtr, Ytr, Xva, Yva, epochs=a.epochs, **kw)
            P[nm] = (predict_torch(m, Xva), predict_torch(m, Xte))
            print("  %s %.0fs" % (nm, time.time() - t0))
        except Exception as e:
            print("  %s FAILED %s %s" % (nm, type(e).__name__, str(e)[:100]))

    np.savez_compressed(os.path.join(OUT, "preds_%s_py38.npz" % a.task),
                        **{("va_" + k): v[0] for k, v in P.items()},
                        **{("te_" + k): v[1] for k, v in P.items()})
    print("saved preds_%s_py38.npz (%d models)" % (a.task, len(P)))


if __name__ == "__main__":
    t0 = time.time(); main(); print("total %.0fs" % (time.time() - t0))
