#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified benchmark harness for TCMCPD tasks B1 / B2 / B4.

Protocol
  * patient-level split 70/15/15 (seed 42) -- prevents visit leakage
  * label space: labels with >=30 train occurrences (B1/B2); >=5 for herb output
  * metrics: P@1/3/5, R@5/10, mAP@5, micro-F1 & macro-F1 at val-tuned threshold
  * baselines by family: floor / linear / tree-ensemble / neighbour / deep

Usage: python3 bench.py {B1|B2|B4} [--quick]
Writes /data/tcmcpd_bench/bench/results_<task>.json
"""
import os, sys, json, math, re, warnings, time, argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
OUT = os.path.join(ROOT, "bench")
os.makedirs(OUT, exist_ok=True)


# ----------------------------- data -----------------------------
def load():
    vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
    pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
    vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
    vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
    vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
    return vis, pv, vsym, vsyn, vher


def ordered(pv):
    pvo = pv.copy()
    pvo["o"] = pvo.visit_id.map(lambda x: int(re.sub(r"\D", "", str(x)) or -1))
    return pvo.sort_values(["patient_id", "o"]).groupby("patient_id").visit_id.apply(list).to_dict()


def build_multihot(vals, rows_index, vocab):
    idx = {x: i for i, x in enumerate(vocab)}
    M = np.zeros((len(rows_index), len(vocab)), np.float32)
    for r, key in enumerate(rows_index):
        for x in vals.get(key, ()):
            j = idx.get(x)
            if j is not None:
                M[r, j] = 1.0
    return M


def split_patients(pats, seed=42):
    p = sorted(pats); rng = np.random.RandomState(seed); rng.shuffle(p)
    n = len(p)
    return set(p[:int(.7 * n)]), set(p[int(.7 * n):int(.85 * n)]), set(p[int(.85 * n):])


# ----------------------------- metrics -----------------------------
def ranking_metrics(S, Y, ks=(1, 3, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hit = np.take_along_axis(Y, topk, axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Y)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Y.sum()))
    ap = []
    for i in range(len(Y)):
        o = order[i, :10]
        rel = Y[i, o]
        if rel.sum() == 0:
            continue
        h = 0; s = 0.0
        for r, q in enumerate(rel, 1):
            if q > 0:
                h += 1; s += h / r
        ap.append(s / rel.sum())
    out["mAP@10"] = float(np.mean(ap)) if ap else 0.0
    out["J@10"] = float(np.mean([
        (Y[i, order[i, :10]].sum()) / max(1.0, (Y[i, order[i, :10]].sum() + Y[i].sum() - Y[i, order[i, :10]].sum()))
        for i in range(len(Y))]))
    return out


def threshold_metrics(P, Y, Pv, Yv):
    best = (-1, .5)
    for t in (.05, .1, .15, .2, .25, .3, .4, .5):
        pr = Pv >= t
        tp = (pr & (Yv > 0)).sum(); fp = (pr & (Yv == 0)).sum(); fn = ((~pr) & (Yv > 0)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        if f1 > best[0]:
            best = (f1, t)
    t = best[1]
    pr = P >= t
    tp = (pr & (Y > 0)).sum(); fp = (pr & (Y == 0)).sum(); fn = ((~pr) & (Y > 0)).sum()
    micro = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    f1s = []
    for j in range(P.shape[1]):
        a = (pr[:, j] & (Y[:, j] > 0)).sum(); b = (pr[:, j] & (Y[:, j] == 0)).sum()
        c = ((~pr[:, j]) & (Y[:, j] > 0)).sum()
        if a + c == 0:
            continue
        pp = a / (a + b) if a + b else 0; rr = a / (a + c) if a + c else 0
        f1s.append(2 * pp * rr / (pp + rr) if pp + rr else 0)
    return {"microF1": float(micro), "macroF1": float(np.mean(f1s)) if f1s else 0.0,
            "thr": float(t), "microF1_val": float(best[0])}


# ----------------------------- deep models -----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_out, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(h // 2, d_out))
    def forward(self, x):
        return self.net(x)


class LabelAttention(nn.Module):
    """Transformer-style over symptom tokens with a compact token budget."""
    def __init__(self, d_in, d_out, d=64, heads=4, max_tok=96):
        super().__init__()
        self.emb = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, max_tok, d))
        self.max_tok = max_tok
        self.att = nn.MultiheadAttention(d, heads, batch_first=True, dropout=.1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, d_out)
    def forward(self, x):
        # keep the strongest max_tok active features as tokens
        if x.shape[1] > self.max_tok:
            idx = torch.topk(x, self.max_tok, dim=1).indices
            xs = torch.gather(x, 1, idx)
        else:
            xs = x[:, :self.max_tok]
            if xs.shape[1] < self.max_tok:
                xs = torch.nn.functional.pad(xs, (0, self.max_tok - xs.shape[1]))
        t = torch.tanh(self.emb(xs.unsqueeze(-1))) + self.pos[:, :xs.shape[1], :]
        a, _ = self.att(t, t, t)
        t = self.norm(t + a)
        m = xs.unsqueeze(-1)
        pooled = (t * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


class LabelGraph(nn.Module):
    """ML-GCN-style: label co-occurrence graph -> propagated label embeddings -> classifier weights."""
    def __init__(self, d_in, adj, d=128):
        super().__init__()
        n = adj.shape[0]
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.proj = nn.Linear(d_in, d)
        self.emb = nn.Parameter(torch.randn(n, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        h = torch.tanh(self.proj(x))               # (B, d)
        W = torch.tanh(self.A @ self.emb)          # (n_lab, d)
        return h @ W.t() + self.bias


def label_adj(Ytr, thr=0.0):
    """Normalised label co-occurrence adjacency with self-loops."""
    C = Ytr.T @ Ytr
    n = C.shape[0]
    d = np.diag(C).copy(); d[d == 0] = 1
    P = C / np.sqrt(np.outer(d, d))
    P[P < thr] = 0.0
    np.fill_diagonal(P, 1.0)
    return P.astype(np.float32)


def nb_scores(Xtr, Ytr, X, alpha=1.0):
    """Bernoulli naive-Bayes log-odds (a strong classical count model)."""
    pos = Ytr.T @ Xtr
    n_pos = Ytr.sum(axis=0) + 2 * alpha
    p = (pos + alpha) / (n_pos[:, None] + 2 * alpha)
    lp, l1p = np.log(p), np.log1p(-p)
    prior = (Ytr.sum(axis=0) + 1.0) / (len(Ytr) + 2.0)
    return X @ (lp - l1p).T + l1p.sum(axis=1)[None, :] + np.log(prior)[None, :]


def train_torch(model, Xtr, Ytr, Xva, Yva, epochs=40, bs=128, lr=1e-3, wd=1e-4):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=DEV); Ytr_t = torch.tensor(Ytr, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV)
    n = len(Xtr)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, Ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva_t), torch.tensor(Yva, device=DEV)).item()
        if vloss < best[0]:
            best = (vloss, {k: v.clone() for k, v in model.state_dict().items()})
    if best[1]:
        model.load_state_dict(best[1])
    return model


def predict_torch(model, X, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs], device=DEV))).cpu().numpy())
    return np.vstack(out)


# ----------------------------- task prep -----------------------------
def prep(task):
    vis, pv, vsym, vsyn, vher = load()
    seq = ordered(pv)
    v2p = dict(zip(pv.visit_id, pv.patient_id))
    V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
    for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
    for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
    for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)

    if task in ("B1", "B2"):
        if task == "B1":
            rows = [v for v in V2M if v in V2Y]
            Xsrc, Ysrc = V2M, V2Y
        else:
            rows = [v for v in V2M if v in V2H]
            Xsrc, Ysrc = V2M, V2H
        pats = {v2p[v] for v in rows}
        trp, vap, tep = split_patients(pats)
        spl = {v: ("train" if v2p[v] in trp else "val" if v2p[v] in vap else "test") for v in rows}
        tr = [v for v in rows if spl[v] == "train"]
        va = [v for v in rows if spl[v] == "val"]
        te = [v for v in rows if spl[v] == "test"]
        xc = Counter(s for v in tr for s in Xsrc[v])
        yc = Counter(y for v in tr for y in Ysrc[v])
        xv = [s for s, c in xc.most_common() if c >= 5]
        yv = [y for y, c in yc.most_common() if c >= (30 if task == "B1" else 5)]
        X = build_multihot(Xsrc, rows, xv)
        Y = build_multihot(Ysrc, rows, yv)
        pos = {v: i for i, v in enumerate(rows)}
        itr = np.array([pos[v] for v in tr]); iva = np.array([pos[v] for v in va])
        ite = np.array([pos[v] for v in te])
        return dict(task=task, X=X, Y=Y, itr=itr, iva=iva, ite=ite,
                    n_lab=len(yv), n_feat=len(xv), rows=rows,
                    meta={"train": len(tr), "val": len(va), "test": len(te),
                          "patients": len(pats), "labels": len(yv), "features": len(xv)})

    if task == "B4":
        pairs = []
        for p, vs in seq.items():
            for a, b in zip(vs, vs[1:]):
                if b in V2H:
                    pairs.append((p, a, b))
        pats = {p for p, _, _ in pairs}
        trp, vap, tep = split_patients(pats)
        def sp(p): return "train" if p in trp else "val" if p in vap else "test"
        tr = [x for x in pairs if sp(x[0]) == "train"]
        va = [x for x in pairs if sp(x[0]) == "val"]
        te = [x for x in pairs if sp(x[0]) == "test"]
        xc = Counter(s for _, a, _ in tr for s in V2M.get(a, ()))
        hc = Counter(h for _, a, _ in tr for h in V2H.get(a, ()))
        gh = Counter(h for _, _, b in tr for h in V2H.get(b, ()))
        xv = [s for s, c in xc.most_common() if c >= 5]
        hv = [h for h, c in hc.most_common() if c >= 5]
        ghv = [h for h, c in gh.most_common() if c >= 5]
        def feats(prs):
            M = np.zeros((len(prs), len(xv) + len(hv)), np.float32)
            xi = {s: i for i, s in enumerate(xv)}
            hi = {h: i for i, h in enumerate(hv)}
            for r, (_, a, _) in enumerate(prs):
                for s in V2M.get(a, ()):
                    j = xi.get(s)
                    if j is not None: M[r, j] = 1.0
                for h in V2H.get(a, ()):
                    j = hi.get(h)
                    if j is not None: M[r, len(xv) + j] = 1.0
            return M
        Ytr = build_multihot({i: V2H[b] for i, (_, _, b) in enumerate(tr)}, range(len(tr)), ghv)
        Yva = build_multihot({i: V2H[b] for i, (_, _, b) in enumerate(va)}, range(len(va)), ghv)
        Yte = build_multihot({i: V2H[b] for i, (_, _, b) in enumerate(te)}, range(len(te)), ghv)
        return dict(task=task, Xtr=feats(tr), Xva=feats(va), Xte=feats(te),
                    Ytr=Ytr, Yva=Yva, Yte=Yte, n_lab=len(ghv), n_feat=len(xv) + len(hv),
                    pairs_tr=tr, pairs_va=va, pairs_te=te, V2H=V2H, vocab=ghv,
                    meta={"train": len(tr), "val": len(va), "test": len(te),
                          "patients": len(pats), "labels": len(ghv),
                          "features": len(xv) + len(hv)})


# ----------------------------- baselines -----------------------------
def run_multilabel(task, quick=False):
    d = prep(task)
    X, Y = d["X"], d["Y"]
    itr, iva, ite = d["itr"], d["iva"], d["ite"]
    Xtr, Ytr = X[itr], Y[itr]
    Xva, Yva = X[iva], Y[iva]
    Xte, Yte = X[ite], Y[ite]
    print("[%s] train/val/test = %d/%d/%d | features=%d labels=%d"
          % (task, len(itr), len(iva), len(ite), d["n_feat"], d["n_lab"]))
    res = {}

    def record(name, Pte, Pva):
        r = ranking_metrics(Pte, Yte)
        r.update(threshold_metrics(Pte, Yte, Pva, Yva))
        res[name] = r
        print("   %-22s P@1=%.3f R@5=%.3f R@10=%.3f mAP@10=%.3f microF1=%.3f macroF1=%.3f"
              % (name, r["P@1"], r["R@5"], r["R@10"], r["mAP@10"], r["microF1"], r["macroF1"]))

    # floor: training prevalence
    prior = Ytr.mean(axis=0)
    record("prior(traincidence)", np.tile(prior, (len(Yte), 1)), np.tile(prior, (len(Yva), 1)))

    # classical count model: Bernoulli NB (log-odds)
    try:
        Sva = nb_scores(Xtr, Ytr, Xva); Ste = nb_scores(Xtr, Ytr, Xte)
        f = lambda S: 1.0 / (1.0 + np.exp(-(S - np.median(S, axis=1, keepdims=True)) / 2.0))
        record("naive-bayes", f(Ste), f(Sva))
    except Exception as e:
        print("   nb failed:", type(e).__name__, str(e)[:80])

    # linear: one-vs-rest logistic
    try:
        t0 = time.time()
        Pva = np.zeros_like(Yva, dtype=np.float32); Pte = np.zeros_like(Yte, dtype=np.float32)
        Xtr_s = sparse.csr_matrix(Xtr)
        for j in range(d["n_lab"]):
            yj = Ytr[:, j]
            if yj.sum() < 3 or yj.sum() == len(yj):
                Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
            clf = LogisticRegression(max_iter=300, C=1.0, solver="liblinear")
            clf.fit(Xtr_s, yj)
            Pva[:, j] = clf.predict_proba(sparse.csr_matrix(Xva))[:, 1]
            Pte[:, j] = clf.predict_proba(sparse.csr_matrix(Xte))[:, 1]
        record("logreg-ovr", Pte, Pva)
        print("      (ovr time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   logreg failed:", type(e).__name__, str(e)[:80])

    # trees: LightGBM per label
    try:
        t0 = time.time()
        Pva = np.zeros_like(Yva, dtype=np.float32); Pte = np.zeros_like(Yte, dtype=np.float32)
        n_est = 120 if quick else 250
        for j in range(d["n_lab"]):
            yj = Ytr[:, j]
            if yj.sum() < 3 or yj.sum() == len(yj):
                Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
            m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=.08, num_leaves=31,
                                   subsample=.9, colsample_bytree=.8, verbose=-1,
                                   n_jobs=8, random_state=42)
            m.fit(Xtr, yj)
            Pva[:, j] = m.predict_proba(Xva)[:, 1]
            Pte[:, j] = m.predict_proba(Xte)[:, 1]
        record("lightgbm-ovr", Pte, Pva)
        print("      (lgbm time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   lightgbm failed:", type(e).__name__, str(e)[:80])

    # trees: XGBoost per label
    try:
        t0 = time.time()
        Pva = np.zeros_like(Yva, dtype=np.float32); Pte = np.zeros_like(Yte, dtype=np.float32)
        n_est = 100 if quick else 200
        for j in range(d["n_lab"]):
            yj = Ytr[:, j]
            if yj.sum() < 3 or yj.sum() == len(yj):
                Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
            m = xgb.XGBClassifier(n_estimators=n_est, max_depth=5, learning_rate=.08,
                                  subsample=.9, colsample_bytree=.8, n_jobs=8,
                                  random_state=42, eval_metric="logloss", verbosity=0)
            m.fit(Xtr, yj)
            Pva[:, j] = m.predict_proba(Xva)[:, 1]
            Pte[:, j] = m.predict_proba(Xte)[:, 1]
        record("xgboost-ovr", Pte, Pva)
        print("      (xgb time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   xgboost failed:", type(e).__name__, str(e)[:80])

    # neighbour: cosine kNN label propagation
    try:
        for k in (10, 25):
            nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xtr)
            def prop(XX):
                dist, ind = nn.kneighbors(XX)
                P = np.zeros((len(XX), d["n_lab"]), np.float32)
                for r in range(len(XX)):
                    w = 1.0 / (dist[r] + 1e-6)
                    for jj, ww in zip(ind[r], w):
                        P[r] += ww * Ytr[jj]
                    s = P[r].sum()
                    if s > 0: P[r] /= s
                return P
            record("knn%d-cosine" % k, prop(Xte), prop(Xva))
    except Exception as e:
        print("   knn failed:", type(e).__name__, str(e)[:80])

    # deep: MLP
    try:
        t0 = time.time()
        m = MLP(d["n_feat"], d["n_lab"])
        m = train_torch(m, Xtr, Ytr, Xva, Yva, epochs=8 if quick else 30)
        record("mlp", predict_torch(m, Xte), predict_torch(m, Xva))
        print("      (mlp time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   mlp failed:", type(e).__name__, str(e)[:80])

    # deep: label-graph classifier (ML-GCN style)
    try:
        t0 = time.time()
        A = label_adj(Ytr)
        m = LabelGraph(d["n_feat"], A)
        m = train_torch(m, Xtr, Ytr, Xva, Yva, epochs=8 if quick else 30)
        record("label-graph(ML-GCN)", predict_torch(m, Xte), predict_torch(m, Xva))
        print("      (labelgraph time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   labelgraph failed:", type(e).__name__, str(e)[:80])

    # deep: label-attention over symptom tokens
    try:
        t0 = time.time()
        m = LabelAttention(d["n_feat"], d["n_lab"])
        m = train_torch(m, Xtr, Ytr, Xva, Yva, epochs=6 if quick else 25, bs=32)
        record("label-attention", predict_torch(m, Xte), predict_torch(m, Xva))
        print("      (attn time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   attention failed:", type(e).__name__, str(e)[:80])

    out = {"task": task, "meta": d["meta"], "results": res,
           "protocol": {"split": "patient-level 70/15/15 seed=42",
                        "metrics": "P@k, R@k, mAP@10, micro/macro-F1 at val-tuned threshold"}}
    with open(os.path.join(OUT, "results_%s.json" % task), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("   saved results_%s.json" % task)
    return out


def run_B4(quick=False):
    d = prep("B4")
    Xtr, Xva, Xte = d["Xtr"], d["Xva"], d["Xte"]
    Ytr, Yva, Yte = d["Ytr"], d["Yva"], d["Yte"]
    V2H, vocab = d["V2H"], d["vocab"]
    print("[B4] train/val/test pairs = %d/%d/%d | features=%d labels=%d"
          % (len(Xtr), len(Xva), len(Xte), d["n_feat"], d["n_lab"]))
    res = {}

    def record(name, Pte, Pva):
        r = ranking_metrics(Pte, Yte)
        r.update(threshold_metrics(Pte, Yte, Pva, Yva))
        res[name] = r
        print("   %-22s P@1=%.3f R@5=%.3f R@10=%.3f mAP@10=%.3f microF1=%.3f macroF1=%.3f"
              % (name, r["P@1"], r["R@5"], r["R@10"], r["mAP@10"], r["microF1"], r["macroF1"]))

    prior = Ytr.mean(axis=0)
    record("prior", np.tile(prior, (len(Yte), 1)), np.tile(prior, (len(Yva), 1)))

    # last-basket: copy herbs of current visit
    hv = {h: i for i, h in enumerate(vocab)}
    def lastbasket(pairs):
        P = np.zeros((len(pairs), len(vocab)), np.float32)
        for r, (_, a, _) in enumerate(pairs):
            for h in V2H.get(a, ()):
                j = hv.get(h)
                if j is not None: P[r, j] = 1.0
        return P
    record("last-basket", lastbasket(d["pairs_te"]), lastbasket(d["pairs_va"]))

    # kNN cosine on features
    for k in (10, 25):
        nn = NearestNeighbors(n_neighbors=min(k, len(Xtr)), metric="cosine").fit(Xtr)
        def prop(XX):
            dist, ind = nn.kneighbors(XX)
            P = np.zeros((len(XX), len(vocab)), np.float32)
            for r in range(len(XX)):
                w = 1.0 / (dist[r] + 1e-6)
                for jj, ww in zip(ind[r], w):
                    P[r] += ww * Ytr[jj]
                s = P[r].sum()
                if s > 0: P[r] /= s
            return P
        record("knn%d-cosine" % k, prop(Xte), prop(Xva))

    # LightGBM per label
    try:
        t0 = time.time()
        Pva = np.zeros_like(Yva, dtype=np.float32); Pte = np.zeros_like(Yte, dtype=np.float32)
        for j in range(len(vocab)):
            yj = Ytr[:, j]
            if yj.sum() < 3 or yj.sum() == len(yj):
                Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
            m = lgb.LGBMClassifier(n_estimators=120 if quick else 250, learning_rate=.08,
                                   num_leaves=31, subsample=.9, colsample_bytree=.8,
                                   verbose=-1, n_jobs=8, random_state=42)
            m.fit(Xtr, yj)
            Pva[:, j] = m.predict_proba(Xva)[:, 1]
            Pte[:, j] = m.predict_proba(Xte)[:, 1]
        record("lightgbm-ovr", Pte, Pva)
        print("      (lgbm time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   lightgbm failed:", type(e).__name__, str(e)[:80])

    # MLP
    try:
        t0 = time.time()
        m = MLP(d["n_feat"], len(vocab))
        m = train_torch(m, Xtr, Ytr, Xva, Yva, epochs=8 if quick else 30)
        record("mlp", predict_torch(m, Xte), predict_torch(m, Xva))
        print("      (mlp time %.0fs)" % (time.time() - t0))
    except Exception as e:
        print("   mlp failed:", type(e).__name__, str(e)[:80])

    out = {"task": "B4", "meta": d["meta"], "results": res,
           "protocol": {"split": "patient-level 70/15/15 seed=42",
                        "target": "herb set of the NEXT visit",
                        "note": "last-basket is the key competitive baseline"}}
    with open(os.path.join(OUT, "results_B4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("   saved results_B4.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["B1", "B2", "B4"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.task == "B4":
        run_B4(a.quick)
    else:
        run_multilabel(a.task, a.quick)
    print("done in %.0fs" % (time.time() - t0))
