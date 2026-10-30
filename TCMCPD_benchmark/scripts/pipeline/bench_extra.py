#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extra recent model: ModernNCA-style (2024) = learned metric + retrieval kNN.

Trains an embedding with a proxy/label-embedding objective on BCE, then predicts by
retrieving top-k training neighbours in the learned space and averaging their label
vectors (this is the core of ModernNCA / TabR-style retrieval models for tabular data).
Saves preds_<task>_extra.npz so the evaluator picks it up automatically.
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42); torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench"), "bench")


class Enc(nn.Module):
    def __init__(self, d_in, d_emb=64, n_lab=1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(.2),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, d_emb))
        self.proxy = nn.Parameter(torch.randn(n_lab, d_emb) * .1)
        self.bias = nn.Parameter(torch.zeros(n_lab))
    def forward(self, x):
        z = nn.functional.normalize(self.enc(x), dim=-1)
        p = nn.functional.normalize(self.proxy, dim=-1)
        return z, z @ p.t() * 5.0 + self.bias


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task", choices=["B1", "B2"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()
    z = np.load(os.path.join(OUT, "cache_%s.npz" % a.task))
    X, Y = z["X"], z["Y"]; itr, iva, ite = z["itr"], z["iva"], z["ite"]
    Xtr, Ytr = X[itr], Y[itr]; Xva, Yva = X[iva], Y[iva]; Xte = X[ite]
    n_lab = Y.shape[1]
    m = Enc(X.shape[1], 64, n_lab).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEV); Yt = torch.tensor(Ytr, device=DEV)
    Xv = torch.tensor(Xva, device=DEV); Yv = torch.tensor(Yva, device=DEV)
    best = (1e9, None)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            _, logits = m(Xt[idx])
            loss = lossf(logits, Yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            _, lv = m(Xv)
            v = lossf(lv, Yv).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    if best[1]: m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad():
        Ztr, _ = m(Xt); Zva, _ = m(Xv); Zte, _ = m(torch.tensor(Xte, device=DEV))
    Ztr = Ztr.cpu().numpy(); Zva = Zva.cpu().numpy(); Zte = Zte.cpu().numpy()

    def retrieve(Zq):
        S = Zq @ Ztr.T                      # cosine (already normalised)
        k = min(a.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        P = np.zeros((len(Zq), n_lab), np.float32)
        r = np.arange(len(Zq))[:, None]
        w = np.exp(S[r, idx] * 8.0)
        w /= w.sum(1, keepdims=True)
        for j in range(k):
            P += w[:, j:j + 1] * Ytr[idx[:, j]]
        return P

    Pva, Pte = retrieve(Zva), retrieve(Zte)
    np.savez_compressed(os.path.join(OUT, "preds_%s_extra.npz" % a.task),
                        **{"va_ModernNCA(2024)": Pva, "te_ModernNCA(2024)": Pte})
    print("[%s] ModernNCA-style done in %.0fs -> preds_%s_extra.npz" % (a.task, time.time() - t0, a.task))


if __name__ == "__main__":
    main()
