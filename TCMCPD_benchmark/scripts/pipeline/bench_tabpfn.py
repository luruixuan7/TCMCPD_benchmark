#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TabPFN v2 (Nature, 2025) runner - must be executed inside the py3.10 conda env.

TabPFN is a tabular foundation model; it has no native multi-label head, so we fit one
binary classifier per label (documented in the benchmark protocol).
Feature cap: TabPFN's context window is limited, so we keep the top-K symptom features
by training frequency (K configurable, default 300).
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
OUT = os.path.join(ROOT, "bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["B1", "B2"])
    ap.add_argument("--feat-cap", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-labels", type=int, default=0)
    a = ap.parse_args()
    from tabpfn import TabPFNClassifier
    import tabpfn
    print("tabpfn version:", getattr(tabpfn, "__version__", "?"))

    c = np.load(os.path.join(OUT, "cache_%s.npz" % a.task))
    X, Y = c["X"], c["Y"]
    itr, iva, ite = c["itr"], c["iva"], c["ite"]
    Xtr, Ytr = X[itr], Y[itr]
    Xva, Xte = X[iva], X[ite]
    # feature cap by training frequency
    if X.shape[1] > a.feat_cap:
        order = np.argsort(-Xtr.sum(0))[:a.feat_cap]
        Xtr, Xva, Xte = Xtr[:, order], Xva[:, order], Xte[:, order]
    print("[%s] X %s | labels %d | features used %d"
          % (a.task, Xtr.shape, Y.shape[1], Xtr.shape[1]))

    n_lab = Y.shape[1]
    labels = list(range(n_lab))
    if a.max_labels:
        labels = labels[:a.max_labels]
    Pva = np.zeros((len(Xva), n_lab), np.float32)
    Pte = np.zeros((len(Xte), n_lab), np.float32)
    prior = Ytr.mean(0)
    t0 = time.time()
    done = 0
    for j in labels:
        yj = Ytr[:, j]
        if yj.sum() < 3 or yj.sum() == len(yj):
            Pva[:, j] = yj.mean(); Pte[:, j] = yj.mean(); continue
        try:
            clf = TabPFNClassifier(device=a.device, ignore_pretraining_limits=True)
            clf.fit(Xtr, yj.astype(int))
            Pva[:, j] = clf.predict_proba(Xva)[:, 1]
            Pte[:, j] = clf.predict_proba(Xte)[:, 1]
            done += 1
        except Exception as e:
            if j == labels[0]:
                print("  TabPFN failed on first label:", type(e).__name__, str(e)[:200])
            Pva[:, j] = prior[j]; Pte[:, j] = prior[j]
        if (j + 1) % 25 == 0:
            print("  %d/%d labels, %.0fs" % (j + 1, len(labels), time.time() - t0))
    np.savez_compressed(os.path.join(OUT, "preds_%s_tabpfn.npz" % a.task),
                        **{"va_TabPFNv2(2025)": Pva, "te_TabPFNv2(2025)": Pte})
    print("done %d/%d labels in %.0fs -> preds_%s_tabpfn.npz"
          % (done, len(labels), time.time() - t0, a.task))


if __name__ == "__main__":
    main()
