#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: build and cache task tensors for B1/B2 (shared by all model runners)."""
import os, sys, json, re, argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
OUT = os.path.join(ROOT, "bench")
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["B1", "B2"])
    a = ap.parse_args()
    pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
    vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
    vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
    vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
    v2p = dict(zip(pv.visit_id, pv.patient_id))
    V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
    for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
    for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
    for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)
    Xsrc, Ysrc = (V2M, V2Y) if a.task == "B1" else (V2M, V2H)
    rows = [v for v in Xsrc if v in Ysrc]
    pats = {v2p[v] for v in rows}
    rng = np.random.RandomState(42); pl = sorted(pats); rng.shuffle(pl)
    n = len(pl)
    trp, vap, tep = set(pl[:int(.7 * n)]), set(pl[int(.7 * n):int(.85 * n)]), set(pl[int(.85 * n):])
    spl = {v: (0 if v2p[v] in trp else 1 if v2p[v] in vap else 2) for v in rows}
    tr = [v for v in rows if spl[v] == 0]
    va = [v for v in rows if spl[v] == 1]
    te = [v for v in rows if spl[v] == 2]
    xc = Counter(s for v in tr for s in Xsrc[v])
    yc = Counter(y for v in tr for y in Ysrc[v])
    min_lab = 30 if a.task == "B1" else 5
    xv = [s for s, c in xc.most_common() if c >= 5]
    yv = [y for y, c in yc.most_common() if c >= min_lab]
    xi = {s: i for i, s in enumerate(xv)}; yi = {y: i for i, y in enumerate(yv)}
    pos = {v: i for i, v in enumerate(rows)}
    X = np.zeros((len(rows), len(xv)), np.float32)
    Y = np.zeros((len(rows), len(yv)), np.float32)
    for r, v in enumerate(rows):
        for s in Xsrc[v]:
            j = xi.get(s)
            if j is not None: X[r, j] = 1.0
        for y in Ysrc[v]:
            j = yi.get(y)
            if j is not None: Y[r, j] = 1.0
    itr = np.array([pos[v] for v in tr]); iva = np.array([pos[v] for v in va])
    ite = np.array([pos[v] for v in te])
    pat_ids = {p: i for i, p in enumerate(sorted(set(v2p[v] for v in rows)))}
    grp = np.array([pat_ids[v2p[v]] for v in te])
    freq = np.array([yc[y] for y in yv], np.int32)
    np.savez_compressed(os.path.join(OUT, "cache_%s.npz" % a.task),
                        X=X, Y=Y, itr=itr, iva=iva, ite=ite, grp=grp, freq=freq)
    meta = {"task": a.task, "train": len(tr), "val": len(va), "test": len(te),
            "patients": len(pats), "labels": len(yv), "features": len(xv),
            "test_patients": int(len(set(grp))), "min_label_support": min_lab}
    json.dump(meta, open(os.path.join(OUT, "meta_%s.json" % a.task), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
