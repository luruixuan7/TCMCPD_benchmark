#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Partial F test for the structural increment in the per-label decomposition (P2).

The paper reports R^2 for two nested regressions: log training frequency alone, and log
frequency plus graph degree plus co-occurrence entropy. Reporting only the two R^2 values
leaves open whether the increment is distinguishable from zero. Both models are nested, so a
partial F test answers that question exactly from quantities already reported (R^2 of the two
models, the number of predictors, and the number of labels) and needs no refitting.

Writes results/e1_e2_r2_increment.json
"""
import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[2]
P = json.load(open(ROOT / "results" / "e1_e2_patterns.json", encoding="utf-8"))["P2"]

OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

K_RED, K_FULL = 1, 3          # log f  |  log f + degree + entropy  (intercepts excluded)
res = {"model": "AP proxy ~ log frequency (+ degree + entropy)", "alpha": 0.05, "tasks": {}}
for task, d in P.items():
    n = int(d["n_labels"])
    r2r, r2f = float(d["r2_log_frequency_only"]), float(d["r2_full"])
    df1, df2 = K_FULL - K_RED, n - K_FULL - 1
    F = ((r2f - r2r) / df1) / ((1.0 - r2f) / df2)
    p = float(fdist.sf(F, df1, df2))
    res["tasks"][task] = {"n_labels": n, "r2_reduced": r2r, "r2_full": r2f,
                          "delta_r2": r2f - r2r, "df": [df1, df2],
                          "F": float(F), "p_value": p,
                          "share_of_explained_from_structure": (r2f - r2r) / r2f}
    print("%s  n=%-4d R2 %.3f -> %.3f  dR2=%.3f  F(%d,%d)=%.2f  p=%.3g  share=%.0f%%"
          % (task, n, r2r, r2f, r2f - r2r, df1, df2, F, p, 100 * (r2f - r2r) / r2f))

json.dump(res, open(OUT / "e1_e2_r2_increment.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwrote results/e1_e2_r2_increment.json")
