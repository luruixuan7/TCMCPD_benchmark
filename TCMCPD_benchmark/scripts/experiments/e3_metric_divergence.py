#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3: metric divergence table for the critique.

For each task and each metric: how many of the 18 comparison models are significantly
BETTER / NOT DIFFERENT / significantly WORSE than the prevalence prior, plus the
equivalence statistics for the best model.

Writes analysis/e3_metric_divergence.json
"""
import os, json

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
B = os.path.join(ROOT, "bench")
A = os.path.join(ROOT, "analysis")

OUT = {"tasks": {}}
EQ = json.load(open(os.path.join(B, "equivalence.json"), encoding="utf-8"))

for task, f in (("B1", "results_B1.json"), ("B2", "results_B2.json")):
    J = json.load(open(os.path.join(B, f), encoding="utf-8"))
    vs = J["vs_prior"]
    n = len(vs)
    block = {"n_comparisons": n, "metrics": {}}
    for k in ("P@1", "R@10", "mAP@10", "microF1", "macroF1"):
        better = sum(1 for v in vs.values() if v[k]["p"] < 0.05 and v[k]["delta"] > 0)
        worse = sum(1 for v in vs.values() if v[k]["p"] < 0.05 and v[k]["delta"] < 0)
        block["metrics"][k] = {"sig_better": better, "sig_worse": worse,
                               "not_different": n - better - worse,
                               "median_delta": float(__import__("numpy").median(
                                   [v[k]["delta"] for v in vs.values()]))}
    # best model by R@10 and its equivalence stats
    best = max(J["point"], key=lambda m: J["point"][m]["R@10"])
    block["best_model_by_R@10"] = {
        "name": best,
        "R@10": round(J["point"][best]["R@10"], 4),
        "macroF1": round(J["point"][best]["macroF1"], 4),
        "mAP@10": round(J["point"][best]["mAP@10"], 4),
        "prior_R@10": round(J["point"]["prior(traincidence)"]["R@10"], 4),
        "prior_mAP": round(J["point"]["prior(traincidence)"]["mAP@10"], 4)}
    eqk = best.lower()
    for cand in EQ[task]["models"]:
        if cand.lower().startswith(eqk.split("(")[0].lower()):
            block["best_model_by_R@10"]["equivalence"] = EQ[task]["models"][cand]
            break
    OUT["tasks"][task] = block
    print("[%s] %d models vs prior" % (task, n))
    for k, v in block["metrics"].items():
        print("   %-8s better %2d | n.s. %2d | worse %2d | median delta %+0.4f"
              % (k, v["sig_better"], v["not_different"], v["sig_worse"], v["median_delta"]))
    print("   best R@10 model: %s (%.4f vs prior %.4f)"
          % (best, block["best_model_by_R@10"]["R@10"], block["best_model_by_R@10"]["prior_R@10"]))

json.dump(OUT, open(os.path.join(A, "e3_metric_divergence.json"), "w"), ensure_ascii=False, indent=1)
print("\nwrote e3_metric_divergence.json")
