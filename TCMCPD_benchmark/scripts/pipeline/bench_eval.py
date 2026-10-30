#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: unified evaluation of all saved predictions (py3.8 models + optional TabPFN).

Produces point estimates, patient-level bootstrap 95% CI, paired tests vs the prevalence
prior, and long-tail strata (head/mid/tail macro-F1 by train label frequency).
"""
import os, sys, json, argparse, glob
import numpy as np

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
OUT = os.path.join(ROOT, "bench")
KEYS = ["P@1", "P@3", "P@5", "R@5", "R@10", "mAP@10", "microF1", "macroF1",
        "macroF1_head", "macroF1_mid", "macroF1_tail"]


def rank_metrics(S, Y, ks=(1, 3, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(Y, order[:, :k], axis=1)
        out["P@%d" % k] = float(hit.sum() / (k * len(Y)))
        out["R@%d" % k] = float(hit.sum() / max(1.0, Y.sum()))
    rel = np.take_along_axis(Y, order[:, :10], axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, 11)[None, :]
    den = rel.sum(1); ok = den > 0
    out["mAP@10"] = float(((prec * rel).sum(1)[ok] / den[ok]).mean()) if ok.any() else 0.0
    return out


def per_label_f1(P, Y, thr):
    pr = P >= thr; yb = Y > 0
    tp = (pr & yb).sum(0).astype(float); fp = (pr & ~yb).sum(0).astype(float)
    fn = (~pr & yb).sum(0).astype(float)
    pp = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * pp * rr, pp + rr, out=np.zeros_like(tp), where=(pp + rr) > 0)


def micro_macro(P, Y, thr):
    pr = P >= thr
    tp = (pr & (Y > 0)).sum(); fp = (pr & (Y == 0)).sum(); fn = ((~pr) & (Y > 0)).sum()
    micro = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return float(micro), float(per_label_f1(P, Y, thr).mean())


def tune_thr(Pv, Yv):
    best = (-1, .5)
    for t in (.03, .05, .08, .1, .15, .2, .25, .3, .4, .5):
        m, _ = micro_macro(Pv, Yv, t)
        if m > best[0]: best = (m, t)
    return best[1]


def strata(P, Y, thr, freq):
    f1 = per_label_f1(P, Y, thr)
    idx = np.argsort(-freq)
    return {("macroF1_" + n): float(f1[p].mean())
            for n, p in zip(("head", "mid", "tail"), np.array_split(idx, 3))}


def full_metrics(P, Y, freq, thr):
    r = rank_metrics(P, Y); mi, ma = micro_macro(P, Y, thr)
    r["microF1"] = mi; r["macroF1"] = ma; r.update(strata(P, Y, thr, freq))
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task", choices=["B1", "B2"])
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    c = np.load(os.path.join(OUT, "cache_%s.npz" % a.task))
    Yva, Yte, grp, freq = c["Y"][c["iva"]], c["Y"][c["ite"]], c["grp"], c["freq"]
    meta = json.load(open(os.path.join(OUT, "meta_%s.json" % a.task)))

    files = sorted(glob.glob(os.path.join(OUT, "preds_%s_*.npz" % a.task)))
    preds = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("te_"):
                nm = k[3:]
                vk = "va_" + nm
                if vk in z.files:
                    preds[nm] = (z[vk], z[k])
    print("[%s] %d models from %d files" % (a.task, len(preds), len(files)))
    thrs = {nm: tune_thr(Pv, Yva) for nm, (Pv, Pt) in preds.items()}
    point = {nm: full_metrics(Pt, Yte, freq, thrs[nm]) for nm, (Pv, Pt) in preds.items()}

    up = np.unique(grp); idx_by = {u: np.where(grp == u)[0] for u in up}
    rng = np.random.RandomState(7)
    boot = {nm: {k: [] for k in KEYS} for nm in preds}
    for b in range(a.boot):
        pick = rng.choice(up, size=len(up), replace=True)
        rows = np.concatenate([idx_by[u] for u in pick])
        Yb = Yte[rows]
        for nm, (Pv, Pt) in preds.items():
            r = full_metrics(Pt[rows], Yb, freq, thrs[nm])
            for k in KEYS: boot[nm][k].append(r[k])
    ci = {nm: {k: [float(np.percentile(boot[nm][k], 2.5)), float(np.percentile(boot[nm][k], 97.5))]
               for k in KEYS} for nm in preds}
    vs = {}
    base = "prior(traincidence)"
    for nm in preds:
        if nm == base: continue
        vs[nm] = {}
        for k in ("P@1", "R@10", "mAP@10", "microF1", "macroF1"):
            d = np.array(boot[nm][k]) - np.array(boot[base][k])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            vs[nm][k] = {"delta": float(np.mean(d)), "p": float(max(p, 1.0 / a.boot))}

    order = sorted(point, key=lambda x: -point[x]["R@10"])
    print("\n%-26s %6s %6s %7s %6s %6s | %5s %5s %5s"
          % ("model", "P@1", "R@10", "mAP@10", "micF1", "macF1", "head", "mid", "tail"))
    for nm in order:
        r = point[nm]
        print("%-26s %.3f %.3f %.3f %.3f %.3f | %.3f %.3f %.3f"
              % (nm[:26], r["P@1"], r["R@10"], r["mAP@10"], r["microF1"], r["macroF1"],
                 r["macroF1_head"], r["macroF1_mid"], r["macroF1_tail"]))
    print("\npaired vs prior:")
    for nm in order:
        if nm == base: continue
        s = vs[nm]
        print("  %-26s dR@10 %+0.3f p=%.3f | dmacF %+0.3f p=%.3f | dmAP %+0.3f p=%.3f"
              % (nm[:26], s["R@10"]["delta"], s["R@10"]["p"], s["macroF1"]["delta"],
                 s["macroF1"]["p"], s["mAP@10"]["delta"], s["mAP@10"]["p"]))

    out = {"task": a.task, "meta": meta,
           "protocol": {"split": "patient-level 70/15/15 seed=42; vocab from train only",
                        "threshold": "tuned on val (micro-F1 grid)",
                        "bootstrap": "%d patient-level resamples" % a.boot,
                        "sources": [os.path.basename(f) for f in files]},
           "point": point, "ci95": ci, "vs_prior": vs}
    json.dump(out, open(os.path.join(OUT, "results_%s.json" % a.task), "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved results_%s.json" % a.task)


if __name__ == "__main__":
    main()
