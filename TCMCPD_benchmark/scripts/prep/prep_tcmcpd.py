#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-prep: canonical preprocessing specification for TCMCPD.

Everything that a re-user must know before touching this corpus, plus the exact
derivation rules used by every downstream experiment. Writes a machine-readable
manifest so that the paper's numbers can be reproduced bit-for-bit.

Rules recorded here
  S1 de-duplication: a visit either has an item or not (edge duplicates collapsed)
  S2 effective vocabulary: entity table vs items actually appearing in edges
  S3 label space: support thresholds (B1 >= 30 train visits, B2 >= 5)
  S4 feature space: symptom support >= 5 in train
  S5 patient-level split 70/15/15 with seed 42 (no patient crosses splits)
  S6 date extraction: full date regex + month-day fallback; interval derivation
  S7 dosage units: g kept as grams, other units flagged not converted
Writes analysis/prep_manifest.json
"""
import os, json, re, math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
A = os.path.join(ROOT, "analysis")
M = {}

vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
esym = pd.read_csv(os.path.join(D, "entities/symptom.csv"))
esyn = pd.read_csv(os.path.join(D, "entities/syndrome.csv"))
eher = pd.read_csv(os.path.join(D, "entities/herb.csv"))
v2p = dict(zip(pv.visit_id, pv.patient_id))

# ---- S1 de-duplication
M["S1_dedup"] = {
    "rule": "a visit either contains an item or not; duplicate edges are collapsed before any statistic",
    "edges_raw": {"visit_symptom": len(vsym), "visit_syndrome": len(vsyn), "visit_herb": len(vher)},
    "edges_unique": {"visit_symptom": int(vsym.drop_duplicates(["visit_id", "TCM_symptom_id"]).shape[0]),
                     "visit_syndrome": int(vsyn.drop_duplicates(["visit_id", "Syndrome_id"]).shape[0]),
                     "visit_herb": int(vher.drop_duplicates(["visit_id", "Herb_id"]).shape[0])}}
M["S1_dedup"]["duplicate_edges"] = {k: M["S1_dedup"]["edges_raw"][k] - M["S1_dedup"]["edges_unique"][k]
                                    for k in M["S1_dedup"]["edges_raw"]}

V2M, V2Y, V2H = (defaultdict(set) for _ in range(3))
for v, s in vsym[["visit_id", "TCM_symptom_id"]].values: V2M[v].add(s)
for v, y in vsyn[["visit_id", "Syndrome_id"]].values: V2Y[v].add(y)
for v, h in vher[["visit_id", "Herb_id"]].values: V2H[v].add(h)

# ---- S2 effective vocabulary
M["S2_vocabulary"] = {
    "rule": "the entity table is a dictionary, not the working vocabulary; report both",
    "symptom": {"entity": int(len(esym)), "used_in_edges": int(len({s for st in V2M.values() for s in st}))},
    "syndrome": {"entity": int(len(esyn)), "used_in_edges": int(len({y for st in V2Y.values() for y in st}))},
    "herb": {"entity": int(len(eher)), "used_in_edges": int(len({h for st in V2H.values() for h in st}))}}
for k in ("symptom", "syndrome", "herb"):
    v = M["S2_vocabulary"][k]
    v["unused"] = v["entity"] - v["used_in_edges"]

# ---- S3 label space + S5 split (recomputed exactly as the benchmark does)
def split_with(task):
    Xs, Ys = (V2M, V2Y) if task == "B1" else (V2M, V2H)
    rows = [v for v in Xs if v in Ys]
    pats = sorted({v2p[v] for v in rows})
    rng = np.random.RandomState(42); p = pats[:]; rng.shuffle(p)
    n = len(p)
    trp, vap, tep = set(p[:int(.7 * n)]), set(p[int(.7 * n):int(.85 * n)]), set(p[int(.85 * n):])
    tr = [v for v in rows if v2p[v] in trp]
    yc = Counter(y for v in tr for y in Ys[v])
    xc = Counter(s for v in tr for s in Xs[v])
    return tr, yc, xc, len(pats)


M["S3_S5_task_specs"] = {}
for task, lmin in (("B1", 30), ("B2", 5)):
    tr, yc, xc, npat = split_with(task)
    M["S3_S5_task_specs"][task] = {
        "label_min_train_support": lmin,
        "feature_min_train_support": 5,
        "split": "patient-level 70/15/15, seed 42",
        "patients": npat,
        "train_visits": len(tr),
        "labels_kept": int(sum(1 for c in yc.values() if c >= lmin)),
        "features_kept": int(sum(1 for c in xc.values() if c >= 5)),
        "preferred_model": "label-graph+ASL" if task == "B1" else "label-graph(ML-GCN)"}

# ---- S6 dates
FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
has_full = vis.content.fillna("").astype(str).str.contains(FULL).sum()
has_md = vis.content.fillna("").astype(str).str.contains(MD).sum()
M["S6_dates"] = {"full_date_regex": FULL.pattern, "month_day_regex": MD.pattern,
                 "visits": int(len(vis)),
                 "visits_with_full_date": int(has_full),
                 "visits_with_month_day_only": int(has_md - has_full),
                 "visits_with_any_date": int(has_md),
                 "visits_without_date": int(len(vis) - has_md),
                 "interval_policy": "intervals computed only when both consecutive visits carry a full date; "
                                    "inferred-year intervals excluded from the primary estimate"}

# ---- S7 dosage units
u = Counter(vher.unit.fillna("NA"))
M["S7_dosage"] = {"units": {k: int(v) for k, v in u.most_common()},
                  "share_g": round(float(u.get("g", 0) / sum(u.values())), 4),
                  "rule": "only grams are used as grams; other units are NOT converted (no reliable conversion)"}

# ---- label multiplicity / coverage
labelled = set(V2Y) & set(v2p)
M["coverage"] = {
    "visits": int(len(vis)),
    "visits_with_syndrome_label": int(len(V2Y)),
    "labelled_share": round(len(V2Y) / len(vis), 4),
    "visits_with_symptom": int(len(V2M)),
    "visits_with_herb": int(len(V2H)),
    "labelled_visits_with_multiple_syndromes": int(sum(1 for v in V2Y if len(V2Y[v]) > 1)),
    "multi_label_share_among_labelled": round(float(np.mean([len(V2Y[v]) > 1 for v in V2Y])), 4),
    "patients_with_ge2_visits": int(sum(1 for p, g in pv.groupby("patient_id") if len(g) >= 2)),
    "max_visits_per_patient": int(pv.groupby("patient_id").size().max())}

json.dump(M, open(os.path.join(A, "prep_manifest.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1)[:2600])
print("\nwrote prep_manifest.json")
