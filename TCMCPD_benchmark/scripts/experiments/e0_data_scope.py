#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0: data-scope facts used in the paper's Section 2.

Only the facts that the critique actually needs: scale, label coverage, multiplicity,
naming granularity, definition reuse, duplicate edges, unit mixing, date availability.
Writes analysis/e0_data_scope.json
"""
import os, json, re
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.environ.get("TCMCPD_ROOT", "/data/tcmcpd_bench")  # set TCMCPD_ROOT to your checkout
D = os.path.join(ROOT, "data", "TCMCPD_TCM_Case_Records_v1.0")
A = os.path.join(ROOT, "analysis")

vis = pd.read_csv(os.path.join(D, "entities/visits.csv"))
pv = pd.read_csv(os.path.join(D, "relationships/patient_visit.csv"))
vsym = pd.read_csv(os.path.join(D, "relationships/visit_symptom.csv"))
vsyn = pd.read_csv(os.path.join(D, "relationships/visit_syndrome.csv"))
vher = pd.read_csv(os.path.join(D, "relationships/visit_herb.csv"))
esym = pd.read_csv(os.path.join(D, "entities/symptom.csv"))
esyn = pd.read_csv(os.path.join(D, "entities/syndrome.csv"))
eher = pd.read_csv(os.path.join(D, "entities/herb.csv"))

E = {}
E["scale"] = {
    "patients_rows": int(len(pd.read_csv(os.path.join(D, "entities/patients.csv")))),
    "patients_with_visits": int(pv.patient_id.nunique()),
    "visits": int(len(vis)),
    "patients_ge2_visits": int((pv.groupby("patient_id").size() >= 2).sum()),
    "max_visits": int(pv.groupby("patient_id").size().max()),
    "symptom_edges": int(len(vsym)), "syndrome_edges": int(len(vsyn)), "herb_edges": int(len(vher))}

V2Y = {}
for v, y in vsyn[["visit_id", "Syndrome_id"]].values:
    V2Y.setdefault(v, set()).add(y)
E["label_coverage"] = {
    "visits": int(len(vis)),
    "labelled_visits": int(len(V2Y)),
    "labelled_share": round(len(V2Y) / len(vis), 4),
    "multi_syndrome_visits": int(sum(1 for s in V2Y.values() if len(s) > 1)),
    "multi_syndrome_share": round(float(np.mean([len(s) > 1 for s in V2Y.values()])), 4),
    "mean_syndromes_per_labelled_visit": round(float(np.mean([len(s) for s in V2Y.values()])), 3)}

# naming granularity + heterographs
names = esyn.Syndrome_name.fillna("").tolist()
short = [n for n in names if 0 < len(n) <= 3]
yu = [n for n in names if "瘀" in n]; yu2 = [n for n in names if "淤" in n]
pairs = [(a, b) for a in yu for b in yu2 if a.replace("瘀", "淤") == b]
E["naming"] = {"syndromes": len(names),
               "name_length_le3": len(short),
               "share_le3": round(len(short) / len(names), 4),
               "heterograph_pairs": [list(p) for p in pairs],
               "examples_short": short[:12]}

# definition reuse
defs = [d.strip() for d in esyn.Syndrome_definition.fillna("").tolist() if str(d).strip()]
dc = Counter(defs)
E["definition_reuse"] = {"syndromes_with_definition": len(defs),
                         "distinct_definitions": len(dc),
                         "max_times_one_definition_reused": int(dc.most_common(1)[0][1]),
                         "share_reused_gt1": round(float(np.mean([v > 1 for v in dc.values()])), 4)}

# duplicate edges + vocabulary
dup = {}
for tag, df, col in (("symptom", vsym, "TCM_symptom_id"), ("syndrome", vsyn, "Syndrome_id"),
                     ("herb", vher, "Herb_id")):
    dup[tag] = {"raw": int(len(df)),
                "unique_visit_item": int(df.drop_duplicates(["visit_id", col]).shape[0]),
                "duplicates": int(len(df) - df.drop_duplicates(["visit_id", col]).shape[0])}
E["duplicate_edges"] = dup
E["vocabulary"] = {
    "symptom": {"entity": int(len(esym)), "used": int(vsym.TCM_symptom_id.nunique())},
    "syndrome": {"entity": int(len(esyn)), "used": int(vsyn.Syndrome_id.nunique())},
    "herb": {"entity": int(len(eher)), "used": int(vher.Herb_id.nunique())}}
for k in E["vocabulary"]:
    E["vocabulary"][k]["unused"] = E["vocabulary"][k]["entity"] - E["vocabulary"][k]["used"]

# units + dates
u = Counter(vher.unit.fillna("NA"))
E["dosage_units"] = {"counts": {k: int(v) for k, v in u.most_common()},
                     "share_g": round(float(u.get("g", 0) / sum(u.values())), 4)}
FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
txt = vis.content.fillna("").astype(str)
E["dates"] = {"visits": int(len(vis)),
              "with_full_date": int(txt.str.contains(FULL).sum()),
              "with_month_day_only": int((txt.str.contains(MD) & ~txt.str.contains(FULL)).sum()),
              "without_any_date": int((~txt.str.contains(MD)).sum()),
              "share_without": round(float((~txt.str.contains(MD)).mean()), 4)}

json.dump(E, open(os.path.join(A, "e0_data_scope.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(E, ensure_ascii=False, indent=1)[:2400])
print("\nwrote e0_data_scope.json")
