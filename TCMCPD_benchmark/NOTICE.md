# NOTICE

## Dataset

This project redistributes **TCMCPD** unmodified.

- Title: TCMCPD — TCM case records (syndrome / symptom / herb entities and relations)
- Source: Zenodo, DOI `10.5281/zenodo.20929034`
- License: **CC BY 4.0** (redistribution permitted with attribution)
- Local copy: `data/TCMCPD_TCM_Case_Records_v1.0/` (18 files, unchanged: `entities/`,
  `relationships/`, data dictionary, provenance and citation)
- Not included: `graph/neo4j_nodes.csv` and `graph/neo4j_edges.csv` (48 MiB, a Neo4j import
  export redundant with `relationships/`, and above the host's single-file upload limit);
  both ship in the Zenodo archive
- Per-file checksums of the local copy: see `CHECKSUMS.sha256`

The dataset's own `PROVENANCE.md` and `CITATION.md` are kept inside the dataset
folder and must be cited together with the dataset.

## Code

All scripts in `scripts/` were written for this project.
They read the dataset and write the JSON artifacts under `results/`.

## Scope statement

The relations (symptoms, syndromes) in the dataset were extracted from case-record
text by the dataset authors. Everything computed here is therefore an **in-dataset
co-occurrence** statistic, not a clinical or causal claim.
