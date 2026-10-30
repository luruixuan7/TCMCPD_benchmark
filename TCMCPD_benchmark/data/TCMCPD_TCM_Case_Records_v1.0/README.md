# TCMCPD dataset

## Description

This dataset contains structured Traditional Chinese Medicine case record entities and pharmacological association data used in the manuscript "TCMCPD: An Integrated TCM Case records and Pharmacology Database".

## Data language

Some clinical entity fields are in Chinese. English, pinyin or Latin names are provided where available. Field definitions are provided in `data_dictionary.xlsx`.

## File structure

- `entities/`: entity tables for patients, visits, TCM syndromes, TCM symptoms, TCM herbs, ingredients and targets.
- `relationships/`: relationship tables linking patients, visits, syndromes, symptoms, herbs, ingredients and targets.
- `graph/`: Neo4j-compatible node and edge files.

## Files

### Entity tables

- `entities/patients.csv`
- `entities/visits.csv`
- `entities/syndrome.csv`
- `entities/symptom.csv`
- `entities/herb.csv`
- `entities/ingredients.csv`
- `entities/target.csv`

### Relationship tables

- `relationships/patient_visit.csv`
- `relationships/visit_syndrome.csv`
- `relationships/visit_symptom.csv`
- `relationships/visit_herb.csv`
- `relationships/herb_ingredient.csv`
- `relationships/ingredient_target.csv`

### Graph import files

- `graph/neo4j_nodes.csv`
- `graph/neo4j_edges.csv`

## Reuse note

Users should cite both the associated Scientific Data article and this dataset DOI when using the dataset.

## Copyright note

Original full case texts from published books are not redistributed if copyright restrictions apply. The released dataset contains structured and standardized entities and relationship tables derived from the case records.

## Data quality note

The `entities/visits.csv` file includes a `content` field containing clinical case-record text available in the local source export. Please review this field before public release if copyright restrictions apply. The `entities/patients.csv` `patient_information` field appears to contain text affected by a prior character-encoding conversion in the source export; patient identifiers remain available for joining with visits.
