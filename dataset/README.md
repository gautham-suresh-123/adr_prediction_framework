# Dataset Documentation — ADR Prediction Framework

## 1. Overview & Data Sources

This research framework integrates clinical safety reports and biological reference databases to evaluate adverse drug reaction (ADR) signals (specifically bleeding risk associated with Warfarin and Amiodarone co-prescription):

1. **FDA Adverse Event Reporting System (FAERS)**:
   - **Type**: Spontaneous patient-level safety reporting database (XML E2B format).
   - **Role**: Source dataset for extracting patient demographics, co-prescribed drugs, and adverse event reaction outcomes.
   - **Coverage Available Locally**: 2021–2025 (54 XML quarterly archives).
   - **Coverage Processed**: All 54 XML files (covering years 2021–2025) contribute to `processed_faers_warfarin.csv`.

2. **PharmGKB (Pharmacogenomics Knowledgebase)**:
   - **Type**: Curated genomic dataset of drug pathways, gene relationships, and clinical dosing guidelines.
   - **Role**: Provides biological reference validation for Warfarin (`CYP2C9`, `VKORC1`) and Amiodarone (`CYP2D6`, `CYP3A4`) metabolic and dosing pathways.

3. **SIDER (Side Effect Resource)**:
   - **Type**: Reference database of recorded side effects extracted from drug labels and MedDRA dictionaries.
   - **Role**: Validates whether bleeding is an established documented side effect of Warfarin.

---

## 2. Raw Data Policy & Governance

To maintain a lightweight, GitHub-compliant repository and adhere to external data distribution terms:

- **Raw Datasets (`dataset/faers/`, `dataset/PharmGKB/`, `dataset/sider/`) ARE NOT TRACKED IN GIT**.
- The raw datasets (approx. 40.2 GB) remain stored **locally** on the researcher's workstation.
- Researchers wishing to re-parse or audit the raw FAERS XML archives must obtain the data directly from the official regulatory/academic portals:
  - **FAERS XML Data**: Available from the FDA FAERS Data Files page ([https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files)).
  - **PharmGKB Data**: Available from the official PharmGKB Download Portal ([https://www.pharmgkb.org/downloads](https://www.pharmgkb.org/downloads)).
  - **SIDER Data**: Available from the SIDER 4.1 Side Effect Resource ([http://sideeffects.embl.de/](http://sideeffects.embl.de/)).

---

## 3. Local Directory Structure

```
dataset/
├── README.md                      # This dataset documentation file
├── DATA_SOURCES.md                # Provenance and dataset manifest
├── processed_faers_warfarin.csv   # Lightweight processed CSV (TRACKED BY GIT ~1.04 MB)
│
├── faers/                         # LOCAL ONLY (IGNORED BY GIT)
│   ├── 2021/                      # 6 XML files
│   ├── 2022/                      # 12 XML files
│   ├── 2023/                      # 12 XML files
│   ├── 2024/                      # 12 XML files
│   └── 2025/                      # 12 XML files
│
├── PharmGKB/                      # LOCAL ONLY (IGNORED BY GIT)
│   ├── drugLabels/
│   ├── guidelineAnnotations.json/
│   ├── pathways-biopax/
│   ├── pathways-tsv/
│   ├── pathways.json/
│   └── relationships/
│
└── sider/                         # LOCAL ONLY (IGNORED BY GIT)
    ├── drug_atc.tsv
    ├── drug_names.tsv
    ├── meddra.tsv/
    ├── meddra_all_indications.tsv/
    ├── meddra_all_label_indications.tsv/
    ├── meddra_all_label_se.tsv/
    ├── meddra_all_se.tsv/
    └── meddra_freq.tsv/
```

---

## 4. Processed Dataset (`processed_faers_warfarin.csv`)

The repository includes `dataset/processed_faers_warfarin.csv` (1.04 MB, 31,192 records before polypharmacy filtering; 30,354 records after filtering `num_drugs > 100`).

This dataset enables **Mode A (Experiment Mode)**, allowing researchers to run all disproportionality analyses, propensity score matching, and machine learning models instantly without downloading 40+ GB of raw data.

### Schema:
- `subject_id`: Unique FAERS safety report identifier.
- `age`: Patient age in years (imputed median: 65.0 for missing values).
- `gender`: Biological sex indicator (`1.0` = Male, `0.0` = Female, mode-imputed for missing).
- `drug_a`: Warfarin inclusion indicator (`1` for all cohort rows).
- `drug_b`: Amiodarone co-prescription treatment variable (`1` = Co-prescribed, `0` = Not co-prescribed).
- `drug_c`: Metformin co-prescription control drug variable (`1` = Present, `0` = Absent).
- `aspirin`: Independent concomitant Aspirin indicator (`1` = Present, `0` = Absent).
- `nsaid`: Non-aspirin NSAID indicator (`1` = Present, `0` = Absent).
- `num_drugs`: Polypharmacy drug count (total reported medications for patient).
- `adr_bleeding`: Target outcome indicator (`1` = MedDRA bleeding PT match, `0` = No bleeding PT match).

---

## 5. Processing Pipeline Architecture

```
Raw FAERS XML Archives (2021–2025)
               │
               ▼
   src/data/parser.py (Parallel XML iterparse)
               │
               ▼
   Warfarin Cohort Filtering (drug_a = 1)
               │
               ▼
   Feature Extraction (Age, Gender, Amiodarone, Metformin, Aspirin, Non-Aspirin NSAIDs, Drug Count)
               │
               ▼
   ADR Phenotype Outcome Extraction (MedDRA Preferred Term Keyword Match)
               │
               ▼
   dataset/processed_faers_warfarin.csv
               │
               ▼
   src/data/loader.py (Outlier Filtering num_drugs <= 100)
               │
               ▼
   run_experiments.py (Disproportionality ROR, PSM, XGBoost, Evaluation, Visualization)
```

---

## 6. Raw Data Regeneration (Mode B Workflow)

If you have local access to the raw FAERS XML archives, you can regenerate `processed_faers_warfarin.csv`:

1. Place the XML directories under `dataset/faers/` (e.g. `dataset/faers/2021/`, `dataset/faers/2022/`, etc.).
2. Execute the parser script directly:
   ```bash
   python -c "from src.data.parser import build_processed_dataset; build_processed_dataset()"
   ```
3. The script will scan all XML files in parallel, extract Warfarin safety reports, perform baseline imputation, and save `dataset/processed_faers_warfarin.csv`.
