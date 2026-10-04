# FAERS-Based ADR Signal Detection and Predictive Modeling Framework

A research framework for adverse drug reaction (ADR) analysis using FDA Adverse Event Reporting System (FAERS) safety reports, disproportionality signal statistics, propensity score matching (PSM), machine learning, and biological reference knowledgebases (**PharmGKB** and **SIDER**).

---

## 1. Study Design Overview

* **Study Target**: Evaluation of bleeding risk associated with concurrent exposure to **Amiodarone** among patients on **Warfarin** therapy.
* **Target Population**: Retrospective patient-level cohort extracted from FAERS XML safety reports covering years **2021 to 2025** ($N = 30,354$ after excluding polypharmacy outliers with $>100$ reported medications).
* **Primary Exposure ($T$)**: Co-prescription of **Amiodarone** ($T = 1$), a strong CYP2C9 metabolic inhibitor that impairs Warfarin clearance.
* **Control Group ($T = 0$)**: Warfarin patients without Amiodarone co-prescription.
* **Covariates ($\mathbf{X}$)**: Patient age, biological sex, concomitant Aspirin, concomitant non-aspirin NSAIDs, polypharmacy drug count, and Metformin exposure (baseline non-interacting control drug).
* **Target Outcome ($Y$)**: **Bleeding** events identified via MedDRA Preferred Term (PT) keyword matching.
* **Reference Databases**:
  - **SIDER**: Cross-references documented side effect terms for Warfarin.
  - **PharmGKB**: Queries pharmacogenomic guidelines and gene relationships (`CYP2C9`, `VKORC1`, `CYP2D6`, `CYP3A4`).

---

## 2. Dataset Strategy & Data Modes

The repository enforces a clean separation between lightweight tracked files and large local datasets:

* **TRACKED IN GITHUB**:
  - `dataset/processed_faers_warfarin.csv` (1.04 MB): Pre-extracted processed dataset containing clinical features, exposures, and bleeding outcomes.
  - `dataset/README.md` and `dataset/DATA_SOURCES.md`: Dataset documentation and provenance manifest.
* **EXCLUDED FROM GITHUB (LOCAL ONLY)**:
  - `dataset/faers/` (~39.8 GB): Raw FAERS XML archives (2021–2025).
  - `dataset/PharmGKB/` (~350 MB): PharmGKB reference files.
  - `dataset/sider/` (~80 MB): SIDER side effect reference files.

### Operating Modes:
* **MODE A — EXPERIMENT MODE (Default)**:
  Runs directly using `dataset/processed_faers_warfarin.csv`. Does NOT require downloading 40+ GB of raw data.
* **MODE B — RAW DATA REGENERATION MODE**:
  Allows researchers with local raw FAERS XML archives under `dataset/faers/` to re-parse and regenerate the processed dataset from scratch using `src/data/parser.py`.

---

## 3. Repository Structure

```
adr_prediction_framework/
│
├── dataset/
│   ├── README.md                      # Dataset documentation and raw data policy
│   ├── DATA_SOURCES.md                # Data provenance manifest
│   └── processed_faers_warfarin.csv   # Compact processed CSV dataset (Tracked, ~1.04 MB)
│
├── src/
│   ├── causal/
│   │   └── discovery.py               # Placeholder for future DAG structure learning
│   ├── data/
│   │   ├── loader.py                  # Clinical data loader & outlier filtering
│   │   └── parser.py                  # Parallelized FAERS XML parser & reference validator
│   ├── evaluation/
│   │   └── metrics.py                 # Classification & calibration (ECE, Brier) metrics
│   ├── models/
│   │   ├── causal.py                  # Disproportionality engine (Crude ROR, Adjusted ROR, PSM ROR, SMD)
│   │   └── predictive.py              # XGBoost predictive risk classifier
│   └── utils/
│       └── visualization.py           # Publication plots (Calibration, ROC, PR, Forest, SMD)
│
├── results/                           # Generated experiment outputs & summary reports
├── scripts/
│   └── check_dataset.py               # Dataset verification & environment check script
├── tests/
│   └── test_pipeline.py               # Automated unit tests for dataset & model logic
│
├── .gitignore                         # Configured git ignore rules for raw data safety
├── README.md                          # Main repository documentation
├── methodology.md                     # Scientific research methodology document
├── requirements.txt                   # Verified Python package dependencies
└── run_experiments.py                 # Main experiment orchestrator script
```

---

## 4. Setup and Installation

### Prerequisites
* Python >= 3.10

### Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/gautham-suresh-123/adr_prediction_framework.git
cd adr_prediction_framework

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 5. Running the Pipeline

### Step 1: Verify Dataset Status
```bash
python scripts/check_dataset.py
```

### Step 2: Run Unit Tests
```bash
python -m unittest discover tests
```

### Step 3: Run Full Experiment Pipeline
```bash
python run_experiments.py
```

Outputs are automatically generated and saved in `./results/`.

---

## 6. Scientific Summary & Results

| Metric / Analysis | Result Value | Interpretation |
|:---|:---|:---|
| **Crude ROR** | 0.6962 (95% CI: 0.5372–0.9023, $p = 7.10 \times 10^{-3}$) | Unadjusted disproportionality signal on test set. |
| **Adjusted ROR** | 0.7100 (95% CI: 0.5444–0.9259, $p = 1.15 \times 10^{-2}$) | Multivariable logistic regression adjusting for age, gender, aspirin, NSAIDs, polypharmacy. |
| **PSM ROR** | 0.7940 (95% CI: 0.5626–1.1208, $p = 0.220$) | 1:1 nearest-neighbor propensity score matched cohort ($N=1,038$). |
| **Matched Signal Test** | Confidence Interval includes 1.0 | Matched analysis does **not** show a statistically significant disproportionality association after PSM balance. |
| **XGBoost ROC-AUC** | 0.6974 | Moderate discrimination for predicting bleeding reports. |
| **XGBoost PR-AUC** | 0.3945 | Precision-Recall AUC under natural class imbalance. |
| **Precision / Recall / F1** | Prec: 0.6734 / Rec: 0.1445 / F1: 0.2379 | High precision but low sensitivity at default 0.50 probability threshold. |
| **ECE / Brier Score** | ECE = 0.87% (0.00874) / Brier = 0.1295 | Excellent probability calibration when evaluated without artificial oversampling. |

---

## 7. Methodological Limitations

1. **Spontaneous Reporting Bias**: FAERS consists of voluntary unverified reports, lacking exposed population denominators and non-adverse event control normalization.
2. **Observational Confounding**: While multivariable logistic regression and PSM control for measured baseline covariates (age, sex, key drugs, polypharmacy), unmeasured confounding (e.g. baseline INR, dose variation, lab values) remains.
3. **Keyword Phenotype Matching**: Bleeding phenotypes are identified via string matching on MedDRA Preferred Terms rather than complete hierarchical MedDRA ontology expansion.
4. **Predictive Sensitivity**: ROC-AUC $\approx 0.70$ demonstrates moderate discrimination; low recall at default threshold highlights that risk prediction scores should be combined with clinical monitoring rather than sole decision rules.

---

## 8. License & Citation

Distributed under the MIT License. If using this framework in academic research, please cite this repository.
