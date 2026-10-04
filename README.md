# Causal Adverse Drug Reaction (ADR) Prediction Framework (Real Clinical Data)

This clinical research project implements a safety report analysis framework to estimate adverse drug reaction (ADR) risk and causal treatment effects. It uses the real clinical datasets provided: **FAERS** (spontaneous patient-level safety reports), **PharmGKB** (pharmacogenomics and clinical dosing relationships), and **SIDER** (reference database of known side effects).

The framework compares traditional machine learning (XGBoost risk prediction) with Causal Inference (Double Machine Learning Causal Forest) to evaluate drug safety risks while adjusting for clinical confounding-by-indication.

---

## Study Design Overview

* **Cohort Inclusion**: Real patient reports from FAERS XML files (covering years 2021 to 2025) of patients exposed to the anticoagulant **Warfarin**.
* **Treatment of Interest ($T$)**: Co-prescription of **Amiodarone** (a strong CYP2C9 inhibitor which impairs Warfarin clearance, increasing bleeding risk).
* **Control Baseline**: Exposure to the non-interacting control medication **Metformin**.
* **Covariates & Confounders ($\mathbf{X}$)**: Patient age, gender, concomitant use of Aspirin or NSAIDs (which independently increase bleeding risk), and overall patient polypharmacy/morbidity (total concurrent drug count).
* **Outcome ($Y$)**: Occurrence of **Bleeding** events (MedDRA preferred terms matching bleeding, haemorrhage, hematoma, etc.).
* **Validation Databases**:
  - **SIDER**: Cross-references SIDER's known side effects for Warfarin to confirm if bleeding is documented.
  - **PharmGKB**: Queries guideline-level dosing and metabolic pathways for Warfarin and Amiodarone with key genes (`CYP2C9`, `VKORC1`, `CYP2D6`, `CYP3A4`).

---

## Project Directory Structure

```
adr_prediction_framework/
│
├── dataset/                   # Input clinical datasets (provided by user)
│   ├── faers/                 # FAERS XML folders grouped by year (2021 to 2025) containing quarterly subfolders
│   ├── PharmGKB/              # PharmGKB pathways, guideline annotations, relationships
│   ├── sider/                 # SIDER drug names, ATC classifications, and side effects
│   └── processed_faers_warfarin.csv  # Cached patient cohort (automatically generated)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py          # Parallelized FAERS XML parser & SIDER/PharmGKB checker
│   │   └── loader.py          # Real data loader, feature builder, and oversampler
│   │
│   └── models/
│       ├── __init__.py
│       ├── predictive.py      # XGBoost predictive risk classifier
│       └── causal.py          # Causal Forest (EconML) DML treatment effect estimator
│
├── results/                   # Generated experiment outputs
│   ├── cohort_demographics.csv # Table 1: Baseline Cohort Characteristics
│   ├── subgroup_analysis.csv  # Table 2: Causal Effect Subgroup Analysis
│   ├── calibration_curve.png  # XGBoost probability calibration curve
│   ├── roc_curve.png          # XGBoost ROC classification curve
│   ├── pr_curve.png           # XGBoost Precision-Recall curve
│   ├── feature_importance.png # XGBoost feature importances plot
│   ├── ite_distribution.png   # Causal Forest treatment effect density plot
│   ├── estimated_ites.csv     # Tabular estimated treatment effects per patient
│   ├── predictive_performance_metrics.csv  # Tabular XGBoost evaluation metrics
│   └── experiment_summary.md  # Comprehensive research summary and reference cross-checks
│
├── run_experiments.py         # Main pipeline orchestrator script
├── methodology.md             # Scientific research methodology document
└── README.md                  # This file
```

---

## Setup and Installation

### Prerequisites
* Python 3.10.x
* Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `econml`, `matplotlib`, `seaborn`, `scipy`, `tabulate`

### Installation
Install the dependencies using:
```bash
pip install pandas numpy scikit-learn xgboost econml matplotlib seaborn scipy tabulate
```

---

## Usage

To execute the entire clinical research experiments pipeline, run:
```bash
python run_experiments.py
```

### Execution Steps:
1. **Data Load & Cache**: Checks if the parsed dataset `dataset/processed_faers_warfarin.csv` exists. If not, it runs a process-level parallel parser over all 54 XML files in the FAERS directory to build the cohort (takes ~5 minutes). If cached, it loads instantly.
2. **Causal Forest Estimation**: Trains a **Causal Forest DML** model on the balanced real dataset to estimate the heterogeneous treatment effect (Risk Difference) of Amiodarone co-prescription on Bleeding. Saves the density plot to `results/ite_distribution.png`.
3. **Predictive Risk Modeling**: Trains an **XGBoost Classifier** to predict bleeding probability. Saves evaluation curves to `results/calibration_curve.png`, `results/roc_curve.png`, `results/pr_curve.png`, and the feature importances to `results/feature_importance.png`.
4. **Reference Database Cross-Check**: Queries SIDER and PharmGKB to obtain validation facts supporting the Warfarin-Amiodarone interaction and bleeding prevalence.
5. **Research Summary Report**: Exports a publication-ready report containing ATE risk differences, classification metrics, and database validation summaries to `results/experiment_summary.md`.
