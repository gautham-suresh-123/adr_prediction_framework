# ADR Disproportionality & Predictive Framework (FAERS Clinical Data) - Academic Summary

## 1. Study Design Overview
This clinical research study analyzes adverse drug reactions (ADRs) and signal disproportionality from spontaneous reporting databases.
- **Cohort**: Patients exposed to **Warfarin** (Anticoagulant) extracted from FAERS XML safety reports (years 2021–2025).
- **Cohort Size**: 30,354 patient records (excluding extreme polypharmacy outliers with >100 medications).
- **Treatment of Interest ($T$)**: Concurrent exposure to **Amiodarone** (CYP2C9 inhibitor) which impairs Warfarin metabolism.
- **Outcome of Interest ($Y$)**: **Bleeding** adverse drug reactions.
- **Confounders ($\mathbf{X}$)**: Patient age, gender, concomitant Aspirin, concomitant non-aspirin NSAIDs, total drug count (polypharmacy), and Metformin exposure (control drug).

## 2. Baseline Demographics & Clinical Characteristics
Baseline demographics of the Treated (Amiodarone) and Untreated groups in the cohort:

| Characteristic                       | Amiodarone Co-Prescribed (Treated)   | No Amiodarone (Untreated)   |
|:-------------------------------------|:-------------------------------------|:----------------------------|
| Sample Size (N)                      | 1677                                 | 28677                       |
| Age (Years, Mean ± SD)               | 70.0 ± 11.6                          | 66.7 ± 14.2                 |
| Male Gender (%)                      | 59.2%                                | 40.4%                       |
| Concomitant Aspirin (%)              | 25.5%                                | 14.3%                       |
| Concomitant Non-Aspirin NSAIDs (%)   | 27.6%                                | 18.6%                       |
| Metformin Control Medication (%)     | 10.4%                                | 7.6%                        |
| Polypharmacy (Drug Count, Mean ± SD) | 20.0 ± 17.7                          | 12.1 ± 12.5                 |
| Bleeding ADR Rate (%)                | 13.1%                                | 18.0%                       |

## 3. Disproportionality Signal Analysis (Reporting Odds Ratios)
Disproportionality statistics (RORs) were estimated on spontaneous safety reports. Confounder adjustment was achieved using multivariable logistic regression and 1:1 nearest-neighbor propensity score matching (PSM):

- **Crude ROR**: 0.6962 (95% CI: 0.5372–0.9023, p = 7.1012e-03)
- **Confounder-Adjusted ROR**: 0.7100 (95% CI: 0.5444–0.9259, p = 1.1459e-02)
- **Propensity Score Matched (PSM) ROR**: 0.7940 (95% CI: 0.5626–1.1208, p = 2.2028e-01)
- **Matched Cohort Size**: N = 1038 (Treated: 519, Untreated: 519)

*Statistical Note*: The 95% confidence interval for the PSM ROR includes 1.0, indicating that after controlling for baseline demographic and polypharmacy balance via PSM, there was no statistically significant disproportionality signal for Amiodarone bleeding reporting in the matched sample.

### Subgroup Analysis (Table 2)
Subgroup disproportionality analysis across clinical characteristics showing Crude vs. Adjusted RORs:

| Subgroup               |   Sample Size (N) |   Crude ROR |   Crude CI Lower |   Crude CI Upper |   Adjusted ROR |   Adjusted CI Lower |   Adjusted CI Upper |   Adjusted p-value |
|:-----------------------|------------------:|------------:|-----------------:|-----------------:|---------------:|--------------------:|--------------------:|-------------------:|
| Full Test Cohort       |              9107 |      0.6962 |           0.5372 |           0.9023 |         0.71   |              0.5444 |              0.9259 |           0.011459 |
| Age < 65               |              2478 |      0.4097 |           0.2053 |           0.8175 |         0.4009 |              0.2001 |              0.8035 |           0.009976 |
| Age >= 65              |              6629 |      0.7806 |           0.5892 |           1.0341 |         0.8589 |              0.642  |              1.1491 |           0.30571  |
| Female                 |              5365 |      0.7875 |           0.5312 |           1.1673 |         0.7688 |              0.7688 |              0.7688 |           1        |
| Male                   |              3742 |      0.6323 |           0.4472 |           0.8939 |         0.7964 |              0.7964 |              0.7964 |           1        |
| Aspirin/NSAID User     |              1726 |      0.6944 |           0.4376 |           1.1018 |         0.6188 |              0.3863 |              0.9914 |           0.045956 |
| Non-Aspirin/NSAID User |              7381 |      0.6695 |           0.4888 |           0.917  |         0.7923 |              0.7923 |              0.7923 |           1        |

## 4. ML Predictive Performance (XGBoost Classifier)
A predictive Gradient Boosted Decision Tree (XGBoost) model was trained on the natural training cohort distribution:

|         |   auc_roc |   pr_auc |   precision |   recall |       f1 |   brier_score |        ece |
|:--------|----------:|---------:|------------:|---------:|---------:|--------------:|-----------:|
| XGBoost |  0.697408 |   0.3945 |     0.67341 | 0.144451 | 0.237876 |      0.129498 | 0.00874355 |

*Note: Evaluated on natural prevalence without outcome oversampling, achieving an Expected Calibration Error (ECE) of **0.87%** (Brier score: 0.1295). Discrimination shows moderate performance (ROC-AUC = 0.6974) with relatively low sensitivity (0.1445) at default classification threshold (0.50).*

## 5. Reference Database Validation
### SIDER Side Effect Validation
- **Result**: SIDER confirms bleeding is a known side effect of Warfarin.
- **Bleeding Terms in SIDER for Warfarin**:
  - *Gastrointestinal haemorrhage*
  - *Subdural haematoma*
  - *Haemorrhage*
  - *Rectal haemorrhage*
  - *Cerebral haemorrhage*

### PharmGKB Pharmacogenomics Guidelines
PharmGKB documents genetic variation and dosing guidelines for these drugs:
- **Warfarin Gene Relationships in PharmGKB**:
  - **CYP2C9** $\leftrightarrow$ **warfarin**: Dosing association evidence (*GuidelineAnnotation,LabelAnnotation,Literature,MultilinkAnnotation,SummaryAnnotation,VariantAnnotation*)
  - **CYP2C19** $\leftrightarrow$ **warfarin**: Dosing association evidence (*MultilinkAnnotation,SummaryAnnotation,VariantAnnotation*)
  - **warfarin** $\leftrightarrow$ **CYP2C19**: Dosing association evidence (*MultilinkAnnotation,SummaryAnnotation,VariantAnnotation*)
  - **warfarin** $\leftrightarrow$ **CYP2C9**: Dosing association evidence (*GuidelineAnnotation,LabelAnnotation,Literature,MultilinkAnnotation,SummaryAnnotation,VariantAnnotation*)
  - **warfarin** $\leftrightarrow$ **CYP3A4**: Dosing association evidence (*MultilinkAnnotation,VariantAnnotation*)
  - **warfarin** $\leftrightarrow$ **VKORC1**: Dosing association evidence (*GuidelineAnnotation,LabelAnnotation,Literature,MultilinkAnnotation,SummaryAnnotation,VariantAnnotation*)
- **Amiodarone Gene Relationships in PharmGKB**:
  - **CYP2D6** $\leftrightarrow$ **amiodarone**: Metabolic association evidence (*GuidelineAnnotation*)
  - **CYP3A4** $\leftrightarrow$ **amiodarone**: Metabolic association evidence (*VariantAnnotation*)
  - **amiodarone** $\leftrightarrow$ **CYP2D6**: Metabolic association evidence (*GuidelineAnnotation*)
  - **amiodarone** $\leftrightarrow$ **CYP3A4**: Metabolic association evidence (*VariantAnnotation*)

---
*Note: FAERS is a spontaneous safety reporting system subject to reporting biases, lack of total exposed population denominators, and potential unmeasured confounding. Results describe reporting disproportionality associations rather than prospective causal risk.*