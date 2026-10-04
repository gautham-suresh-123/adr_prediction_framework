# Methodology: Causal Adverse Drug Reaction (ADR) Prediction Framework

This document details the formal mathematical, computational, and statistical methodology of the Adverse Drug Reaction (ADR) prediction and treatment effect estimation framework using real observational clinical data. The framework combines traditional predictive machine learning with Double Machine Learning (DML) to analyze drug-drug interactions (DDIs) and safety outcomes from spontaneous safety reporting systems (FAERS), validated against biological reference databases (PharmGKB and SIDER).

---

## 1. Clinical Cohort Extraction (FAERS XML)

Spontaneous reporting databases (such as the FDA Adverse Event Reporting System - FAERS) suffer from severe reporting bias and confounding-by-indication. To evaluate predictive and causal models on real-world safety reports, we construct a retrospective patient-level cohort from FAERS E2B (R2) XML files covering all quarters of **2024 and 2025**.

### 1.1 Study Population and Inclusion Criteria
The target study population consists of patients exposed to the narrow-therapeutic-index anticoagulant **Warfarin**:
$$\text{Cohort} = \{i \mid \text{Warfarin or Coumadin} \in \mathbf{D}_i\}$$
where $\mathbf{D}_i$ is the set of reported active drug substances and medicinal products for patient $i$.

### 1.2 Exposure and Treatment Variable ($T$)
The treatment of interest ($T_i$) is the concurrent prescription of the antiarrhythmic **Amiodarone** ($T_i = 1$), which is a strong CYP2C9 inhibitor known to significantly reduce Warfarin clearance and increase blood levels. The control comparison is patients not exposed to Amiodarone ($T_i = 0$), including those exposed to the non-interacting control drug **Metformin**.
$$T_i = \mathbb{I}(\text{Amiodarone or Cordarone} \in \mathbf{D}_i)$$

### 1.3 Confounders and Covariates ($\mathbf{X}$)
To control for confounding-by-indication and multi-morbidity bias, we define a baseline covariate vector $\mathbf{X}_i$ for each patient:
1. **Age**: Patient age at adverse event onset, normalized to years:
   $$X_{\text{age}} \in \mathbb{R}^+$$
2. **Gender**: Binary biological sex:
   $$X_{\text{gender}} \in \{0, 1\} \quad (0: \text{Female}, 1: \text{Male})$$
3. **Concomitant Aspirin Use**: Binary indicator for antiplatelet co-therapy, which independently increases bleeding risk:
   $$X_{\text{aspirin}} = \mathbb{I}(\text{Aspirin or Acetylsalicylic Acid} \in \mathbf{D}_i)$$
4. **Concomitant NSAID Use**: Binary indicator for non-steroidal anti-inflammatory drug co-therapy, which also elevates bleeding risk:
   $$X_{\text{nsaid}} = \mathbb{I}(\text{NSAIDs} \in \mathbf{D}_i)$$
5. **Morbidity Index (Total Drugs)**: Count of all concurrent drugs reported, serving as a proxy for patient health complexity and polypharmacy:
   $$X_{\text{num\_drugs}} = |\mathbf{D}_i|$$
6. **Control Drug Exposure**: Exposure to the anti-diabetic drug Metformin (as a baseline safety control):
   $$X_{\text{control}} = \mathbb{I}(\text{Metformin or Glucophage} \in \mathbf{D}_i)$$

### 1.4 Target Safety Outcome ($Y$)
The binary target outcome $Y_i$ is the occurrence of a **Bleeding** event, defined by the presence of any MedDRA Preferred Term (PT) matching haemorrhagic event keywords (e.g., *Gastrointestinal haemorrhage, Subdural haematoma, Epistaxis, Haematuria, Melaena*):
$$Y_i = \mathbb{I}(\mathbf{R}_i \cap \mathbf{K}_{\text{bleed}} \neq \emptyset)$$
where $\mathbf{R}_i$ is the set of MedDRA reaction terms for patient $i$, and $\mathbf{K}_{\text{bleed}}$ is the set of bleeding-related preferred terms.

---

## 2. Predictive Risk Modeling (XGBoost)

We train a standard Gradient Boosted Decision Tree (GBDT) model (XGBoost Classifier) to predict the probability of a bleeding event:
$$\hat{P}(Y_i = 1 \mid \mathbf{X}_i, T_i)$$
In this predictive formulation, the treatment assignment $T_i$ (Amiodarone) is treated as a standard input feature alongside the baseline covariates $\mathbf{X}_i$. While this maximizes association-based classification accuracy, it does not separate the drug's true causal risk from the clinical indications that led to its prescription, making it susceptible to confounding bias.

---

## 3. Causal Inference (Double Machine Learning)

To isolate the actual risk difference caused by Amiodarone co-prescription under confounding-by-indication, we estimate heterogeneous treatment effects on the binary bleeding outcome $Y_i$ using **Double Machine Learning (DML)**.

We define the causal treatment effect for patient $i$ as the risk difference:
$$\text{ITE}_i = E[Y_i(1) - Y_i(0) \mid \mathbf{X}_i]$$
$$\text{ATE} = E[Y(1) - Y(0)]$$
where $Y(1)$ and $Y(0)$ are the potential outcomes under Amiodarone exposure and non-exposure, respectively.

### 3.1 Causal Forest DML
DML first orthogonalizes the treatment ($T$) and the outcome ($Y$) against the baseline features ($\mathbf{X}$) using machine learning estimators (Random Forest Regressor and Gradient Boosting Classifier) to isolate the residual variation:

1. **Outcome Residualization**:
   Estimate the conditional expectation of bleeding $m(\mathbf{X}) = \mathbb{E}[Y \mid \mathbf{X}]$ using a Random Forest regressor, and compute outcome residuals:
   $$\tilde{Y}_i = Y_i - \hat{m}(\mathbf{X}_i)$$

2. **Treatment Residualization (Propensity Score)**:
   Estimate the propensity score $e(\mathbf{X}) = \mathbb{E}[T \mid \mathbf{X}] = P(T=1 \mid \mathbf{X})$ using a Gradient Boosting classifier, and compute treatment residuals:
   $$\tilde{T}_i = T_i - \hat{e}(\mathbf{X}_i)$$

3. **Causal Forest Fit**:
   Fit a Causal Forest regressor on the residuals to estimate the conditional average treatment effect (CATE) $\theta(\mathbf{X})$:
   $$\tilde{Y}_i = \theta(\mathbf{X}_i) \tilde{T}_i + \epsilon_i$$
This formulation separates the confounding influence of baseline covariates $\mathbf{X}_i$ from the treatment effect of interest, providing a causally valid risk difference of Amiodarone on Bleeding in the presence of strong observational bias.
