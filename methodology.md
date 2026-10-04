# Research Methodology: FAERS-Based ADR Signal Detection and Predictive Modeling Framework

This document details the formal mathematical, computational, and statistical methodology of the Adverse Drug Reaction (ADR) disproportionality detection and predictive modeling framework using observational clinical safety data. The framework integrates spontaneous reporting analysis from the FDA Adverse Event Reporting System (FAERS) with biological reference validation against PharmGKB and SIDER.

---

## 1. Study Population & Cohort Construction

Spontaneous reporting databases suffer from reporting biases, under-reporting, stimulated reporting, and lack of true exposed population denominators. To analyze safety signals in real-world reports, we construct a retrospective patient-level cohort from FAERS E2B XML quarterly safety report files covering years **2021 to 2025** (54 XML quarterly archives).

### 1.1 Inclusion Criteria
The target study cohort consists of safety reports where patients were exposed to the narrow-therapeutic-index anticoagulant **Warfarin**:
$$\text{Cohort} = \{i \mid \text{Warfarin or Coumadin} \in \mathbf{D}_i\}$$
where $\mathbf{D}_i$ is the set of reported active drug substances and medicinal products for patient $i$. Reports with extreme polypharmacy ($|\mathbf{D}_i| > 100$) are excluded as recording anomalies.

### 1.2 Exposure and Primary Treatment ($T$)
The primary exposure of interest ($T_i$) is concurrent co-prescription of the antiarrhythmic **Amiodarone** ($T_i = 1$), a potent CYP2C9 inhibitor known to impair Warfarin metabolism and elevate systemic exposure:
$$T_i = \mathbb{I}(\text{Amiodarone or Cordarone} \in \mathbf{D}_i)$$
Patients exposed to Warfarin without Amiodarone constitute the unexposed comparison group ($T_i = 0$).

### 1.3 Baseline Covariates & Confounders ($\mathbf{X}$)
To control for demographic and clinical multi-morbidity baseline differences between exposed and unexposed patients, we extract:
1. **Age ($X_{\text{age}}$)**: Onset age normalized to years (missing values imputed via median fill = 65.0).
2. **Biological Sex ($X_{\text{gender}}$)**: Binary indicator ($1$: Male, $0$: Female, mode imputed).
3. **Concomitant Aspirin ($X_{\text{aspirin}}$)**: Binary indicator for Aspirin co-therapy ($\mathbb{I}(\text{Aspirin or Acetylsalicylic Acid} \in \mathbf{D}_i)$).
4. **Concomitant Non-Aspirin NSAIDs ($X_{\text{nsaid}}$)**: Binary indicator for non-aspirin NSAID co-therapy ($\mathbb{I}(\text{Ibuprofen, Naproxen, Celecoxib, Diclofenac, Meloxicam, Ketoprofen, Indomethacin} \in \mathbf{D}_i)$).
5. **Polypharmacy Index ($X_{\text{num\_drugs}}$)**: Count of all concurrent reported medications ($|\mathbf{D}_i|$), serving as a proxy for multi-morbidity burden.
6. **Metformin Control Medication ($X_{\text{drug\_c}}$)**: Binary indicator for exposure to Metformin, included as a baseline metabolic control drug feature.

### 1.4 Target Outcome Phenotype ($Y$)
The binary outcome $Y_i$ is the occurrence of a **Bleeding** event, identified via string keyword matching against MedDRA Preferred Terms (PTs) (e.g., *Gastrointestinal haemorrhage, Subdural haematoma, Epistaxis, Haematuria, Melaena*):
$$Y_i = \mathbb{I}(\mathbf{R}_i \cap \mathbf{K}_{\text{bleed}} \neq \emptyset)$$
where $\mathbf{R}_i$ is the set of reported MedDRA PT strings for report $i$, and $\mathbf{K}_{\text{bleed}}$ is the set of bleeding keyword terms.

---

## 2. Disproportionality Signal Analysis (Reporting Odds Ratios)

Spontaneous reporting safety disproportionality is quantified using the Reporting Odds Ratio (ROR).

### 2.1 Crude Reporting Odds Ratio
The crude ROR evaluates the reporting association between Amiodarone exposure ($T$) and Bleeding ($Y$) using a $2 \times 2$ contingency table:

$$\begin{array}{c|c|c}
& \text{Bleed } (Y=1) & \text{No Bleed } (Y=0) \\
\hline
\text{Amiodarone } (T=1) & A & B \\
\hline
\text{No Amiodarone } (T=0) & C & D \\
\end{array}$$

$$\text{ROR}_{\text{crude}} = \frac{A \cdot D}{B \cdot C}$$

The standard error of $\ln(\text{ROR})$ and 95% Wald confidence interval are computed as:
$$\text{SE}(\ln \text{ROR}) = \sqrt{\frac{1}{A} + \frac{1}{B} + \frac{1}{C} + \frac{1}{D}}$$
$$\text{CI}_{95\%} = \exp\left(\ln(\text{ROR}) \pm 1.96 \cdot \text{SE}(\ln \text{ROR})\right)$$

### 2.2 Confounder-Adjusted ROR (Multivariable Logistic Regression)
To adjust for baseline covariates $\mathbf{X}_i$, we specify a multivariable logistic regression model:
$$\text{logit}(P(Y_i = 1 \mid T_i, \mathbf{X}_i)) = \beta_0 + \beta_T T_i + \boldsymbol{\beta}_X^T \mathbf{X}_i$$
$$\text{ROR}_{\text{adjusted}} = \exp(\hat{\beta}_T)$$
Standard error $\text{SE}(\hat{\beta}_T)$ is derived from the inverse Fisher Information matrix $\mathbf{I}^{-1} = (\mathbf{X}_{\text{design}}^T \mathbf{W} \mathbf{X}_{\text{design}})^{-1}$, yielding Wald $z$-statistics and $p$-values.

### 2.3 Propensity Score Matched (PSM) ROR
To address confounding-by-indication and non-linear baseline imbalances, a propensity score model is estimated:
$$e(\mathbf{X}_i) = P(T_i = 1 \mid \mathbf{X}_i) = \frac{1}{1 + \exp(-\boldsymbol{\gamma}^T \mathbf{X}_i)}$$
1:1 nearest-neighbor matching without replacement is executed within a caliper width of $0.10$ on the propensity score scale. Covariate balance before and after matching is evaluated using Standardized Mean Differences (SMD):
$$\text{SMD} = \frac{\bar{X}_{T=1} - \bar{X}_{T=0}}{\sqrt{\frac{s_{T=1}^2 + s_{T=0}^2}{2}}}$$
The matched ROR is estimated on the balanced matched sample ($N = 1,038$).

---

## 3. Predictive Risk Modeling (XGBoost Classifier)

To evaluate predictive risk scoring for bleeding events, a Gradient Boosted Decision Tree (XGBoost) model is trained:
$$\hat{P}(Y_i = 1 \mid \mathbf{X}_i, T_i)$$
- **Training Strategy**: Models are trained on the natural unbalanced training distribution ($70/30$ train/test split) without outcome oversampling to preserve natural probability calibration.
- **Evaluation Metrics**:
  - Discrimination: Area Under the ROC Curve (ROC-AUC) and Area Under the Precision-Recall Curve (PR-AUC).
  - Classification: Precision, Recall, F1-score at default $0.50$ decision threshold.
  - Calibration: Expected Calibration Error (ECE) across 10 probability bins and Brier score loss:
    $$\text{ECE} = \sum_{b=1}^{B} \frac{|N_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
    $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

---

## 4. Methodological Interpretation & Limitations

1. **PSM Result Interpretation**: In the matched sample, the PSM ROR is $0.7940$ with a 95% confidence interval of $0.5626\text{--}1.1208$. Because the confidence interval spans 1.0, the matched disproportionality analysis does not demonstrate a statistically significant reporting association after controlling for baseline covariates.
2. **Predictive Performance**: ROC-AUC $\approx 0.6974$ indicates moderate discrimination. Recall is relatively low ($0.1445$) at the default 0.50 threshold due to class imbalance, demonstrating that predictive risk scores should complement clinical oversight rather than serve as standalone rules.
3. **Observational Limitations**: FAERS safety reports lack total exposed population denominators, suffer from reporting and indication bias, and contain unmeasured clinical confounders.
