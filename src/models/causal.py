import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import norm, chi2_contingency

class DisproportionalityAnalysisEngine:
    """
    Performs Disproportionality Analysis on spontaneous safety reports (FAERS),
    including Crude Reporting Odds Ratio (ROR), Confounder-Adjusted ROR (via Logistic Regression),
    and Propensity Score Matched (PSM) ROR.
    """
    def __init__(self, features, treatment_col, outcome_col, seed=42):
        self.features = features
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.seed = seed
        
    def compute_crude_ror(self, df):
        """
        Computes the Crude Reporting Odds Ratio (ROR) and 95% Confidence Intervals.
        """
        # Contingency Table:
        #                 Bleed (Y=1)   No Bleed (Y=0)
        # Treated (T=1)       A               B
        # Untreated (T=0)     C               D
        
        t_col = self.treatment_col
        o_col = self.outcome_col
        
        a = int(df[(df[t_col] == 1) & (df[o_col] == 1)].shape[0])
        b = int(df[(df[t_col] == 1) & (df[o_col] == 0)].shape[0])
        c = int(df[(df[t_col] == 0) & (df[o_col] == 1)].shape[0])
        d = int(df[(df[t_col] == 0) & (df[o_col] == 0)].shape[0])
        
        # Avoid division by zero
        if a == 0 or b == 0 or c == 0 or d == 0:
            ror = 1.0
            se_ln_ror = 0.0
            ci_lower = 1.0
            ci_upper = 1.0
            p_val = 1.0
        else:
            ror = (a * d) / (b * c)
            se_ln_ror = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_lower = np.exp(np.log(ror) - 1.96 * se_ln_ror)
            ci_upper = np.exp(np.log(ror) + 1.96 * se_ln_ror)
            
            # Chi-square test
            contingency = [[a, b], [c, d]]
            _, p_val, _, _ = chi2_contingency(contingency)
            
        return {
            'A': a, 'B': b, 'C': c, 'D': d,
            'ROR': ror,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_val
        }
        
    def compute_adjusted_ror(self, df):
        """
        Computes the Confounder-Adjusted ROR using Logistic Regression.
        Also calculates standard errors and Wald p-values using pure NumPy covariance.
        """
        # Design matrix X: Intercept + Treatment + Confounders
        # Outcome y: Bleeding
        X_vars = [self.treatment_col] + self.features
        X = df[X_vars].copy()
        y = df[self.outcome_col].values
        
        # Fit logistic regression in sklearn
        # Use penalty=None for unbiased maximum likelihood estimates
        model = LogisticRegression(penalty=None, max_iter=1000, random_state=self.seed)
        model.fit(X, y)
        
        # Coefficients
        beta = np.concatenate([model.intercept_, model.coef_[0]])
        
        # Calculate standard errors of coefficients
        # Add column of ones for intercept to design matrix
        X_design = np.hstack([np.ones((X.shape[0], 1)), X.values])
        
        # Predicted probabilities
        p = model.predict_proba(X)[:, 1]
        
        # Weight matrix diagonal elements w_i = p_i * (1 - p_i)
        w = p * (1 - p)
        
        # Fisher Information Matrix X^T * W * X
        # Avoid memory issues by doing matrix multiplication with weights
        XTWX = np.dot(X_design.T * w, X_design)
        
        try:
            # Covariance matrix is the inverse of Fisher Information
            cov_beta = np.linalg.inv(XTWX)
            se_beta = np.sqrt(np.diag(cov_beta))
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            se_beta = np.zeros(beta.shape)
            
        # The treatment coefficient is the second term (index 1)
        treatment_beta = beta[1]
        treatment_se = se_beta[1]
        
        adj_ror = np.exp(treatment_beta)
        ci_lower = np.exp(treatment_beta - 1.96 * treatment_se)
        ci_upper = np.exp(treatment_beta + 1.96 * treatment_se)
        
        # Wald z-statistic
        z_stat = treatment_beta / treatment_se if treatment_se > 0 else 0.0
        p_val = 2 * (1 - norm.cdf(np.abs(z_stat))) if treatment_se > 0 else 1.0
        
        return {
            'Adjusted_ROR': adj_ror,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_val,
            'beta': treatment_beta,
            'se': treatment_se
        }
        
    def perform_psm_ror(self, df):
        """
        Performs 1:1 nearest neighbor propensity score matching (without replacement)
        and estimates the ROR on the matched balanced cohort.
        Returns matched df, before/after propensity scores, and matched ROR results.
        """
        # Fit Propensity Score model using Logistic Regression
        ps_model = LogisticRegression(max_iter=1000, random_state=self.seed)
        ps_model.fit(df[self.features], df[self.treatment_col])
        
        df_temp = df.copy()
        df_temp['propensity_score'] = ps_model.predict_proba(df[self.features])[:, 1]
        
        # Separate treated and untreated
        treated = df_temp[df_temp[self.treatment_col] == 1].reset_index(drop=True)
        untreated = df_temp[df_temp[self.treatment_col] == 0].reset_index(drop=True)
        
        # Shuffle treated to prevent matching order bias
        treated_shuffled = treated.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        
        matched_treated_indices = []
        matched_untreated_indices = []
        
        # Available pool of untreated indices
        untreated_pool = untreated.copy()
        
        caliper = 0.1 # Caliper width of 0.1 on the propensity score scale
        
        for idx, tr_row in treated_shuffled.iterrows():
            if len(untreated_pool) == 0:
                break
                
            # Compute absolute difference in propensity score
            diffs = np.abs(untreated_pool['propensity_score'] - tr_row['propensity_score'])
            min_idx = diffs.idxmin()
            min_diff = diffs.loc[min_idx]
            
            if min_diff <= caliper:
                matched_treated_indices.append(tr_row.name) # relative to treated_shuffled
                matched_untreated_indices.append(min_idx)   # relative to untreated_pool
                
                # Remove matched untreated from pool
                untreated_pool = untreated_pool.drop(min_idx)
                
        matched_treated = treated_shuffled.iloc[matched_treated_indices]
        matched_untreated = untreated.iloc[matched_untreated_indices]
        
        matched_df = pd.concat([matched_treated, matched_untreated], axis=0).reset_index(drop=True)
        
        # Compute ROR on matched cohort
        matched_ror_results = self.compute_crude_ror(matched_df)
        
        # Collect propensity distributions for plotting
        ps_before = {
            'treated': treated['propensity_score'].values,
            'untreated': untreated['propensity_score'].values
        }
        
        ps_after = {
            'treated': matched_treated['propensity_score'].values,
            'untreated': matched_untreated['propensity_score'].values
        }
        
        return matched_df, ps_before, ps_after, matched_ror_results
