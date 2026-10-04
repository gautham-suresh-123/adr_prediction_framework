import pandas as pd
import numpy as np
import os

def clean_and_align_events(demographics_df, labevents_df, prescriptions_df, faers_df):
    """
    Cleans, imputes, and aligns demographics, lab events, prescriptions, 
    and FAERS adverse event reports into a single, analysis-ready patient-level dataset.
    """
    # 1. Start with demographics as patient base
    df = demographics_df.copy()
    
    # 2. Extract and pivot baseline and post-treatment lab values
    # Baseline labs (baseline_flag = 1)
    baseline_labs = labevents_df[labevents_df['baseline_flag'] == 1].pivot(
        index='subject_id', columns='lab_name', values='valuenum'
    ).reset_index()
    baseline_labs.columns = ['subject_id', 'baseline_ALT', 'baseline_Creatinine', 'baseline_INR']
    
    # Post-treatment labs (baseline_flag = 0)
    post_labs = labevents_df[labevents_df['baseline_flag'] == 0].pivot(
        index='subject_id', columns='lab_name', values='valuenum'
    ).reset_index()
    post_labs.columns = ['subject_id', 'post_ALT', 'post_Creatinine', 'post_INR']
    
    # Merge labs
    df = df.merge(baseline_labs, on='subject_id', how='left')
    df = df.merge(post_labs, on='subject_id', how='left')
    
    # 3. Extract prescriptions and pivot to treatments (Drug A: Warfarin, Drug B: Amiodarone, Drug C: Metformin)
    # Pivot prescription presence
    presc_pivoted = prescriptions_df.pivot_table(
        index='subject_id', 
        columns='drug_name', 
        values='dosage', 
        aggfunc='first'
    ).reset_index()
    
    # Rename columns to show treatment and dosage
    presc_pivoted = presc_pivoted.rename(columns={
        'Warfarin': 'dosage_a',
        'Amiodarone': 'dosage_b',
        'Metformin': 'dosage_c'
    })
    
    # Fill missing dosages with 0.0 (meaning drug was not prescribed)
    for col in ['dosage_a', 'dosage_b', 'dosage_c']:
        if col not in presc_pivoted.columns:
            presc_pivoted[col] = 0.0
        else:
            presc_pivoted[col] = presc_pivoted[col].fillna(0.0)
            
    # Add binary indicators for treatment assignment
    presc_pivoted['drug_a'] = (presc_pivoted['dosage_a'] > 0).astype(int)
    presc_pivoted['drug_b'] = (presc_pivoted['dosage_b'] > 0).astype(int)
    presc_pivoted['drug_c'] = (presc_pivoted['dosage_c'] > 0).astype(int)
    
    df = df.merge(presc_pivoted, on='subject_id', how='left')
    
    # Fill missing drug variables for patients who got no prescriptions at all
    for col in ['dosage_a', 'dosage_b', 'dosage_c', 'drug_a', 'drug_b', 'drug_c']:
        df[col] = df[col].fillna(0.0)
        
    # 4. Map outcomes from FAERS safety reports
    # ADR Bleeding: 1 if patient has Gastrointestinal Bleeding report in FAERS, 0 otherwise
    bleeding_patients = faers_df[faers_df['adr_name'] == 'Gastrointestinal Bleeding']['subject_id'].unique()
    df['adr_bleeding'] = df['subject_id'].isin(bleeding_patients).astype(int)
    
    # 5. Missing value treatment (Imputation)
    # Baseline ALT: median-fill
    df['baseline_ALT'] = df['baseline_ALT'].fillna(df['baseline_ALT'].median())
    # Baseline Creatinine: median-fill
    df['baseline_Creatinine'] = df['baseline_Creatinine'].fillna(df['baseline_Creatinine'].median())
    # Baseline INR: median-fill
    df['baseline_INR'] = df['baseline_INR'].fillna(df['baseline_INR'].median())
    
    # Post labs imputation (if missing, carry forward baseline values - represent "no change")
    df['post_INR'] = df['post_INR'].fillna(df['baseline_INR'])
    df['post_ALT'] = df['post_ALT'].fillna(df['baseline_ALT'])
    df['post_Creatinine'] = df['post_Creatinine'].fillna(df['baseline_Creatinine'])
    
    return df

def correct_class_imbalance(df, target_col='adr_bleeding', method='oversample', ratio=1.0, seed=42):
    """
    Corrects class imbalance in the training data using up-sampling or down-sampling.
    Designed without external dependencies for maximum robustness.
    """
    np.random.seed(seed)
    
    class_0 = df[df[target_col] == 0]
    class_1 = df[df[target_col] == 1]
    
    n_0 = len(class_0)
    n_1 = len(class_1)
    
    if n_1 == 0:
        return df # Can't do anything if there are no cases
        
    if method == 'oversample':
        # Target size for minority class
        target_size = int(n_0 * ratio)
        oversampled_minority = class_1.sample(target_size, replace=True, random_state=seed)
        balanced_df = pd.concat([class_0, oversampled_minority], axis=0).reset_index(drop=True)
    elif method == 'undersample':
        target_size = int(n_1 / ratio)
        undersampled_majority = class_0.sample(min(target_size, n_0), replace=False, random_state=seed)
        balanced_df = pd.concat([undersampled_majority, class_1], axis=0).reset_index(drop=True)
    else:
        balanced_df = df.copy()
        
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df

class ClinicalDataLoader:
    """
    Orchestrates the data pipeline: loading the real clinical cohort, 
    balancing, splitting, and preparing for ML and causal estimators.
    """
    def __init__(self, data_generator_fn=None):
        # Kept for backward compatibility but ignored
        self.data_generator_fn = None
        self.aligned_data = None
        
    def load_and_preprocess(self, n_patients=None, seed=42, imbalance_method='oversample'):
        processed_path = "dataset/processed_faers_warfarin.csv"
        
        if not os.path.exists(processed_path):
            print("Processed real dataset not found. Running FAERS XML parser...")
            from src.data.parser import build_processed_dataset
            self.aligned_data = build_processed_dataset()
        else:
            print(f"Loading cached real dataset from {processed_path}...")
            self.aligned_data = pd.read_csv(processed_path)
            
        # Filter polypharmacy outliers (num_drugs > 100)
        n_before = len(self.aligned_data)
        self.aligned_data = self.aligned_data[self.aligned_data['num_drugs'] <= 100].reset_index(drop=True)
        print(f"Filtered out polypharmacy outliers (drugs > 100): {n_before - len(self.aligned_data)} records removed. Remaining: {len(self.aligned_data)}.")
            
        # Feature matrix representing real patient clinical baseline
        feature_cols = ['age', 'gender', 'aspirin', 'nsaid', 'num_drugs', 'drug_c']
        
        # Treatment of interest: Amiodarone co-prescription
        treatment_col = 'drug_b'
        dosage_col = None
        outcome_col = 'adr_bleeding'
        mediator_col = None # FAERS does not have structured baseline/post labs (like INR)
        
        return self.aligned_data, feature_cols, treatment_col, dosage_col, outcome_col, mediator_col
            
if __name__ == '__main__':
    loader = ClinicalDataLoader()
    df, features, treatment, dosage, outcome, mediator = loader.load_and_preprocess()
    print("Features list:", features)
    print("Treatment column:", treatment)
    print("Outcome column:", outcome)
    print("ADR event count in aligned data:\n", df[outcome].value_counts())
    
    balanced = correct_class_imbalance(df, target_col=outcome, method='oversample')
    print("Balanced ADR event count:\n", balanced[outcome].value_counts())
