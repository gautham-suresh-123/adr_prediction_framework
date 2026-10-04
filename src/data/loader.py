import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_CSV = PROJECT_ROOT / "dataset" / "processed_faers_warfarin.csv"

def clean_and_align_events(demographics_df, labevents_df, prescriptions_df, faers_df):
    """
    [HISTORICAL / HELPER UTILITY]
    Cleans, imputes, and aligns demographics, lab events, prescriptions, 
    and FAERS adverse event reports into a single patient-level dataset.
    Note: Preserved for multi-source alignment workflows. Not used in the primary FAERS XML pipeline.
    """
    df = demographics_df.copy()
    
    baseline_labs = labevents_df[labevents_df['baseline_flag'] == 1].pivot(
        index='subject_id', columns='lab_name', values='valuenum'
    ).reset_index()
    baseline_labs.columns = ['subject_id', 'baseline_ALT', 'baseline_Creatinine', 'baseline_INR']
    
    post_labs = labevents_df[labevents_df['baseline_flag'] == 0].pivot(
        index='subject_id', columns='lab_name', values='valuenum'
    ).reset_index()
    post_labs.columns = ['subject_id', 'post_ALT', 'post_Creatinine', 'post_INR']
    
    df = df.merge(baseline_labs, on='subject_id', how='left')
    df = df.merge(post_labs, on='subject_id', how='left')
    
    presc_pivoted = prescriptions_df.pivot_table(
        index='subject_id', 
        columns='drug_name', 
        values='dosage', 
        aggfunc='first'
    ).reset_index()
    
    presc_pivoted = presc_pivoted.rename(columns={
        'Warfarin': 'dosage_a',
        'Amiodarone': 'dosage_b',
        'Metformin': 'dosage_c'
    })
    
    for col in ['dosage_a', 'dosage_b', 'dosage_c']:
        if col not in presc_pivoted.columns:
            presc_pivoted[col] = 0.0
        else:
            presc_pivoted[col] = presc_pivoted[col].fillna(0.0)
            
    presc_pivoted['drug_a'] = (presc_pivoted['dosage_a'] > 0).astype(int)
    presc_pivoted['drug_b'] = (presc_pivoted['dosage_b'] > 0).astype(int)
    presc_pivoted['drug_c'] = (presc_pivoted['dosage_c'] > 0).astype(int)
    
    df = df.merge(presc_pivoted, on='subject_id', how='left')
    
    for col in ['dosage_a', 'dosage_b', 'dosage_c', 'drug_a', 'drug_b', 'drug_c']:
        df[col] = df[col].fillna(0.0)
        
    bleeding_patients = faers_df[faers_df['adr_name'] == 'Gastrointestinal Bleeding']['subject_id'].unique()
    df['adr_bleeding'] = df['subject_id'].isin(bleeding_patients).astype(int)
    
    df['baseline_ALT'] = df['baseline_ALT'].fillna(df['baseline_ALT'].median())
    df['baseline_Creatinine'] = df['baseline_Creatinine'].fillna(df['baseline_Creatinine'].median())
    df['baseline_INR'] = df['baseline_INR'].fillna(df['baseline_INR'].median())
    
    df['post_INR'] = df['post_INR'].fillna(df['baseline_INR'])
    df['post_ALT'] = df['post_ALT'].fillna(df['baseline_ALT'])
    df['post_Creatinine'] = df['post_Creatinine'].fillna(df['baseline_Creatinine'])
    
    return df

def correct_class_imbalance(df, target_col='adr_bleeding', method='oversample', ratio=1.0, seed=42):
    """
    [HELPER UTILITY]
    Corrects class imbalance in training data using up-sampling or down-sampling.
    Note: To prevent calibration degradation in epidemiological prediction, 
    the main pipeline evaluates models on natural event prevalence without oversampling.
    """
    np.random.seed(seed)
    
    class_0 = df[df[target_col] == 0]
    class_1 = df[df[target_col] == 1]
    
    n_0 = len(class_0)
    n_1 = len(class_1)
    
    if n_1 == 0:
        return df
        
    if method == 'oversample':
        target_size = int(n_0 * ratio)
        oversampled_minority = class_1.sample(target_size, replace=True, random_state=seed)
        balanced_df = pd.concat([class_0, oversampled_minority], axis=0).reset_index(drop=True)
    elif method == 'undersample':
        target_size = int(n_1 / ratio)
        undersampled_majority = class_0.sample(min(target_size, n_0), replace=False, random_state=seed)
        balanced_df = pd.concat([undersampled_majority, class_1], axis=0).reset_index(drop=True)
    else:
        balanced_df = df.copy()
        
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df

class ClinicalDataLoader:
    """
    Orchestrates the data loading pipeline: loading the real clinical cohort,
    filtering polypharmacy outliers, and returning feature/treatment/outcome metadata.
    """
    def __init__(self, data_generator_fn=None):
        self.data_generator_fn = None
        self.aligned_data = None
        
    def load_and_preprocess(self, processed_path=None, seed=42):
        if processed_path is None:
            processed_path = DEFAULT_PROCESSED_CSV
        else:
            processed_path = Path(processed_path)
            
        if not processed_path.exists():
            print(f"Processed dataset not found at {processed_path}. Running FAERS XML parser...")
            from src.data.parser import build_processed_dataset
            self.aligned_data = build_processed_dataset(output_path=processed_path)
        else:
            print(f"Loading cached real dataset from {processed_path}...")
            self.aligned_data = pd.read_csv(processed_path)
            
        # Filter polypharmacy outliers (num_drugs > 100)
        n_before = len(self.aligned_data)
        self.aligned_data = self.aligned_data[self.aligned_data['num_drugs'] <= 100].reset_index(drop=True)
        print(f"Filtered out polypharmacy outliers (drugs > 100): {n_before - len(self.aligned_data)} records removed. Remaining: {len(self.aligned_data)}.")
            
        # Baseline feature matrix
        feature_cols = ['age', 'gender', 'aspirin', 'nsaid', 'num_drugs', 'drug_c']
        
        # Primary treatment of interest: Amiodarone co-prescription
        treatment_col = 'drug_b'
        dosage_col = None
        outcome_col = 'adr_bleeding'
        mediator_col = None
        
        return self.aligned_data, feature_cols, treatment_col, dosage_col, outcome_col, mediator_col

if __name__ == '__main__':
    loader = ClinicalDataLoader()
    df, features, treatment, dosage, outcome, mediator = loader.load_and_preprocess()
    print("Features list:", features)
    print("Treatment column:", treatment)
    print("Outcome column:", outcome)
    print("ADR event count in aligned data:\n", df[outcome].value_counts())
