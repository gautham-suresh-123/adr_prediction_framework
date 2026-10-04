import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import ClinicalDataLoader, DEFAULT_PROCESSED_CSV
from src.models.causal import DisproportionalityAnalysisEngine, compute_standardized_mean_differences
from src.models.predictive import PredictiveClassifierEngine
from src.evaluation.metrics import compute_all_predictive_metrics, expected_calibration_error

class TestADRPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.processed_csv_path = cls.project_root / "dataset" / "processed_faers_warfarin.csv"
        
    def test_processed_dataset_existence_and_schema(self):
        """Verifies that processed_faers_warfarin.csv exists and contains required columns."""
        self.assertTrue(self.processed_csv_path.exists(), f"Processed CSV missing at {self.processed_csv_path}")
        df = pd.read_csv(self.processed_csv_path)
        
        required_cols = {'subject_id', 'age', 'gender', 'drug_a', 'drug_b', 'drug_c', 'aspirin', 'nsaid', 'num_drugs', 'adr_bleeding'}
        self.assertTrue(required_cols.issubset(set(df.columns)), f"Missing columns in CSV: {required_cols - set(df.columns)}")
        self.assertGreater(len(df), 1000, "Dataset contains fewer records than expected.")
        
    def test_clinical_data_loader(self):
        """Verifies ClinicalDataLoader load and preprocess pipeline."""
        loader = ClinicalDataLoader()
        df, features, treatment, dosage, outcome, mediator = loader.load_and_preprocess(processed_path=self.processed_csv_path)
        
        self.assertIn('age', features)
        self.assertEqual(treatment, 'drug_b')
        self.assertEqual(outcome, 'adr_bleeding')
        self.assertTrue((df['num_drugs'] <= 100).all(), "Outlier polypharmacy values (>100) were not filtered.")
        
    def test_disproportionality_engine_synthetic(self):
        """Tests DisproportionalityAnalysisEngine on a synthetic cohort."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'age': np.random.normal(65, 10, n),
            'gender': np.random.binomial(1, 0.5, n),
            'aspirin': np.random.binomial(1, 0.2, n),
            'nsaid': np.random.binomial(1, 0.15, n),
            'num_drugs': np.random.poisson(10, n),
            'drug_c': np.random.binomial(1, 0.1, n),
            'drug_b': np.random.binomial(1, 0.3, n),
            'adr_bleeding': np.random.binomial(1, 0.2, n)
        })
        
        features = ['age', 'gender', 'aspirin', 'nsaid', 'num_drugs', 'drug_c']
        engine = DisproportionalityAnalysisEngine(features, 'drug_b', 'adr_bleeding', seed=42)
        
        crude = engine.compute_crude_ror(df)
        self.assertIn('ROR', crude)
        self.assertIn('CI_lower', crude)
        self.assertIn('CI_upper', crude)
        
        adj = engine.compute_adjusted_ror(df)
        self.assertIn('Adjusted_ROR', adj)
        
        matched_df, ps_before, ps_after, matched_res, smd_summary = engine.perform_psm_ror(df)
        self.assertIn('SMD_Before', smd_summary.columns)
        self.assertIn('SMD_After', smd_summary.columns)
        
    def test_predictive_metrics(self):
        """Tests classification performance metric functions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.6, 0.3, 0.7])
        
        metrics = compute_all_predictive_metrics(y_true, y_prob)
        self.assertGreaterEqual(metrics['auc_roc'], 0.0)
        self.assertLessEqual(metrics['auc_roc'], 1.0)
        self.assertGreaterEqual(metrics['ece'], 0.0)
        self.assertLessEqual(metrics['ece'], 1.0)
        
    def test_smd_computation(self):
        """Tests Standardized Mean Difference calculation."""
        df = pd.DataFrame({
            'drug_b': [1, 1, 0, 0],
            'age': [70, 72, 60, 62]
        })
        smd = compute_standardized_mean_differences(df, 'drug_b', ['age'])
        self.assertGreater(smd['age'], 0.0)

if __name__ == '__main__':
    unittest.main()
