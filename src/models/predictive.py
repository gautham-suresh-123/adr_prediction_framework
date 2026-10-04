import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class PredictiveClassifierEngine:
    """
    Trains a standard predictive machine learning classifier (XGBoost)
    to predict the probability of an Adverse Drug Reaction (ADR).
    
    Standard ML treats the treatment variable (drug_a) as a feature, which does not 
    guarantee causal validity under intervention but serves as the predictive baseline.
    """
    def __init__(self, features, outcome_col, seed=42):
        self.features = features
        self.outcome_col = outcome_col
        self.seed = seed
        self.xgb_model = None
        
    def fit(self, train_df):
        """
        Trains XGBoost on the aligned clinical features.
        """
        X = train_df[self.features].copy()
        y = train_df[self.outcome_col].copy()
        
        print("Training XGBoost Classifier...")
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=self.seed,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X, y)
        
    def predict_probs(self, df):
        """
        Outputs prediction probabilities for the XGBoost classifier.
        """
        X = df[self.features].copy()
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        
        probs_df = pd.DataFrame({
            'xgb_pred_prob': xgb_probs
        })
        return probs_df

if __name__ == '__main__':
    # Simple verification run
    n = 200
    p = 5
    X = np.random.normal(0, 1, (n, p))
    # Outcome is binary
    y = np.random.binomial(1, 1 / (1 + np.exp(-X[:, 0] - X[:, 1])))
    
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(p)])
    df['adr_bleeding'] = y
    
    features = [f"X{i}" for i in range(p)]
    engine = PredictiveClassifierEngine(features, 'adr_bleeding')
    engine.fit(df)
    probs = engine.predict_probs(df)
    print("XGBoost prediction probabilities shape:", probs.shape)
    print("Sample prediction probabilities:\n", probs.head(3))
