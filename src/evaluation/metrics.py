import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) of a classifier.
    ECE measures the difference between prediction confidence and actual accuracy.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Patients in this probability bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_all_predictive_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes classification performance and calibration metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    # Calibration Curve points
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    return {
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier_score': brier,
        'ece': ece,
        'calibration_points': (prob_true, prob_pred)
    }

if __name__ == '__main__':
    y_true = [0, 0, 1, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.6, 0.3, 0.7]
    metrics = compute_all_predictive_metrics(y_true, y_prob)
    print("Test metrics:")
    for k, v in metrics.items():
        if k != 'calibration_points':
            print(f"  {k}: {v:.4f}")
