import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_calibration_curve(prob_true, prob_pred, ece, save_path=None):
    """
    Plots the calibration curve (reliability diagram) for the predictive classifier.
    """
    plt.figure(figsize=(8, 7), dpi=150)
    
    # Diagonal perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='#9CA3AF', label='Perfect Calibration')
    
    plt.plot(
        prob_pred, prob_true, 
        marker='s', 
        color='#3B82F6',
        label=f"XGBoost Classifier (ECE: {ece:.4f})",
        linewidth=2,
        markersize=6
    )
        
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight='semibold')
    plt.ylabel("Fraction of Positives", fontsize=11, fontweight='semibold')
    plt.title("ADR Probability Calibration Curve", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved calibration curve plot to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plots the ROC curve for the predictive classifier.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 7), dpi=150)
    plt.plot(fpr, tpr, color='#10B981', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='semibold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='semibold')
    plt.title('ADR Prediction ROC Curve (XGBoost)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ROC curve plot to {save_path}")
    plt.close()

def plot_propensity_score_overlap(ps_before, ps_after, save_path=None):
    """
    Plots the density curves of propensity scores for Treated and Untreated patients
    before and after Propensity Score Matching (PSM).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # Before Matching
    sns.kdeplot(ps_before['treated'], color='#EF4444', fill=True, alpha=0.2, ax=axes[0], label='Treated (Amiodarone)', linewidth=2)
    sns.kdeplot(ps_before['untreated'], color='#3B82F6', fill=True, alpha=0.1, ax=axes[0], label='Untreated (No Amiodarone)', linewidth=2)
    axes[0].set_title('Propensity Score Distribution (Before Matching)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Estimated Propensity Score P(T=1 | X)', fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('Density', fontsize=10, fontweight='semibold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    
    # After Matching
    sns.kdeplot(ps_after['treated'], color='#EF4444', fill=True, alpha=0.2, ax=axes[1], label='Treated (Amiodarone)', linewidth=2)
    sns.kdeplot(ps_after['untreated'], color='#3B82F6', fill=True, alpha=0.2, ax=axes[1], label='Untreated (No Amiodarone)', linewidth=2)
    axes[1].set_title('Propensity Score Distribution (After Matching)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Estimated Propensity Score P(T=1 | X)', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('Density', fontsize=10, fontweight='semibold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    
    plt.suptitle('Propensity Score Balance & Overlap Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved propensity score balance plot to {save_path}")
    plt.close()

def plot_subgroup_forest(subgroups_df, save_path=None):
    """
    Plots a standard clinical Forest Plot showing the Crude and Adjusted RORs (with 95% CIs)
    across different subgroups.
    """
    plt.figure(figsize=(10, 7), dpi=150)
    
    y_positions = np.arange(len(subgroups_df))
    
    # Plot Crude RORs
    plt.errorbar(
        subgroups_df['Crude ROR'].values, 
        y_positions + 0.15, 
        xerr=[(subgroups_df['Crude ROR'] - subgroups_df['Crude CI Lower']).values, 
              (subgroups_df['Crude CI Upper'] - subgroups_df['Crude ROR']).values],
        fmt='o', 
        color='#EF4444', 
        ecolor='#EF4444', 
        elinewidth=1.5, 
        capsize=4,
        label='Crude ROR (95% CI)',
        markersize=6
    )
    
    # Plot Adjusted RORs
    plt.errorbar(
        subgroups_df['Adjusted ROR'].values, 
        y_positions - 0.15, 
        xerr=[(subgroups_df['Adjusted ROR'] - subgroups_df['Adjusted CI Lower']).values, 
              (subgroups_df['Adjusted CI Upper'] - subgroups_df['Adjusted ROR']).values],
        fmt='s', 
        color='#10B981', 
        ecolor='#10B981', 
        elinewidth=1.5, 
        capsize=4,
        label='Adjusted ROR (95% CI)',
        markersize=6
    )
    
    # Null effect line (ROR = 1)
    plt.axvline(x=1.0, color='#9CA3AF', linestyle='--', linewidth=1.5, label='Null Effect (ROR = 1.0)')
    
    plt.yticks(y_positions, subgroups_df['Subgroup'].values, fontsize=10, fontweight='semibold')
    plt.xlabel('Reporting Odds Ratio (ROR) - Log Scale', fontsize=11, fontweight='semibold')
    plt.xscale('log')
    
    # Clean up x-axis ticks to make it readable in log scale
    ticks = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    plt.xticks(ticks, [str(t) for t in ticks])
    
    plt.title('Subgroup Analysis of Amiodarone Bleeding Signal (Reporting Odds Ratios)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    plt.gca().invert_yaxis() # Put first subgroup at the top
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved subgroup forest plot to {save_path}")
    plt.close()

def plot_pr_curve(y_true, y_prob, save_path=None):
    """
    Plots the Precision-Recall curve for the predictive classifier.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 7), dpi=150)
    plt.plot(recall, precision, color='#8B5CF6', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline line (positive rate)
    pos_rate = np.mean(y_true)
    plt.axhline(y=pos_rate, color='#9CA3AF', lw=1.5, linestyle='--', label=f'Random Guess (Rate = {pos_rate:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11, fontweight='semibold')
    plt.ylabel('Precision', fontsize=11, fontweight='semibold')
    plt.title('ADR Prediction Precision-Recall Curve (XGBoost)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved PR curve plot to {save_path}")
    plt.close()

def plot_feature_importances(feature_names, importances, save_path=None):
    """
    Plots the feature importances of the predictive model.
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(9, 6), dpi=150)
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis', hue=sorted_features, legend=False)
    
    plt.xlabel('Feature Importance Score', fontsize=11, fontweight='semibold')
    plt.ylabel('Clinical Features', fontsize=11, fontweight='semibold')
    plt.title('Predictive Model (XGBoost) - Feature Importances', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.4, axis='x')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved Feature Importances plot to {save_path}")
    plt.close()
