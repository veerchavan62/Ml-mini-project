"""
Evaluation and Threshold Optimization Module
Evaluates regression accuracy and popularity assessment classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def evaluate_regression(y_true_log, y_pred_log, y_true_raw=None):
    """
    Compute comprehensive regression metrics both on log scale and original shares.
    """
    rmse_log = root_mean_squared_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)
    evs = explained_variance_score(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    
    # Back-transform to original shares: S_pred = exp(y_log) - 1
    y_pred_raw = np.expm1(np.clip(y_pred_log, 0, 15))
    
    if y_true_raw is not None:
        mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
        medae_raw = median_absolute_error(y_true_raw, y_pred_raw)
    else:
        mae_raw = np.nan
        medae_raw = np.nan
        
    return {
        "RMSLE (RMSE on log)": rmse_log,
        "MAE (log-shares)": mae_log,
        "R2 Score": r2,
        "Explained Variance": evs,
        "MAE (original shares)": mae_raw,
        "MedAE (original shares)": medae_raw,
        "y_pred_raw": y_pred_raw
    }


def optimize_classification_threshold(y_true_class, y_pred_continuous, metric="f1", n_thresholds=200):
    """
    Perform data-driven threshold optimization on continuous prediction scores
    to identify high-impact content. Searches thresholds to maximize target metric (e.g. F1).
    """
    # Candidate thresholds between 5th and 95th percentiles of predictions
    t_min = np.percentile(y_pred_continuous, 5)
    t_max = np.percentile(y_pred_continuous, 95)
    thresholds = np.linspace(t_min, t_max, n_thresholds)
    
    best_threshold = thresholds[len(thresholds) // 2]
    best_score = -1.0
    history = []
    
    for t in thresholds:
        preds = (y_pred_continuous >= t).astype(int)
        prec = precision_score(y_true_class, preds, zero_division=0)
        rec = recall_score(y_true_class, preds, zero_division=0)
        f1 = f1_score(y_true_class, preds, zero_division=0)
        acc = accuracy_score(y_true_class, preds)
        
        score = f1 if metric == "f1" else acc
        if score > best_score:
            best_score = score
            best_threshold = t
            
        history.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc
        })
        
    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "threshold_history": pd.DataFrame(history)
    }


def evaluate_classification(y_true, y_pred_binary, y_scores=None):
    """
    Compute full suite of classification metrics for popularity assessment.
    """
    acc = accuracy_score(y_true, y_pred_binary)
    bal_acc = balanced_accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    roc_auc = roc_auc_score(y_true, y_scores) if y_scores is not None else np.nan
    pr_auc = average_precision_score(y_true, y_scores) if y_scores is not None else np.nan
    
    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion Matrix": cm
    }
