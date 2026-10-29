"""
Interpretability and Explainability Module
Analyzes feature importances, linear model coefficients, domain groupings,
and generates publication-quality visualization figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Set clean aesthetic style for matplotlib / seaborn
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})


def categorize_features(feature_names):
    """
    Categorize heterogeneous article features into functional domain groups.
    """
    categories = {
        "Keyword Popularity": [f for f in feature_names if f.startswith("kw_")],
        "LDA Topic Modeling": [f for f in feature_names if f.startswith("LDA_")],
        "Digital Channels": [f for f in feature_names if f.startswith("data_channel_")],
        "Temporal / Day-of-Week": [f for f in feature_names if f.startswith("weekday_") or f == "is_weekend"],
        "Content & Media Length": [
            f for f in feature_names if any(w in f for w in ["token", "href", "img", "video", "keyword", "empty", "length"])
            and not f.startswith("kw_")
        ],
        "Sentiment & Polarity": [
            f for f in feature_names if any(w in f for w in ["sentiment", "polarity", "subjectivity", "positive", "negative"])
        ],
        "Self References": [f for f in feature_names if f.startswith("self_reference_")]
    }
    return categories


def extract_linear_coefficients(model, feature_names):
    """
    Extract standardized linear regression coefficients showing directional impact.
    """
    coefs = pd.Series(model.coef_, index=feature_names)
    sorted_coefs = coefs.sort_values(ascending=False)
    return sorted_coefs


def extract_tree_importances(model, feature_names):
    """
    Extract tree-based Gini / Gain feature importances.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)


def plot_top_feature_importances(linear_coefs, xgb_importances, rf_importances, save_path="figures/feature_importance.png"):
    """
    Plot side-by-side comparison of:
    1. ElasticNet / Ridge directional coefficients (top positive & negative drivers)
    2. XGBoost Gain feature importance
    3. Random Forest feature importance
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. Linear Coefficients (top 10 positive and top 10 negative)
    top_pos = linear_coefs.head(8)
    top_neg = linear_coefs.tail(8)
    top_linear = pd.concat([top_pos, top_neg]).sort_values()
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in top_linear.values]
    
    axes[0].barh(range(len(top_linear)), top_linear.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_yticks(range(len(top_linear)))
    axes[0].set_yticklabels(top_linear.index, fontsize=9)
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_title("Standardized ElasticNet Coefficients\n(Directional Driver of Log-Shares)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Standardized Coefficient")
    
    # 2. XGBoost Feature Importance
    top_xgb = xgb_importances.head(15).sort_values()
    axes[1].barh(range(len(top_xgb)), top_xgb.values, color="#3498db", edgecolor="black", alpha=0.85)
    axes[1].set_yticks(range(len(top_xgb)))
    axes[1].set_yticklabels(top_xgb.index, fontsize=9)
    axes[1].set_title("XGBoost Regressor\nFeature Importance (Gain)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Relative Gain Importance")
    
    # 3. Random Forest Feature Importance
    top_rf = rf_importances.head(15).sort_values()
    axes[2].barh(range(len(top_rf)), top_rf.values, color="#9b59b6", edgecolor="black", alpha=0.85)
    axes[2].set_yticks(range(len(top_rf)))
    axes[2].set_yticklabels(top_rf.index, fontsize=9)
    axes[2].set_title("Random Forest Regressor\nFeature Importance (MDI / Gini)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Mean Decrease in Impurity")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance figure to: {save_path}")


def plot_threshold_optimization(history_df, best_threshold, save_path="figures/threshold_optimization.png"):
    """
    Plot metric curves across decision thresholds for popularity assessment.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(history_df["threshold"], history_df["f1"], label="F1-Score", color="#e67e22", lw=2.5)
    ax.plot(history_df["threshold"], history_df["precision"], label="Precision", color="#2980b9", lw=2, linestyle="--")
    ax.plot(history_df["threshold"], history_df["recall"], label="Recall", color="#27ae60", lw=2, linestyle="--")
    ax.plot(history_df["threshold"], history_df["accuracy"], label="Accuracy", color="#8e44ad", lw=1.5, linestyle=":")
    
    ax.axvline(best_threshold, color="#c0392b", linestyle="-.", lw=2, label=f"Optimal Threshold = {best_threshold:.3f}")
    ax.set_title("Popularity Assessment: Decision Threshold Optimization", fontsize=13, fontweight="bold")
    ax.set_xlabel("Continuous Prediction Threshold (log-shares scale)")
    ax.set_ylabel("Metric Score")
    ax.legend(loc="lower center", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold optimization plot to: {save_path}")


def plot_roc_pr_curves(eval_data_dict, save_path="figures/roc_pr_curves.png"):
    """
    Plot ROC and Precision-Recall curves comparing models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    
    for i, (name, data) in enumerate(eval_data_dict.items()):
        y_true = data["y_true"]
        scores = data["scores"]
        color = colors[i % len(colors)]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {data['roc_auc']:.3f})", color=color, lw=2)
        
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, scores)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {data['pr_auc']:.3f})", color=color, lw=2)
        
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    axes[0].set_title("ROC Curves for High-Impact Content Detection", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    baseline_pr = eval_data_dict[list(eval_data_dict.keys())[0]]["y_true"].mean()
    axes[1].axhline(baseline_pr, color="black", linestyle="--", alpha=0.5, label=f"No Skill ({baseline_pr:.2f})")
    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC/PR curves to: {save_path}")


def plot_confusion_matrices(cm_dict, save_path="figures/confusion_matrices.png"):
    """
    Plot confusion matrices for different thresholding strategies and models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
        
    labels = ["Low-Impact\n(<1400)", "High-Impact\n(>=1400)"]
    for ax, (title, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {save_path}")


def plot_residuals_and_predictions(y_test_log, y_pred_log, save_path="figures/residuals_and_predictions.png"):
    """
    Plot prediction scatter and residual distribution for continuous share estimation.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    residuals = y_test_log - y_pred_log
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test_log, y_pred_log, alpha=0.15, color="#2980b9", s=15)
    axes[0].plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], "r--", lw=2, label="Perfect Fit (y=x)")
    axes[0].set_title("Actual vs. Predicted Log-Shares (Test Set)", fontweight="bold")
    axes[0].set_xlabel("Actual Log(1 + Shares)")
    axes[0].set_ylabel("Predicted Log(1 + Shares)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#27ae60", bins=50)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_title(f"Residual Distribution (Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f})", fontweight="bold")
    axes[1].set_xlabel("Residuals: Log(Actual) - Log(Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residuals plot to: {save_path}")
