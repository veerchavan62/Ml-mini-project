"""
Master Execution Pipeline
Online News Popularity: End-to-End Interpretable Machine Learning Framework
"""

import os
import json
import time
import numpy as np
import pandas as pd

from src.data_pipeline import load_and_clean_data, split_and_scale_data
from src.models import train_regression_models, train_direct_classifiers
from src.evaluation import (
    evaluate_regression,
    optimize_classification_threshold,
    evaluate_classification
)
from src.interpretability import (
    categorize_features,
    extract_linear_coefficients,
    extract_tree_importances,
    plot_top_feature_importances,
    plot_threshold_optimization,
    plot_roc_pr_curves,
    plot_confusion_matrices,
    plot_residuals_and_predictions
)


def main():
    start_time = time.time()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 80)
    print("ONLINE NEWS POPULARITY: MACHINE LEARNING & POPULARITY ASSESSMENT FRAMEWORK")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # 1. DATA CLEANING & PREPROCESSING
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading and cleaning dataset...")
    data_dict = load_and_clean_data("OnlineNewsPopularity.csv")
    print(f"  Raw data shape:     {data_dict['initial_shape']}")
    print(f"  Cleaned data shape: {data_dict['clean_shape']}")
    print(f"  Feature count:      {len(data_dict['feature_names'])}")
    print(f"  Median shares:      {np.median(data_dict['y_raw']):.1f}")
    print(f"  Mean shares:        {np.mean(data_dict['y_raw']):.1f} (Std: {np.std(data_dict['y_raw']):.1f})")
    print(f"  Log(shares) Mean:   {np.mean(data_dict['y_log']):.3f} (Std: {np.std(data_dict['y_log']):.3f})")
    
    # -------------------------------------------------------------
    # 2. DATA SPLITTING & SCALING
    # -------------------------------------------------------------
    print("\n[Step 2/5] Splitting into Train (80%) and Test (20%) with stratification...")
    split_data = split_and_scale_data(
        data_dict["X"],
        data_dict["y_log"],
        data_dict["y_raw"],
        data_dict["y_class_median"],
        test_size=0.2,
        random_state=42
    )
    print(f"  Train sample size:  {len(split_data['idx_train'])}")
    print(f"  Test sample size:   {len(split_data['idx_test'])}")
    
    X_train_scaled = split_data["X_train_scaled"]
    X_test_scaled = split_data["X_test_scaled"]
    X_train_df = split_data["X_train_df"]
    X_test_df = split_data["X_test_df"]
    y_train_log = split_data["y_train_log"]
    y_test_log = split_data["y_test_log"]
    y_train_raw = split_data["y_train_raw"]
    y_test_raw = split_data["y_test_raw"]
    y_train_class = split_data["y_train_class"]
    y_test_class = split_data["y_test_class"]
    feature_names = data_dict["feature_names"]
    
    # -------------------------------------------------------------
    # 3. REGRESSION MODEL TRAINING & TEST EVALUATION
    # -------------------------------------------------------------
    print("\n[Step 3/5] Training Regression Models for Expected Share Estimation...")
    reg_models = train_regression_models(
        X_train_scaled=X_train_scaled,
        y_train_log=y_train_log,
        X_train_raw=X_train_df,
        feature_names=feature_names,
        random_state=42
    )
    
    reg_results = {}
    preds_dict_log = {}
    preds_dict_raw = {}
    
    for name, item in reg_models.items():
        m = item["model"]
        use_scaled = item["use_scaled"]
        
        # Predictions
        if use_scaled:
            y_pred_log = m.predict(X_test_scaled)
        else:
            y_pred_log = m.predict(X_test_df)
            
        preds_dict_log[name] = y_pred_log
        metrics = evaluate_regression(y_test_log, y_pred_log, y_true_raw=y_test_raw)
        preds_dict_raw[name] = metrics.pop("y_pred_raw")
        reg_results[name] = metrics
        
    df_reg_results = pd.DataFrame(reg_results).T
    print("\n" + "-" * 75)
    print("REGRESSION TEST PERFORMANCE METRICS (EXPECTED SHARES ESTIMATION)")
    print("-" * 75)
    print(df_reg_results.to_string())
    df_reg_results.to_csv("results/regression_metrics.csv")
    
    # -------------------------------------------------------------
    # 4. THRESHOLD OPTIMIZATION & POPULARITY ASSESSMENT CONVERSION
    # -------------------------------------------------------------
    print("\n[Step 4/5] Converting Continuous Share Estimates into Optimized Popularity Assessment...")
    
    # We optimize the threshold on the training set for XGBoost Regressor
    best_reg_model = reg_models["XGBoost Regressor"]["model"]
    y_train_pred_xgb_log = best_reg_model.predict(X_train_df)
    
    opt_result = optimize_classification_threshold(
        y_true_class=y_train_class,
        y_pred_continuous=y_train_pred_xgb_log,
        metric="f1"
    )
    best_tau_log = opt_result["best_threshold"]
    best_tau_raw = np.expm1(best_tau_log)
    default_tau_log = np.log1p(1400.0)
    
    print(f"  Benchmark Default Decision Threshold (Median 1400 shares): {default_tau_log:.3f}")
    print(f"  F1-Optimized Decision Threshold (Train set):               {best_tau_log:.3f} (approx {best_tau_raw:.0f} shares)")
    
    # Train direct classifiers for benchmark comparison
    direct_classifiers = train_direct_classifiers(
        X_train_scaled=X_train_scaled,
        X_train_raw=X_train_df,
        y_train_class=y_train_class,
        random_state=42
    )
    
    # Generate assessment strategies on test set
    y_test_pred_xgb_log = preds_dict_log["XGBoost Regressor"]
    
    # Strategy A: XGBoost Regressor + Default Threshold (log(1400))
    xgb_pred_default = (y_test_pred_xgb_log >= default_tau_log).astype(int)
    # Strategy B: XGBoost Regressor + F1-Optimized Threshold
    xgb_pred_opt = (y_test_pred_xgb_log >= best_tau_log).astype(int)
    # Strategy C: ElasticNet Regressor + Default Threshold
    elastic_pred_default = (preds_dict_log["ElasticNet"] >= default_tau_log).astype(int)
    # Strategy D: Direct Logistic Regression
    direct_lr_prob = direct_classifiers["Direct Logistic Regression"]["model"].predict_proba(X_test_scaled)[:, 1]
    direct_lr_pred = (direct_lr_prob >= 0.5).astype(int)
    # Strategy E: Direct XGBoost Classifier
    direct_xgb_prob = direct_classifiers["Direct XGBoost Classifier"]["model"].predict_proba(X_test_df)[:, 1]
    direct_xgb_pred = (direct_xgb_prob >= 0.5).astype(int)
    
    clf_evals = {
        "XGBoost Regressor (Default Threshold = 1400)": evaluate_classification(
            y_test_class, xgb_pred_default, y_test_pred_xgb_log
        ),
        "XGBoost Regressor (Optimized Threshold tau*)": evaluate_classification(
            y_test_class, xgb_pred_opt, y_test_pred_xgb_log
        ),
        "ElasticNet Regressor (Default Threshold)": evaluate_classification(
            y_test_class, elastic_pred_default, preds_dict_log["ElasticNet"]
        ),
        "Direct Logistic Regression": evaluate_classification(
            y_test_class, direct_lr_pred, direct_lr_prob
        ),
        "Direct XGBoost Classifier": evaluate_classification(
            y_test_class, direct_xgb_pred, direct_xgb_prob
        )
    }
    
    # Format classification table
    clf_table_data = {}
    cm_dict = {}
    for name, res in clf_evals.items():
        cm = res["Confusion Matrix"]
        cm_dict[name.split("(")[0].strip()] = cm
        clf_table_data[name] = {
            "Accuracy": res["Accuracy"],
            "Balanced Acc": res["Balanced Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1-Score": res["F1-Score"],
            "ROC-AUC": res["ROC-AUC"],
            "PR-AUC": res["PR-AUC"],
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1]
        }
        
    df_clf_results = pd.DataFrame(clf_table_data).T
    print("\n" + "-" * 85)
    print("POPULARITY ASSESSMENT TEST PERFORMANCE METRICS (HIGH-IMPACT IDENTIFICATION)")
    print("-" * 85)
    print(df_clf_results.to_string())
    df_clf_results.to_csv("results/classification_metrics.csv")
    
    # -------------------------------------------------------------
    # 5. INTERPRETABILITY & VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n[Step 5/5] Extracting Interpretability Insights and Generating Visualizations...")
    
    # 5.1 Coefficients and Importances
    elastic_model = reg_models["ElasticNet"]["model"]
    linear_coefs = extract_linear_coefficients(elastic_model, feature_names)
    
    xgb_reg_model = reg_models["XGBoost Regressor"]["model"]
    xgb_importances = extract_tree_importances(xgb_reg_model, feature_names)
    
    rf_reg_model = reg_models["Random Forest"]["model"]
    rf_importances = extract_tree_importances(rf_reg_model, feature_names)
    
    # Save feature importances to CSV
    df_importance = pd.DataFrame({
        "ElasticNet_Coef": linear_coefs,
        "XGBoost_Gain": xgb_importances,
        "RandomForest_MDI": rf_importances
    })
    df_importance.to_csv("results/feature_importance_summary.csv")
    
    print("\nTop 10 Positive Drivers of Popularity (ElasticNet Standardized Coef):")
    print(linear_coefs.head(10).to_string())
    print("\nTop 10 Negative Drivers of Popularity (ElasticNet Standardized Coef):")
    print(linear_coefs.tail(10).to_string())
    
    print("\nTop 10 Most Influential Features (XGBoost Gain):")
    print(xgb_importances.head(10).to_string())
    
    # 5.2 Category Importance Breakdown
    categories = categorize_features(feature_names)
    cat_summary = {}
    for cat_name, cols in categories.items():
        cat_summary[cat_name] = {
            "Feature Count": len(cols),
            "Total XGBoost Gain": xgb_importances[cols].sum(),
            "Avg XGBoost Gain": xgb_importances[cols].mean(),
            "Total RF MDI": rf_importances[cols].sum()
        }
    df_cat_summary = pd.DataFrame(cat_summary).T.sort_values("Total XGBoost Gain", ascending=False)
    print("\nDomain Category Importance Summary:")
    print(df_cat_summary.to_string())
    df_cat_summary.to_csv("results/category_importance_summary.csv")
    
    # 5.3 Plotting
    print("\nRendering publication figures...")
    plot_top_feature_importances(linear_coefs, xgb_importances, rf_importances)
    plot_threshold_optimization(opt_result["threshold_history"], best_tau_log)
    
    roc_pr_dict = {
        "XGBoost Regressor (Continuous)": {
            "y_true": y_test_class,
            "scores": y_test_pred_xgb_log,
            "roc_auc": clf_evals["XGBoost Regressor (Optimized Threshold tau*)"]["ROC-AUC"],
            "pr_auc": clf_evals["XGBoost Regressor (Optimized Threshold tau*)"]["PR-AUC"]
        },
        "Direct XGBoost Classifier": {
            "y_true": y_test_class,
            "scores": direct_xgb_prob,
            "roc_auc": clf_evals["Direct XGBoost Classifier"]["ROC-AUC"],
            "pr_auc": clf_evals["Direct XGBoost Classifier"]["PR-AUC"]
        },
        "Direct Logistic Regression": {
            "y_true": y_test_class,
            "scores": direct_lr_prob,
            "roc_auc": clf_evals["Direct Logistic Regression"]["ROC-AUC"],
            "pr_auc": clf_evals["Direct Logistic Regression"]["PR-AUC"]
        },
        "ElasticNet Regressor (Continuous)": {
            "y_true": y_test_class,
            "scores": preds_dict_log["ElasticNet"],
            "roc_auc": clf_evals["ElasticNet Regressor (Default Threshold)"]["ROC-AUC"],
            "pr_auc": clf_evals["ElasticNet Regressor (Default Threshold)"]["PR-AUC"]
        }
    }
    plot_roc_pr_curves(roc_pr_dict)
    
    cm_plot_dict = {
        "XGBoost (Default tau=1400)": clf_evals["XGBoost Regressor (Default Threshold = 1400)"]["Confusion Matrix"],
        "XGBoost (Optimized tau*)": clf_evals["XGBoost Regressor (Optimized Threshold tau*)"]["Confusion Matrix"],
        "Direct XGB Classifier": clf_evals["Direct XGBoost Classifier"]["Confusion Matrix"]
    }
    plot_confusion_matrices(cm_plot_dict)
    plot_residuals_and_predictions(y_test_log, y_test_pred_xgb_log)
    
    elapsed = time.time() - start_time
    print(f"\nPipeline execution completed in {elapsed:.1f} seconds!")
    print("All models, test metrics, and visualizations successfully generated.")


if __name__ == "__main__":
    main()
