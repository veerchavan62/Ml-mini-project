"""
Quick Results Viewer for Terminal
Displays all test performance metrics cleanly formatted without line-wrapping.
"""

import pandas as pd

def show_results():
    print("\n" + "=" * 88)
    print(" 1. REGRESSION TEST METRICS: EXPECTED SHARES ESTIMATION")
    print("=" * 88)
    df_reg = pd.read_csv("results/regression_metrics.csv", index_col=0)
    # Format nicely
    fmt_reg = df_reg.copy()
    fmt_reg["RMSLE (log)"] = fmt_reg["RMSLE (RMSE on log)"].map(lambda x: f"{x:.4f}")
    fmt_reg["MAE (log)"] = fmt_reg["MAE (log-shares)"].map(lambda x: f"{x:.4f}")
    fmt_reg["R2 Score"] = fmt_reg["R2 Score"].map(lambda x: f"{x:.4f}")
    fmt_reg["MAE (shares)"] = fmt_reg["MAE (original shares)"].map(lambda x: f"{x:,.1f}")
    fmt_reg["MedAE (shares)"] = fmt_reg["MedAE (original shares)"].map(lambda x: f"{x:,.1f}")
    
    cols_reg = ["RMSLE (log)", "MAE (log)", "R2 Score", "MAE (shares)", "MedAE (shares)"]
    print(fmt_reg[cols_reg].to_string())

    print("\n" + "=" * 88)
    print(" 2. POPULARITY ASSESSMENT: HIGH-IMPACT ARTICLE IDENTIFICATION (>= 1,400 SHARES)")
    print("=" * 88)
    df_clf = pd.read_csv("results/classification_metrics.csv", index_col=0)
    
    # Core performance metrics
    fmt_clf = pd.DataFrame(index=df_clf.index)
    fmt_clf["Accuracy"] = df_clf["Accuracy"].map(lambda x: f"{x*100:.2f}%")
    fmt_clf["Bal Acc"] = df_clf["Balanced Acc"].map(lambda x: f"{x*100:.2f}%")
    fmt_clf["Precision"] = df_clf["Precision"].map(lambda x: f"{x*100:.2f}%")
    fmt_clf["Recall"] = df_clf["Recall"].map(lambda x: f"{x*100:.2f}%")
    fmt_clf["F1-Score"] = df_clf["F1-Score"].map(lambda x: f"{x:.4f}")
    fmt_clf["ROC-AUC"] = df_clf["ROC-AUC"].map(lambda x: f"{x:.4f}")
    fmt_clf["PR-AUC"] = df_clf["PR-AUC"].map(lambda x: f"{x:.4f}")
    print(fmt_clf.to_string())

    print("\n--- Confusion Matrix Breakdown (Total Test Articles = 7,929) ---")
    fmt_cm = pd.DataFrame(index=df_clf.index)
    fmt_cm["True Neg (TN)"] = df_clf["TN"].astype(int)
    fmt_cm["False Pos (FP)"] = df_clf["FP"].astype(int)
    fmt_cm["False Neg (FN)"] = df_clf["FN"].astype(int)
    fmt_cm["True Pos (TP)"] = df_clf["TP"].astype(int)
    print(fmt_cm.to_string())

    print("\n" + "=" * 88)
    print(" 3. TOP 10 INFLUENTIAL FEATURES (XGBOOST GAIN & ELASTICNET COEFFICIENTS)")
    print("=" * 88)
    df_feat = pd.read_csv("results/feature_importance_summary.csv", index_col=0)
    top_xgb = df_feat.sort_values("XGBoost_Gain", ascending=False).head(10).copy()
    top_xgb["XGBoost Gain"] = top_xgb["XGBoost_Gain"].map(lambda x: f"{x:.4f}")
    top_xgb["ElasticNet Coef"] = top_xgb["ElasticNet_Coef"].map(lambda x: f"{x:+.4f}")
    print(top_xgb[["XGBoost Gain", "ElasticNet Coef"]].to_string())

    print("\n" + "=" * 88)
    print(" 4. DOMAIN CATEGORY IMPORTANCE BREAKDOWN")
    print("=" * 88)
    df_cat = pd.read_csv("results/category_importance_summary.csv", index_col=0).copy()
    df_cat["Feature Count"] = df_cat["Feature Count"].astype(int)
    df_cat["Total XGB Gain"] = df_cat["Total XGBoost Gain"].map(lambda x: f"{x:.4f}")
    df_cat["Avg XGB Gain"] = df_cat["Avg XGBoost Gain"].map(lambda x: f"{x:.4f}")
    df_cat["Total RF MDI"] = df_cat["Total RF MDI"].map(lambda x: f"{x:.4f}")
    print(df_cat[["Feature Count", "Total XGB Gain", "Avg XGB Gain", "Total RF MDI"]].to_string())
    print("=" * 88 + "\n")

if __name__ == "__main__":
    show_results()
