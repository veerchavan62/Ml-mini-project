"""
Model Training and Prediction Modules
Contains regression models for estimating expected social shares
and direct classification models for popularity assessment comparison.
"""

import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier


def train_regression_models(X_train_scaled, y_train_log, X_train_raw, feature_names, random_state=42):
    """
    Train a suite of regression models on log-shares:
    1. Ridge Regression (L2 regularization, highly interpretable)
    2. ElasticNet Regression (L1 + L2 regularization for sparsity and interpretability)
    3. Random Forest Regressor (Non-linear ensemble)
    4. XGBoost Regressor (State-of-the-art gradient boosted trees)
    """
    models = {}
    
    # 2. Ridge Regression
    print("Fitting Ridge Regressor (L2)...")
    ridge = Ridge(alpha=10.0, random_state=random_state)
    ridge.fit(X_train_scaled, y_train_log)
    models["Ridge Regression"] = {
        "model": ridge,
        "use_scaled": True
    }
    
    # 3. ElasticNet Regression
    print("Fitting ElasticNet Regressor (L1+L2)...")
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=2000, random_state=random_state)
    elastic.fit(X_train_scaled, y_train_log)
    models["ElasticNet"] = {
        "model": elastic,
        "use_scaled": True
    }
    
    # 4. Random Forest Regressor
    print("Fitting Random Forest Regressor (150 trees, max_depth=16)...")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.6,
        n_jobs=-1,
        random_state=random_state
    )
    # Tree models can use raw feature DataFrame
    rf.fit(X_train_raw, y_train_log)
    models["Random Forest"] = {
        "model": rf,
        "use_scaled": False
    }
    
    # 5. XGBoost Regressor
    print("Fitting XGBoost Regressor (300 trees, eta=0.04, max_depth=6)...")
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist"
    )
    xgb.fit(X_train_raw, y_train_log)
    models["XGBoost Regressor"] = {
        "model": xgb,
        "use_scaled": False
    }
    
    return models


def train_direct_classifiers(X_train_scaled, X_train_raw, y_train_class, random_state=42):
    """
    Train direct binary classifiers to identify high-impact articles,
    serving as benchmark comparison to our continuous estimation + conversion framework.
    """
    classifiers = {}
    
    print("Fitting Direct Logistic Regression Classifier...")
    log_reg = LogisticRegression(C=0.1, max_iter=1000, random_state=random_state)
    log_reg.fit(X_train_scaled, y_train_class)
    classifiers["Direct Logistic Regression"] = {
        "model": log_reg,
        "use_scaled": True
    }
    
    print("Fitting Direct XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train_raw, y_train_class)
    classifiers["Direct XGBoost Classifier"] = {
        "model": xgb_clf,
        "use_scaled": False
    }
    
    return classifiers
