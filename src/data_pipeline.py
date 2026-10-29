"""
Data Cleaning, Preprocessing, and Feature Engineering Pipeline
for Online News Popularity Framework
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


def load_and_clean_data(filepath="OnlineNewsPopularity.csv"):
    """
    Load raw UCI Online News Popularity dataset, clean column names,
    handle corrupt rows and sentinels, and prepare feature matrices.
    """
    df = pd.read_csv(filepath)
    
    # 1. Clean whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    initial_shape = df.shape
    
    # 2. Handle known data corruption:
    # Article with n_unique_tokens == 701 is an erroneous scraper artifact
    df = df[df["n_unique_tokens"] <= 1.0].copy()
    
    # 3. Handle zero content tokens and token rate anomalies
    # Flag articles with no content body
    df["is_empty_content"] = (df["n_tokens_content"] == 0).astype(float)
    
    # 4. Handle negative keyword sentinels (-1 indicates no historical keyword data)
    kw_cols = [
        "kw_min_min", "kw_max_min", "kw_avg_min",
        "kw_min_max", "kw_max_max", "kw_avg_max",
        "kw_min_avg", "kw_max_avg", "kw_avg_avg"
    ]
    for col in kw_cols:
        # Create a sentinel flag if negative
        df[f"{col}_missing"] = (df[col] < 0).astype(float)
        # Clip negative sentinels to zero for numerical stability
        df[col] = df[col].clip(lower=0)
    
    # 5. Handle self-reference sentinels
    self_ref_cols = [
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess"
    ]
    for col in self_ref_cols:
        df[col] = df[col].clip(lower=0)
        
    # 6. Separate features and target
    # Non-predictive metadata: url, timedelta
    drop_cols = ["url", "timedelta", "shares"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols].copy()
    y_raw = df["shares"].values
    
    # 7. Target transformation: log(1 + shares) to stabilize variance and right-skew
    y_log = np.log1p(y_raw)
    
    # Binary classification target based on standard benchmark median threshold (1400 shares)
    y_class_median = (y_raw >= 1400).astype(int)
    # Viral / Upper-quartile target (2800 shares)
    y_class_top25 = (y_raw >= 2800).astype(int)
    
    return {
        "df_clean": df,
        "X": X,
        "y_raw": y_raw,
        "y_log": y_log,
        "y_class_median": y_class_median,
        "y_class_top25": y_class_top25,
        "feature_names": feature_cols,
        "initial_shape": initial_shape,
        "clean_shape": df.shape
    }


def split_and_scale_data(X, y_log, y_raw, y_class, test_size=0.2, random_state=42):
    """
    Split into train and test sets, and fit standard/robust scalers.
    Stratify using the binary popularity class to preserve distribution balance.
    """
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class
    )
    
    X_train_df = X.iloc[idx_train].copy()
    X_test_df = X.iloc[idx_test].copy()
    
    y_train_log = y_log[idx_train]
    y_test_log = y_log[idx_test]
    
    y_train_raw = y_raw[idx_train]
    y_test_raw = y_raw[idx_test]
    
    y_train_class = y_class[idx_train]
    y_test_class = y_class[idx_test]
    
    # Standard scaler for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_log": y_train_log,
        "y_test_log": y_test_log,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "y_train_class": y_train_class,
        "y_test_class": y_test_class,
        "scaler": scaler,
        "idx_train": idx_train,
        "idx_test": idx_test
    }
