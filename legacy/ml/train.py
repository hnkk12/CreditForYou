"""
Model Training Module for CreditForYou ML Pipeline.

Trains the Primary Model (Logistic Regression)
using Stratified Train/Test split and Cross-Validation while handling class imbalance.
Saves model artifacts to models/ directory.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from ml.data_loader import load_combined_dataset
from ml.target_generation import generate_target_label
from ml.feature_engineering import engineer_features
from ml.preprocessing import PipelinePreprocessor
from ml.feature_selection import perform_feature_selection

#Model Training
def train_models(
    test_size: float = 0.20,
    random_state: int = 42,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Executes the end-to-end model training pipeline:
    1. Loads raw dataset
    2. Generates target labels
    3. Engineers features
    4. Preprocesses & scales data
    5. Performs statistical feature selection
    6. Splits data using Stratified Train/Test Split
    7. Trains Logistic Regression primary credit risk model
    8. Performs 5-Fold Stratified Cross-Validation
    9. Saves model, preprocessor, and feature list
    """
    print("[Train] Starting CreditForYou ML Training Pipeline (Logistic Regression)...")

    # 1. Load data
    df_raw = load_combined_dataset(".")

    # 2. Target Generation
    y = generate_target_label(df_raw, default_threshold=5.0)

    # 3. Feature Engineering
    df_feat = engineer_features(df_raw)

    # 4. Preprocessing & Scaling (Fit on full dataset before split or fit within train)
    preprocessor = PipelinePreprocessor()
    X_scaled, _ = preprocessor.fit_transform(df_feat)

    # 5. Statistical Feature Selection
    X_selected, selected_features = perform_feature_selection(
        X_scaled, y, save_path=os.path.join(models_dir, "selected_features.json")
    )

    # 6. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Train] Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples.")

    # Calculate class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
    print(f"[Train] Class imbalance ratio (Neg/Pos): {scale_pos_weight:.2f}")

    # 7. Model Initialization
    # Primary Model: Logistic Regression with balanced class weights
    logistic_model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state
    )

    # 8. Stratified Cross-Validation (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores_log = cross_val_score(logistic_model, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"[Train] 5-Fold Stratified CV ROC-AUC -> Logistic Regression: {cv_scores_log.mean():.4f} +/- {cv_scores_log.std():.4f}")

    # Fit final model on full X_train
    logistic_model.fit(X_train, y_train)

    # 9. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)

    # Save Logistic Regression
    joblib.dump(logistic_model, os.path.join(models_dir, "logistic_model.pkl"))

    # Save Preprocessor
    preprocessor.save(os.path.join(models_dir, "preprocessor.pkl"))

    print(f"[Train] Successfully saved model and preprocessor to {models_dir}/")

    return {
        "logistic_model": logistic_model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv_scores_logistic": cv_scores_log
    }


if __name__ == "__main__":
    train_results = train_models()

