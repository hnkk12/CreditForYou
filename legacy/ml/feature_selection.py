"""
Feature Selection Module for CreditForYou ML Pipeline.

Performs rigorous statistical feature selection using:
1. Redundancy / Correlation Filtering
2. Mutual Information (mutual_info_classif)
3. Recursive Feature Elimination (RFE)

Saves the selected feature list to models/selected_features.json.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression


def remove_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.85) -> List[str]:
    """
    Step 1: Identifies and removes highly correlated / redundant features.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    keep_features = [col for col in X.columns if col not in to_drop]
    
    print(f"[FeatureSelection] Step 1 (Correlation Filter > {threshold}): Dropped {len(to_drop)} redundant features. Remaining: {len(keep_features)}.")
    return keep_features


def select_by_mutual_information(X: pd.DataFrame, y: pd.Series, top_k: int = 25) -> List[str]:
    """
    Step 2: Uses Mutual Information (mutual_info_classif) to rank informative features.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # Select features with non-zero mutual info, up to top_k
    informative_feats = mi_series[mi_series > 0.001].head(top_k).index.tolist()
    print(f"[FeatureSelection] Step 2 (Mutual Information): Selected top {len(informative_feats)} informative features.")
    return informative_feats


def select_by_rfe(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15) -> List[str]:
    """
    Step 3: Uses Recursive Feature Elimination (RFE) to isolate the final optimal feature set.
    """
    estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    
    actual_n_select = min(n_features_to_select, X.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=actual_n_select, step=1)
    rfe.fit(X, y)
    
    selected_mask = rfe.support_
    selected_feats = X.columns[selected_mask].tolist()
    print(f"[FeatureSelection] Step 3 (RFE): Selected {len(selected_feats)} final features.")
    return selected_feats


def perform_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.85,
    mi_top_k: int = 25,
    rfe_n_select: int = 15,
    save_path: str = "models/selected_features.json"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes the full 3-step statistical feature selection workflow and saves output.
    """
    initial_features = list(X.columns)
    initial_count = len(initial_features)
    
    # Step 1: Remove redundant features
    step1_feats = remove_highly_correlated_features(X[initial_features], threshold=corr_threshold)
    
    # Step 2: Mutual Information selection
    step2_feats = select_by_mutual_information(X[step1_feats], y, top_k=mi_top_k)
    
    # Step 3: RFE selection
    final_feats = select_by_rfe(X[step2_feats], y, n_features_to_select=rfe_n_select)
    final_count = len(final_feats)
    
    # Clearly print required output format
    print("\n" + "="*50)
    print("FEATURE SELECTION RESULTS:")
    print("Before feature selection:")
    print(initial_count)
    print("\nAfter feature selection:")
    print(final_count)
    print("\nFinal selected features:")
    print(final_feats)
    print("="*50 + "\n")
    
    # Save selected features to JSON
    save_selected_features(final_feats, save_path)

    return X[final_feats], final_feats


def save_selected_features(feature_list: List[str], filepath: str):
    """
    Saves selected feature names array to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=4)
    print(f"[FeatureSelection] Saved selected feature list to {filepath}")


def load_selected_features(filepath: str = "models/selected_features.json") -> List[str]:
    """
    Loads selected feature names from JSON.
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join("ml", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Selected features file not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    return feature_list


if __name__ == "__main__":
    from ml.data_loader import load_combined_dataset
    from ml.target_generation import generate_target_label
    from ml.feature_engineering import engineer_features
    from ml.preprocessing import PipelinePreprocessor

    df_raw = load_combined_dataset(".")
    y = generate_target_label(df_raw)
    df_feat = engineer_features(df_raw)
    
    prep = PipelinePreprocessor()
    X_scaled, _ = prep.fit_transform(df_feat)
    
    X_selected, selected_feats = perform_feature_selection(X_scaled, y)
