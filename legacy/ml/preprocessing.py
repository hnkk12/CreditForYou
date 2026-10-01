"""
Preprocessing Module for CreditForYou ML Pipeline.

Handles data cleaning, missing value imputation, duplicate removal, identifier separation,
and feature scaling while strictly preventing data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

# Identifiers that MUST be excluded from ML model features
IDENTIFIER_COLUMNS = [
    "user_id",
    "applicant_id",
    "transaction_id",
    "default",  # target column, handled separately
    "education",  # raw text string mapped to numeric
    "education_level",  # raw text string mapped to numeric
    "housing",  # raw text string mapped to numeric
    "employment_status"  # raw text string mapped to numeric
]

#Pipeline
class PipelinePreprocessor:
    """
    Stateful preprocessor class that fits on training data and transforms 
    evaluation/inference data without leakage.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer_values: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    #Raw Data Clean
    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types, drops duplicates, and handles invalid values.
        """
        cleaned_df = df.copy()

        # Remove duplicate records based on applicant_id if present
        if "applicant_id" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=["applicant_id"]).reset_index(drop=True)

        return cleaned_df

    def get_candidate_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extracts numeric candidate feature column names excluding identifiers.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in IDENTIFIER_COLUMNS]
        return feature_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fits imputers and scalers on training dataset and returns transformed DataFrame.
        """
        df_clean = self.clean_raw_data(df)
        self.feature_names = self.get_candidate_feature_columns(df_clean)

        X = df_clean[self.feature_names].copy()

        # Calculate median imputation values for numeric columns
        for col in self.feature_names:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            self.imputer_values[col] = float(median_val)
            X[col] = X[col].fillna(median_val)

        # Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=df_clean.index)

        self.is_fitted = True
        print(f"[Preprocessing] Fitted preprocessor on {len(X_scaled_df)} samples across {len(self.feature_names)} features.")
        return X_scaled_df, self.feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or single applicant inference data using fitted parameters.
        """
        if not self.is_fitted:
            raise ValueError("PipelinePreprocessor must be fitted before calling transform().")

        X = df.copy()

        # Ensure all required features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = self.imputer_values.get(col, 0.0)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.imputer_values.get(col, 0.0))

        X_subset = X[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_subset)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def save(self, filepath: str):
        """
        Saves fitted preprocessor to file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessing] Saved fitted preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PipelinePreprocessor":
        """
        Loads preprocessor from file.
        """
        return joblib.load(filepath)


if __name__ == "__main__":
    from ml.data_loader import load_combined_dataset
    from ml.feature_engineering import engineer_features
    df_raw = load_combined_dataset(".")
    df_feat = engineer_features(df_raw)
    
    prep = PipelinePreprocessor()
    X_scaled, feats = prep.fit_transform(df_feat)
    print("Preprocessed features shape:", X_scaled.shape)
    print("Candidate features count:", len(feats))
