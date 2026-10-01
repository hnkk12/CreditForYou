"""
Prediction API & Credit Scoring Module for CreditForYou ML Pipeline.

Calculates Probability of Default (PD) and ML Credit Score:
    ML Score = 1000 * (1 - PD)

Exposes the clean primary API function:
    predict_credit_risk(applicant_data)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

from ml.feature_engineering import engineer_features
from ml.preprocessing import PipelinePreprocessor
from ml.feature_selection import load_selected_features
from ml.rule_engine import compute_rule_based_credit_score


# Pipeline Cache to prevent repeated file reloads
_PIPELINE_CACHE: Dict[str, Any] = {}


def load_inference_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """
    Loads trained Logistic Regression model, preprocessor, and selected feature list.
    Uses caching for instant performance.
    """
    if "logistic_model" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE

    # Check alternative paths if executed from different relative directories
    base_dir = models_dir
    if not os.path.exists(os.path.join(base_dir, "logistic_model.pkl")):
        base_dir = os.path.join("ml", models_dir)
        if not os.path.exists(os.path.join(base_dir, "logistic_model.pkl")):
            raise FileNotFoundError(f"Trained models not found in {models_dir} or {base_dir}")

    model_path = os.path.join(base_dir, "logistic_model.pkl")
    prep_path = os.path.join(base_dir, "preprocessor.pkl")
    feats_path = os.path.join(base_dir, "selected_features.json")

    logistic_model = joblib.load(model_path)
    preprocessor = PipelinePreprocessor.load(prep_path)
    selected_features = load_selected_features(feats_path)

    _PIPELINE_CACHE["logistic_model"] = logistic_model
    _PIPELINE_CACHE["preprocessor"] = preprocessor
    _PIPELINE_CACHE["selected_features"] = selected_features
    _PIPELINE_CACHE["models_dir"] = base_dir

    return _PIPELINE_CACHE


def predict_credit_risk(applicant_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]], models_dir: str = "models") -> Dict[str, Any]:
    """
    Main Credit Risk Integration API:
    Calculates Rule-Based Credit Score, Probability of Default (PD), ML Credit Score, and Product Recommendations.

    Args:
        applicant_data: Dictionary, list of dictionaries, or pandas DataFrame of applicant attributes.
        models_dir: Directory where trained artifacts are stored.
    """
    artifacts = load_inference_artifacts(models_dir)
    logistic_model = artifacts["logistic_model"]
    preprocessor = artifacts["preprocessor"]
    selected_features = artifacts["selected_features"]

    # Convert dictionary or list to DataFrame
    if isinstance(applicant_data, dict):
        df_input = pd.DataFrame([applicant_data])
    elif isinstance(applicant_data, list):
        df_input = pd.DataFrame(applicant_data)
    elif isinstance(applicant_data, pd.DataFrame):
        df_input = applicant_data.copy()
    else:
        raise TypeError("applicant_data must be a dict, list of dicts, or pandas DataFrame")

    applicant_id = str(df_input.get("applicant_id", pd.Series(["UNKNOWN"])).iloc[0])

    # Step 1: Feature Engineering & Preprocessing
    df_feat = engineer_features(df_input)
    X_scaled = preprocessor.transform(df_feat)
    X_inference = X_scaled[selected_features]

    # Step 2: ML Model Prediction (Probability of Default via Logistic Regression)
    pd_probs = logistic_model.predict_proba(X_inference)[:, 1]

    # Batch or Single output processing
    if len(df_input) == 1:
        single_dict = df_input.iloc[0].to_dict()
        rule_res = compute_rule_based_credit_score(single_dict)

        pd_val = round(float(pd_probs[0]), 4)
        ml_score_val = max(0, min(1000, int(round(1000.0 * (1.0 - pd_val)))))

        result = {
            "applicant_id": applicant_id,
            "rule_credit_score": rule_res["rule_credit_score"],
            "raw_rule_points": rule_res["raw_points"],
            "pd": pd_val,
            "ml_score": ml_score_val,
            "risk_info": rule_res["risk_info"],
            "component_scores": rule_res["component_scores"],
            "subfactor_breakdown": rule_res["subfactor_breakdown"],
            "factor_analysis": rule_res["factor_analysis"],
            "recommended_products": rule_res["recommended_products"],
            "model": "Logistic Regression + Rule-Based Engine",
            "risk_probability": pd_val
        }
        return result
    else:
        # Multiple applicants batch output
        results = []
        for idx in range(len(df_input)):
            single_row = df_input.iloc[idx].to_dict()
            app_id = str(single_row.get("applicant_id", f"APP_{idx:03d}"))
            rule_res = compute_rule_based_credit_score(single_row)

            pd_val = round(float(pd_probs[idx]), 4)
            ml_score_val = max(0, min(1000, int(round(1000.0 * (1.0 - pd_val)))))

            results.append({
                "applicant_id": app_id,
                "rule_credit_score": rule_res["rule_credit_score"],
                "pd": pd_val,
                "ml_score": ml_score_val,
                "risk_info": rule_res["risk_info"],
                "component_scores": rule_res["component_scores"],
                "recommended_products": rule_res["recommended_products"],
                "model": "Logistic Regression + Rule-Based Engine",
                "risk_probability": pd_val
            })
        return {"batch_predictions": results, "count": len(results)}


if __name__ == "__main__":
    # Test prediction on a sample applicant
    sample_applicant = {
        "applicant_id": "8a0d6320-e982-445b-afa8-bb9e0054aca9",
        "age": 28,
        "monthly_income": 6500,
        "months_at_job": 36,
        "housing": "owner",
        "education": "bachelor",
        "employment_status": "employed",
        "monthly_spend": 2100,
        "essential_pct": 0.45,
        "cashflow_volatility": 0.12,
        "savings_days": 180,
        "on_time_rate": 0.95,
        "dti": 0.22,
        "credit_util": 0.15,
        "delinq_90plus": 0,
        "delinq_60plus": 0,
        "delinq_30plus": 0,
        "positive_habits": 3,
        "risk_flags": 0,
        "tx_late_ratio": 0.0,
        "tx_debit_credit_ratio": 0.8
    }

    res = predict_credit_risk(sample_applicant)
    print("Sample Prediction Output:\n", json.dumps(res, indent=4))
