"""
Model Explainability Module for CreditForYou ML Pipeline.

Implements SHAP (TreeExplainer) for the trained XGBoost model.
Provides per-applicant risk explanations (top positive and negative risk contributors,
SHAP values, and human-readable narratives) as well as global feature importance.
"""

import os
import json
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

from ml.predict import load_inference_artifacts
from ml.feature_engineering import engineer_features
from ml.rule_engine import compute_rule_based_credit_score


# Human-readable template mapping for feature impact
HUMAN_EXPLANATION_TEMPLATES = {
    "credit_util": {
        "pos": "High credit utilization -> increased predicted default risk.",
        "neg": "Low credit utilization -> reduced predicted default risk."
    },
    "risk_flags": {
        "pos": "Elevated risk flags present -> increased predicted default risk.",
        "neg": "Low/zero risk flags -> reduced predicted default risk."
    },
    "dti": {
        "pos": "High debt-to-income (DTI) ratio -> increased predicted default risk.",
        "neg": "Low debt-to-income (DTI) ratio -> reduced predicted default risk."
    },
    "on_time_rate": {
        "pos": "Lower on-time payment performance -> increased predicted default risk.",
        "neg": "High on-time payment rate -> reduced predicted default risk."
    },
    "digital_payment_rate": {
        "pos": "Low digital payment adoption -> increased predicted default risk.",
        "neg": "Strong digital payment habits -> reduced predicted default risk."
    },
    "housing_encoded": {
        "pos": "Non-property ownership / unverified housing -> increased predicted default risk.",
        "neg": "Stable housing / property ownership -> reduced predicted default risk."
    },
    "spend_income_ratio": {
        "pos": "High spend-to-income ratio -> increased predicted default risk.",
        "neg": "Healthy spend-to-income buffer -> reduced predicted default risk."
    },
    "city_tier": {
        "pos": "Tier 3 city location -> slightly adjusted default risk model.",
        "neg": "Tier 1/2 city location -> favorable geographic factor."
    },
    "delinq_total": {
        "pos": "Historical repayment delinquencies -> increased predicted default risk.",
        "neg": "Zero historical delinquencies -> reduced predicted default risk."
    },
    "monthly_spend": {
        "pos": "High monthly spending volume -> increased expenditure pressure.",
        "neg": "Controlled monthly spending volume -> stable cashflow."
    },
    "savings_days": {
        "pos": "Low savings reserve (days) -> increased default vulnerability.",
        "neg": "Substantial savings days reserve -> reduced predicted default risk."
    },
    "tx_late_count": {
        "pos": "Late transactional payment records -> increased predicted default risk.",
        "neg": "Zero late transactional payments -> reduced predicted default risk."
    },
    "tx_debit_credit_ratio": {
        "pos": "High debit relative to credit cashflow -> increased risk pressure.",
        "neg": "Balanced cashflow credit inflow -> reduced predicted default risk."
    },
    "positive_habits": {
        "pos": "Fewer demonstrated positive financial habits -> increased risk score.",
        "neg": "Strong positive financial habits -> reduced predicted default risk."
    },
    "habit_risk_balance": {
        "pos": "Negative habit-to-risk balance -> increased predicted default risk.",
        "neg": "Positive habit-to-risk balance -> reduced predicted default risk."
    }
}


def get_human_explanation(feature_name: str, shap_val: float) -> str:
    """
    Generates a clear human-readable narrative explanation for a given SHAP contribution.
    """
    templates = HUMAN_EXPLANATION_TEMPLATES.get(feature_name)
    if templates:
        return templates["pos"] if shap_val > 0 else templates["neg"]
    
    # Fallback template
    if shap_val > 0:
        return f"Feature '{feature_name}' elevated -> increased predicted default risk."
    else:
        return f"Feature '{feature_name}' favorable -> reduced predicted default risk."


def explain_applicant_risk(
    applicant_data: Union[Dict[str, Any], pd.DataFrame],
    top_n: int = 3,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Computes SHAP explanations & Rule-Based Factor Analysis for an individual applicant using Logistic Regression.
    """
    artifacts = load_inference_artifacts(models_dir)
    logistic_model = artifacts["logistic_model"]
    preprocessor = artifacts["preprocessor"]
    selected_features = artifacts["selected_features"]

    if isinstance(applicant_data, dict):
        df_input = pd.DataFrame([applicant_data])
        raw_dict = applicant_data
    else:
        df_input = applicant_data.copy()
        raw_dict = df_input.iloc[0].to_dict()

    applicant_id = str(df_input.get("applicant_id", pd.Series(["UNKNOWN"])).iloc[0])

    # Rule-Based Score & Factor Analysis
    rule_res = compute_rule_based_credit_score(raw_dict)

    # Transform applicant input
    df_feat = engineer_features(df_input)
    X_scaled = preprocessor.transform(df_feat)
    X_inference = X_scaled[selected_features]

    # Predict PD & ML score
    pd_prob = float(logistic_model.predict_proba(X_inference)[0, 1])
    pd_val = round(pd_prob, 4)
    ml_score_val = max(0, min(1000, int(round(1000.0 * (1.0 - pd_val)))))

    # Calculate Linear Feature Impact (Log-Odds Contribution = Model Coefficient * Scaled Feature Value)
    coefs = logistic_model.coef_[0]
    scaled_vals = X_inference.values[0]
    shap_vals_row = coefs * scaled_vals

    all_shap_dict = {}
    for feat_name, s_val in zip(selected_features, shap_vals_row):
        all_shap_dict[feat_name] = round(float(s_val), 4)

    # Sort contributions
    pos_items = sorted([(f, v) for f, v in all_shap_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
    neg_items = sorted([(f, v) for f, v in all_shap_dict.items() if v < 0], key=lambda x: x[1])

    top_pos_contributors = []
    for f_name, val in pos_items[:top_n]:
        top_pos_contributors.append({
            "feature": f_name,
            "shap_value": val,
            "explanation": get_human_explanation(f_name, val)
        })

    top_neg_contributors = []
    for f_name, val in neg_items[:top_n]:
        top_neg_contributors.append({
            "feature": f_name,
            "shap_value": val,
            "explanation": get_human_explanation(f_name, val)
        })

    return {
        "applicant_id": applicant_id,
        "rule_credit_score": rule_res["rule_credit_score"],
        "pd": pd_val,
        "ml_score": ml_score_val,
        "risk_info": rule_res["risk_info"],
        "component_scores": rule_res["component_scores"],
        "rule_factor_analysis": rule_res["factor_analysis"],
        "model": "Logistic Regression + Explainable Coefficients",
        "top_positive_risk_contributors": top_pos_contributors,
        "top_negative_risk_contributors": top_neg_contributors,
        "all_shap_values": all_shap_dict
    }


def generate_global_feature_importance(df_sample: pd.DataFrame = None, models_dir: str = "models") -> Dict[str, float]:
    """
    Computes global feature importance for Logistic Regression model.
    """
    artifacts = load_inference_artifacts(models_dir)
    logistic_model = artifacts["logistic_model"]
    preprocessor = artifacts["preprocessor"]
    selected_features = artifacts["selected_features"]

    if df_sample is None:
        from ml.data_loader import load_combined_dataset
        df_sample = load_combined_dataset(".")

    df_feat = engineer_features(df_sample)
    X_scaled = preprocessor.transform(df_feat)
    X_inference = X_scaled[selected_features]

    coefs = np.abs(logistic_model.coef_[0])
    importance_series = pd.Series(coefs, index=selected_features).sort_values(ascending=False)

    global_importance = {feat: round(float(val), 4) for feat, val in importance_series.items()}
    print("\n[Explainability] Global Feature Importance (Absolute Linear Coefficients):")
    for feat, score in global_importance.items():
        print(f"  - {feat:25s}: {score:.4f}")

    return global_importance


if __name__ == "__main__":
    sample_applicant = {
        "applicant_id": "8a0d6320-e982-445b-afa8-bb9e0054aca9",
        "age": 24,
        "monthly_income": 2500,
        "months_at_job": 12,
        "housing": "rent",
        "education": "bachelor",
        "employment_status": "employed",
        "monthly_spend": 2100,
        "essential_pct": 0.75,
        "cashflow_volatility": 0.28,
        "savings_days": 10,
        "on_time_rate": 0.58,
        "dti": 0.55,
        "credit_util": 0.78,
        "delinq_90plus": 1,
        "delinq_60plus": 0,
        "delinq_30plus": 1,
        "positive_habits": 0,
        "risk_flags": 3,
        "tx_late_ratio": 0.15,
        "tx_debit_credit_ratio": 2.5
    }

    explanation = explain_applicant_risk(sample_applicant)
    print("\nSample SHAP Risk Explanation Output:\n", json.dumps(explanation, indent=4))
    
    global_imp = generate_global_feature_importance()
