"""
Feature Engineering Module for CreditForYou ML Pipeline.

Constructs core and derived financial, behavioral, and transactional metrics
for credit risk assessment.
"""

import pandas as pd
import numpy as np


# Standardized ordinal mapping dictionaries
EDUCATION_MAP = {
    "none": 0,
    "highschool": 1,
    "high school": 1,
    "diploma": 2,
    "cert": 2,
    "bachelor": 3,
    "bachelor's": 3,
    "master": 4,
    "master's": 4,
    "phd": 5
}

HOUSING_MAP = {
    "none": 0,
    "rent": 1,
    "owner": 2
}

EMPLOYMENT_MAP = {
    "unemployed": 0,
    "student": 1,
    "self-employed": 2,
    "employed": 3
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw or partially cleaned DataFrame and generates structured financial
    and behavioral features for ML modeling.
    """
    df_feat = df.copy()

    # 1. Standardize and encode categorical attributes
    if "education" in df_feat.columns:
        df_feat["education_level_num"] = df_feat["education"].astype(str).str.lower().str.strip().map(EDUCATION_MAP).fillna(1)
    elif "education_level" in df_feat.columns:
        df_feat["education_level_num"] = df_feat["education_level"].astype(str).str.lower().str.strip().map(EDUCATION_MAP).fillna(1)
    else:
        df_feat["education_level_num"] = 1

    if "housing" in df_feat.columns:
        df_feat["housing_encoded"] = df_feat["housing"].astype(str).str.lower().str.strip().map(HOUSING_MAP).fillna(0)
    else:
        df_feat["housing_encoded"] = 0

    if "employment_status" in df_feat.columns:
        df_feat["employment_encoded"] = df_feat["employment_status"].astype(str).str.lower().str.strip().map(EMPLOYMENT_MAP).fillna(2)
    else:
        df_feat["employment_encoded"] = 2

    # City Tier numerical standard
    if "city_tier" in df_feat.columns:
        df_feat["city_tier"] = pd.to_numeric(df_feat["city_tier"], errors="coerce").fillna(2)

    # 2. Financial ratios & liquidity metrics
    income = pd.to_numeric(df_feat["monthly_income"], errors="coerce").fillna(1000.0) if "monthly_income" in df_feat.columns else pd.Series(1000.0, index=df_feat.index)
    spend = pd.to_numeric(df_feat["monthly_spend"], errors="coerce").fillna(500.0) if "monthly_spend" in df_feat.columns else pd.Series(500.0, index=df_feat.index)
    income = np.maximum(income, 1.0)
    spend = np.maximum(spend, 0.0)

    # Spend to income ratio
    df_feat["spend_income_ratio"] = np.clip(spend / (income + 1e-5), 0.0, 5.0)

    # Savings ratio (net unspent income ratio)
    df_feat["savings_ratio"] = np.clip((income - spend) / (income + 1e-5), -2.0, 1.0)

    # 3. Delinquency Aggregations
    delinq_30 = df_feat["delinq_30plus"].fillna(0).astype(int) if "delinq_30plus" in df_feat.columns else pd.Series(0, index=df_feat.index)
    delinq_60 = df_feat["delinq_60plus"].fillna(0).astype(int) if "delinq_60plus" in df_feat.columns else pd.Series(0, index=df_feat.index)
    delinq_90 = df_feat["delinq_90plus"].fillna(0).astype(int) if "delinq_90plus" in df_feat.columns else pd.Series(0, index=df_feat.index)

    df_feat["delinq_total"] = delinq_30 + delinq_60 + delinq_90
    df_feat["delinq_weighted_score"] = delinq_30 * 1.0 + delinq_60 * 2.0 + delinq_90 * 3.0

    # 4. Behavioral & Cashflow Features
    if "cashflow_volatility" in df_feat.columns:
        df_feat["cashflow_volatility"] = np.clip(df_feat["cashflow_volatility"].astype(float), 0.0, 1.0)
    
    if "essential_pct" in df_feat.columns:
        df_feat["essential_pct"] = np.clip(df_feat["essential_pct"].astype(float), 0.0, 1.0)

    if "on_time_rate" in df_feat.columns:
        df_feat["on_time_rate"] = np.clip(df_feat["on_time_rate"].astype(float), 0.0, 1.0)

    if "credit_util" in df_feat.columns:
        df_feat["credit_util"] = np.clip(df_feat["credit_util"].astype(float), 0.0, 1.0)

    if "dti" in df_feat.columns:
        df_feat["dti"] = np.clip(df_feat["dti"].astype(float), 0.0, 2.0)

    # 5. Interaction features
    # Positive habit buffer vs risk flag deficit
    pos_habits = pd.to_numeric(df_feat["positive_habits"], errors="coerce").fillna(0.0) if "positive_habits" in df_feat.columns else pd.Series(0.0, index=df_feat.index)
    risk_flags = pd.to_numeric(df_feat["risk_flags"], errors="coerce").fillna(0.0) if "risk_flags" in df_feat.columns else pd.Series(0.0, index=df_feat.index)
    df_feat["habit_risk_balance"] = pos_habits - risk_flags

    # Job stability to age ratio
    age = pd.to_numeric(df_feat["age"], errors="coerce").fillna(25.0) if "age" in df_feat.columns else pd.Series(25.0, index=df_feat.index)
    months_job = pd.to_numeric(df_feat["months_at_job"], errors="coerce").fillna(12.0) if "months_at_job" in df_feat.columns else pd.Series(12.0, index=df_feat.index)
    age = np.maximum(age, 18.0)
    df_feat["job_tenure_years"] = months_job / 12.0
    df_feat["work_age_ratio"] = df_feat["job_tenure_years"] / (age - 17.0 + 1e-5)

    print(f"[FeatureEngineering] Processed {len(df_feat)} records into {df_feat.shape[1]} features.")
    return df_feat


if __name__ == "__main__":
    from ml.data_loader import load_combined_dataset
    df_raw = load_combined_dataset(".")
    df_feat = engineer_features(df_raw)
    print("Engineered feature columns:", list(df_feat.columns))
