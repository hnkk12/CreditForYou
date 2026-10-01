"""
Target Generation Module for CreditForYou ML Pipeline.

Generates a clearly documented synthetic ground-truth default target (0 or 1)
based on financial vulnerability and payment delinquency rules.

METHODOLOGY:
- Raw datasets provided (demographic_data, new_age_sample_data, merged_data, transactional_data)
  do NOT contain an explicit binary default target.
- For this hackathon prototype, we define ground-truth default (1 = default, 0 = successful repayment)
  using objective multi-factor risk rules based on historical delinquencies, low payment on-time rate,
  high credit utilization, high debt-to-income (DTI), cashflow volatility, and transactional late payments.
- This logic is isolated in this module to prevent data leakage and ensure modularity.
"""

import pandas as pd
import numpy as np


def generate_target_label(df: pd.DataFrame, default_threshold: float = 5.0) -> pd.Series:
    """
    Computes a composite financial risk score and assigns binary target:
    default = 1 (high risk / default)
    default = 0 (successful repayment)
    
    Args:
        df: Input raw combined DataFrame.
        default_threshold: Risk score threshold above which default = 1.
        
    Returns:
        pd.Series containing binary default targets.
    """
    df_calc = df.copy()
    
    # Initialize risk score accumulator
    risk_score = np.zeros(len(df_calc), dtype=float)
    
    # 1. Historical Delinquency Indicators
    if "delinq_90plus" in df_calc.columns:
        risk_score += (df_calc["delinq_90plus"] > 0).astype(int) * 3.0
    if "delinq_60plus" in df_calc.columns:
        risk_score += (df_calc["delinq_60plus"] > 0).astype(int) * 2.0
    if "delinq_30plus" in df_calc.columns:
        risk_score += (df_calc["delinq_30plus"] > 0).astype(int) * 1.0
        
    # 2. Payment On-Time Performance & Transactional Lateness
    if "on_time_rate" in df_calc.columns:
        risk_score += (df_calc["on_time_rate"] < 0.65).astype(int) * 2.0
    if "tx_late_ratio" in df_calc.columns:
        risk_score += (df_calc["tx_late_ratio"] > 0.08).astype(int) * 2.0
        
    # 3. Leverage and Credit Stress
    if "credit_util" in df_calc.columns:
        risk_score += (df_calc["credit_util"] > 0.65).astype(int) * 2.0
    if "dti" in df_calc.columns:
        risk_score += (df_calc["dti"] > 0.45).astype(int) * 1.5
        
    # 4. Financial Cushion and Vulnerability
    if "savings_days" in df_calc.columns:
        risk_score += (df_calc["savings_days"] < 30).astype(int) * 1.0
    if "cashflow_volatility" in df_calc.columns:
        risk_score += (df_calc["cashflow_volatility"] > 0.22).astype(int) * 1.0
    if "risk_flags" in df_calc.columns:
        risk_score += (df_calc["risk_flags"] >= 2).astype(int) * 2.0
        
    # Assign binary default label
    target = (risk_score >= default_threshold).astype(int)
    
    default_cnt = target.sum()
    total_cnt = len(target)
    default_rate = default_cnt / total_cnt
    
    print(f"[TargetGeneration] Generated target labels: {default_cnt}/{total_cnt} defaults ({default_rate:.2%} default rate).")
    return target


if __name__ == "__main__":
    from ml.data_loader import load_combined_dataset
    df_raw = load_combined_dataset(".")
    target = generate_target_label(df_raw)
    print("Target value counts:\n", target.value_counts())
