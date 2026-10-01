"""
CreditForYou ML / Credit-Scoring End-to-End Pipeline Execution Script.

Runs Data Loading -> Target Generation -> Feature Engineering -> Statistical Feature Selection ->
Model Training (Logistic Regression) -> Evaluation -> Inference (PD & ML Score) -> SHAP Explainability.
"""

import json
from ml.train import train_models
from ml.evaluate import evaluate_models
from ml.predict import predict_credit_risk
from ml.explain import explain_applicant_risk, generate_global_feature_importance
from ml.rule_engine import run_what_if_simulation


def main():
    print("\n" + "="*80)
    print("      CreditForYou RULE-BASED & ML CREDIT-SCORING PIPELINE - END-TO-END")
    print("="*80 + "\n")

    # Step 1: Train Models
    print("--- [STEP 1-5] TRAINING ML PIPELINE ---")
    train_results = train_models()

    # Step 2: Evaluate Models
    print("\n--- [STEP 5 EVALUATION] MODEL EVALUATION ---")
    eval_results = evaluate_models(train_results)

    # Step 3: Test Sample Inference (Rule-Based Score & ML PD Score)
    print("\n--- [STEP 6-8 INFERENCE] RULE-BASED & ML APPLICANT SCORING ---")
    sample_low_risk = {
        "applicant_id": "APP_LOW_RISK_001",
        "age": 29,
        "monthly_income": 8500,
        "months_at_job": 48,
        "housing": "owner",
        "education": "master",
        "employment_status": "employed",
        "monthly_spend": 2400,
        "essential_pct": 0.72,
        "cashflow_volatility": 0.04,
        "savings_days": 240,
        "on_time_rate": 0.98,
        "dti": 0.18,
        "credit_util": 0.12,
        "delinq_90plus": 0,
        "delinq_60plus": 0,
        "delinq_30plus": 0,
        "positive_habits": 3,
        "risk_flags": 0,
        "tx_late_ratio": 0.0,
        "tx_debit_credit_ratio": 0.7
    }

    sample_high_risk = {
        "applicant_id": "APP_HIGH_RISK_002",
        "age": 22,
        "monthly_income": 1800,
        "months_at_job": 6,
        "housing": "rent",
        "education": "highschool",
        "employment_status": "employed",
        "monthly_spend": 1700,
        "essential_pct": 0.82,
        "cashflow_volatility": 0.29,
        "savings_days": 10,
        "on_time_rate": 0.59,
        "dti": 0.58,
        "credit_util": 0.82,
        "delinq_90plus": 1,
        "delinq_60plus": 0,
        "delinq_30plus": 1,
        "positive_habits": 0,
        "risk_flags": 3,
        "tx_late_ratio": 0.18,
        "tx_debit_credit_ratio": 2.4
    }

    pred_low = predict_credit_risk(sample_low_risk)
    pred_high = predict_credit_risk(sample_high_risk)

    print("\nLow Risk Applicant Result:")
    print(json.dumps(pred_low, indent=4))

    print("\nHigh Risk Applicant Result:")
    print(json.dumps(pred_high, indent=4))

    # Step 4: What-If Simulator
    print("\n--- [WHAT-IF SIMULATION ENGINE] ---")
    scenarios = ["saving_boost", "autopay_builder", "discretionary_spike", "delinquency_spike"]
    for scen in scenarios:
        sim = run_what_if_simulation(sample_high_risk, scen)
        print(f"Scenario [{scen:20s}]: Base={sim['baseline_score']} -> Sim={sim['simulated_score']} (Delta={sim['score_delta']:+d} pts)")

    # Step 5: SHAP Explainability & Factor Analysis
    print("\n--- [STEP 9 EXPLAINABILITY] SHAP & RULE FACTOR ANALYSIS ---")
    explain_low = explain_applicant_risk(sample_low_risk)
    explain_high = explain_applicant_risk(sample_high_risk)

    print("\nLow Risk XAI Explanation:")
    print(json.dumps(explain_low, indent=4))

    print("\nHigh Risk XAI Explanation:")
    print(json.dumps(explain_high, indent=4))

    # Global SHAP feature importance
    print("\n--- GLOBAL FEATURE IMPORTANCE ---")
    generate_global_feature_importance()

    print("\n" + "="*80)
    print("      CreditForYou PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
