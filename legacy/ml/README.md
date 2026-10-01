# CreditForYou ML / Credit-Scoring Pipeline

This module implements the end-to-end Machine Learning pipeline for the **CreditForYou** hackathon project. It processes alternative credit data, demographic data, and transactional history to estimate **Probability of Default (PD)**, convert PD into an **ML Credit Score**, and generate **SHAP-based risk explanations**.

---

## 📁 Pipeline Architecture & Code Structure

```
ml/
├── data_loader.py           # Loads raw JSON/CSV datasets & aggregates transactional metrics
├── preprocessing.py          # Data cleaning, missing value imputation, scaling & leakage prevention
├── feature_engineering.py    # Derives financial ratios, liquidity, and behavioral metrics
├── target_generation.py      # Synthetic ground-truth default target generator
├── feature_selection.py     # Statistical 3-step feature selection (Correlation, MI, RFE)
├── train.py                  # Trains XGBoost (Primary) & Logistic Regression (Baseline)
├── evaluate.py               # Evaluates ROC-AUC, Precision, Recall, F1, & Confusion Matrix
├── predict.py                # Credit Scoring API returning PD and ML Score (1000 * (1 - PD))
├── explain.py                # SHAP TreeExplainer & human-readable risk narratives
└── models/
    ├── xgboost_model.pkl    # Serialized Primary XGBoost Model
    ├── logistic_model.pkl   # Serialized Baseline Logistic Regression Model
    ├── preprocessor.pkl     # Fitted StandardScaler & Imputer
    └── selected_features.json # Selected statistical feature list
```

---

## 🎯 Target Generation Methodology

The raw datasets (`demographic_data.json`, `new_age_sample_data.json`, `merged_data.json`, `transactional_data.csv`) do not contain a pre-existing default binary target.

To build a realistic ML model for the hackathon prototype, a ground-truth default target (`default = 1`, `successful repayment = 0`) is constructed in `target_generation.py` using multi-factor financial distress rules:
- **Historical Delinquency:** `delinq_90plus` (+3 pts), `delinq_60plus` (+2 pts), `delinq_30plus` (+1 pt).
- **Payment Performance:** `on_time_rate < 0.65` (+2 pts), `tx_late_ratio > 0.08` (+2 pts).
- **Credit Stress:** `credit_util > 0.65` (+2 pts), `dti > 0.45` (+1.5 pts).
- **Liquidity Buffer:** `savings_days < 30` (+1 pt), `cashflow_volatility > 0.22` (+1 pt), `risk_flags >= 2` (+2 pts).

Applicants exceeding the calibrated threshold ($\text{Score} \ge 5.0$) are assigned `default = 1` (~21.4% default rate, consistent with standard credit risk benchmarks).

> **Note:** Target generation is kept strictly isolated from the feature selection and training pipelines.

---

## 🔬 Statistical 3-Step Feature Selection

Feature selection is performed statistically before model training to eliminate redundancy and isolate key predictive signals:

1. **Correlation Filtering:** Removes features with pairwise correlation $|r| > 0.85$.
2. **Mutual Information (`mutual_info_classif`):** Ranks informative non-linear features against the target.
3. **Recursive Feature Elimination (RFE):** Isolates the top 15 optimal features using a linear estimator.

### Summary
- **Before Feature Selection:** 40 features
- **After Feature Selection:** 15 features
- **Final Selected Features:**
  `['credit_util', 'risk_flags', 'dti', 'digital_payment_rate', 'housing_encoded', 'on_time_rate', 'spend_income_ratio', 'city_tier', 'delinq_total', 'monthly_spend', 'savings_days', 'tx_late_count', 'tx_debit_credit_ratio', 'positive_habits', 'habit_risk_balance']`

---

## 📊 Model Performance Comparison

Training uses **Stratified 5-Fold Cross-Validation** and an 80/20 train/test split. Class imbalance is handled using `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Logistic Regression.

| Model | ROC-AUC | Precision | Recall | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9415 | 0.5714 | 0.9524 | 0.7143 | `[[64, 15], [1, 20]]` |
| **XGBoost Classifier (Primary)** | **0.9910** | **0.9091** | **0.9524** | **0.9302** | `[[77, 2], [1, 20]]` |

---

## 🧮 Probability of Default (PD) & ML Credit Score

For any applicant, the module calculates:
- **PD (Probability of Default):** Estimated risk $\in [0, 1]$.
- **ML Credit Score:** $\text{ML Score} = 1000 \times (1 - \text{PD})$ bounded between $0$ and $1000$.

### Integration API Function

```python
from ml.predict import predict_credit_risk

result = predict_credit_risk(applicant_data)
```

**Output Schema:**
```json
{
    "applicant_id": "APP_001",
    "pd": 0.0109,
    "ml_score": 989,
    "model": "XGBoost",
    "selected_features": [...],
    "risk_probability": 0.0109
}
```

---

## 🔍 Model Explainability (SHAP)

Individual applicant risk predictions are explained using **SHAP (TreeExplainer)**.

```python
from ml.explain import explain_applicant_risk

explanation = explain_applicant_risk(applicant_data)
```

**Output Example:**
```json
{
    "applicant_id": "APP_HIGH_RISK_002",
    "pd": 0.9857,
    "ml_score": 14,
    "top_positive_risk_contributors": [
        {
            "feature": "credit_util",
            "shap_value": 1.1357,
            "explanation": "High credit utilization -> increased predicted default risk."
        },
        {
            "feature": "on_time_rate",
            "shap_value": 0.9198,
            "explanation": "Lower on-time payment performance -> increased predicted default risk."
        }
    ],
    "top_negative_risk_contributors": [
        {
            "feature": "spend_income_ratio",
            "shap_value": -0.1637,
            "explanation": "Healthy spend-to-income buffer -> reduced predicted default risk."
        }
    ]
}
```

---

## 🚀 How to Run the Pipeline

Run the complete pipeline end-to-end:

```bash
python run_pipeline.py
```
