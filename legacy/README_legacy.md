# 💳 CreditForYou — Next-Gen AI Alternative Credit-Scoring & Risk Engine

**CreditForYou** is an enterprise-grade hybrid **Rule-Based & Machine Learning Credit Risk Platform** designed to evaluate non-traditional financial behavior, income stability, payment discipline, and cash flow patterns.

It enables financial institutions to evaluate thin-file and credit-invisible consumers accurately by processing **Bank Account Statements** and **Salary Payslip PDFs** into instant **0–1000 point credit scores**, default probabilities, and pre-approved credit offers.



## 🌟 Core Features

### 1. 📄 Financial PDF Document Extractor
- Automatic text parsing for **Bank Statements**, **Salary Payslips**, and **Credit Bureau Reports** using `pypdf`.
- Extracts key financial metrics: Monthly Income, Monthly Spend, Savings Days, Debt-to-Income (DTI), Credit Utilization, Rent Payment History, Digital Footprint Rate, and Delinquencies.

### 2. 🎯 Hybrid Scoring & Risk Engine
- **Rule-Based Scoring Engine (0–1000 pts):** Computes scoring across 4 financial pillars (Lifestyle, Spending Behavior, Repayment Discipline, Bonus & Risk Factors).
- **Machine Learning Probability of Default (PD):** Trained **XGBoost Classifier** ($\text{ROC-AUC} = 0.9910$) predicting risk of default ($PD \in [0, 1]$).
- **ML Credit Score:** Calculated as $\text{ML Score} = 1000 \times (1 - PD)$.

### 3. ⚙️ Verification & Missing Fields Editor
- **Locked Read-Only PDF Extraction:** Values parsed directly from uploaded PDFs are locked as read-only to preserve document integrity.
- **Missing Parameters Handling:** Unextracted or missing parameters are initialized to `0` or `empty` and can be manually edited before score computation.

### 4. 👥 Dual-Role Enterprise Portal
- **Borrower / Applicant Dashboard:**
  - Composite CreditForYou Scorecard & Risk Tier Badge (`EXCELLENT`, `GOOD`, `FAIR`, `HIGH RISK`).
  - Score Component Contribution Bar Chart (Altair).
  - **Explainable AI (XAI):** Positive Drivers (Score Boosters) & High Risk Factors (Areas To Improve).
  - Pre-Approved Credit Offers matched against candidate eligibility.
- **Lender Officer Workspace:**
  - Filterable Applicant Registry (Filter by Score Range, Income, Max DTI, and Approval Status).
  - Portfolio Risk Metrics (Cohort Average Score, Total Capital Exposure, Approval Counts).
  - Quick **Approve** / **Reject** decision controls with database persistence.

### 5. 🔄 Real-Time What-If Score Simulator
- Interactively test financial scenarios and observe real-time credit score deltas ($\Delta \text{pts}$):
  - 🟢 **Autopay Builder:** Simulates setting up automated recurring payments.
  - 🟢 **Savings Reserve Booster:** Simulates increasing liquid emergency fund reserves.
  - 🔴 **Discretionary Spending Spike:** Simulates high monthly expenditure surges.
  - 🔴 **Delinquency Impact:** Simulates missed payment delinquencies.

### 6. 🏦 Partner Banking Checkout Gateway
- Matches applicants with pre-approved credit products (Credit Cards, Personal Loans, Micro-Credit Lines) from `product_catalog.json`.
- Interactive payment checkout gateway supporting Credit/Debit Cards, Net Banking, UPI, and Auto-Debit.
- Generates official downloadable **Payment Receipts (`.TXT`)** and **Transaction Reports (`.CSV`)**.

### 7. 🗄️ SQLite Database Persistence
- Built-in SQLite database (`creditforyou_legacy.db`) storing user accounts, applicant profiles, credit evaluations, lender decision logs, and payment transactions (`db.py`).



## 📌 Rule Engine Scoring Framework Architecture

The Rule-Based Engine (`ml/rule_engine.py`) scores applicants across **4 Core Pillars**:

| Pillar | Max Points | Key Evaluated Parameters |
| :--- | :---: | :--- |
| **🏢 Lifestyle** | **350 pts** | Employment Stability (150 pts), Housing Status (80 pts), Digital Footprint (70 pts), Education Level (50 pts) |
| **💰 Spending Behavior** | **350 pts** | Spend-to-Income Ratio (120 pts), Expense Diversity (80 pts), Cashflow Volatility (70 pts), Savings Reserve Days (80 pts) |
| **💳 Repayment Discipline** | **570 pts** | On-Time Payment Rate (200 pts), Debt-to-Income / DTI (120 pts), Credit Utilization (100 pts), Delinquencies (150 pts) |
| **🎁 Bonus & Risk Factors** | **±50 pts** | Positive Financial Habits (+20 pts each, max +50), Risk Flags (-20 pts each, max -50) |

---

## 📂 Project Structure

```
CreditForYou/
├── app.py                      # Main Streamlit Web Application (UI & Portals)
├── db.py                       # SQLite Database Layer (creditforyou_legacy.db)
├── run_pipeline.py             # End-to-end ML Pipeline Execution Script
├── requirements.txt            # Project Python dependencies
├── creditforyou_legacy.db                # SQLite Persistent Database
├── hero_image.jpg              # Landing page hero visual asset
├── product_catalog.json        # Financial products & credit offers catalog
├── demographic_data.json       # Raw demographic dataset
├── id_mapping.json             # User ID to Applicant ID mapping
├── merged_data.json            # Consolidated alternative dataset
├── new_age_sample_data.json    # Sample alternative credit data
├── transactional_data.csv      # Transactional history logs
├── models/                     # Saved Machine Learning Models & Artifacts
│   ├── xgboost_model.pkl       # Trained XGBoost Risk Classifier
│   ├── logistic_model.pkl      # Baseline Logistic Regression Model
│   ├── preprocessor.pkl        # Data Preprocessing Pipeline
│   └── selected_features.json  # Feature selection mapping
└── ml/                         # Modular Machine Learning Codebase
    ├── __init__.py
    ├── rule_engine.py          # Rule Engine Scoring Logic (0-1000 pts)
    ├── data_loader.py          # Dataset Loader & Integrator
    ├── preprocessing.py        # Feature Transformer & Scaler
    ├── feature_engineering.py # Financial Ratio & Metric Derivation
    ├── feature_selection.py   # Feature Importance Selection
    ├── target_generation.py    # Synthetic Default Target Generator
    ├── train.py                # Model Training Script
    ├── evaluate.py             # ROC-AUC & Model Evaluation Metrics
    ├── predict.py              # Credit Score & PD Inference API
    └── explain.py              # SHAP Model Explainability (XAI)
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed, then install the required packages:

```bash
pip install -r requirements.txt
pip install pypdf altair
```

### 2. Run the Web Application

Launch the interactive CreditForYou portal:

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at:
👉 **`http://localhost:8501`**

### 3. Run the End-to-End ML Pipeline

To re-train the models and execute data pre-processing, target generation, and evaluation:

```bash
python run_pipeline.py
```
# CreditForYou Hackathon - Alternative Credit-Scoring Platform

## 🚀 Live Deployment

### 🌐 Access the CreditForYou Application

👉 **Live Application:**  
**https://creditforyou.streamlit.app/**

> The complete CreditForYou Alternative Credit-Scoring Platform is deployed and available online.  
> Click the link above to access the live application directly.

---

This repository contains the unified **Rule-Based & ML Credit-Scoring Platform** for the **CreditForYou** hackathon project.

## 💻 Programmatic Usage & API Example

You can use the modular `ml` engine directly in your Python code:

```python
from ml.predict import predict_credit_risk
from ml.explain import explain_applicant_risk
from ml.rule_engine import run_what_if_simulation

# Define applicant financial profile
applicant_data = {
    "applicant_id": "APP_8842",
    "months_at_job": 24,
    "housing": "rent",
    "employment_status": "employed",
    "monthly_income": 4500.0,
    "monthly_spend": 1800.0,
    "essential_pct": 0.60,
    "cashflow_volatility": 0.08,
    "savings_days": 90,
    "on_time_rate": 0.96,
    "dti": 0.20,
    "credit_util": 0.25,
    "delinq_30plus": 0,
    "delinq_60plus": 0,
    "delinq_90plus": 0,
    "positive_habits": 2,
    "risk_flags": 0
}

# 1. Compute Rule-Based Score & ML Probability of Default
res = predict_credit_risk(applicant_data)
print(f"Rule Score : {res['rule_credit_score']} / 1000")
print(f"Default Risk: {res['pd']*100:.1f}%")
print(f"Risk Tier   : {res['risk_info']['tier']}")

# 2. Run What-If Simulation (Autopay Builder)
sim = run_what_if_simulation(applicant_data, "autopay_builder")
print(f"Autopay Builder Delta: {sim['score_delta']:+d} pts (New Score: {sim['simulated_score']})")

# 3. Get Model Explainability (SHAP & Factor Drivers)
explanation = explain_applicant_risk(applicant_data)
print("Top Positive Drivers:", explanation["top_negative_risk_contributors"])


## 🛡️ Data Security & Compliance

- **Mask Sensitive Numbers Toggle:** Built-in masking toggle for financial values and scores during presentations or screen-sharing.
- **256-Bit SSL/TLS Encryption Concept:** Secure handling of PDF text parsing without transmitting external files to 3rd-party services.
- **Read-Only Lock:** Extracted document parameters are locked as read-only to prevent unauthorized tampering.

---

### 📄 License & Attribution
Developed for the **BNP Paribas / CreditForYou Hackathon Project**. Built with Python, Streamlit, XGBoost, SHAP, Altair, and SQLite.
