"""CreditForYou academic prototype dashboard."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from creditforyou.config import DEFAULT_MODEL_PATH, SYNTHETIC_DATA_DIR
from creditforyou.datasets import load_synthetic_table, load_transactions
from creditforyou.explain import explain_pd
from creditforyou.income import prepare_transactions
from creditforyou.model import build_features
from creditforyou.pipeline import evaluate_synthetic_customer
from db import assessment_history, init_db, save_assessment

st.set_page_config(page_title="CreditForYou", page_icon="💳", layout="wide")
st.title("CreditForYou — Income & Credit Risk Prototype")
st.caption("Academic prototype using synthetic banking activity and a separate US Lending Club PD model; not a bank policy.")


@st.cache_data(show_spinner="Reading transaction history…")
def customer_transactions(customer_id):
    return load_transactions(customers=[customer_id])


@st.cache_data(show_spinner="Running income, affordability and PD pipeline…")
def assess(customer_id, amount, term, purpose, experience, home):
    return evaluate_synthetic_customer([customer_id], amount, term, purpose, experience, home)


customers = load_synthetic_table("customers", SYNTHETIC_DATA_DIR)
accounts = load_synthetic_table("accounts", SYNTHETIC_DATA_DIR)
loans = load_synthetic_table("loans", SYNTHETIC_DATA_DIR)
customer_id = st.sidebar.selectbox("Synthetic customer", customers["customer_id"].tolist())
evaluation_mode = st.sidebar.checkbox("Evaluation mode (show ground truth)", value=False)
profile = customers[customers["customer_id"] == customer_id].iloc[0]
customer_accounts = accounts[accounts["customer_id"] == customer_id]
customer_loans = loans[loans["customer_id"] == customer_id]
tx = customer_transactions(customer_id)
prepared, tx_warnings = prepare_transactions(tx)

st.subheader("Applicant / customer selection")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customer", customer_id)
c2.metric("Accounts", len(customer_accounts))
c3.metric("Transactions", f"{len(tx):,}")
c4.metric("Existing loans", len(customer_loans))
st.write({"profession": profile["profession"], "segment": profile["customer_segment"],
          "coverage": f"{tx['timestamp'].min()} → {tx['timestamp'].max()}"})

with st.form("application"):
    st.subheader("Credit application")
    f1, f2, f3 = st.columns(3)
    requested = f1.number_input("Requested amount (USD, synthetic demo)", min_value=100.0, value=10_000.0, step=500.0)
    term = f2.selectbox("Term (months)", [12, 24, 36, 48, 60], index=2)
    purpose = f3.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"])
    f4, f5 = st.columns(2)
    experience = f4.number_input("Employment tenure (years)", min_value=0.0, max_value=50.0, value=5.0)
    home = f5.selectbox("Home ownership", ["rent", "mortgage", "own", "family", "other"])
    submitted = st.form_submit_button("Evaluate application", type="primary")

if submitted:
    results, warnings, model = assess(customer_id, requested, term, purpose, experience, home)
    row = results.iloc[0]
    st.subheader("Income analysis")
    a, b, c, d = st.columns(4)
    a.metric("Estimated monthly income", f"${row['estimated_monthly_income']:,.2f}")
    b.metric("Stability", f"{row['income_stability_score']:.1%}")
    c.metric("Confidence", f"{row['income_confidence_score']:.1%} ({row['income_confidence_label']})")
    d.metric("Active months", int(row["active_months"]))
    if evaluation_mode:
        truth = float(profile["monthly_income"])
        st.info(f"Evaluation-only ground truth: ${truth:,.2f}; absolute error: ${abs(row['estimated_monthly_income'] - truth):,.2f}")
    metrics = pd.DataFrame({"metric": ["Average expense", "Expense / income", "Average net cash flow", "Positive cash-flow months"],
                            "value": [row["avg_monthly_expense"], row["expense_income_ratio"],
                                      row["avg_monthly_net_cashflow"], row["positive_cashflow_month_ratio"]]})
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    monthly = prepared.assign(month=prepared["date"].dt.to_period("M").astype(str),
                               income=prepared["inflow"].where(prepared["is_income"], 0),
                               expense=prepared["outflow"]).groupby("month", as_index=False)[["income", "expense"]].sum()
    chart_data = monthly.melt("month", var_name="flow", value_name="amount")
    st.altair_chart(alt.Chart(chart_data).mark_line(point=True).encode(x="month:N", y="amount:Q", color="flow:N"),
                     use_container_width=True)

    st.subheader("Credit risk and final decision")
    r1, r2, r3, r4 = st.columns(4)
    pd_value = row["pd_used"] if pd.notna(row["pd_used"]) else row.get("pd_challenger")
    pd_label = ("Model-predicted default probability" if pd.notna(row["pd_used"])
                else "PD challenger (Lending Club, not used in decision)")
    r1.metric(pd_label, f"{pd_value:.1%}" if pd.notna(pd_value) else "Unavailable")
    r2.metric("Credit score", f"{row['credit_score']:.2f}/100")
    r3.metric("Tier", row["tier"])
    r4.metric("Recommendation", row["decision"])
    st.caption(f"Model: {model.get('model_version', model['model_name']) if model else 'unavailable'}")
    breakdown = pd.DataFrame({"component": ["Income stability", "Affordability", "Tenure", "Repayment history", "ML PD"],
                              "points": [row["score_income_stability"], row["score_debt_to_income"], row["score_experience"],
                                         row["score_repayment_history"], row["score_ml"]]})
    st.bar_chart(breakdown.set_index("component"))
    st.write("Reason codes:", row["reason_codes"] or "No reason code")
    if row["missing_components"]:
        st.warning("Unavailable components were disabled and remaining weights normalized: " + ", ".join(row["missing_components"]))
    if model:
        model_features = build_features(pd.DataFrame([row]).assign(monthly_income=row["estimated_monthly_income"]))
        st.write("Local model sensitivity (not SHAP or causal attribution):")
        st.dataframe(pd.DataFrame(explain_pd(model, model_features)), hide_index=True, use_container_width=True)
    for warning in tx_warnings + warnings:
        st.warning(warning)
    row_dict = row.to_dict()
    row_dict["model_version"] = model.get("model_version") if model else None
    if st.button("Save assessment to SQLite"):
        init_db()
        app_id = save_assessment(row_dict, profile.to_dict())
        st.success(f"Saved application {app_id}")

history = assessment_history(customer_id)
if history:
    with st.expander("Saved assessment history"):
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
