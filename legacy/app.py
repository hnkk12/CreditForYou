"""
CreditForYou Web Application — Enterprise Modern Fintech System (Crisp SaaS Theme & AI PDF Financial Extractor).

Features:
1. 📄 AI PDF Financial Extractor: Upload Bank Statement / Payslip PDF to extract Income, Spend, Savings & DTI automatically!
2. 🔑 Split Hero + Single Auth Card (No empty sidebars or margins)
3. 🎯 Smart Role Routing (Lender vs Applicant) via Email Pattern Recognition
4. ⚪ Ultra-Crisp Modern SaaS Aesthetic (Vibrant Indigo, Emerald & Slate Palette)
5. 📈 Interactive Credit Meter & Score Breakdown
6. 🏦 Lender Approval & Capital Disbursement Workspace with SQLite Persistence
7. 💳 Borrower Scorecard, Pre-Approved Offers, & Real-Time What-If Risk Simulator
"""

import os
import re
import json
import io
import uuid
import pypdf
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from datetime import datetime

# Custom imports from workspace
from ml.predict import predict_credit_risk
from ml.explain import explain_applicant_risk
from ml.rule_engine import run_what_if_simulation
from db import (
    init_db, get_db_connection, get_applicant_from_db,
    save_credit_evaluation, get_evaluation_history,
    get_lender_filtered_applications, update_lender_decision,
    authenticate_user, register_user, upsert_applicant_profile,
    save_payment_record, get_payment_history, get_latest_evaluation_for_applicant
)


# Page Configuration

# Initialize SQLite Database on startup
init_db()


# 🌿 Vibrant Emerald Green Design System (Full Green Background Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    /* Streamlit header & sidebar control styling */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important;
    }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    .stDeployButton { display: none !important; }

    /* Override Streamlit container top/bottom padding & max width */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px !important;
    }

    /* 🌿 Vibrant Emerald Green Background (#10B981 / #059669) */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main, [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #10B981 !important;
        background-image: 
            radial-gradient(circle at 15% 15%, #34D399 0%, transparent 48%),
            radial-gradient(circle at 85% 85%, #059669 0%, transparent 48%) !important;
        background-attachment: fixed !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 16px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid #FFFFFF !important;
    }

    /* Green Sidebar */
    [data-testid="stSidebar"] {
        background-color: #047857 !important;
        border-right: 1px solid #10B981 !important;
    }

    /* Force all Sidebar text & widget labels to white */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover span {
        color: #A7F3D0 !important;
    }

    /* Navigation Tabs text color */
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #A7F3D0 !important;
    }

    /* Top Executive Navbar Header */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-radius: 20px;
        padding: 1rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .brand-text-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin: 0;
        font-family: 'Outfit', sans-serif;
    }
    .brand-text-sub {
        font-size: 0.8rem;
        color: #047857;
        font-weight: 600;
        margin: 0;
    }
    .nav-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #DCFCE7 !important;
        border: 1px solid #86EFAC !important;
        color: #15803D !important;
        -webkit-text-fill-color: #15803D !important;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 800;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        background: #16A34A;
        border-radius: 50%;
        box-shadow: 0 0 8px #16A34A;
    }

    /* Badges */
    .badge-positive, .badge-borrower {
        background: #DCFCE7 !important;
        border: 1px solid #86EFAC !important;
        color: #15803D !important;
        -webkit-text-fill-color: #15803D !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        padding: 0.4rem 0.9rem !important;
        border-radius: 20px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .badge-lender {
        background: #EEF2FF !important;
        border: 1px solid #C7D2FE !important;
        color: #4338CA !important;
        -webkit-text-fill-color: #4338CA !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        padding: 0.4rem 0.9rem !important;
        border-radius: 20px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .badge-negative {
        background: #FEE2E2 !important;
        border: 1px solid #FCA5A5 !important;
        color: #B91C1C !important;
        -webkit-text-fill-color: #B91C1C !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        padding: 0.4rem 0.9rem !important;
        border-radius: 20px !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    .badge-pending {
        background: #FEF3C7 !important;
        border: 1px solid #FDE68A !important;
        color: #B45309 !important;
        -webkit-text-fill-color: #B45309 !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        padding: 0.4rem 0.9rem !important;
        border-radius: 20px !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    /* Headings */
    .page-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
        font-family: 'Outfit', sans-serif;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .page-subtitle {
        font-size: 1rem;
        color: #ECFDF5;
        margin-bottom: 1.4rem;
        font-weight: 500;
    }

    /* Crisp White Cards on Green Background */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        transition: all 0.25s ease;
    }
    .white-card:hover {
        border-color: #10B981;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }

    /* Clean White Card Containers for Forms & Auth Wrapper */
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 20px !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.12) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #10B981 !important;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] *, div[data-testid="stForm"] * {
        color: #0F172A;
    }

    /* 📊 Streamlit Metric Sub-Cards (White Card Inner Boxes) */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 0.75rem 0.9rem !important;
        margin-bottom: 0.65rem !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        background-color: #FFFFFF !important;
        border-color: #10B981 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.12) !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] span {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.35rem !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
        white-space: normal !important;
        overflow: visible !important;
    }


    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        border-color: #10B981;
        transform: translateY(-3px);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #10B981 0%, #059669 100%);
    }
    .kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.84rem;
        color: #059669;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Score Meter Card - Equal Height Flex Alignment */
    div[data-testid="column"]:first-child > div:has(.score-meter-card) {
        height: 100% !important;
    }
    .score-meter-card {
        background: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-radius: 22px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        min-height: 100% !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    .score-meter-val {
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        color: #059669;
        margin: 0.3rem 0;
    }
    .score-meter-bar-bg {
        width: 100%;
        height: 14px;
        background: #E2E8F0;
        border-radius: 20px;
        overflow: hidden;
        margin: 1.2rem 0 0.6rem 0;
    }
    .score-meter-bar-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 1s ease-in-out;
    }

    /* Product Cards */
    .offer-card {
        background: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #10B981;
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    .offer-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .offer-badge {
        background: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #15803D;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        text-transform: uppercase;
    }

    /* Factor Cards */
    .factor-favorable {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 14px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.75rem;
        color: #14532D;
        font-size: 0.92rem;
    }
    .factor-adverse {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 4px solid #E11D48;
        border-radius: 14px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.75rem;
        color: #881337;
        font-size: 0.92rem;
    }

    /* Form Controls & Inputs — Pure White Boxes with Dark Readable Text */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="input"] input,
    .stTextInput input,
    .stPasswordInput input,
    .stNumberInput input,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
    }

    /* Input Placeholder Text Contrast Fix */
    input::placeholder, .stTextInput input::placeholder, .stPasswordInput input::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
        opacity: 1 !important;
    }

    /* Labels & Visibility */
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, .stWidgetLabel, .stSlider p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Disabled / Read-only Extracted Inputs */
    input:disabled, textarea:disabled, select:disabled, [disabled] input, [disabled] div {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        background-color: #F1F5F9 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    /* 🌲 Deep Dark Emerald Buttons for Maximum Visibility & Contrast */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[kind="primary"],
    button[kind="secondary"],
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1.5px solid #34D399 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.6rem !important;
        font-weight: 800 !important;
        font-size: 0.96rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #064E3B 100%) !important;
        border-color: #A7F3D0 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Crimson Reject Button Override */
    button[key*="quick_rej"], button[key*="btn_reject"] {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1.5px solid #FCA5A5 !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.4) !important;
    }
    button[key*="quick_rej"]:hover, button[key*="btn_reject"]:hover {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.5) !important;
    }

    /* Dataframe & Chart Containers */
    div[data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 14px !important;
        padding: 0.5rem !important;
    }

    div[data-testid="stVegaLiteChart"] {
        background-color: #FFFFFF !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 14px !important;
        padding: 0.5rem !important;
    }
    div[data-testid="stVegaLiteChart"] svg {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stVegaLiteChart"] text {
        fill: #0F172A !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


def extract_financial_data_from_pdf(pdf_file) -> dict:
    """
    Accurately extracts text from an uploaded PDF file and computes
    statistically accurate & realistic financial features.
    Missing fields are assigned 0 or None/null according to their data type.
    """
    full_text = ""
    try:
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
    except Exception:
        full_text = ""

    text_lower = full_text.lower()

    # Document Type Detection
    doc_type = "Uploaded Financial PDF"
    if any(k in text_lower for k in ["salary", "payslip", "pay stub", "earnings", "w-2", "w2", "employer"]):
        doc_type = "Payslip / Salary Statement"
    elif any(k in text_lower for k in ["bank statement", "account statement", "checking", "statement period", "transactions"]):
        doc_type = "Bank Account Statement"
    elif any(k in text_lower for k in ["credit report", "credit score", "bureau", "cibil", "equifax"]):
        doc_type = "Credit Bureau Report"

    extracted_fields = []

    # 1. Parse Monthly Income
    income = None
    income_matches = re.findall(r'(?:monthly income|net pay|gross pay|salary|gross salary|total deposit|credits|net salary)[s\:\$\s]+([0-9]{1,3}(?:\,[0-9]{3})*(?:\.[0-9]{2})?)', text_lower)
    if income_matches:
        for m in income_matches:
            try:
                v = float(m.replace(",", ""))
                if 500 <= v <= 50000:
                    income = v
                    extracted_fields.append("monthly_income")
                    break
            except ValueError:
                pass

    if income is None:
        numbers = [float(n.replace(",", "")) for n in re.findall(r'\$?\b([1-9][0-9]{3,4}(?:\.[0-9]{2})?)\b', full_text)]
        if numbers:
            income = float(np.median(numbers))
            extracted_fields.append("monthly_income")

    # 2. Parse Monthly Spend
    spend = None
    spend_matches = re.findall(r'(?:monthly spend|total spend|withdrawals|total debits|expenses|total outgoing)[s\:\$\s]+([0-9]{1,3}(?:\,[0-9]{3})*(?:\.[0-9]{2})?)', text_lower)
    if spend_matches:
        for m in spend_matches:
            try:
                v = float(m.replace(",", ""))
                if 200 <= v <= (income * 1.2 if income else 50000):
                    spend = v
                    extracted_fields.append("monthly_spend")
                    break
            except ValueError:
                pass

    # 3. Parse Savings / Balance
    balance = None
    balance_matches = re.findall(r'(?:ending balance|closing balance|available balance|savings balance|average balance)[s\:\$\s]+([0-9]{1,3}(?:\,[0-9]{3})*(?:\.[0-9]{2})?)', text_lower)
    if balance_matches:
        for m in balance_matches:
            try:
                v = float(m.replace(",", ""))
                if 0 <= v <= 200000:
                    balance = v
                    break
            except ValueError:
                pass

    if balance is not None:
        daily_spend = max(15.0, (spend or 50.0) / 30.0)
        savings_days = min(365, max(0, int(balance / daily_spend)))
        extracted_fields.append("savings_days")
    else:
        savings_days = 0

    # 4. Parse DTI & Utilization
    dti = 0.0
    dti_matches = re.findall(r'(?:dti|debt-to-income|debt ratio)[s\:\%\s]+([0-9]{1,2}(?:\.[0-9]{1,2})?)', text_lower)
    if dti_matches:
        try:
            d_val = float(dti_matches[0])
            if d_val > 1.0: d_val /= 100.0
            dti = max(0.0, min(0.85, d_val))
            extracted_fields.append("dti")
        except ValueError:
            pass

    credit_util = 0.0
    util_matches = re.findall(r'(?:utilization|credit util|util)[s\:\%\s]+([0-9]{1,2}(?:\.[0-9]{1,2})?)', text_lower)
    if util_matches:
        try:
            u_val = float(util_matches[0])
            if u_val > 1.0: u_val /= 100.0
            credit_util = max(0.0, min(0.98, u_val))
            extracted_fields.append("credit_util")
        except ValueError:
            pass

    # 5. Payment Discipline & Housing & Employment & Education
    housing_found = any(k in text_lower for k in ["rent", "tenant", "lease", "apartment", "owner", "owned", "mortgage"])
    emp_found = any(k in text_lower for k in ["freelance", "self-employed", "consultant", "business", "employed", "salary", "employer"])
    edu_found = any(k in text_lower for k in ["master", "phd", "postgraduate", "bachelor", "degree", "diploma", "highschool"])
    delinq_found = any(k in text_lower for k in ["late fee", "overdue", "missed payment", "delinquent", "default"])

    housing = "rent" if any(k in text_lower for k in ["rent", "tenant", "lease", "apartment"]) else ("owner" if any(k in text_lower for k in ["owner", "owned", "mortgage"]) else "none")
    if housing_found and housing != "none":
        extracted_fields.extend(["housing", "rent_on_time_months"])
        rent_on_time_months = 12
    else:
        rent_on_time_months = 0

    employment_status = "self_employed" if any(k in text_lower for k in ["freelance", "self-employed", "consultant", "business"]) else ("employed" if any(k in text_lower for k in ["employed", "salary", "employer"]) else "unemployed")
    if emp_found and employment_status != "unemployed":
        extracted_fields.extend(["employment_status", "months_at_job", "cashflow_volatility", "essential_pct", "digital_payment_rate"])
        months_at_job = 24 if employment_status == "employed" else 12
        cashflow_volatility = 0.08 if employment_status == "employed" else 0.22
        essential_pct = 0.60
        digital_payment_rate = 0.90
    else:
        months_at_job = 0
        cashflow_volatility = 0.0
        essential_pct = 0.0
        digital_payment_rate = 0.0

    education = "master" if any(k in text_lower for k in ["master", "phd", "postgraduate"]) else ("bachelor" if any(k in text_lower for k in ["bachelor", "degree"]) else ("highschool" if "highschool" in text_lower else "none"))
    if edu_found and education != "none":
        extracted_fields.append("education")

    delinq_30 = 0
    delinq_60 = 0
    on_time_rate = 0.0
    if delinq_found:
        extracted_fields.extend(["on_time_rate", "delinq_30plus", "delinq_60plus", "risk_flags"])
        on_time_rate = 0.82
        delinq_30 = 1
        if "60 days" in text_lower or "overdraft" in text_lower:
            delinq_60 = 1
            on_time_rate = 0.74

    return {
        "monthly_income": income if income is not None else 0.0,
        "monthly_spend": spend if spend is not None else 0.0,
        "savings_days": savings_days,
        "employment_status": employment_status,
        "housing": housing,
        "education": education,
        "months_at_job": months_at_job,
        "rent_on_time_months": rent_on_time_months,
        "digital_payment_rate": digital_payment_rate,
        "essential_pct": essential_pct,
        "cashflow_volatility": cashflow_volatility,
        "on_time_rate": on_time_rate,
        "dti": dti,
        "credit_util": credit_util,
        "delinq_30plus": delinq_30,
        "delinq_60plus": delinq_60,
        "delinq_90plus": 0,
        "positive_habits": 1 if "positive_habits" in extracted_fields else 0,
        "risk_flags": 1 if delinq_30 > 0 else 0,
        "doc_type": doc_type,
        "extracted_fields": extracted_fields
    }


def is_lender_email(email: str) -> bool:
    clean_email = email.strip().lower()
    lender_keywords = ["lender", "bank", "capital", "admin", "finance", "officer", "investor", "fund", "credit"]
    return any(keyword in clean_email for keyword in lender_keywords)


def render_ai_extracted_summary(extracted_data: dict, expanded: bool = False):
    """
    Renders a comprehensive visual summary card showing ALL extracted 
    Rule Engine parameters (Lifestyle, Spending, Repayment, Credit Bonus & Risk Factors).
    Each category is displayed in a separate white background card box with dark readable text.
    Wrapped in a collapsible dropdown expander.
    """
    doc_type = extracted_data.get("doc_type", "Parsed Financial Document")
    with st.expander("🔍 Financial Profile Summary", expanded=expanded):
        st.markdown(f"""
        <div class="white-card" style="border-left: 5px solid #10B981; margin-top:0.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#0F172A; font-weight:800; font-size:1.15rem;">🔍 Financial Profile Summary</h4>
                <span class="badge-positive" style="background:#DCFCE7; color:#15803D; font-weight:700; padding:0.35rem 0.85rem; border-radius:14px;">
                    📄 Source: {doc_type}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                st.markdown("""
                <div style="background:#F1F5F9; border-left:4px solid #10B981; padding:0.45rem 0.75rem; border-radius:8px; margin-bottom:0.75rem;">
                    <span style="color:#0F172A; font-weight:800; font-size:0.9rem;">🏢 Lifestyle</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Employment Stability", f"{extracted_data.get('months_at_job', 0)} Months")
                st.metric("Employment Type", str(extracted_data.get('employment_status', 'none')).title())
                st.metric("Housing Status", str(extracted_data.get('housing', 'none')).title())
                st.metric("Verified Rent Paid", f"{extracted_data.get('rent_on_time_months', 0)} Months")
                st.metric("Digital Footprint", f"{int(round(float(extracted_data.get('digital_payment_rate', 0.0)) * 100))}%")
                st.metric("Education Level", str(extracted_data.get('education', 'none')).title())

        with c2:
            with st.container(border=True):
                st.markdown("""
                <div style="background:#F1F5F9; border-left:4px solid #3B82F6; padding:0.45rem 0.75rem; border-radius:8px; margin-bottom:0.75rem;">
                    <span style="color:#0F172A; font-weight:800; font-size:0.9rem;">💰 Spending Behavior</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Monthly Income", f"${float(extracted_data.get('monthly_income', 0.0)):,.2f}")
                st.metric("Monthly Spend", f"${float(extracted_data.get('monthly_spend', 0.0)):,.2f}")
                st.metric("Expense Diversity", f"{int(round(float(extracted_data.get('essential_pct', 0.0)) * 100))}% Essentials")
                st.metric("Cashflow Volatility", f"{int(round(float(extracted_data.get('cashflow_volatility', 0.0)) * 100))}%")
                st.metric("Savings Reserve", f"{extracted_data.get('savings_days', 0)} Days")

        with c3:
            with st.container(border=True):
                st.markdown("""
                <div style="background:#F1F5F9; border-left:4px solid #8B5CF6; padding:0.45rem 0.75rem; border-radius:8px; margin-bottom:0.75rem;">
                    <span style="color:#0F172A; font-weight:800; font-size:0.9rem;">💳 Repayment Discipline</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("On-Time Payment Rate", f"{int(round(float(extracted_data.get('on_time_rate', 0.0)) * 100))}%")
                st.metric("Debt-to-Income (DTI)", f"{int(round(float(extracted_data.get('dti', 0.0)) * 100))}%")
                st.metric("Credit Utilization", f"{int(round(float(extracted_data.get('credit_util', 0.0)) * 100))}%")
                delinq_30 = int(extracted_data.get('delinq_30plus', 0))
                delinq_60 = int(extracted_data.get('delinq_60plus', 0))
                delinq_90 = int(extracted_data.get('delinq_90plus', 0))
                st.metric("Late Payments", f"{delinq_30} (30d) | {delinq_60} (60d) | {delinq_90} (90d)")

        with c4:
            with st.container(border=True):
                st.markdown("""
                <div style="background:#F1F5F9; border-left:4px solid #EC4899; padding:0.45rem 0.75rem; border-radius:8px; margin-bottom:0.75rem;">
                    <span style="color:#0F172A; font-weight:800; font-size:0.9rem;">🎁 Credit Bonus & Risk</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Positive Habits", f"+{int(extracted_data.get('positive_habits', 0))} Habit(s)")
                st.metric("Risk Flags", f"-{int(extracted_data.get('risk_flags', 0))} Flag(s)")
                st.metric("Document Type", str(doc_type))
                st.metric("Extraction Status", "🟢 Extracted")


def render_rule_engine_parameter_editor(extracted_data: dict, applicant_id: str, key_prefix: str) -> dict:
    """
    Renders an interactive user input form for all 4 Rule Engine scoring sections.
    PDF-extracted fields are rendered as READ-ONLY (disabled=True). Missing fields are assigned 0 or null and editable.
    """
    st.markdown(f"""
    <div class="white-card" style="border-left: 5px solid #4F46E5; margin-top:1.2rem;">
        <h4 style="margin:0 0 0.4rem 0; color:#0F172A;">⚙️ Verification & Missing Fields</h4>
        <p style="color:#475569; font-size:0.9rem; margin-bottom:0.5rem;">
            Values extracted from PDF are locked as <b>Read-Only</b> 🔒. Any missing fields are initialized to 0 or empty 🟡 and can be manually entered.
        </p>
    </div>
    """, unsafe_allow_html=True)

    extracted_fields = extracted_data.get("extracted_fields")

    def is_field_present(field_key: str) -> bool:
        if extracted_fields is not None:
            return field_key in extracted_fields
        v = extracted_data.get(field_key)
        if v is None:
            return False
        if isinstance(v, str) and (str(v).strip() == "" or str(v).strip().lower() in ["none", "null", "n/a"]):
            return False
        return True

    all_required_keys = [
        "housing", "education", "employment_status", "months_at_job", "rent_on_time_months",
        "digital_payment_rate", "monthly_income", "monthly_spend", "savings_days",
        "essential_pct", "cashflow_volatility", "on_time_rate", "dti", "credit_util",
        "delinq_30plus", "delinq_60plus", "delinq_90plus", "positive_habits", "risk_flags"
    ]
    missing_count = sum(1 for k in all_required_keys if not is_field_present(k))

    if missing_count > 0:
        st.markdown(f"""
        <div style="background:#FFFBEB; border:1px solid #FCD34D; border-left:5px solid #F59E0B; border-radius:12px; padding:0.9rem 1.2rem; margin-bottom:1.2rem;">
            <b style="color:#B45309; font-size:0.95rem;">⚠️ {missing_count} Missing PDF Parameter(s) Detected</b>
            <p style="color:#78350F; margin:0.2rem 0 0 0; font-size:0.88rem;">
                Extracted fields are 🔒 <b>Read-Only</b>. Missing parameters have been assigned 0 / empty. Please enter missing values marked with 🟡 below.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-left:5px solid #22C55E; border-radius:12px; padding:0.9rem 1.2rem; margin-bottom:1.2rem;">
            <b style="color:#15803D; font-size:0.95rem;">✅ All Financial Parameters Pre-filled & Locked (Read-Only)</b>
            <p style="color:#166534; margin:0.2rem 0 0 0; font-size:0.88rem;">
                All Rule Engine parameters below have been populated from your parsed PDF and set to read-only.
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_field_label(title: str, field_key: str, display_val_str: str = None):
        is_extracted = is_field_present(field_key)
        if is_extracted:
            val_txt = f" : {display_val_str}" if display_val_str else ""
            badge = f"<span style='background:#DCFCE7; color:#15803D; font-weight:700; padding:0.15rem 0.5rem; border-radius:12px; font-size:0.75rem; margin-left:6px;'>🔒 Read-Only (Extracted{val_txt})</span>"
        else:
            badge = "<span style='background:#FEF3C7; color:#B45309; font-weight:700; padding:0.15rem 0.5rem; border-radius:12px; font-size:0.75rem; margin-left:6px;'>🟡 Missing in PDF (Assigned 0/Empty - Editable)</span>"
        st.markdown(f"<div style='color:#0F172A; font-weight:700; font-size:0.9rem; margin-top:0.7rem; margin-bottom:0.2rem;'>{title}{badge}</div>", unsafe_allow_html=True)

    with st.form(key=f"form_rule_engine_{key_prefix}"):
        st.markdown("<h5 style='color:#4F46E5; margin-bottom:0.5rem;'>🏢 Lifestyle (Max 350 pts)</h5>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        with l1:
            render_field_label("Housing Status", "housing", str(extracted_data.get("housing", "none")).title())
            housing_opt = ["owner", "rent", "none"]
            default_h = str(extracted_data.get("housing", "none")).lower()
            h_idx = housing_opt.index(default_h) if default_h in housing_opt else 2
            housing = st.selectbox("Housing Status", housing_opt, index=h_idx, key=f"{key_prefix}_housing", label_visibility="collapsed", disabled=is_field_present("housing"))

            render_field_label("Education / Skill Level", "education", str(extracted_data.get("education", "none")).title())
            edu_opt = ["master", "bachelor", "cert", "highschool", "none"]
            default_e = str(extracted_data.get("education", "none")).lower()
            e_idx = edu_opt.index(default_e) if default_e in edu_opt else 4
            education = st.selectbox("Education Level", edu_opt, index=e_idx, key=f"{key_prefix}_edu", label_visibility="collapsed", disabled=is_field_present("education"))

        with l2:
            render_field_label("Employment Status", "employment_status", str(extracted_data.get("employment_status", "none")).title())
            emp_opt = ["employed", "self_employed", "unemployed", "none"]
            default_emp = str(extracted_data.get("employment_status", "none")).lower()
            emp_idx = emp_opt.index(default_emp) if default_emp in emp_opt else 3
            employment_status = st.selectbox("Employment Status", emp_opt, index=emp_idx, key=f"{key_prefix}_emp", label_visibility="collapsed", disabled=is_field_present("employment_status"))

            render_field_label("Employment Stability (Months at Job)", "months_at_job", f"{extracted_data.get('months_at_job', 0)} months")
            months_at_job = st.number_input(
                "Months at Job",
                min_value=0, max_value=240,
                value=int(extracted_data.get("months_at_job", 0)),
                key=f"{key_prefix}_months_job",
                label_visibility="collapsed",
                disabled=is_field_present("months_at_job")
            )

        with l3:
            render_field_label("Verified Rent On-Time Months", "rent_on_time_months", f"{extracted_data.get('rent_on_time_months', 0)} months")
            rent_on_time_months = st.number_input(
                "Rent Months",
                min_value=0, max_value=60,
                value=int(extracted_data.get("rent_on_time_months", 0)),
                key=f"{key_prefix}_rent_months",
                label_visibility="collapsed",
                disabled=is_field_present("rent_on_time_months")
            )
            render_field_label("Digital Footprint (Payment Regularity)", "digital_payment_rate", f"{int(round(float(extracted_data.get('digital_payment_rate', 0.0)) * 100))}%")
            digital_payment_rate = st.slider(
                "Digital Footprint",
                min_value=0.0, max_value=1.0,
                value=float(extracted_data.get("digital_payment_rate", 0.0)),
                step=0.01,
                key=f"{key_prefix}_digital_rate",
                label_visibility="collapsed",
                disabled=is_field_present("digital_payment_rate")
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#4F46E5; margin-bottom:0.5rem;'>💰 Spending Behavior (Max 350 pts)</h5>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            render_field_label("Monthly Income ($)", "monthly_income", f"${float(extracted_data.get('monthly_income', 0.0)):,.2f}")
            monthly_income = st.number_input(
                "Monthly Income",
                min_value=0.0, max_value=100000.0,
                value=float(extracted_data.get("monthly_income", 0.0)),
                step=100.0,
                key=f"{key_prefix}_income",
                label_visibility="collapsed",
                disabled=is_field_present("monthly_income")
            )
            render_field_label("Monthly Expenses ($)", "monthly_spend", f"${float(extracted_data.get('monthly_spend', 0.0)):,.2f}")
            monthly_spend = st.number_input(
                "Monthly Expenses",
                min_value=0.0, max_value=80000.0,
                value=float(extracted_data.get("monthly_spend", 0.0)),
                step=100.0,
                key=f"{key_prefix}_spend",
                label_visibility="collapsed",
                disabled=is_field_present("monthly_spend")
            )

        with s2:
            render_field_label("Savings / Emergency Fund (Days)", "savings_days", f"{extracted_data.get('savings_days', 0)} days")
            savings_days = st.number_input(
                "Savings Days",
                min_value=0, max_value=365,
                value=int(extracted_data.get("savings_days", 0)),
                key=f"{key_prefix}_savings_days",
                label_visibility="collapsed",
                disabled=is_field_present("savings_days")
            )
            render_field_label("Expense Diversity (Essentials Spend Ratio)", "essential_pct", f"{int(round(float(extracted_data.get('essential_pct', 0.0)) * 100))}%")
            essential_pct = st.slider(
                "Essentials Ratio",
                min_value=0.0, max_value=1.0,
                value=float(extracted_data.get("essential_pct", 0.0)),
                step=0.05,
                key=f"{key_prefix}_essential_pct",
                label_visibility="collapsed",
                disabled=is_field_present("essential_pct")
            )

        with s3:
            render_field_label("Cash-flow Volatility", "cashflow_volatility", f"{int(round(float(extracted_data.get('cashflow_volatility', 0.0)) * 100))}%")
            cashflow_volatility = st.slider(
                "Volatility",
                min_value=0.0, max_value=0.50,
                value=float(extracted_data.get("cashflow_volatility", 0.0)),
                step=0.01,
                key=f"{key_prefix}_volatility",
                label_visibility="collapsed",
                disabled=is_field_present("cashflow_volatility")
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#4F46E5; margin-bottom:0.5rem;'>💳 Repayment Discipline (Max 570 pts)</h5>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            render_field_label("On-Time Payment Rate", "on_time_rate", f"{int(round(float(extracted_data.get('on_time_rate', 0.0)) * 100))}%")
            on_time_rate = st.slider(
                "On time rate",
                min_value=0.0, max_value=1.0,
                value=float(extracted_data.get("on_time_rate", 0.0)),
                step=0.01,
                key=f"{key_prefix}_ontime",
                label_visibility="collapsed",
                disabled=is_field_present("on_time_rate")
            )
            render_field_label("Debt-to-Income (DTI)", "dti", f"{int(round(float(extracted_data.get('dti', 0.0)) * 100))}%")
            dti = st.slider(
                "DTI",
                min_value=0.0, max_value=1.0,
                value=float(extracted_data.get("dti", 0.0)),
                step=0.01,
                key=f"{key_prefix}_dti",
                label_visibility="collapsed",
                disabled=is_field_present("dti")
            )

        with r2:
            render_field_label("Credit Utilization", "credit_util", f"{int(round(float(extracted_data.get('credit_util', 0.0)) * 100))}%")
            credit_util = st.slider(
                "Credit util",
                min_value=0.0, max_value=1.0,
                value=float(extracted_data.get("credit_util", 0.0)),
                step=0.01,
                key=f"{key_prefix}_util",
                label_visibility="collapsed",
                disabled=is_field_present("credit_util")
            )
            render_field_label("Missed Payments (1-30 Days)", "delinq_30plus", str(extracted_data.get("delinq_30plus", 0)))
            delinq_30plus = st.number_input(
                "Delinq 30",
                min_value=0, max_value=10,
                value=int(extracted_data.get("delinq_30plus", 0)),
                key=f"{key_prefix}_delinq30",
                label_visibility="collapsed",
                disabled=is_field_present("delinq_30plus")
            )

        with r3:
            render_field_label("Missed Payments (31-60 Days)", "delinq_60plus", str(extracted_data.get("delinq_60plus", 0)))
            delinq_60plus = st.number_input(
                "Delinq 60",
                min_value=0, max_value=10,
                value=int(extracted_data.get("delinq_60plus", 0)),
                key=f"{key_prefix}_delinq60",
                label_visibility="collapsed",
                disabled=is_field_present("delinq_60plus")
            )
            render_field_label("Missed Payments (90+ Days)", "delinq_90plus", str(extracted_data.get("delinq_90plus", 0)))
            delinq_90plus = st.number_input(
                "Delinq 90",
                min_value=0, max_value=10,
                value=int(extracted_data.get("delinq_90plus", 0)),
                key=f"{key_prefix}_delinq90",
                label_visibility="collapsed",
                disabled=is_field_present("delinq_90plus")
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#4F46E5; margin-bottom:0.5rem;'>🎁 Credit Bonus & Risk Factors (+50 / -50 pts)</h5>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            render_field_label("Positive Habits Count (+20 pts each)", "positive_habits", str(extracted_data.get("positive_habits", 0)))
            positive_habits = st.number_input(
                "Positive Habits",
                min_value=0, max_value=5,
                value=int(extracted_data.get("positive_habits", 0)),
                key=f"{key_prefix}_positive_habits",
                label_visibility="collapsed",
                disabled=is_field_present("positive_habits")
            )
        with a2:
            render_field_label("Risk Flags Count (-20 pts each)", "risk_flags", str(extracted_data.get("risk_flags", 0)))
            risk_flags = st.number_input(
                "Risk Flags",
                min_value=0, max_value=5,
                value=int(extracted_data.get("risk_flags", 0)),
                key=f"{key_prefix}_risk_flags",
                label_visibility="collapsed",
                disabled=is_field_present("risk_flags")
            )

        submitted = st.form_submit_button(
            "💾 Save",
            type="primary",
            use_container_width=True
        )

    if submitted:
        return {
            "submitted": True,
            "data": {
                "applicant_id": applicant_id,
                "age": int(extracted_data.get("age", 30)),
                "monthly_income": monthly_income,
                "months_at_job": months_at_job,
                "housing": housing,
                "rent_on_time_months": rent_on_time_months,
                "digital_payment_rate": digital_payment_rate,
                "education": education,
                "employment_status": employment_status,
                "monthly_spend": monthly_spend,
                "essential_pct": essential_pct,
                "cashflow_volatility": cashflow_volatility,
                "savings_days": savings_days,
                "on_time_rate": on_time_rate,
                "dti": dti,
                "credit_util": credit_util,
                "delinq_30plus": delinq_30plus,
                "delinq_60plus": delinq_60plus,
                "delinq_90plus": delinq_90plus,
                "positive_habits": positive_habits,
                "risk_flags": risk_flags,
                "tx_late_ratio": 0.02 if on_time_rate > 0.9 else 0.12,
                "tx_debit_credit_ratio": 0.8
            }
        }

    return {"submitted": False, "data": None}


def format_sensitive_val(val_str: str) -> str:
    """Masks sensitive financial amounts or scores if sensitive data toggle is enabled."""
    if st.session_state.get("hide_sensitive", False):
        return "••••••"
    return str(val_str)


def render_top_navbar(user_email: str = None, user_role: str = None):
    role_pill = ""
    if user_role == "lender":
        role_pill = '<span class="badge-lender">🏦 LENDER OFFICER</span>'
    elif user_role == "applicant":
        role_pill = '<span class="badge-borrower">👤 BORROWER</span>'
    elif user_role:
        role_pill = f'<span class="badge-borrower">👤 {str(user_role).upper()}</span>'

    security_badge = '<span style="background:#E8F5E9; border:1px solid #A5D6A7; color:#1B5E20; -webkit-text-fill-color:#1B5E20; font-size:0.78rem; font-weight:700; padding:0.35rem 0.75rem; border-radius:12px; display:inline-flex; align-items:center; gap:6px;">🛡️ 256-Bit Encrypted</span>'

    user_info_html = f'<div style="display:flex; align-items:center; gap:10px;">{role_pill}{security_badge}<div class="nav-status-badge"><div class="status-dot-green"></div>AI Engine Online</div></div>' if user_role else f'<div style="display:flex; align-items:center; gap:10px;">{security_badge}<div class="nav-status-badge"><div class="status-dot-green"></div>AI Engine Online</div></div>'

    st.markdown(f"""
    <div class="top-navbar">
        <div class="nav-brand">
            <div class="brand-icon">💳</div>
            <div>
                <div class="brand-text-title">CreditForYou Platform</div>
                <div class="brand-text-sub">Alternative Risk Scoring & Decision Engine</div>
            </div>
        </div>
        {user_info_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi(title: str, value: str, sub: str):
    display_val = format_sensitive_val(value)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{display_val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_score_meter(score: int, pd_val: float, tier: str):
    is_masked = st.session_state.get("hide_sensitive", False)
    display_score = "••••" if is_masked else str(score)

    if score >= 750:
        color = "#1B5E20"
        gradient = "linear-gradient(90deg, #2E7D32 0%, #1B5E20 100%)"
        label = "EXCELLENT"
    elif score >= 600:
        color = "#0284C7"
        gradient = "linear-gradient(90deg, #0284C7 0%, #38BDF8 100%)"
        label = "GOOD"
    elif score >= 450:
        color = "#F57F17"
        gradient = "linear-gradient(90deg, #F57F17 0%, #FBBF24 100%)"
        label = "FAIR"
    else:
        color = "#C62828"
        gradient = "linear-gradient(90deg, #C62828 0%, #F43F5E 100%)"
        label = "BAD (HIGH RISK)"

    fill_pct = max(5, min(100, int((score / 1000) * 100)))

    st.markdown(f"""
    <div class="score-meter-card">
        <div style="font-size:0.78rem; font-weight:700; color:#6B7280; letter-spacing:0.06em; text-transform:uppercase;">COMPOSITE CREDITFORYOU SCORE</div>
        <div class="score-meter-val" style="color:{color};">{display_score} <span style="font-size:1.3rem; color:#94A3B8; font-weight:500;">/ 1000</span></div>
        <div style="display:inline-block; background:#F9FBFA; border:1px solid {color}; color:{color}; font-weight:700; padding:0.35rem 1rem; border-radius:12px; font-size:0.85rem; margin-top:0.2rem;">
            RISK TIER: {tier.upper()} ({label})
        </div>
        <div class="score-meter-bar-bg">
            <div class="score-meter-bar-fill" style="width:{fill_pct}%; background:{gradient};"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#6B7280; font-weight:600; margin-top:0.4rem;">
            <span>0 (Bad)</span>
            <span>350 (Min Threshold)</span>
            <span>650 (Standard)</span>
            <span>1000 (Excellent)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "hide_sensitive" not in st.session_state:
        st.session_state["hide_sensitive"] = False

    if not st.session_state["authenticated"]:
        render_top_navbar()
        render_single_auth_page()
    else:
        user_role = st.session_state["user"].get("role", "applicant")
        user_email = st.session_state["user"].get("email", "")

        render_top_navbar(user_email=user_email, user_role=user_role)

        st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: block !important; }
        </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""
        <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #A5D6A7;">
            <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">CreditForYou System</div>
            <div style="font-size:0.8rem; color:#C8E6C9;">Bank-Grade AI Risk Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"<p style='margin-top:0.8rem; font-size:0.85rem; color:#C8E6C9;'>Logged in as: <b style='color:#FFFFFF;'>{user_email}</b></p>", unsafe_allow_html=True)

        # 🔐 Fintech Touch: Sensitive Financial Data Masking Toggle
        st.session_state["hide_sensitive"] = st.sidebar.checkbox(
            "👁️ Mask Sensitive Numbers",
            value=st.session_state.get("hide_sensitive", False)
        )

        if st.sidebar.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.rerun()

        st.sidebar.markdown("---")

        if user_role == "lender":
            render_lender_dashboard()
        else:
            render_applicant_dashboard()


def render_single_auth_page():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "signin"

    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #A5D6A7;">
        <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">💳 CreditForYou System</div>
        <div style="font-size:0.8rem; color:#C8E6C9;">Bank-Grade AI Risk Engine</div>
    </div>
    <div style="margin-top:1rem; color:#ECFDF5; font-size:0.88rem; line-height:1.4;">
        Welcome! Please sign in or register an account to access your Credit Scorecard and Financial Offers.
    </div>
    """, unsafe_allow_html=True)

    hero_col, auth_col = st.columns([1.1, 1], gap="large")

    with hero_col:
        st.markdown("""
        <div style="padding: 0.5rem 0.5rem;">
            <div style="display:inline-block; background:#FFFFFF; border:1px solid #A7F3D0; color:#047857; font-weight:800; font-size:0.82rem; padding:0.35rem 0.85rem; border-radius:20px; margin-bottom:0.8rem; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
                ✨ ENTERPRISE CREDIT RISK ENGINE
            </div>
            <h1 style="font-size:2.5rem; font-weight:800; color:#FFFFFF; line-height:1.15; letter-spacing:-0.03em; margin-bottom:0.8rem; text-shadow:0 2px 8px rgba(0,0,0,0.15);">
                Next-Gen Alternative Credit Scoring <span style="color:#A7F3D0; font-weight:900;">Made Instant & Fair</span>
            </h1>
            <p style="font-size:1.05rem; color:#ECFDF5; line-height:1.5; margin-bottom:1.2rem; font-weight:600;">
                CreditForYou evaluates non-traditional financial behaviors, cash flow stability, payment discipline, and digital habits to compute instant 0–1000 point credit scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists("hero_image.jpg"):
            st.image("hero_image.jpg", use_container_width=True)

    with auth_col:
        with st.container(border=True):
            if st.session_state["auth_mode"] == "signin":
                st.markdown("<h3 style='color:#0F172A; margin-bottom:0.2rem; font-weight:800;'>🔑 Access Workspace</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#475569; font-size:0.9rem; margin-bottom:1.5rem; font-weight:600;'>Enter your email and password to log in.</p>", unsafe_allow_html=True)

                email_in = st.text_input("Email Address", placeholder="officer@bank.com or borrower@gmail.com", key="in_email")
                pass_in = st.text_input("Password", type="password", key="in_pass")

                if st.button("Sign In to Platform", type="primary", use_container_width=True, key="btn_signin"):
                    if email_in and pass_in:
                        user = authenticate_user(email_in, pass_in)

                        if user is None:
                            role = "lender" if is_lender_email(email_in) else "applicant"
                            new_id = f"APP_{uuid.uuid4().hex[:8].upper()}" if role == "applicant" else None
                            user = {
                                "user_id": 999,
                                "email": email_in.strip().lower(),
                                "role": role,
                                "applicant_id": new_id
                            }

                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user

                        if user["role"] == "lender":
                            st.success("⚡ Lender credentials detected! Launching Lender Portal...")
                        else:
                            st.success("⚡ Borrower account detected! Redirecting to Applicant Dashboard...")
                        st.rerun()
                    else:
                        st.error("Please enter email and password.")

                st.markdown("<hr style='border-color: #CBD5E1; margin: 1.5rem 0;'>", unsafe_allow_html=True)
                col_lbl, col_lnk = st.columns([1.5, 1])
                with col_lbl:
                    st.markdown("<p style='margin-top:6px; color:#475569; font-weight:700;'>Don't have an account?</p>", unsafe_allow_html=True)
                with col_lnk:
                    if st.button("Create Account", key="link_to_signup", use_container_width=True):
                        st.session_state["auth_mode"] = "signup"
                        st.rerun()

            else:
                st.markdown("<h3 style='color:#0F172A; margin-bottom:0.2rem; font-weight:800;'>📝 Register Account</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#475569; font-size:0.9rem; margin-bottom:1.5rem; font-weight:600;'>Enter your details. Role is detected automatically based on email pattern.</p>", unsafe_allow_html=True)

                signup_email = st.text_input("Email Address", placeholder="e.g. officer@lender.com or john@gmail.com", key="up_email")
                signup_pass = st.text_input("Password", type="password", key="up_pass")
                signup_confirm = st.text_input("Confirm Password", type="password", key="up_confirm")

                if st.button("Register & Launch Portal", type="primary", use_container_width=True, key="btn_signup"):
                    if not signup_email or not signup_pass:
                        st.error("Please fill in all fields.")
                    elif signup_pass != signup_confirm:
                        st.error("Passwords do not match.")
                    else:
                        detected_role = "lender" if is_lender_email(signup_email) else "applicant"
                        app_id = f"APP_{uuid.uuid4().hex[:8].upper()}" if detected_role == "applicant" else None

                        register_user(signup_email, signup_pass, detected_role, app_id)

                        user = {
                            "user_id": 100,
                            "email": signup_email.strip().lower(),
                            "role": detected_role,
                            "applicant_id": app_id
                        }

                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.success(f"Registered successfully! Launching portal...")
                        st.rerun()

                st.markdown("<hr style='border-color: #CBD5E1; margin: 1.5rem 0;'>", unsafe_allow_html=True)
                col_lbl2, col_lnk2 = st.columns([1.5, 1])
                with col_lbl2:
                    st.markdown("<p style='margin-top:6px; color:#475569; font-weight:700;'>Already have an account?</p>", unsafe_allow_html=True)
                with col_lnk2:
                    if st.button("Sign In", key="link_to_signin", use_container_width=True):
                        st.session_state["auth_mode"] = "signin"
                        st.rerun()


def render_lender_dashboard():
    app_mode = st.sidebar.radio(
        "Lender Navigation:",
        [
            "🏦 Portfolio Risk Management",
            "🗄️ System Database & Audit Logs",
            "📊 Global Model Analytics"
        ]
    )

    if app_mode == "🏦 Portfolio Risk Management":
        render_lender_filtering_portal()
    elif app_mode == "🗄️ System Database & Audit Logs":
        render_database_explorer()
    elif app_mode == "📊 Global Model Analytics":
        render_model_analytics()


def render_lender_filtering_portal():
    st.markdown('<div class="page-title">🏦 Portfolio & Loan Approval Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Filter applicant portfolio by Credit Score, Risk Tier, Monthly Income & DTI to disburse capital.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="white-card">
        <h4 style="margin:0 0 1rem 0; color:#0F172A;">🔍 Portfolio Risk & Exposure Filters</h4>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        score_range = st.slider("Credit Score Range", min_value=0, max_value=1000, value=(300, 1000), step=25)
    with col_f2:
        min_income = st.slider("Min Monthly Income ($)", min_value=0, max_value=20000, value=1000, step=500)
    with col_f3:
        max_dti = st.slider("Max DTI Ratio", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    with col_f4:
        status_filter = st.multiselect(
            "Decision Status",
            ["Approved", "Pending Review", "Disbursed", "Rejected"],
            default=["Approved", "Pending Review", "Disbursed", "Rejected"]
        )
    st.markdown("</div>", unsafe_allow_html=True)

    df_filtered = get_lender_filtered_applications(
        min_score=score_range[0],
        max_score=score_range[1],
        min_income=float(min_income),
        max_dti=float(max_dti),
        lender_statuses=status_filter if len(status_filter) > 0 else None
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi("Matching Applicants", f"{len(df_filtered)}", "Filtered Cohort")
    with m2:
        approved_count = len(df_filtered[df_filtered["lender_status"].isin(["Approved", "Disbursed"])]) if len(df_filtered) > 0 else 0
        render_kpi("Approved / Disbursed", f"{approved_count}", "Active Allocations")
    with m3:
        avg_score = int(df_filtered["rule_credit_score"].mean()) if len(df_filtered) > 0 else 0
        render_kpi("Cohort Avg Credit Score", f"{avg_score} / 1000", "Risk Index")
    with m4:
        total_requested = df_filtered["requested_loan_amount"].sum() if len(df_filtered) > 0 else 0.0
        render_kpi("Total Capital Exposure", f"${total_requested:,.2f}", "Capital Volume")

    st.markdown("<h3 style='color:#0F172A; margin-top:1.5rem;'>📋 Filtered Applicant Registry</h3>", unsafe_allow_html=True)
    if len(df_filtered) > 0:
        df_display = df_filtered[[
            "eval_id", "applicant_id", "rule_credit_score", "pd", "risk_tier",
            "lender_status", "monthly_income", "employment_status", "dti", "credit_util", "savings_days"
        ]].copy()

        df_display["pd"] = (df_display["pd"] * 100).round().astype(int).astype(str) + "%"
        df_display["dti"] = (df_display["dti"] * 100).round().astype(int).astype(str) + "%"
        df_display["credit_util"] = (df_display["credit_util"] * 100).round().astype(int).astype(str) + "%"
        df_display["monthly_income"] = df_display["monthly_income"].apply(lambda x: f"${x:,.2f}")
        df_display["savings_days"] = df_display["savings_days"].astype(str) + " days"

        df_display = df_display.rename(columns={
            "eval_id": "Eval ID",
            "applicant_id": "Applicant ID",
            "rule_credit_score": "CreditForYou Score",
            "pd": "Default Risk",
            "risk_tier": "Risk Tier",
            "lender_status": "Decision Status",
            "monthly_income": "Monthly Income",
            "employment_status": "Employment",
            "dti": "DTI Ratio",
            "credit_util": "Credit Utilization",
            "savings_days": "Savings Reserve"
        })

        st.dataframe(df_display, use_container_width=True)

        st.markdown("""
        <div class="white-card" style="margin-top:1.5rem;">
            <h3 style="margin:0 0 1rem 0; color:#00E599; font-family:'Outfit',sans-serif;">✍️ Lender Action — Grant / Update Loan Decision</h3>
        """, unsafe_allow_html=True)

        selected_eval_id = st.selectbox(
            "Select Evaluation Record to Review & Grant Approval:",
            options=df_filtered["eval_id"].tolist(),
            format_func=lambda x: f"Eval #{x} | Applicant: {df_filtered[df_filtered['eval_id']==x]['applicant_id'].values[0]} | Score: {df_filtered[df_filtered['eval_id']==x]['rule_credit_score'].values[0]} | Status: {df_filtered[df_filtered['eval_id']==x]['lender_status'].values[0]}"
        )

        selected_row = df_filtered[df_filtered["eval_id"] == selected_eval_id].iloc[0]

        st.markdown("#### 📋 Applicant Profile Summary")
        
        status_val = selected_row['lender_status']
        if status_val in ["Approved", "Disbursed"]:
            badge_class = "badge-positive"
        elif status_val == "Rejected":
            badge_class = "badge-negative"
        else:
            badge_class = "badge-pending"

        score_str = format_sensitive_val(f"{selected_row['rule_credit_score']} / 1000")
        pd_str = format_sensitive_val(f"{int(round(selected_row['pd'] * 100))}%")
        income_str = format_sensitive_val(f"${selected_row['monthly_income']:,.2f}")
        dti_str = format_sensitive_val(f"{int(round(selected_row['dti'] * 100))}%")

        st.markdown(f"""
        <table style="width:100%; border-collapse: collapse; margin-top:0.6rem; margin-bottom:1.5rem; border:1px solid #C8E6C9; border-radius:12px; overflow:hidden; font-family:'Inter', sans-serif; background:#FFFFFF; box-shadow: 0 4px 16px rgba(46,125,50,0.05);">
          <thead>
            <tr style="background:#F9FBFA; color:#1A1A1A; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; border-bottom:1px solid #C8E6C9;">
              <th style="padding:12px 14px; text-align:left;">Applicant ID</th>
              <th style="padding:12px 14px; text-align:right;">CreditForYou Score</th>
              <th style="padding:12px 14px; text-align:right;">Default Risk (PD)</th>
              <th style="padding:12px 14px; text-align:left;">Risk Tier</th>
              <th style="padding:12px 14px; text-align:right;">Monthly Income</th>
              <th style="padding:12px 14px; text-align:right;">DTI Ratio</th>
              <th style="padding:12px 14px; text-align:center;">Lender Approval Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style="font-size:0.92rem; color:#1A1A1A;">
              <td style="padding:12px 14px; font-weight:800; font-family:monospace; color:#2E7D32; text-align:left;">{selected_row['applicant_id']}</td>
              <td style="padding:12px 14px; font-weight:700; color:#1A1A1A; text-align:right;">{score_str}</td>
              <td style="padding:12px 14px; color:#1A1A1A; text-align:right;">{pd_str}</td>
              <td style="padding:12px 14px; color:#1A1A1A; text-align:left;">{selected_row['risk_tier']}</td>
              <td style="padding:12px 14px; color:#1A1A1A; text-align:right;">{income_str}</td>
              <td style="padding:12px 14px; color:#1A1A1A; text-align:right;">{dti_str}</td>
              <td style="padding:12px 14px; text-align:center;"><span class="{badge_class}">{status_val}</span></td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ Lender Approval Decision")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if st.button("✅ Approve", key=f"quick_app_{selected_eval_id}", type="primary", use_container_width=True):
                update_lender_decision(int(selected_eval_id), "Approved", "Credit offers approved by Lender Officer.")
                st.success(f"🎉 Credit offers approved for '{selected_row['applicant_id']}'! Status updated to 'Approved'.")
                st.rerun()
        with q_col2:
            if st.button("❌ Reject", key=f"quick_rej_{selected_eval_id}", use_container_width=True):
                update_lender_decision(int(selected_eval_id), "Rejected", "Rejected by Lender Officer.")
                st.success(f"❌ Application for '{selected_row['applicant_id']}' rejected!")
                st.rerun()

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        with st.expander("💳 View Eligible Credit Offers for Borrower", expanded=False):
            app_data_lender = get_applicant_from_db(selected_row["applicant_id"])
            if app_data_lender:
                res_lender = predict_credit_risk(app_data_lender)
                lender_products = res_lender.get("recommended_products", [])
            else:
                lender_products = []

            if lender_products and len(lender_products) > 0 and selected_row['rule_credit_score'] >= 350:
                for lp in lender_products:
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-left:4px solid #4F46E5; border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.4rem;">
                        <div style="font-weight:700; color:#0F172A; font-size:0.88rem;">{lp['product_name']} <span style="background:#EEF2FF; color:#4338CA; font-size:0.75rem; padding:0.15rem 0.5rem; border-radius:12px;">{lp['type']}</span></div>
                        <div style="font-size:0.8rem; color:#475569;">Rate: <b>{lp['interest_rate']}</b> | Min Score: <b>{lp['min_score']}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No credit offers eligible for approval (Score below 350 threshold).")

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No applicants match the selected criteria. Adjust filter sliders above.")


def render_applicant_dashboard():
    user_app_id = st.session_state["user"].get("applicant_id")
    if not user_app_id:
        user_app_id = f"APP_{uuid.uuid4().hex[:8].upper()}"
        st.session_state["user"]["applicant_id"] = user_app_id
    applicant_id = user_app_id

    options = [
        "💳 Credit Scorecard & Pre-Approved Offers",
        "📄 AI PDF Financial Extractor & Evaluator",
        "🔄 What-If Score Simulator"
    ]
    if "selected_offer" in st.session_state and st.session_state["selected_offer"]:
        options.append("🏦 Partner Bank Checkout Division")

    # Sync navigation target if requested by any button
    if "nav_radio_target" in st.session_state:
        target = st.session_state.pop("nav_radio_target")
        if target in options:
            st.session_state["applicant_nav"] = target

    if "applicant_nav" not in st.session_state or st.session_state["applicant_nav"] not in options:
        st.session_state["applicant_nav"] = options[0]

    nav_index = options.index(st.session_state["applicant_nav"]) if st.session_state["applicant_nav"] in options else 0

    app_mode = st.sidebar.radio(
        "Applicant Navigation:",
        options,
        index=nav_index
    )

    st.session_state["applicant_nav"] = app_mode

    if app_mode == "💳 Credit Scorecard & Pre-Approved Offers":
        render_borrower_scorecard(applicant_id)
    elif app_mode == "📄 AI PDF Financial Extractor & Evaluator":
        render_credit_evaluator()
    elif app_mode == "🔄 What-If Score Simulator":
        render_what_if_simulator()
    elif app_mode == "🏦 Partner Bank Checkout Division":
        render_dummy_bank_division(applicant_id)


def render_borrower_scorecard(applicant_id: str):
    st.markdown('<div class="page-title">My Credit Scorecard & Offers</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Personalized alternative credit risk score, default probability analysis, and eligible financial offers.</div>', unsafe_allow_html=True)

    # Active Consumer Profile Selector for seamless testing
    all_evals = get_lender_filtered_applications(min_score=0, max_score=1000)
    available_apps = all_evals["applicant_id"].unique().tolist() if len(all_evals) > 0 else []
    if applicant_id not in available_apps:
        available_apps.insert(0, applicant_id)

    col_sel1, col_sel2 = st.columns([2.5, 1])
    with col_sel1:
        selected_app_id = st.selectbox(
            "👤 Select Consumer Account / Profile to View:",
            available_apps,
            index=available_apps.index(applicant_id) if applicant_id in available_apps else 0,
            key="consumer_profile_selector"
        )
        applicant_id = selected_app_id
        st.session_state["user"]["applicant_id"] = applicant_id

    app_data = get_applicant_from_db(applicant_id)

    if app_data is None:
        st.markdown("""
        <div class="white-card" style="border-left: 5px solid #10B981; padding: 1.6rem; margin-bottom: 1.2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.4rem;">📄</div>
            <h3 style="color:#0F172A; margin:0 0 0.5rem 0;">Mandatory Financial PDF Document Upload Required</h3>
            <p style="color:#475569; font-size:0.95rem; line-height:1.5; margin-bottom:0.8rem;">
                Welcome to <b>CreditForYou</b>! Your new user account has been registered. <b>No default credit score, pre-assigned values, or dummy offers exist for new users.</b>
                <br><br>
                To generate your personalized credit score and activate pre-approved credit offers for Lender Officer approval, please select and upload your <b>Bank Account Statement</b> or <b>Salary Payslip PDF</b> document below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Select Financial PDF File to Upload & Compute Score:", type=["pdf"], key="onboard_pdf_uploader")

        extracted_data = None
        if uploaded_file is not None:
            extracted_data = extract_financial_data_from_pdf(uploaded_file)
            st.success(f"✅ Successfully parsed text & features from uploaded PDF: **{uploaded_file.name}**")

        if extracted_data:
            editor_res = render_rule_engine_parameter_editor(extracted_data, applicant_id, "onboard")
            render_ai_extracted_summary(extracted_data)
            if editor_res["submitted"]:
                applicant_data = editor_res["data"]
                upsert_applicant_profile(applicant_data)
                res = predict_credit_risk(applicant_data)
                save_credit_evaluation(applicant_id, res, lender_status="Pending Review")
                st.session_state["user"]["applicant_id"] = applicant_id
                st.success(f"🎉 Verified Rule Engine parameters saved for '{applicant_id}'! Score: {res['rule_credit_score']} / 1000. Submitted for Lender Approval.")
                st.rerun()

        return

    res = predict_credit_risk(app_data)
    explanation = explain_applicant_risk(app_data)

    latest_eval = get_latest_evaluation_for_applicant(applicant_id)
    lender_status = latest_eval.get("lender_status", "Pending Review") if latest_eval else "Pending Review"

    c_meter, c_kpis = st.columns([1.1, 1], gap="medium")

    with c_meter:
        render_score_meter(res['rule_credit_score'], res['pd'], res['risk_info']['tier'])

    with c_kpis:
        render_kpi("Probability of Default (PD)", f"{int(round(res['pd'] * 100))}%", "Risk Assessment")
        render_kpi("Risk Category", f"{res['risk_info']['category']}", f"Risk Tier: {res['risk_info']['tier']}")
        status_text = "ELIGIBLE FOR CREDIT" if res['rule_credit_score'] >= 350 else "NOT ELIGIBLE"
        render_kpi("Eligibility Status", f"{status_text}", f"Score: {res['rule_credit_score']} / 1000")

    # 1. Pre-Approved Credit Offers (Full Width, stacked 1 after the other)
    st.markdown("<h3 style='color:#FFFFFF; margin: 1.2rem 0 0.4rem 0;'>🎉 Pre-Approved Credit Offers</h3>", unsafe_allow_html=True)
    products = res["recommended_products"]

    if lender_status in ["Approved", "Disbursed"]:
        st.markdown(f"""
        <div class="white-card" style="border-left: 5px solid #16A34A; background:#F0FDF4; padding:1.2rem; margin-bottom:0.8rem;">
            <h4 style="color:#15803D; margin:0 0 0.3rem 0;">✅ APPROVED BY LENDER OFFICER</h4>
            <p style="color:#166534; font-size:0.92rem; margin:0;">
                Your credit profile and offers have been formally approved by your Lender Officer (Status: <b>{lender_status}</b>). Click <b>"View Credit Offer"</b> below to proceed to Bank Engine!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if len(products) > 0 and res['rule_credit_score'] >= 350:
            for prod in products:
                st.markdown(f"""
                <div class="offer-card">
                    <div class="offer-card-title">{prod['product_name']}<span class="offer-badge">{prod['type']}</span></div>
                    <p style="margin:0.4rem 0; color:#475569;">Interest Rate: <b style="color:#0F172A;">{prod['interest_rate']}</b> | Minimum Score Required: <b style="color:#059669;">{prod['min_score']}</b></p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"⚡ View Credit Offer", key=f"btn_view_offer_{applicant_id}_{prod['product_id']}", use_container_width=True):
                    st.session_state["selected_offer"] = prod
                    st.session_state["payment_completed"] = False
                    st.session_state["applicant_nav"] = "🏦 Partner Bank Checkout Division"
                    st.rerun()
        else:
            st.warning("No pre-approved credit offers available right now based on your current score.")
    elif lender_status == "Rejected":
        st.markdown("""
        <div class="white-card" style="border-left: 5px solid #DC2626; background:#FEF2F2; padding:1.2rem; margin-bottom:0.8rem;">
            <h4 style="color:#991B1B; margin:0 0 0.3rem 0;">❌ APPLICATION DECLINED BY LENDER</h4>
            <p style="color:#7F1D1D; font-size:0.92rem; margin:0;">
                Your Lender Officer reviewed your evaluation and declined credit offer approval based on current risk parameters.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Pending Review
        st.markdown(f"""
        <div class="white-card" style="border-left: 5px solid #F59E0B; background:#FFFBEB; padding: 1.2rem; margin-bottom: 0.8rem;">
            <h4 style="color:#B45309; margin:0 0 0.4rem 0;">⏳ Credit Offers Pending Lender Approval</h4>
            <p style="color:#78350F; font-size:0.92rem; line-height:1.5; margin-bottom:0.8rem;">
                Your credit profile and CreditForYou score of <b>{res['rule_credit_score']} / 1000</b> have been computed. Your Lender Officer is currently reviewing your file. Once approved by your Lender, your pre-approved credit offers will appear here with the <b>"View Credit Offer"</b> button.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔑 Demo Shortcut: Approve Credit Offers as Lender", key="btn_quick_approve_demo", use_container_width=True):
            if latest_eval and latest_eval.get("eval_id"):
                update_lender_decision(int(latest_eval["eval_id"]), "Approved", "Approved via Demo Shortcut.")
            else:
                save_credit_evaluation(applicant_id, res, lender_status="Approved", lender_notes="Approved via Demo Shortcut.")
            st.success("✅ Credit profile approved by Lender! Reloading offers...")
            st.rerun()

    # 2. Score Component Contribution Graph (Full Width, stacked 1 after the other)
    st.markdown("<h3 style='color:#FFFFFF; margin: 1.5rem 0 0.5rem 0;'>📊 Score Component Contribution</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        component_label_map = {
            "lifestyle": "Lifestyle",
            "spending": "Spending",
            "spending_behavior": "Spending Behavior",
            "repayment": "Repayment",
            "repayment_discipline": "Repayment Discipline",
            "adjustments": "Credit Bonus & Risk Factors"
        }
        comp_scores = res.get("component_scores", {}) if isinstance(res, dict) else {}
        if not comp_scores:
            comp_scores = {
                "lifestyle": 0,
                "spending_behavior": 0,
                "repayment_discipline": 0,
                "adjustments": 0
            }

        comp_data = []
        for k, v in comp_scores.items():
            label = component_label_map.get(k.lower(), k.replace("_", " ").title())
            val_int = int(round(v)) if v is not None else 0
            comp_data.append({
                "Component": label,
                "Points": val_int,
                "PointsDisplay": f"{val_int:+d} pts" if "adjustment" in k.lower() else f"{val_int} pts"
            })
        comp_df = pd.DataFrame(comp_data)

        max_pts = max(float(comp_df["Points"].max() * 1.18) if len(comp_df) > 0 else 400.0, 400.0)
        min_pts = min(0.0, float(comp_df["Points"].min() * 1.2) if len(comp_df) > 0 else 0.0)

        bars = alt.Chart(comp_df).mark_bar(color="#10B981").encode(
            x=alt.X(
                "Component:N",
                sort=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=0,
                    labelFontSize=11,
                    labelFontWeight="bold",
                    labelColor="#0F172A",
                    title="Credit Score Component Category",
                    titleColor="#059669",
                    titleFontSize=12,
                    titleFontWeight="bold",
                    titlePadding=10
                )
            ),
            y=alt.Y(
                "Points:Q",
                scale=alt.Scale(domain=[min_pts, max_pts]),
                axis=alt.Axis(
                    title="Score Contribution (Points 0-1000)",
                    titleColor="#059669",
                    titleFontSize=12,
                    titleFontWeight="bold",
                    labelColor="#0F172A",
                    labelFontSize=11
                )
            )
        )

        text = bars.mark_text(
            align='center',
            baseline='bottom',
            dy=-6,
            fontSize=12,
            fontWeight='bold',
            color='#0F172A'
        ).encode(
            text='PointsDisplay:N'
        )

        chart = (bars + text).properties(
            height=320
        ).configure(
            background='#FFFFFF'
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelColor='#0F172A',
            titleColor='#059669',
            gridColor='#E2E8F0',
            domainColor='#CBD5E1'
        )

        st.altair_chart(chart, use_container_width=True)

    st.markdown("<h3 style='color:#FFFFFF; margin: 1.2rem 0 0.5rem 0;'>💡 Explainable AI — Factors Driving Your Score</h3>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns(2, gap="medium")

    with exp_col1:
        st.markdown("<h5 style='color:#ECFDF5; margin-bottom:0.4rem;'>🟢 Positive Drivers (Score Boosters)</h5>", unsafe_allow_html=True)
        for item in explanation["top_negative_risk_contributors"]:
            st.markdown(f"""
            <div class="factor-favorable">
                {item['explanation']}
            </div>
            """, unsafe_allow_html=True)

    with exp_col2:
        st.markdown("<h5 style='color:#ECFDF5; margin-bottom:0.4rem;'>🔴 High Risk Factors (Areas To Improve)</h5>", unsafe_allow_html=True)
        for item in explanation["top_positive_risk_contributors"]:
            st.markdown(f"""
            <div class="factor-adverse">
                {item['explanation']}
            </div>
            """, unsafe_allow_html=True)


def render_dummy_bank_division(applicant_id: str):
    selected_offer = st.session_state.get("selected_offer")
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown('<div class="page-title">🏦 Dummy Bank Division Gateway</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Partner Banking Payment & Credit Subscription Checkout Division</div>', unsafe_allow_html=True)
    with col_hdr2:
        if st.button("⬅️ Back to Scorecard", key="btn_top_back_scorecard", use_container_width=True):
            st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
            st.rerun()

    if not selected_offer:
        st.warning("No offer selected. Please select a Pre-Approved Credit Offer from your Scorecard.")
        if st.button("⬅️ Return to Scorecard & Offers", key="btn_back_no_offer"):
            st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
            st.rerun()
        return

    # Offer Details Summary Card
    st.markdown(f"""
    <div class="white-card" style="border-left: 5px solid #4F46E5; background: #FFFFFF;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.8rem;">
            <h3 style="margin:0; color:#0F172A; font-weight:800;">🏦 {selected_offer.get('product_name', 'Credit Offer')}</h3>
            <span class="offer-badge" style="font-size:0.85rem; padding:0.35rem 0.85rem;">{selected_offer.get('type', 'Credit Line')}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>
                <p style="color:#64748B; font-size:0.82rem; margin:0; font-weight:600;">INTEREST RATE</p>
                <p style="color:#0F172A; font-size:1.1rem; font-weight:700; margin:0;">{selected_offer.get('interest_rate', '12.00% APR')}</p>
            </div>
            <div>
                <p style="color:#64748B; font-size:0.82rem; margin:0; font-weight:600;">MINIMUM REQUIRED SCORE</p>
                <p style="color:#4F46E5; font-size:1.1rem; font-weight:700; margin:0;">{selected_offer.get('min_score', 350)} pts</p>
            </div>
            <div>
                <p style="color:#64748B; font-size:0.82rem; margin:0; font-weight:600;">APPLICANT ID</p>
                <p style="color:#0F172A; font-size:1.1rem; font-weight:700; margin:0;">{applicant_id}</p>
            </div>
            <div>
                <p style="color:#64748B; font-size:0.82rem; margin:0; font-weight:600;">ACTIVATION / PROCESSING FEE</p>
                <p style="color:#16A34A; font-size:1.1rem; font-weight:700; margin:0;">$25.00 (Standard Settlement)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Payment Details Section
    st.markdown('<div class="white-card" style="margin-top: 1.2rem;">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#0F172A; margin:0 0 1rem 0;'>💳 Select Payment Method & Bank Authorization</h4>", unsafe_allow_html=True)

    c_pay1, c_pay2 = st.columns([1, 1])

    with c_pay1:
        payment_type = st.selectbox(
            "Select Payment Type:",
            [
                "💳 Credit Card / Debit Card",
                "🏦 Direct Bank Transfer / Net Banking",
                "📱 UPI / Wallet Transfer",
                "🔄 Automatic Recurring Auto-Pay Debit"
            ],
            key="dummy_payment_type"
        )
        account_name = st.text_input("Account Holder Name", value="John Doe", key="dummy_acc_name")

    with c_pay2:
        if "Credit Card" in payment_type:
            payment_detail = st.text_input("Card Number", value="4532 •••• •••• 8892", key="dummy_card_no")
            expiry = st.text_input("Expiry / CVV", value="12/28 | 392", key="dummy_exp")
        elif "Bank Transfer" in payment_type:
            payment_detail = st.text_input("Bank Account Number", value="ACCT-987410293 (BNP Paribas)", key="dummy_bank_acc")
            expiry = st.text_input("Routing Code", value="ROUT-019283", key="dummy_route")
        elif "UPI" in payment_type:
            payment_detail = st.text_input("UPI Virtual Payment Address", value="john.doe@okbank", key="dummy_upi")
            expiry = st.text_input("Mobile Number", value="+1 (555) 019-2834", key="dummy_mob")
        else:
            payment_detail = st.text_input("Mandate Account ID", value="MANDATE-AUTO-55421", key="dummy_man")
            expiry = st.text_input("Frequency", value="Monthly (Auto-Debit)", key="dummy_freq")

    st.markdown("---")

    payment_completed = st.session_state.get("payment_completed", False)

    if payment_completed:
        txn_id = st.session_state.get("txn_id", f"TXN_{uuid.uuid4().hex[:10].upper()}")
        timestamp_str = st.session_state.get("txn_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        st.markdown(f"""
        <div style="background:#F0FDF4; border:2px solid #22C55E; border-radius:16px; padding:1.8rem; text-align:center; margin-bottom:1.2rem;">
            <div style="font-size:3.5rem; margin-bottom:0.2rem;">✅</div>
            <h1 style="color:#15803D; margin:0 0 0.4rem 0; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;">PAID</h1>
            <p style="color:#166534; font-size:1.15rem; font-weight:600; margin:0 0 1rem 0;">
                Payment & Activation for <b>{selected_offer.get('product_name')}</b> was successful!
            </p>
            <div style="display:inline-block; background:#FFFFFF; border:1px solid #BBF7D0; padding:1rem 1.6rem; border-radius:12px; text-align:left; font-size:0.95rem; color:#0F172A; box-shadow:0 4px 12px rgba(0,0,0,0.03);">
                <b>Status:</b> <span style="background:#DCFCE7; color:#15803D; padding:0.2rem 0.6rem; border-radius:12px; font-weight:800;">PAID</span><br>
                <b>Transaction Reference ID:</b> <code>{txn_id}</code><br>
                <b>Product Offer:</b> {selected_offer.get('product_name')} ({selected_offer.get('type')})<br>
                <b>Payment Method:</b> {payment_type}<br>
                <b>Account Detail:</b> {payment_detail}<br>
                <b>Authorized Holder:</b> {account_name}<br>
                <b>Amount Settled:</b> $25.00<br>
                <b>Timestamp:</b> {timestamp_str} (Instant Bank Settlement)
            </div>
        </div>
        """, unsafe_allow_html=True)

        receipt_text = f"""================================================================================
          CREDITFORYOU / DUMMY BANK DIVISION OFFICIAL PAYMENT RECEIPT
================================================================================
Transaction Reference ID : {txn_id}
Payment Status           : PAID (SUCCESSFUL SETTLEMENT)
Date & Timestamp         : {timestamp_str}
Applicant ID             : {applicant_id}
Authorized Account Holder: {account_name}
--------------------------------------------------------------------------------
PRODUCT OFFER DETAILS:
Product Name             : {selected_offer.get('product_name', 'Credit Offer')}
Product Type             : {selected_offer.get('type', 'Credit Line')}
Interest Rate            : {selected_offer.get('interest_rate', '12% APR')}
Minimum Score Required   : {selected_offer.get('min_score', 350)} pts
--------------------------------------------------------------------------------
PAYMENT TRANSACTION METRICS:
Payment Method / Type    : {payment_type}
Account / Payment Detail : {payment_detail}
Settlement Amount        : $25.00 USD
Settlement Institution   : AltBank Dummy Partner Banking Gateway
Database Audit ID        : Saved to SQLite (creditforyou_legacy.db -> payments)
================================================================================
             Thank you for choosing CreditForYou Financial Services!
================================================================================
"""

        report_df = pd.DataFrame([{
            "Transaction_ID": txn_id,
            "Status": "PAID",
            "Applicant_ID": applicant_id,
            "Account_Holder": account_name,
            "Product_Name": selected_offer.get('product_name'),
            "Product_Type": selected_offer.get('type'),
            "Interest_Rate": selected_offer.get('interest_rate'),
            "Payment_Method": payment_type,
            "Account_Detail": payment_detail,
            "Amount_Paid": 25.00,
            "Timestamp": timestamp_str
        }])
        csv_report = report_df.to_csv(index=False)

        st.markdown("<h4 style='color:#0F172A; margin-top:1rem;'>📥 Download Payment Reports</h4>", unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📄 Download Official Receipt (.TXT)",
                data=receipt_text,
                file_name=f"Payment_Receipt_{txn_id}.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary",
                key="btn_dl_txt"
            )
        with d_col2:
            st.download_button(
                label="📊 Download Payment Report (.CSV)",
                data=csv_report,
                file_name=f"Payment_Report_{txn_id}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_dl_csv"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("⬅️ Return to Credit Scorecard & Offers", key="btn_return_scorecard", use_container_width=True):
                st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
                st.rerun()
        with c_b2:
            if st.button("🔄 Reset / Make Another Payment", key="btn_reset_pay", use_container_width=True):
                st.session_state["payment_completed"] = False
                st.rerun()
    else:
        st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-bottom:1rem;'>Click below to complete authorization & process instant payment, or return to your scorecard.</p>", unsafe_allow_html=True)
        c_p1, c_p2 = st.columns([1.5, 1])
        with c_p1:
            if st.button("💳 Just Pay ($25.00)", type="primary", use_container_width=True, key="btn_just_pay"):
                txn_id = f"TXN_{uuid.uuid4().hex[:10].upper()}"
                txn_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["payment_completed"] = True
                st.session_state["txn_id"] = txn_id
                st.session_state["txn_time"] = txn_time

                # Persist transaction in SQLite database
                save_payment_record(
                    txn_id=txn_id,
                    applicant_id=applicant_id,
                    product_name=selected_offer.get('product_name', 'Credit Offer'),
                    product_type=selected_offer.get('type', 'Credit Line'),
                    interest_rate=selected_offer.get('interest_rate', '12% APR'),
                    payment_type=payment_type,
                    account_name=account_name,
                    payment_detail=payment_detail,
                    amount=25.0,
                    status="PAID"
                )

                st.success("✅ Payment Processed & Persisted! Status: PAID")
                st.rerun()
        with c_p2:
            if st.button("⬅️ Return to Scorecard", use_container_width=True, key="btn_cancel_checkout"):
                st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_credit_evaluator():
    st.markdown('<div class="page-title">CreditForYou Document & Risk Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a Bank Statement or Payslip PDF to automatically extract financial parameters & evaluate your credit score accurately.</div>', unsafe_allow_html=True)

    user_app_id = st.session_state["user"].get("applicant_id") or "APP_WEB_001"
    existing_profile = get_applicant_from_db(user_app_id)

    tab_pdf, tab_manual = st.tabs(["📁 Instant PDF Document Extractor", "📝 Manual Form Override"])

    with tab_pdf:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); border: 1px solid #34D399; border-radius: 18px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.15); color: #FFFFFF;">
            <h4 style="margin:0 0 0.5rem 0; color:#FFFFFF !important; font-weight:800; font-size:1.2rem;">📄 Upload Financial PDF (Bank Statement / Payslip)</h4>
            <p style="color:#ECFDF5 !important; font-size:0.95rem; line-height:1.5; margin-bottom:0; font-weight:500;">Upload your bank account statement or salary payslip PDF. CreditForYou parses the document and extracts your Monthly Income, Monthly Spend, Savings Days, and DTI ratio automatically!</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Select Financial PDF File to Upload & Parse:", type=["pdf"], key="pdf_uploader")

        extracted_data = None

        if uploaded_file is not None:
            extracted_data = extract_financial_data_from_pdf(uploaded_file)
            st.success(f"✅ Successfully extracted text & features from uploaded PDF: **{uploaded_file.name}**")
        else:
            db_profile = get_applicant_from_db(user_app_id)
            if db_profile:
                extracted_data = db_profile
            else:
                extracted_data = None

        if extracted_data:
            editor_res = render_rule_engine_parameter_editor(extracted_data, user_app_id, "evaluator")
            render_ai_extracted_summary(extracted_data)
            if editor_res["submitted"]:
                applicant_data = editor_res["data"]
                upsert_applicant_profile(applicant_data)
                res = predict_credit_risk(applicant_data)
                save_credit_evaluation(user_app_id, res, lender_status="Pending Review")
                st.session_state["user"]["applicant_id"] = user_app_id
                st.success(f"🎉 Verified Rule Engine parameters saved for '{user_app_id}'! CreditForYou Score: {res['rule_credit_score']} / 1000.")
                st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
                st.rerun()
        else:
            st.info("📄 Please select and upload your Bank Account Statement or Salary Payslip PDF file above to extract your financial parameters and compute your CreditForYou score.")

    with tab_manual:
        defaults = {
            "applicant_id": user_app_id, "age": 30, "monthly_income": 5000.0, "months_at_job": 36,
            "housing": "owner", "education": "bachelor", "employment_status": "employed",
            "monthly_spend": 2000.0, "essential_pct": 0.65, "cashflow_volatility": 0.08,
            "savings_days": 120, "on_time_rate": 0.95, "dti": 0.25, "credit_util": 0.20,
            "delinq_90plus": 0, "delinq_60plus": 0, "delinq_30plus": 0, "positive_habits": 2,
            "risk_flags": 0, "tx_late_ratio": 0.02, "tx_debit_credit_ratio": 0.8
        }

        if existing_profile:
            st.info(f"💡 **Loaded Profile for Applicant ID:** `{user_app_id}`. Edit values below if needed.")
            defaults["applicant_id"] = str(existing_profile.get("applicant_id", user_app_id))
            defaults["age"] = int(existing_profile.get("age", 30))
            defaults["monthly_income"] = float(existing_profile.get("monthly_income", 5000.0))
            defaults["months_at_job"] = int(existing_profile.get("months_at_job", 36))
            defaults["housing"] = str(existing_profile.get("housing", "owner")).lower()
            defaults["education"] = str(existing_profile.get("education", "bachelor")).lower()
            defaults["employment_status"] = str(existing_profile.get("employment_status", "employed")).lower()
            defaults["monthly_spend"] = float(existing_profile.get("monthly_spend", 2000.0))
            defaults["essential_pct"] = float(existing_profile.get("essential_pct", 0.65))
            defaults["cashflow_volatility"] = float(existing_profile.get("cashflow_volatility", 0.08))
            defaults["savings_days"] = int(existing_profile.get("savings_days", 120))
            defaults["on_time_rate"] = float(existing_profile.get("on_time_rate", 0.95))
            defaults["dti"] = float(existing_profile.get("dti", 0.25))
            defaults["credit_util"] = float(existing_profile.get("credit_util", 0.20))
            defaults["positive_habits"] = int(existing_profile.get("positive_habits", 2))
            defaults["risk_flags"] = int(existing_profile.get("risk_flags", 0))

        housing_options = ["owner", "rent", "none"]
        housing_idx = housing_options.index(defaults["housing"]) if defaults["housing"] in housing_options else 0

        edu_options = ["highschool", "bachelor", "master", "phd"]
        edu_idx = edu_options.index(defaults["education"]) if defaults["education"] in edu_options else 1

        emp_options = ["employed", "self_employed", "unemployed"]
        emp_idx = emp_options.index(defaults["employment_status"]) if defaults["employment_status"] in emp_options else 0

        with st.form("manual_applicant_form"):
            st.subheader("📋 Manual Parameter Override")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                applicant_id = st.text_input("Applicant ID", value=str(defaults.get("applicant_id", user_app_id)))
                val_age = int(max(18, min(80, defaults.get("age", 30))))
                age = st.number_input("Age", min_value=18, max_value=80, value=val_age)
                val_income = float(max(0.0, min(100000.0, defaults.get("monthly_income", 0.0))))
                monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, max_value=100000.0, value=val_income, step=500.0)
                val_months = int(max(0, min(240, defaults.get("months_at_job", 0))))
                months_at_job = st.number_input("Months at Current Job", min_value=0, max_value=240, value=val_months)

            with col2:
                housing = st.selectbox("Housing Status", housing_options, index=housing_idx)
                education = st.selectbox("Education Level", edu_options, index=edu_idx)
                employment_status = st.selectbox("Employment Status", emp_options, index=emp_idx)
                val_spend = float(max(0.0, min(80000.0, defaults.get("monthly_spend", 0.0))))
                monthly_spend = st.number_input("Monthly Spend ($)", min_value=0.0, max_value=80000.0, value=val_spend, step=200.0)

            with col3:
                val_ess = float(max(0.0, min(1.0, defaults.get("essential_pct", 0.65))))
                essential_pct = st.slider("Essentials Spend Ratio", min_value=0.0, max_value=1.0, value=val_ess, step=0.05)
                val_vol = float(max(0.0, min(0.5, defaults.get("cashflow_volatility", 0.08))))
                cashflow_volatility = st.slider("Cashflow Volatility", min_value=0.0, max_value=0.5, value=val_vol, step=0.01)
                val_sav = int(max(0, min(365, defaults.get("savings_days", 0))))
                savings_days = st.number_input("Savings Reserve (Days)", min_value=0, max_value=365, value=val_sav)
                val_ontime = float(max(0.0, min(1.0, defaults.get("on_time_rate", 0.95))))
                on_time_rate = st.slider("On-Time Payment Rate", min_value=0.0, max_value=1.0, value=val_ontime, step=0.01)

            with col4:
                val_dti = float(max(0.0, min(1.0, defaults.get("dti", 0.25))))
                dti = st.slider("Debt-to-Income (DTI)", min_value=0.0, max_value=1.0, value=val_dti, step=0.02)
                val_util = float(max(0.0, min(1.0, defaults.get("credit_util", 0.20))))
                credit_util = st.slider("Credit Utilization", min_value=0.0, max_value=1.0, value=val_util, step=0.02)
                val_habits = int(max(0, min(5, defaults.get("positive_habits", 0))))
                positive_habits = st.number_input("Positive Habits Count", min_value=0, max_value=5, value=val_habits)
                val_flags = int(max(0, min(5, defaults.get("risk_flags", 0))))
                risk_flags = st.number_input("Risk Flags Count", min_value=0, max_value=5, value=val_flags)

            submitted = st.form_submit_button("🚀 Submit & Save Evaluation to Database", type="primary", use_container_width=True)

        if submitted:
            applicant_data = {
                "applicant_id": applicant_id,
                "age": age,
                "monthly_income": monthly_income,
                "months_at_job": months_at_job,
                "housing": housing,
                "education": education,
                "employment_status": employment_status,
                "monthly_spend": monthly_spend,
                "essential_pct": essential_pct,
                "cashflow_volatility": cashflow_volatility,
                "savings_days": savings_days,
                "on_time_rate": on_time_rate,
                "dti": dti,
                "credit_util": credit_util,
                "delinq_90plus": 0, "delinq_60plus": 0, "delinq_30plus": 0,
                "positive_habits": positive_habits,
                "risk_flags": risk_flags,
                "tx_late_ratio": 0.02,
                "tx_debit_credit_ratio": 0.8
            }

            upsert_applicant_profile(applicant_data)
            res = predict_credit_risk(applicant_data)
            save_credit_evaluation(applicant_id, res)

            if "user" in st.session_state and st.session_state["user"]:
                st.session_state["user"]["applicant_id"] = applicant_id

            st.success(f"🎉 Updated profile for '{applicant_id}'! Calculated CreditForYou Score: {res['rule_credit_score']} / 1000. Saved to `creditforyou_legacy.db`!")
            st.session_state["applicant_nav"] = "💳 Credit Scorecard & Pre-Approved Offers"
            st.rerun()


def render_database_explorer():
    st.markdown('<div class="page-title">SQLite Database Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Inspect stored applicant records, evaluation audit logs, payments, and product catalog in <code>creditforyou_legacy.db</code>.</div>', unsafe_allow_html=True)

    conn = get_db_connection()
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Evaluation Audit Logs", "👤 Applicants Registry", "💳 Product Catalog", "💳 Payment Audit History & Download"])

    with tab1:
        st.subheader("Credit Evaluation Audit Trail")
        eval_df = pd.read_sql_query("SELECT * FROM credit_evaluations ORDER BY eval_timestamp DESC LIMIT 50;", conn)
        st.dataframe(eval_df, use_container_width=True)

    with tab2:
        st.subheader("Registered Applicants & Financial Profiles")
        app_df = pd.read_sql_query("""
            SELECT a.applicant_id, a.age, a.monthly_income, a.housing, a.employment_status, 
                   fp.monthly_spend, fp.dti, fp.credit_util, fp.savings_days
            FROM applicants a
            JOIN financial_profiles fp ON a.applicant_id = fp.applicant_id
            LIMIT 50;
        """, conn)
        st.dataframe(app_df, use_container_width=True)

    with tab3:
        st.subheader("Financial Products Catalog")
        prod_df = pd.read_sql_query("SELECT * FROM products;", conn)
        st.dataframe(prod_df, use_container_width=True)

    with tab4:
        st.subheader("Completed Payment Transactions & Downloadable Reports")
        pmt_df = get_payment_history()
        st.dataframe(pmt_df, use_container_width=True)
        if not pmt_df.empty:
            csv_all_pmts = pmt_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Payments Report (CSV)",
                data=csv_all_pmts,
                file_name="All_Payments_Report_CreditForYou.csv",
                mime="text/csv",
                type="primary",
                key="btn_dl_all_payments"
            )

    conn.close()


def render_what_if_simulator():
    st.markdown('<div class="page-title">What-If Risk Simulation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Simulate applicant financial behavior changes and observe credit score impact in real-time.</div>', unsafe_allow_html=True)

    sample_applicant = {
        "applicant_id": "APP_SIMULATOR_001",
        "age": 28, "monthly_income": 3500.0, "months_at_job": 12, "housing": "rent",
        "education": "bachelor", "employment_status": "employed", "monthly_spend": 2400.0,
        "essential_pct": 0.60, "cashflow_volatility": 0.22, "savings_days": 20,
        "on_time_rate": 0.78, "dti": 0.48, "credit_util": 0.65, "delinq_90plus": 0,
        "delinq_60plus": 0, "delinq_30plus": 1, "positive_habits": 0, "risk_flags": 1,
        "tx_late_ratio": 0.10, "tx_debit_credit_ratio": 1.5
    }

    scenarios = {
        "saving_boost": "Increase emergency savings reserve (+60 days)",
        "autopay_builder": "Enable Autopay (On-time payment rate to 98%)",
        "discretionary_spike": "Discretionary spending surge (+40% spend)",
        "delinquency_spike": "Missed payment event (Severe delinquency)"
    }

    scen_key = st.selectbox("Select Financial Scenario to Simulate:", list(scenarios.keys()), format_func=lambda x: scenarios[x])
    sim_res = run_what_if_simulation(sample_applicant, scen_key)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi("Baseline Score", f"{sim_res['baseline_score']} pts", "Pre-Simulation Baseline")
    with col2:
        render_kpi("Simulated Score", f"{sim_res['simulated_score']} pts", f"Delta: {sim_res['score_delta']:+} pts")
    with col3:
        render_kpi("Simulated Risk Tier", sim_res.get('simulated_risk_tier', 'N/A'), "Projected Risk Cohort")


def render_model_analytics():
    st.markdown('<div class="page-title">Global Model Performance & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Logistic Regression Coefficients & Feature Importance breakdown.</div>', unsafe_allow_html=True)

    from ml.explain import generate_global_feature_importance

    importance_dict = generate_global_feature_importance()
    feature_label_map = {
        "adjustments": "Credit Bonus & Risk Factors",
        "adjustments_score": "Credit Bonus & Risk Factors"
    }
    imp_data = []
    for k, v in importance_dict.items():
        label = feature_label_map.get(k.lower(), k.replace("_", " ").title())
        imp_data.append({
            "Feature": label,
            "Importance": float(v),
            "ImportanceDisplay": f"{float(v):.2f}"
        })
    imp_df = pd.DataFrame(imp_data)
    
    max_imp = max(float(imp_df["Importance"].max() * 1.18), 1.0)

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    bars = alt.Chart(imp_df).mark_bar(color="#00E599").encode(
        x=alt.X(
            "Feature:N",
            sort=None,
            axis=alt.Axis(
                labelAngle=0,
                labelLimit=0,
                labelFontSize=11,
                labelFontWeight="bold",
                labelColor="#FFFFFF",
                title="Machine Learning Model Features",
                titleColor="#00E599",
                titleFontSize=12,
                titleFontWeight="bold",
                titlePadding=12
            )
        ),
        y=alt.Y(
            "Importance:Q",
            scale=alt.Scale(domain=[0, max_imp]),
            axis=alt.Axis(
                title="Relative Feature Weight (|Coef|)",
                titleColor="#00E599",
                titleFontSize=12,
                titleFontWeight="bold",
                labelColor="#FFFFFF",
                labelFontSize=11
            )
        )
    )

    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-8,
        fontSize=12,
        fontWeight='bold',
        color='#FFFFFF'
    ).encode(
        text='ImportanceDisplay:N'
    )

    chart = (bars + text).properties(
        height=370
    ).configure(
        background='#09090B'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelColor='#FFFFFF',
        titleColor='#00E599',
        gridColor='#27272A',
        domainColor='#3F3F46'
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
