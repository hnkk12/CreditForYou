import sqlite3

import pandas as pd
import pytest

from creditforyou.income import estimate_income, prepare_transactions
from creditforyou.lendingclub import assert_no_leakage, build_default_target
from db import init_db, save_assessment


def test_lendingclub_target_excludes_unresolved_statuses():
    target = build_default_target(pd.Series([
        "Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)", "In Grace Period"
    ]))
    assert target.iloc[:3].tolist() == [0, 1, 1]
    assert target.iloc[3:].isna().all()


def test_leakage_guard():
    assert_no_leakage(["loan_to_income", "dti", "purpose"])
    with pytest.raises(ValueError):
        assert_no_leakage(["dti", "total_pymnt"])
    with pytest.raises(ValueError):
        assert_no_leakage(["hardship_amount"])


def test_rich_schema_excludes_transfer_and_handles_no_income():
    tx = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "timestamp": ["2025-01-01", "2025-01-02", "2025-01-01"],
        "amount": [1000, 5000, -100],
        "transaction_type": ["ach_credit", "internal_transfer", "card_payment"],
        "category": ["income", "finance", "shopping"],
        "subcategory": ["salary", "internal_transfer", "retail"],
        "label": ["SALARY", "TRANSFER", "SHOP"],
        "is_salary": [True, False, False],
        "is_transfer": [False, True, False],
        "balance_after_transaction": [1000, 6000, 900],
    })
    prepared, _ = prepare_transactions(tx)
    assert prepared["is_income"].tolist() == [True, False, False]
    result = estimate_income(prepared).set_index("customer_id")
    assert result.loc["A", "estimated_monthly_income"] == 1000
    assert result.loc["B", "estimated_monthly_income"] == 0


def test_database_schema_and_save(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    row = {
        "customer_id": "C1", "requested_amount": 1000, "term_months": 12, "purpose": "other",
        "estimated_monthly_income": 500, "income_stability_score": 0.8, "income_confidence_score": 0.9,
        "pd_used": 0.2, "post_loan_debt_ratio": 0.3, "credit_score": 70, "tier": "B",
        "decision": "AUTO_APPROVE", "reason_codes": "Stable income", "missing_components": [],
    }
    application_id = save_assessment(row, path=path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications WHERE application_id=?", (application_id,)).fetchone()[0] == 1
