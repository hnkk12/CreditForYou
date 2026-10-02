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
    result = estimate_income(prepared, calibration={}).set_index("customer_id")
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


# --------------------------------------------------------------------------- các sửa đổi sau review

def test_model_dti_excludes_mortgage(tmp_path):
    from creditforyou.datasets import existing_debt_by_customer
    from creditforyou.model import build_features
    pd.DataFrame({"customer_id": ["C1"]}).assign(monthly_income=1).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame({"account_id": ["A1"], "customer_id": ["C1"]}).to_csv(tmp_path / "accounts.csv", index=False)
    pd.DataFrame({"transaction_id": [], "account_id": [], "customer_id": [], "timestamp": [], "amount": []}) \
        .to_csv(tmp_path / "transactions.csv", index=False)
    pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"], "customer_id": ["C1"] * 3, "account_id": ["A1"] * 3,
        "loan_type": ["credit_immobilier", "credit_auto", "credit_conso"],
        "monthly_payment": [2000.0, 400.0, 100.0], "status": ["active", "active", "closed"],
    }).to_csv(tmp_path / "loans.csv", index=False)
    debt = existing_debt_by_customer(tmp_path).iloc[0]
    assert debt["existing_monthly_debt"] == 2400          # khả năng chi trả: tính mọi khoản đang vay
    assert debt["existing_monthly_debt_ex_mortgage"] == 400  # đặc trưng PD: giống định nghĩa dti Lending Club
    feats = build_features(pd.DataFrame([{**debt.to_dict(), "monthly_income": 4000, "requested_amount": 1000}]))
    assert feats["dti"].iloc[0] == 0.1


def test_own_account_transfers_are_not_expenses():
    tx = pd.DataFrame({
        "customer_id": ["A"] * 6,
        "timestamp": ["2025-01-01", "2025-01-05", "2025-01-10", "2025-02-01", "2025-02-05", "2025-02-10"],
        "amount": [1000, -300, -500, 1000, -300, -500],
        "transaction_type": ["ach_credit", "card_payment", "internal_transfer"] * 2,
        "category": ["income", "groceries", "finance"] * 2,
        "label": ["SALARY", "SHOP", "TO SAVINGS"] * 2,
        "is_salary": [True, False, False] * 2,
        "is_transfer": [False, False, True] * 2,
        "balance_after_transaction": [1000, 700, 200] * 2,
    })
    prepared, _ = prepare_transactions(tx)
    r = estimate_income(prepared, calibration={}).iloc[0]
    assert r["avg_monthly_expense"] == 300
    assert r["expense_income_ratio"] == 0.3


def test_normalize_series_matches_normalize_name():
    from creditforyou.io import normalize_name, normalize_series
    s = pd.Series(["Ngày Giao Dịch", "WM SUPERCENTER #12", None, "Ngày Giao Dịch", "đồng"])
    assert normalize_series(s).tolist() == [normalize_name(x) for x in s.astype(str)]


def test_transaction_cache_matches_csv(tmp_path, monkeypatch):
    from creditforyou import datasets
    monkeypatch.setattr(datasets, "PROCESSED_DIR", tmp_path / "processed")
    csv = tmp_path / "transactions.csv"
    pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3"], "account_id": ["A1", "A2", "A1"],
        "customer_id": ["C2", "C1", "C2"], "timestamp": ["2025-01-02", "2025-01-01", "2025-01-01"],
        "amount": [10.0, -5.0, 7.0], "is_salary": [True, False, False], "is_transfer": [False, False, True],
    }).to_csv(csv, index=False)
    assert not datasets.transaction_cache_is_fresh(csv)
    datasets.build_transaction_cache(csv, log=lambda *_: None)
    assert datasets.transaction_cache_is_fresh(csv)
    from_cache = datasets.load_transactions(csv, customers=["C2"]).sort_values("transaction_id").reset_index(drop=True)
    assert from_cache["transaction_id"].tolist() == ["t1", "t3"]
    assert from_cache["is_salary"].tolist() == [True, False]
    csv.write_text(csv.read_text() + "t4,A1,C1,2025-01-03,1.0,False,False\n")
    assert not datasets.transaction_cache_is_fresh(csv)   # CSV đổi -> cache tự hết hạn


def test_income_type_overrides_wrong_salary_flag():
    """Dataset ISB gắn is_salary=True cả cho tiền freelance/lương hưu -> phải phân loại theo subcategory."""
    tx = pd.DataFrame({
        "customer_id": ["F"] * 4,
        "timestamp": ["2025-01-05", "2025-02-05", "2025-03-05", "2025-03-20"],
        "amount": [3000, 0.0001, 5000, 100],
        "transaction_type": ["ach_credit"] * 4,
        "category": ["income"] * 4,
        "subcategory": ["freelance", "freelance", "freelance", "pension"],
        "label": ["ACH DEPOSIT INVOICE"] * 3 + ["DIRECT DEP SOCIAL SECURITY"],
        "is_salary": [True, True, True, True],
        "is_transfer": [False] * 4,
        "balance_after_transaction": [0, 0, 0, 0],
    })
    prepared, _ = prepare_transactions(tx)
    assert prepared["income_type"].tolist() == ["self_employed"] * 3 + ["pension"]
    r = estimate_income(prepared, calibration={"self_employed": 0.5}).iloc[0]
    assert r["main_income_type"] == "self_employed"
    assert r["estimated_monthly_income"] == pytest.approx(r["gross_self_employed_income"] * 0.5 + r["gross_pension_income"], abs=0.01)


def test_calibration_learns_per_type_factor():
    from creditforyou.evaluation import cross_validated_estimates, fit_income_calibration
    rows = []
    for i in range(30):
        rows.append({"gross_salary_income": 1000 + i, "gross_pension_income": 0.0, "gross_self_employed_income": 0.0,
                     "gross_other_income": 0.0, "monthly_income": 0.9 * (1000 + i)})
        rows.append({"gross_salary_income": 0.0, "gross_pension_income": 0.0, "gross_self_employed_income": 2000 + i,
                     "gross_other_income": 0.0, "monthly_income": 0.5 * (2000 + i)})
    df = pd.DataFrame(rows)
    df["gross_monthly_income"] = df.filter(like="gross_").sum(axis=1)
    f = fit_income_calibration(df)["factors"]
    assert f["salary"] == pytest.approx(0.9) and f["self_employed"] == pytest.approx(0.5)
    assert f["pension"] == 1.0                                  # không đủ mẫu -> giữ nguyên
    oof = cross_validated_estimates(df)
    assert ((oof - df["monthly_income"]).abs() / df["monthly_income"]).max() < 1e-9


def test_data_quality_report_flags_noise(tmp_path, monkeypatch):
    from creditforyou import datasets, quality
    monkeypatch.setattr(datasets, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(quality, "REPORTS_DIR", tmp_path / "reports")
    pd.DataFrame({"customer_id": ["C1", "C2"], "monthly_income": [1000, 2000], "age": [30, 40],
                  "birth_date": ["1996-01-01", "1986-01-01"]}).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame({"account_id": ["A1", "A2"], "customer_id": ["C1", "C2"],
                  "opened_at": ["2020-01-01", "2026-03-31"]}).to_csv(tmp_path / "accounts.csv", index=False)
    pd.DataFrame({"loan_id": ["L1"], "customer_id": ["C1"], "account_id": ["A1"], "loan_type": ["credit_immobilier"],
                  "principal": [10000.0], "annual_rate": [0.06], "term_months": [120],
                  "monthly_payment": [111.02], "status": ["active"]}).to_csv(tmp_path / "loans.csv", index=False)
    pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"], "account_id": ["A1", "A1", "A1", "A2"],
        "customer_id": ["C1", "C1", "C1", "C2"],
        "timestamp": ["2026-01-05", "2026-02-05", "2026-02-10", "2026-03-31"],
        "amount": [1000.0, 900.0, -200.0, -50.0], "category": ["income", "income", "finance", "shopping"],
        "subcategory": ["freelance", "salary", "internal_transfer", "retail"], "label": ["INV", "PAY", "TR", "S"],
        "is_salary": [True, True, False, False], "is_transfer": [False, False, True, False],
        "balance_after_transaction": [1000.0, 1900.0, 1700.0, 5.0],
    }).to_csv(tmp_path / "transactions.csv", index=False)
    report = quality.data_quality_report(tmp_path)
    issues = {f["issue"].split(" {")[0]: f for f in report["findings"]}
    assert issues["is_salary=True nhưng không phải lương"]["count"] == 1
    assert issues["chuyển khoản giữa các tài khoản của chính khách"]["count"] == 1
    assert issues["khoản vay mua nhà trong nợ đang có"]["count"] == 1
    stale = [f for f in report["findings"] if f["issue"].startswith("monthly_income > 0")][0]
    assert stale["count"] == 1                     # C2 không có khoản thu nhập nào
    assert all(f["handling"] for f in report["findings"])
