import numpy as np
import pandas as pd
import pytest

from creditforyou.decision import assign_tier, decide, interest_rate
from creditforyou.income import estimate_income, prepare_transactions
from creditforyou.io import map_columns, normalize_id, parse_amount, parse_dates, parse_direction, parse_label, parse_years
from creditforyou.scoring import compute_score


# --------------------------------------------------------------------------- ví dụ trong bản ý tưởng

def test_doc_example_score():
    """Ví dụ mục 2A: ổn định 95%, vay 5M/thu nhập 8.28M, 5 năm KN, 4/5 đúng hạn, PD 0.22.
    Bản ý tưởng ghi thành phần 2 = 15.0 và tổng 68.95, nhưng (1 - 5/8.28) × 25 = 9.90 -> tổng đúng 63.85."""
    s = compute_score({
        "income_stability": 0.95, "estimated_monthly_income": 8_280_000, "monthly_installment": 5_000_000,
        "existing_monthly_debt": 0, "years_experience": 5, "past_loans_total": 5, "past_loans_on_time": 4,
        "pd": 0.22,
    })
    assert s["score_income_stability"] == 23.75
    assert s["score_debt_to_income"] == pytest.approx(9.90, abs=0.01)
    assert s["score_experience"] == 2.5
    assert s["score_repayment_history"] == 16.0
    assert s["score_ml"] == 11.7
    assert s["credit_score"] == pytest.approx(63.85, abs=0.01)
    assert assign_tier(s["credit_score"])["tier"] == "B"


def test_score_bounds():
    s = compute_score({"income_stability": 1.5, "estimated_monthly_income": 1, "monthly_installment": 10,
                       "years_experience": 99, "past_loans_total": 2, "past_loans_on_time": 5, "pd": -1})
    assert s["score_income_stability"] == 25 and s["score_debt_to_income"] == 0
    assert s["score_experience"] == 15 and s["score_repayment_history"] == 20 and s["score_ml"] == 15


def test_thin_file_neutral():
    s = compute_score({"income_stability": 1, "estimated_monthly_income": 10e6, "pd": 0.1})
    assert s["repayment_ratio"] == 0.5
    assert any("lịch sử" in n for n in s["score_notes"])


def test_tiers_and_rates():
    assert [assign_tier(x)["tier"] for x in (100, 80, 79.99, 60, 59, 40, 39.9, 0)] == list("AABBCCDD")
    assert interest_rate(80, assign_tier(80)) == 10.0
    assert interest_rate(100, assign_tier(100)) == 9.0
    assert interest_rate(60, assign_tier(60)) == 13.0


def test_decision_rules():
    base = {"estimated_monthly_income": 10e6, "months_of_data": 12, "income_confidence": 0.95,
            "requested_amount": 50e6, "age": 30}
    good = compute_score({"income_stability": 1, "estimated_monthly_income": 10e6, "monthly_installment": 1e6,
                          "years_experience": 30, "past_loans_total": 3, "past_loans_on_time": 3, "pd": 0.05})
    assert decide(base, good)["decision"] == "AUTO_APPROVE"
    assert decide({**base, "age": 17}, good)["decision"] == "REJECT"
    assert decide({**base, "months_of_data": 2}, good)["decision"] == "MANUAL_REVIEW"
    c = decide(base, {**good, "credit_score": 50})
    assert c["tier"] == "C" and c["approved_amount"] == 40e6


# --------------------------------------------------------------------------- parse

def test_parse_amount_formats():
    s = pd.Series(["1.234.567", "1,234,567", "8.2M", "5tr", "(500,000)", "-1.000", "12.5", "500.000", None])
    out = parse_amount(s).tolist()
    assert out[:8] == [1234567, 1234567, 8.2e6, 5e6, -500000, -1000, 12.5, 500000]
    assert np.isnan(out[8])


def test_parse_dates_auto():
    assert parse_dates(pd.Series(["25/03/2024", "01/04/2024"])).dt.month.tolist() == [3, 4]
    assert parse_dates(pd.Series(["03/25/2024", "04/01/2024"])).dt.month.tolist() == [3, 4]
    assert parse_dates(pd.Series(["2024-03-25"])).dt.day.tolist() == [25]


def test_normalize_id():
    assert normalize_id(pd.Series(["1001.0", " KH01 ", 1002, "12.5"])).tolist() == ["1001", "KH01", "1002", "12.5"]


def test_parse_misc():
    assert parse_direction(pd.Series(["CREDIT", "Ghi có", "DR", "Nợ", "?"])).tolist()[:4] == [1, 1, -1, -1]
    assert parse_years(pd.Series(["10+ years", "< 1 year", "3 years", "18 months"])).tolist() == [10, 0.5, 3, 1.5]
    lab = parse_label(pd.Series(["Fully Paid", "Charged Off", "Current"]))
    assert lab.tolist()[:2] == [0, 1] and np.isnan(lab.iloc[2])


def test_map_columns_vietnamese_headers():
    cm = map_columns(["Mã KH", "Ngày giao dịch", "Số tiền", "Loại", "Nội dung"], "transactions")
    assert cm.ok
    assert cm.mapping == {"customer_id": "Mã KH", "date": "Ngày giao dịch", "amount": "Số tiền",
                          "direction": "Loại", "description": "Nội dung"}
    assert not map_columns(["foo", "bar"], "transactions").ok


# --------------------------------------------------------------------------- thu nhập

def _tx(monthly, desc="GRAB DOANH THU", extra=()):
    rows = [("C1", f"2025-{m:02d}-01", v, "credit", desc) for m, v in enumerate(monthly, start=1)]
    rows += [("C1", f"2025-{len(monthly):02d}-28", 100_000, "debit", "MUA SAM")]
    rows += list(extra)
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount", "direction", "description"])


def test_income_stable_grab_driver():
    monthly = [8.2e6, 8.5e6, 8.1e6, 8.3e6, 8.4e6, 8.2e6, 8.3e6, 8.25e6, 8.35e6, 8.2e6, 8.4e6, 8.3e6]
    prepared, _ = prepare_transactions(_tx(monthly))
    r = estimate_income(prepared).iloc[0]
    assert r["months_of_data"] == 12
    assert r["estimated_monthly_income"] == pytest.approx(np.mean(monthly), rel=1e-6)
    assert r["income_stability"] > 0.95 and r["income_confidence"] == 0.95
    assert r["repayment_capacity"] == "Cao"


def test_income_excludes_non_income_inflows():
    monthly = [10e6] * 6
    extra = [("C1", "2025-03-10", 50e6, "credit", "GIAI NGAN KHOAN VAY"),
             ("C1", "2025-04-10", 20e6, "credit", "Chuyen khoan noi bo tu TK tiet kiem"),
             ("C1", "2025-05-10", 3e6, "credit", "Hoan tien don hang")]
    prepared, _ = prepare_transactions(_tx(monthly, extra=extra))
    r = estimate_income(prepared).iloc[0]
    assert r["estimated_monthly_income"] == 10e6
    assert r["months_of_data"] == 6 and r["income_confidence"] == 0.95


def test_income_volatile_is_conservative_and_zero_months_count():
    monthly = [2e6, 15e6, 0, 12e6, 3e6, 14e6]
    prepared, _ = prepare_transactions(_tx(monthly))
    r = estimate_income(prepared).iloc[0]
    assert r["months_zero_income"] == 1
    assert r["income_stability"] < 0.8
    assert 0 < r["estimated_monthly_income"] < np.mean(monthly)


def test_income_detects_debt_payment():
    extra = [("C1", f"2025-{m:02d}-15", 2e6, "debit", "TRA GOP KHOAN VAY") for m in range(1, 7)]
    prepared, _ = prepare_transactions(_tx([10e6] * 6, extra=extra))
    assert estimate_income(prepared).iloc[0]["detected_monthly_debt"] == 2e6


def test_unsigned_amounts_without_direction_warns():
    df = _tx([1e6] * 3).drop(columns="direction")
    _, warnings = prepare_transactions(df)
    assert any("TIỀN VÀO" in w for w in warnings)
