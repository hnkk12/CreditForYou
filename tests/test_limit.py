import pandas as pd
import pytest

from creditforyou.config import scoring_config
from creditforyou.limit import (annuity_factor, checkout_decision, compute_limits, fx, graduate_limit,
                                installment_for, monthly_capacity)


def _row(**kw):
    base = {"customer_id": "C1", "tier": "B", "decision": "AUTO_APPROVE", "decision_reasons": "",
            "estimated_monthly_income": 4000.0, "conservative_cash_income": 4000.0,
            "avg_monthly_committed_expense": 1500.0, "debt_service_monthly": 0.0,
            "months_of_data": 12, "income_confidence": 0.95, "has_credit_history": True,
            "customer_segment": "STANDARD"}
    base.update(kw)
    return base


def test_capacity_uses_tightest_constraint():
    cfg = scoring_config()["pos_limit"]
    cap, binding = monthly_capacity(_row())
    # (đơn vị dataset × fx -> VND) 50% × (4000 - 1500) = 1250 ; 20% × 4000 = 800 ; 50% × 4000 - 0 = 2000 -> 800
    assert cap == pytest.approx(cfg["max_new_installment_to_income"] * 4000 * fx())
    assert binding == "trả góp mới / thu nhập"
    cap, binding = monthly_capacity(_row(avg_monthly_committed_expense=3800.0))
    assert cap == pytest.approx(cfg["disposable_share"] * 200 * fx())
    assert monthly_capacity(_row(avg_monthly_committed_expense=5000.0))[0] == 0   # thâm hụt -> 0


def test_limit_statuses_and_caps():
    cfg = scoring_config()["pos_limit"]
    res = compute_limits(pd.DataFrame([
        _row(customer_id="ok"),
        _row(customer_id="thin", has_credit_history=False),
        _row(customer_id="soft", decision="MANUAL_REVIEW", decision_reasons="chi tiêu cao"),
        _row(customer_id="rej", decision="REJECT", decision_reasons="tuổi"),
        _row(customer_id="short", months_of_data=2),
        _row(customer_id="vuln", customer_segment="VULNERABLE"),
        _row(customer_id="broke", avg_monthly_committed_expense=5000.0),
    ])).set_index("customer_id")
    assert res.loc["ok", "limit_status"] == "PREAPPROVED"
    assert res.loc["ok", "pos_limit"] <= cfg["tier_caps"]["B"]
    assert res.loc["thin", "pos_limit"] <= cfg["thin_file_cap"]
    assert res.loc["soft", "limit_status"] == "PREAPPROVED_REDUCED"
    assert res.loc["soft", "pos_limit"] < res.loc["ok", "pos_limit"]
    assert res.loc["rej", "limit_status"] == "NOT_ELIGIBLE" and res.loc["rej", "pos_limit"] == 0
    assert res.loc["short", "limit_status"] == "OFFLINE_REVIEW"
    assert res.loc["vuln", "limit_status"] == "OFFLINE_REVIEW"
    assert res.loc["broke", "limit_status"] == "NOT_ELIGIBLE"
    assert (res["pos_limit_max_term"] >= res["pos_limit"]).all()


def test_checkout_is_always_instant_and_respects_caps():
    assert checkout_decision(0, 100)["checkout_decision"] == "DECLINE"
    ok = checkout_decision(750, 700)
    assert ok["checkout_decision"] == "APPROVE" and ok["down_payment"] == 0
    # hạn mức bị chặn bởi trần cứng (max_term = limit) -> không được kéo kỳ hạn để vượt trần
    capped = checkout_decision(750, 800, available_limit_max_term=750)
    assert capped["down_payment"] == 50
    # hạn mức bị chặn bởi khả năng trả -> kéo dài kỳ hạn hợp lệ
    longer = checkout_decision(750, 800, available_limit_max_term=1400)
    assert longer["checkout_decision"] == "COUNTER_OFFER" and longer["down_payment"] == 0
    assert longer["offered_term"] > scoring_config()["pos_limit"]["term_months"]
    for d in (ok, capped, longer):
        assert d["checkout_decision"] in {"APPROVE", "COUNTER_OFFER", "DECLINE"}   # không có REVIEW


def test_annuity_roundtrip():
    assert installment_for(1000 * annuity_factor(6, 0.18), 6, 0.18) == pytest.approx(1000)
    assert annuity_factor(6, 0.0) == 6


def test_graduation():
    assert graduate_limit(1000, "B", 3, 0)["action"] == "STEP_UP"
    assert graduate_limit(1000, "B", 1, 0)["action"] == "HOLD"
    assert graduate_limit(1000, "B", 6, 30)["new_limit"] == 0
    cap_b = scoring_config()["pos_limit"]["tier_caps"]["B"]
    assert graduate_limit(cap_b * 0.95, "B", 3, 0)["new_limit"] == cap_b   # không vượt trần nhóm
