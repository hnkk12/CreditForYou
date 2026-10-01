"""BƯỚC 3 — Quyết định phê duyệt: quy tắc nghiệp vụ + xếp hạng A/B/C/D + lãi suất + hạn mức."""

from .config import scoring_config

AUTO_APPROVE, MANUAL_REVIEW, REJECT = "AUTO_APPROVE", "MANUAL_REVIEW", "REJECT"
_DECISION_VN = {AUTO_APPROVE: "Phê duyệt tự động", MANUAL_REVIEW: "Review thủ công", REJECT: "Từ chối"}
_SEVERITY = {AUTO_APPROVE: 0, MANUAL_REVIEW: 1, REJECT: 2}


def assign_tier(score: float) -> dict:
    tiers = sorted(scoring_config()["tiers"], key=lambda t: -t["min_score"])
    for t in tiers:
        if score >= t["min_score"]:
            return t
    return tiers[-1]


def interest_rate(score: float, tier: dict):
    """Điểm cao trong nhóm -> lãi suất thấp trong khoảng của nhóm."""
    if not tier.get("rate_range"):
        return None
    lo, hi = tier["rate_range"]
    tiers = sorted(scoring_config()["tiers"], key=lambda t: t["min_score"])
    idx = [t["tier"] for t in tiers].index(tier["tier"])
    upper = tiers[idx + 1]["min_score"] if idx + 1 < len(tiers) else 100
    span = max(upper - tier["min_score"], 1e-9)
    pos = min(max((score - tier["min_score"]) / span, 0), 1)
    return round(hi - pos * (hi - lo), 2)


def decide(app: dict, score: dict) -> dict:
    """app: thông tin hồ sơ + thu nhập; score: kết quả compute_score."""
    rules = scoring_config()["business_rules"]
    tier = assign_tier(score["credit_score"])
    decision = tier["decision"]
    reasons = []

    def escalate(new, reason):
        nonlocal decision
        reasons.append(reason)
        if _SEVERITY[new] > _SEVERITY[decision]:
            decision = new

    age = app.get("age")
    if age is not None and age == age:
        if rules.get("min_age") is not None and age < rules["min_age"]:
            escalate(REJECT, f"Tuổi {age:.0f} < {rules['min_age']}")
        if rules.get("max_age") is not None and age > rules["max_age"]:
            escalate(REJECT, f"Tuổi {age:.0f} > {rules['max_age']}")

    income = app.get("estimated_monthly_income") or 0
    if rules.get("min_income_monthly") is not None and income < rules["min_income_monthly"]:
        escalate(REJECT, f"Thu nhập ước tính {income:,.0f}đ < tối thiểu {rules['min_income_monthly']:,.0f}đ".replace(",", "."))

    dti = score.get("dti")
    if rules.get("max_dti_hard") is not None and (dti is None or dti > rules["max_dti_hard"]):
        escalate(REJECT, f"Tỷ lệ trả nợ/thu nhập {'∞' if dti is None else f'{dti:.0%}'} > trần {rules['max_dti_hard']:.0%}")

    expense_ratio = app.get("expense_income_ratio")
    if (rules.get("max_expense_income_ratio_hard") is not None and expense_ratio is not None
            and expense_ratio == expense_ratio and expense_ratio > rules["max_expense_income_ratio_hard"]):
        escalate(REJECT, f"Tỷ lệ chi tiêu/thu nhập {expense_ratio:.0%} > trần cứng "
                         f"{rules['max_expense_income_ratio_hard']:.0%}")

    months = app.get("months_of_data") or 0
    if months < scoring_config()["income"]["min_months_required"]:
        escalate(MANUAL_REVIEW, f"Chỉ có {int(months)} tháng dữ liệu giao dịch")

    conf = app.get("income_confidence")
    if rules.get("low_confidence_to_review") is not None and conf is not None and conf < rules["low_confidence_to_review"]:
        escalate(MANUAL_REVIEW, f"Mức tin tưởng thu nhập {conf:.0%} < {rules['low_confidence_to_review']:.0%}")

    if (rules.get("max_pd_auto_approve") is not None and score.get("pd_used") is not None
            and score["pd_used"] > rules["max_pd_auto_approve"]):
        escalate(MANUAL_REVIEW, f"Xác suất vỡ nợ {score['pd_used']:.0%} > {rules['max_pd_auto_approve']:.0%}")

    installment_ratio = score.get("installment_income_ratio")
    if (rules.get("max_installment_income_ratio_auto") is not None and installment_ratio is not None
            and installment_ratio > rules["max_installment_income_ratio_auto"]):
        escalate(MANUAL_REVIEW, f"Khoản trả mới/thu nhập {installment_ratio:.0%} > ngưỡng tự động "
                                f"{rules['max_installment_income_ratio_auto']:.0%}")
    if (rules.get("max_post_loan_debt_ratio_auto") is not None and dti is not None
            and dti > rules["max_post_loan_debt_ratio_auto"]):
        escalate(MANUAL_REVIEW, f"Tổng trả nợ sau vay/thu nhập {dti:.0%} > ngưỡng tự động "
                                f"{rules['max_post_loan_debt_ratio_auto']:.0%}")
    if (rules.get("max_expense_income_ratio_auto") is not None and expense_ratio is not None
            and expense_ratio == expense_ratio and expense_ratio > rules["max_expense_income_ratio_auto"]):
        escalate(MANUAL_REVIEW, f"Tỷ lệ chi tiêu/thu nhập {expense_ratio:.0%} > ngưỡng tự động "
                                f"{rules['max_expense_income_ratio_auto']:.0%}")

    requested = app.get("requested_amount")
    if decision == REJECT:
        approved, rate = 0, None
    else:
        ratio = tier.get("max_amount_ratio")
        approved = None if requested is None else (requested if ratio is None else requested * ratio)
        rate = interest_rate(score["credit_score"], tier)

    return {
        "tier": tier["tier"],
        "tier_label": tier["label"],
        "decision": decision,
        "decision_vn": _DECISION_VN[decision],
        "interest_rate_pct": rate,
        "approved_amount": None if approved is None else round(approved),
        "decision_reasons": reasons,
    }
