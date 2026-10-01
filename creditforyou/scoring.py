"""Configurable hybrid credit score (prototype, not a bank policy)."""

from __future__ import annotations

import math

import numpy as np

from .config import scoring_config


def annuity_payment(principal, annual_rate, term_months):
    """Equal monthly payment in the same currency as ``principal``."""
    if not principal or not term_months or principal <= 0 or term_months <= 0:
        return 0.0
    rate = annual_rate / 12.0
    if rate == 0:
        return principal / term_months
    return principal * rate / (1 - (1 + rate) ** (-term_months))


def _num(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        return default if math.isnan(value) else value
    except (TypeError, ValueError):
        return default


def compute_score(app: dict) -> dict:
    """Score available evidence and normalize enabled component weights to 100.

    Missing evidence is never replaced with random/neutral values. Its component
    is disabled and listed in ``missing_components``.
    """
    cfg = scoring_config()["scoring"]
    weights = cfg["weights"]
    raw, maxima, notes, missing = {}, {}, [], []

    stability = _num(app.get("income_stability_score"), _num(app.get("income_stability")))
    if stability is None:
        missing.append("income_stability")
    else:
        maxima["income_stability"] = weights["income_stability"]
        raw["income_stability"] = float(np.clip(stability, 0, 1)) * weights["income_stability"]

    income = _num(app.get("estimated_monthly_income"), 0.0)
    installment = _num(app.get("monthly_installment"))
    existing = _num(app.get("existing_monthly_debt"), 0.0)
    if income <= 0 or installment is None:
        dti = None
        installment_ratio = None
        missing.append("affordability")
    else:
        installment_ratio = installment / income
        dti = (installment + existing) / income
        maxima["debt_to_income"] = weights["debt_to_income"]
        raw["debt_to_income"] = float(np.clip(1 - dti, 0, 1)) * weights["debt_to_income"]

    years = _num(app.get("financial_tenure_years"), _num(app.get("years_experience")))
    if years is None:
        missing.append("financial_tenure_or_experience")
    else:
        maxima["experience"] = weights["experience"]
        raw["experience"] = min(max(years, 0) / cfg["experience_full_years"], 1.0) * weights["experience"]

    total = _num(app.get("past_loans_total"))
    on_time = _num(app.get("past_loans_on_time"))
    repayment_ratio = _num(app.get("on_time_rate"))
    derived_repayment = False
    if total is not None and total > 0 and on_time is not None:
        repayment_ratio = on_time / total
        derived_repayment = True
    if repayment_ratio is None:
        missing.append("credit_or_repayment_history")
    else:
        if not derived_repayment and repayment_ratio > 1:
            repayment_ratio /= 100
        repayment_ratio = float(np.clip(repayment_ratio, 0, 1))
        maxima["repayment_history"] = weights["repayment_history"]
        raw["repayment_history"] = repayment_ratio * weights["repayment_history"]

    pd_value = _num(app.get("pd"))
    if pd_value is None:
        missing.append("ml_pd")
    else:
        pd_value = float(np.clip(pd_value, 0, 1))
        maxima["ml"] = weights["ml_probability"]
        raw["ml"] = (1 - pd_value) * weights["ml_probability"]

    available_weight = float(sum(maxima.values()))
    if not available_weight:
        raise ValueError("Không có component hợp lệ để chấm điểm.")
    scale = 100.0 / available_weight
    scores = {key: value * scale for key, value in raw.items()}
    normalized_max = {key: value * scale for key, value in maxima.items()}
    if missing:
        notes.append("Thiếu dữ liệu; đã vô hiệu hóa component và chuẩn hóa trọng số còn lại: " + ", ".join(missing))

    return {
        "score_income_stability": round(scores.get("income_stability", 0.0), 2),
        "score_debt_to_income": round(scores.get("debt_to_income", 0.0), 2),
        "score_experience": round(scores.get("experience", 0.0), 2),
        "score_repayment_history": round(scores.get("repayment_history", 0.0), 2),
        "score_ml": round(scores.get("ml", 0.0), 2),
        "credit_score": round(sum(scores.values()), 2),
        "available_weight": round(available_weight, 2),
        "normalized_component_max": {k: round(v, 2) for k, v in normalized_max.items()},
        "missing_components": missing,
        "post_loan_debt_ratio": round(dti, 4) if dti is not None else None,
        "installment_income_ratio": round(installment_ratio, 4) if installment_ratio is not None else None,
        "dti": round(dti, 4) if dti is not None else None,
        "repayment_ratio": None if repayment_ratio is None else round(repayment_ratio, 4),
        "pd_used": None if pd_value is None else round(pd_value, 4),
        "score_notes": notes,
    }


def reason_codes(app: dict, score: dict) -> list[str]:
    """Human-readable, non-deterministic wording for prototype explanations."""
    reasons = []
    stability = _num(app.get("income_stability_score"), _num(app.get("income_stability")))
    confidence = _num(app.get("income_confidence_score"), _num(app.get("income_confidence")))
    recurring = _num(app.get("recurring_income_ratio"))
    if stability is not None:
        reasons.append("High income volatility" if stability < 0.5 else "Stable income pattern")
    if confidence is not None and confidence < 0.7:
        reasons.append("Limited income evidence or data coverage")
    if recurring is not None and recurring >= 0.8:
        reasons.append("Strong recurring income pattern")
    if score.get("post_loan_debt_ratio") is not None and score["post_loan_debt_ratio"] > 0.5:
        reasons.append("High post-loan debt ratio")
    if score.get("pd_used") is not None:
        reasons.append("Low model-predicted default risk" if score["pd_used"] < 0.15
                       else "Elevated model-predicted default risk" if score["pd_used"] > 0.35
                       else "Moderate model-predicted default risk")
    return reasons
