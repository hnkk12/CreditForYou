"""Hạn mức trả góp tại điểm bán (point-of-purchase, POP).

Ba bước tách rời:
  [A] compute_limits     - cấp TRƯỚC hạn mức theo lô (ngoài giờ mua hàng): ai được, tối đa bao nhiêu.
  [B] checkout_decision  - tại quầy: so khoản mua với hạn mức -> APPROVE / COUNTER_OFFER / DECLINE, tức thì,
                           không có bước người duyệt.
  [C] graduate_limit     - sau khi cho vay: trả đúng hạn thì nâng hạn mức, trễ hạn thì khoá.

Trạng thái cấp hạn mức:
  PREAPPROVED          cấp đủ hạn mức tính được
  PREAPPROVED_REDUCED  vướng guardrail mềm -> vẫn cấp ngay nhưng hạn mức × reduced_limit_factor
  OFFLINE_REVIEW       thiếu dữ liệu (ít tháng / tin cậy thấp) -> chuyên viên xem xét NGOÀI giờ mua hàng
  NOT_ELIGIBLE         bị từ chối bởi quy tắc cứng hoặc nhóm D, hoặc không còn khả năng trả
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import scoring_config

PREAPPROVED, REDUCED, OFFLINE_REVIEW, NOT_ELIGIBLE = (
    "PREAPPROVED", "PREAPPROVED_REDUCED", "OFFLINE_REVIEW", "NOT_ELIGIBLE")
STATUS_VN = {PREAPPROVED: "Cấp hạn mức", REDUCED: "Cấp hạn mức (giảm)",
             OFFLINE_REVIEW: "Xem xét ngoài giờ", NOT_ELIGIBLE: "Không đủ điều kiện"}


def annuity_factor(term_months: int, annual_rate: float) -> float:
    """Số tiền gốc vay được trên mỗi 1 đơn vị trả góp hằng tháng."""
    r = annual_rate / 12
    return term_months if r == 0 else (1 - (1 + r) ** (-term_months)) / r


def installment_for(amount: float, term_months: int, annual_rate: float) -> float:
    return amount / annuity_factor(term_months, annual_rate)


def fx() -> float:
    """Hệ số quy đổi đơn vị tiền của dataset sang đơn vị hạn mức (VND)."""
    return float(scoring_config()["pos_limit"].get("fx_dataset_to_vnd", 1.0))


def monthly_capacity(row) -> tuple[float, str]:
    """Khoản trả góp mới tối đa mỗi tháng (VND) và ràng buộc nào đang chặn."""
    cfg = scoring_config()["pos_limit"]
    k = fx()
    cash_income = float(row.get("conservative_cash_income") or 0.0) * k
    net_income = float(row.get("estimated_monthly_income") or 0.0) * k
    committed = float(row.get("avg_monthly_committed_expense") or 0.0) * k
    debt = float(row.get("debt_service_monthly", row.get("existing_monthly_debt")) or 0.0) * k
    disposable = cash_income - committed - debt
    candidates = {
        "dư khả dụng sau chi phí bắt buộc & nợ": cfg["disposable_share"] * disposable,
        "trả góp mới / thu nhập": cfg["max_new_installment_to_income"] * net_income,
        "tổng nợ / thu nhập": cfg["max_total_debt_to_income"] * net_income - debt,
    }
    binding = min(candidates, key=candidates.get)
    return max(0.0, candidates[binding]), binding


def compute_limits(scored: pd.DataFrame) -> pd.DataFrame:
    """[A] scored: output của _score_rows (có tier, decision, các cột dòng tiền)."""
    cfg = scoring_config()["pos_limit"]
    rows = []
    for rec in scored.to_dict("records"):
        capacity, binding = monthly_capacity(rec)
        tier = rec["tier"]
        reasons = []
        hard_caps = {f"trần nhóm {tier}": cfg["tier_caps"][tier],
                     f"{cfg['tier_income_multiple'][tier]}× thu nhập tháng":
                         cfg["tier_income_multiple"][tier] * float(rec.get("estimated_monthly_income") or 0) * fx()}
        if not rec.get("has_credit_history", False):
            hard_caps["trần khách mới (chưa có lịch sử vay)"] = cfg["thin_file_cap"]

        def capped(term):
            caps = {"khả năng trả": capacity * annuity_factor(term, cfg["annual_rate"]), **hard_caps}
            name = min(caps, key=caps.get)
            return max(0.0, caps[name]), name

        limit, cap_name = capped(cfg["term_months"])
        # Hạn mức nếu khách chọn kỳ hạn dài nhất: chỉ lớn hơn khi đang bị chặn bởi khả năng trả, không vượt trần cứng.
        limit_max_term, _ = capped(cfg["max_term_months"])

        months = rec.get("months_of_data") or 0
        confidence = rec.get("income_confidence") or 0
        if rec["decision"] == "REJECT" or tier == "D":
            status = NOT_ELIGIBLE
            reasons.append(rec.get("decision_reasons") or f"Nhóm {tier}")
        elif str(rec.get("customer_segment", "")).upper() in {s.upper() for s in cfg.get("offline_review_segments", [])}:
            status = OFFLINE_REVIEW
            reasons.append(f"Phân khúc {rec['customer_segment']} (dễ tổn thương) -> tư vấn & xem xét ngoài giờ")
        elif months < scoring_config()["income"]["min_months_required"] or confidence < cfg["offline_review_min_confidence"]:
            status = OFFLINE_REVIEW
            reasons.append(f"Dữ liệu chưa đủ ({int(months)} tháng, tin cậy {confidence:.0%}) -> xem xét ngoài giờ")
        elif rec["decision"] == "MANUAL_REVIEW" or tier == "C":
            status = REDUCED
            limit *= cfg["reduced_limit_factor"]
            limit_max_term *= cfg["reduced_limit_factor"]
            reasons.append(rec.get("decision_reasons") or f"Nhóm {tier}")
        else:
            status = PREAPPROVED
        if status in (PREAPPROVED, REDUCED) and limit < cfg["min_limit"]:
            reasons.append(f"Hạn mức tính được {limit:,.0f} < tối thiểu {cfg['min_limit']:,.0f} (giới hạn: {binding})")
            status, limit = NOT_ELIGIBLE, 0.0
        if status in (NOT_ELIGIBLE, OFFLINE_REVIEW):
            limit = limit_max_term = 0.0
        rows.append({
            "monthly_capacity": round(capacity, 2),
            "capacity_binding_constraint": binding,
            "limit_binding_cap": cap_name,
            "pos_limit": float(np.floor(limit / 100000) * 100000),   # làm tròn xuống bội số 100.000 VND
            "pos_limit_max_term": float(np.floor(max(limit_max_term, limit) / 100000) * 100000),
            "limit_status": status,
            "limit_status_vn": STATUS_VN[status],
            "limit_reasons": " | ".join(r for r in reasons if r),
        })
    return pd.concat([scored.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def checkout_decision(available_limit: float, purchase_amount: float, term_months: int | None = None,
                      available_limit_max_term: float | None = None) -> dict:
    """[B] Quyết định tại quầy - luôn tức thì, không chuyển cho người duyệt.

    available_limit         : hạn mức còn lại ở kỳ hạn chuẩn
    available_limit_max_term: hạn mức còn lại nếu kéo dài tới max_term_months (đã áp mọi trần cứng)
    """
    cfg = scoring_config()["pos_limit"]
    term = int(term_months or cfg["term_months"])
    rate = cfg["annual_rate"]
    if available_limit <= 0:
        return {"checkout_decision": "DECLINE", "message": "Chưa có hạn mức trả góp", "installment": None,
                "down_payment": None, "offered_term": None}
    if purchase_amount <= available_limit:
        return {"checkout_decision": "APPROVE", "message": f"Duyệt trả góp {term} tháng",
                "installment": round(installment_for(purchase_amount, term, rate), 2),
                "down_payment": 0.0, "offered_term": term}
    # Đề xuất 1: kéo dài kỳ hạn - chỉ khi hạn mức đang bị chặn bởi khả năng trả hằng tháng
    max_term_limit = available_limit_max_term or available_limit
    if purchase_amount <= max_term_limit:
        base = annuity_factor(cfg["term_months"], rate)
        for longer in [t for t in cfg.get("allowed_terms", range(1, 61)) if term < t <= cfg["max_term_months"]]:
            # hạn mức tăng theo hệ số niên kim nhưng không vượt hạn mức kỳ hạn dài nhất (đã gồm trần cứng)
            allowed = min(available_limit * annuity_factor(longer, rate) / base, max_term_limit)
            if purchase_amount <= allowed:
                return {"checkout_decision": "COUNTER_OFFER",
                        "message": f"Đề xuất kéo dài kỳ hạn lên {longer} tháng",
                        "installment": round(installment_for(purchase_amount, longer, rate), 2),
                        "down_payment": 0.0, "offered_term": longer}
    # Đề xuất 2: trả trước phần vượt hạn mức
    down = purchase_amount - available_limit
    return {"checkout_decision": "COUNTER_OFFER",
            "message": f"Trả trước {down:,.0f}, trả góp phần còn lại {available_limit:,.0f}",
            "installment": round(installment_for(available_limit, term, rate), 2),
            "down_payment": round(down, 2), "offered_term": term}


def graduate_limit(current_limit: float, tier: str, on_time_installments: int, max_dpd: int) -> dict:
    """[C] Nâng/khoá hạn mức theo hành vi trả nợ thực tế."""
    cfg = scoring_config()["pos_limit"]
    g = cfg["graduation"]
    if max_dpd >= g["freeze_on_dpd"]:
        return {"new_limit": 0.0, "action": "FREEZE", "message": f"Trễ hạn {max_dpd} ngày -> khoá hạn mức"}
    if on_time_installments >= g["on_time_installments_required"]:
        new = min(current_limit * (1 + g["step_up"]), cfg["tier_caps"][tier])
        return {"new_limit": round(new, 2), "action": "STEP_UP" if new > current_limit else "HOLD",
                "message": f"{on_time_installments} kỳ đúng hạn -> nâng hạn mức"}
    return {"new_limit": current_limit, "action": "HOLD", "message": "Chưa đủ lịch sử trả để nâng"}
