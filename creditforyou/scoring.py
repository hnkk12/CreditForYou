"""BƯỚC #5 — Chấm điểm tín dụng thang 100 (dùng kết quả #2).

| Thành phần             | Max | Công thức                                         |
|------------------------|-----|---------------------------------------------------|
| 1. Ổn định thu nhập     | 25  | độ ổn định × 25                                    |
| 2. Tỷ lệ nợ/thu nhập    | 25  | (1 - trả nợ hằng tháng / thu nhập tháng) × 25      |
| 3. Kinh nghiệm          | 15  | min(năm kinh nghiệm / 30, 1) × 15                  |
| 4. Lịch sử trả nợ       | 20  | (số lần trả đúng hạn / tổng số khoản vay) × 20     |
| 5. Mô hình ML           | 15  | (1 - xác suất vỡ nợ) × 15                          |

Lưu ý so với bản ý tưởng:
- Thành phần 2 dùng KHOẢN TRẢ HẰNG THÁNG (khoản vay mới + nợ đang có) chia THU NHẬP THÁNG,
  vì "số tiền vay / thu nhập tháng" sẽ > 1 với hầu hết khoản vay -> luôn 0 điểm.
- Mọi tỷ lệ được kẹp trong [0, 1] để điểm không âm / không vượt trần.
- Khách chưa từng vay (thin-file): lịch sử trả nợ = mức trung tính (config), không cho 0 điểm.
"""

import math

import numpy as np

from .config import scoring_config


def annuity_payment(principal, annual_rate, term_months):
    """Khoản trả góp đều hằng tháng."""
    if not principal or not term_months or principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r == 0:
        return principal / term_months
    return principal * r / (1 - (1 + r) ** (-term_months))


def _num(x, default=None):
    try:
        if x is None:
            return default
        x = float(x)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def compute_score(app: dict) -> dict:
    """app cần các khoá: income_stability, estimated_monthly_income, monthly_installment,
    existing_monthly_debt, years_experience, past_loans_total, past_loans_on_time / on_time_rate, pd.
    Thiếu khoá nào -> dùng mặc định an toàn và ghi vào 'notes'."""
    cfg = scoring_config()["scoring"]
    w = cfg["weights"]
    notes = []

    # 1. Ổn định thu nhập
    stability = float(np.clip(_num(app.get("income_stability"), 0.0), 0, 1))
    s1 = stability * w["income_stability"]

    # 2. Tỷ lệ nợ / thu nhập
    income = _num(app.get("estimated_monthly_income"), 0.0)
    installment = _num(app.get("monthly_installment"), 0.0)
    existing = _num(app.get("existing_monthly_debt"), 0.0) if cfg.get("dti_includes_existing_debt", True) else 0.0
    if income > 0:
        dti = (installment + existing) / income
    else:
        dti = np.inf
        notes.append("Thu nhập ước tính = 0 -> tỷ lệ nợ/thu nhập = 0 điểm")
    s2 = float(np.clip(1 - dti, 0, 1)) * w["debt_to_income"] if np.isfinite(dti) else 0.0

    # 3. Kinh nghiệm
    years = _num(app.get("years_experience"))
    if years is None:
        years = 0.0
        notes.append("Thiếu số năm kinh nghiệm -> tính 0")
    s3 = min(max(years, 0) / cfg["experience_full_years"], 1.0) * w["experience"]

    # 4. Lịch sử trả nợ
    total = _num(app.get("past_loans_total"))
    on_time = _num(app.get("past_loans_on_time"))
    rate = _num(app.get("on_time_rate"))
    if total and total > 0 and on_time is not None:
        repay_ratio = min(max(on_time / total, 0), 1)
    elif rate is not None:
        repay_ratio = min(max(rate if rate <= 1 else rate / 100, 0), 1)
    else:
        repay_ratio = cfg["no_history_repayment_ratio"]
        notes.append(f"Chưa có lịch sử vay -> dùng mức trung tính {repay_ratio:.0%}")
    s4 = repay_ratio * w["repayment_history"]

    # 5. ML
    pd_ = _num(app.get("pd"))
    if pd_ is None:
        pd_ = 0.5
        notes.append("Chưa có mô hình ML -> PD trung tính 50% (chạy `train` để có PD thật)")
    pd_ = float(np.clip(pd_, 0, 1))
    s5 = (1 - pd_) * w["ml_probability"]

    total_score = s1 + s2 + s3 + s4 + s5
    return {
        "score_income_stability": round(s1, 2),
        "score_debt_to_income": round(s2, 2),
        "score_experience": round(s3, 2),
        "score_repayment_history": round(s4, 2),
        "score_ml": round(s5, 2),
        "credit_score": round(total_score, 2),
        "dti": round(dti, 4) if np.isfinite(dti) else None,
        "repayment_ratio": round(repay_ratio, 4),
        "pd_used": round(pd_, 4),
        "score_notes": notes,
    }
