"""BƯỚC #2 — Ước tính thu nhập từ lịch sử giao dịch.

Quy trình (theo bản ý tưởng, có bổ sung các điểm còn thiếu):
  1. Lấy tiền vào — LOẠI BỎ tiền vào không phải thu nhập (chuyển khoản nội bộ, giải ngân vay,
     hoàn tiền, tất toán tiết kiệm) để tránh thổi phồng thu nhập.
  2. Gom theo tháng; tháng không có tiền vào tính là 0 (không bỏ qua -> phản ánh đúng độ ổn định).
     Trung bình chia cho SỐ THÁNG THỰC CÓ DỮ LIỆU (không cố định 12).
  3. Độ ổn định = 1 - CV (CV = độ lệch chuẩn / trung bình), kẹp trong [0, 1].
  4. Ổn định >= ngưỡng -> lấy trung bình; ngược lại lấy giá trị bảo thủ hơn:
     TB × (0.5 + 0.5 × độ ổn định) (mặc định) hoặc bách phân vị 25 (tuỳ chọn trong config).
  5. Mức tin tưởng theo số tháng dữ liệu (6-12 tháng: 95%, 3-6: 70%, <3: 40%).
"""

import re

import numpy as np
import pandas as pd

from .config import scoring_config
from .io import normalize_id, normalize_name, parse_amount, parse_dates, parse_direction


def prepare_transactions(tx: pd.DataFrame, dayfirst=None):
    """Chuẩn hoá giao dịch (đã map cột) -> cột: customer_id, date, inflow, outflow,
    text, is_income, is_debt_payment. Trả về (df, warnings)."""
    cfg = scoring_config()["income"]
    warnings = []
    df = pd.DataFrame(index=tx.index)
    df["customer_id"] = normalize_id(tx["customer_id"])
    df["date"] = parse_dates(tx["date"], dayfirst=dayfirst)

    if "inflow" in tx.columns:
        df["inflow"] = parse_amount(tx["inflow"]).abs().fillna(0.0)
        df["outflow"] = parse_amount(tx["outflow"]).abs().fillna(0.0) if "outflow" in tx.columns else 0.0
    else:
        amount = parse_amount(tx["amount"])
        if "direction" in tx.columns:
            d = parse_direction(tx["direction"])
            unknown = d.isna() & amount.notna()
            if unknown.any():
                sample = tx.loc[unknown, "direction"].astype(str).unique()[:5]
                warnings.append(f"{unknown.sum()} giao dịch có hướng không nhận diện được {list(sample)} "
                                f"-> dùng dấu của số tiền.")
            signed = amount.abs() * d.fillna(np.sign(amount))
        else:
            signed = amount
            if (amount.dropna() >= 0).all():
                warnings.append("KHÔNG có cột hướng giao dịch và mọi số tiền đều dương -> toàn bộ bị coi là "
                                "TIỀN VÀO. Kiểm tra lại dataset (cần cột type/direction hoặc tiền ra mang dấu âm).")
        df["inflow"] = signed.clip(lower=0).fillna(0.0)
        df["outflow"] = (-signed).clip(lower=0).fillna(0.0)

    text = pd.Series("", index=tx.index)
    for col in ("description", "category"):
        if col in tx.columns:
            text = text + " " + tx[col].fillna("").astype(str)
    df["text"] = text.map(normalize_name)

    # loại giao dịch lỗi / huỷ
    keep = pd.Series(True, index=tx.index)
    if "status" in tx.columns:
        bad_status = {normalize_name(s) for s in cfg["excluded_status"]}
        keep &= ~tx["status"].astype(str).map(normalize_name).isin(bad_status)
    bad_date = df["date"].isna()
    if bad_date.any():
        warnings.append(f"{bad_date.sum()} dòng không đọc được ngày -> bỏ qua.")
    keep &= ~bad_date
    dropped = (~keep).sum()
    df = df[keep].copy()

    excl_pattern = _keyword_regex(cfg["exclude_inflow_keywords"])
    excl_cats = {normalize_name(c) for c in cfg["exclude_inflow_categories"]}
    is_excluded = df["text"].str.contains(excl_pattern, regex=True) if excl_pattern else False
    if "category" in tx.columns:
        is_excluded = is_excluded | tx.loc[df.index, "category"].astype(str).map(normalize_name).isin(excl_cats)
    df["is_income"] = (df["inflow"] > 0) & ~is_excluded
    debt_pattern = _keyword_regex(cfg["debt_payment_keywords"])
    df["is_debt_payment"] = (df["outflow"] > 0) & (df["text"].str.contains(debt_pattern, regex=True)
                                                   if debt_pattern else False)
    if dropped:
        warnings.append(f"Đã bỏ {dropped} giao dịch (lỗi ngày hoặc trạng thái huỷ/thất bại).")
    return df, warnings


def _keyword_regex(keywords):
    kws = [normalize_name(k) for k in keywords if normalize_name(k)]
    if not kws:
        return None
    # khớp theo ranh giới từ (đã chuẩn hoá thành dạng a_b_c)
    return r"(?:^|_)(?:" + "|".join(re.escape(k) for k in kws) + r")(?:_|$)"



def estimate_income(tx: pd.DataFrame) -> pd.DataFrame:
    """tx: kết quả prepare_transactions. Trả về 1 dòng / khách hàng (tính vector hoá cho dataset lớn)."""
    cfg = scoring_config()["income"]
    tx = tx.copy()
    tx["m"] = tx["date"].dt.year * 12 + tx["date"].dt.month - 1        # chỉ số tháng (số nguyên)

    # --- Khung thời gian của từng khách: [start, end] (chỉ số tháng)
    span = tx.groupby("customer_id", sort=False)["date"].agg(first="min", last="max")
    start = span["first"].dt.year * 12 + span["first"].dt.month - 1
    end = span["last"].dt.year * 12 + span["last"].dt.month - 1
    if cfg.get("trim_partial_months", True):
        # Bỏ tháng đầu/cuối nếu dữ liệu không trọn tháng (tránh kéo thu nhập xuống),
        # chỉ khi sau khi bỏ vẫn còn đủ số tháng tối thiểu.
        s2 = start + ((span["first"].dt.day > 7) & (end > start)).astype(int)
        e2 = end - ((span["last"].dt.day < span["last"].dt.days_in_month - 6) & (end > s2)).astype(int)
        ok = (e2 - s2 + 1) >= cfg["min_months_required"]
        start, end = start.where(~ok, s2), end.where(~ok, e2)
    start = np.maximum(start, end - cfg["lookback_months"] + 1)
    n_months = (end - start + 1).astype(int)

    # --- Lưới khách × tháng (tháng không có tiền vào = 0)
    grid = pd.DataFrame({"customer_id": np.repeat(span.index.values, n_months.values)})
    grid["m"] = np.concatenate([np.arange(s, e + 1) for s, e in zip(start, end)]) if len(span) else []
    tx = tx.merge(pd.DataFrame({"customer_id": span.index, "_s": start.values, "_e": end.values}), on="customer_id")
    win = tx[(tx["m"] >= tx["_s"]) & (tx["m"] <= tx["_e"])]
    monthly = (win.assign(income=win["inflow"].where(win["is_income"], 0.0),
                          debt=win["outflow"].where(win["is_debt_payment"], 0.0))
               .groupby(["customer_id", "m"])[["income", "inflow", "outflow", "debt"]].sum())
    grid = grid.merge(monthly, left_on=["customer_id", "m"], right_index=True, how="left").fillna(0.0)

    g = grid.groupby("customer_id", sort=False)
    out = pd.DataFrame({
        "months_of_data": n_months,
        "first_month": [f"{s // 12}-{s % 12 + 1:02d}" for s in start],
        "last_month": [f"{e // 12}-{e % 12 + 1:02d}" for e in end],
        "n_transactions": win.groupby("customer_id").size().reindex(span.index, fill_value=0),
        "mean": g["income"].mean(),
        "std": g["income"].std(ddof=1),
        "p": g["income"].quantile(cfg["low_stability_percentile"] / 100),
        "months_zero_income": g["income"].apply(lambda s: int((s <= 0).sum())),
        "total_income": g["income"].sum(),
        "total_inflow": g["inflow"].sum(),
        "avg_monthly_outflow": g["outflow"].mean(),
        "detected_monthly_debt": g["debt"].mean(),
        "monthly_income_series": g.apply(lambda d: ";".join(
            f"{m // 12}-{m % 12 + 1:02d}:{v:.0f}" for m, v in zip(d["m"], d["income"])), include_groups=False),
    }, index=span.index)

    # --- Bước 3: độ ổn định = 1 - CV
    valid = (out["mean"] > 0) & out["std"].notna()
    out["income_cv"] = (out["std"] / out["mean"]).where(valid)
    out["income_stability"] = (1 - out["income_cv"]).clip(0, 1).fillna(0.0)

    # --- Bước 4: điều chỉnh theo độ ổn định
    stable = out["income_stability"] >= cfg["high_stability_threshold"]
    if cfg.get("low_stability_method", "haircut") == "percentile":
        conservative = np.minimum(out["mean"], out["p"])
        method_low = f"p{cfg['low_stability_percentile']}_bao_thu"
    else:
        floor = cfg.get("haircut_floor", 0.5)
        conservative = out["mean"] * (floor + (1 - floor) * out["income_stability"])
        method_low = "giam_tru_theo_on_dinh"
    out["estimated_monthly_income"] = np.where(stable, out["mean"], conservative)
    out["estimation_method"] = np.where(stable, "trung_binh", method_low)

    # --- Bước 5: mức tin tưởng theo số tháng dữ liệu
    out["income_confidence"] = 0.0
    for rule in sorted(cfg["confidence_by_months"], key=lambda r: r["min_months"]):
        out.loc[out["months_of_data"] >= rule["min_months"], "income_confidence"] = rule["confidence"]

    est, stab, conf = out["estimated_monthly_income"], out["income_stability"], out["income_confidence"]
    out["repayment_capacity"] = np.select(
        [(est > 0) & (stab >= 0.8) & (conf >= 0.95), (est > 0) & (stab >= 0.5) & (conf >= 0.70)],
        ["Cao", "Trung bình"], "Thấp")
    out["excluded_inflow_share"] = (1 - out["total_income"] / out["total_inflow"]).where(out["total_inflow"] > 0, 0.0)

    out = out.rename(columns={"mean": "avg_monthly_inflow", "std": "income_std"})
    for c in ("avg_monthly_inflow", "income_std", "estimated_monthly_income", "avg_monthly_outflow",
              "detected_monthly_debt"):
        out[c] = out[c].round(0)
    for c in ("income_cv", "income_stability", "excluded_inflow_share"):
        out[c] = out[c].round(4)
    cols = ["months_of_data", "first_month", "last_month", "n_transactions", "avg_monthly_inflow", "income_std",
            "income_cv", "income_stability", "estimated_monthly_income", "estimation_method", "income_confidence",
            "repayment_capacity", "months_zero_income", "excluded_inflow_share", "avg_monthly_outflow",
            "detected_monthly_debt", "monthly_income_series"]
    return out[cols].rename_axis("customer_id").reset_index()
