"""Quy trình hoàn chỉnh #2 + #5: giao dịch -> thu nhập -> điểm -> quyết định."""

import time

import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_PATH, scoring_config
from .decision import decide
from .income import estimate_income, prepare_transactions
from .io import load_dataset, normalize_id, parse_amount, parse_label, parse_percent_ratio, parse_term_months, parse_years
from .model import build_features, load_model, predict_pd
from .scoring import annuity_payment, compute_score


def run_income(tx_path, dayfirst=None, customers=None):
    tx = load_dataset(tx_path, "transactions")
    if customers:
        tx = tx[normalize_id(tx["customer_id"]).isin(set(normalize_id(pd.Series(customers, dtype=str))))]
        if tx.empty:
            raise KeyError(f"Không tìm thấy khách hàng {customers} trong file giao dịch.")
    prepared, warnings = prepare_transactions(tx, dayfirst=dayfirst)
    return estimate_income(prepared), warnings


def _prepare_applicants(path) -> pd.DataFrame:
    a = load_dataset(path, "applicants", keep_extra=True)
    a["customer_id"] = normalize_id(a["customer_id"])
    parsers = {
        "age": lambda s: pd.to_numeric(s, errors="coerce"),
        "years_experience": parse_years,
        "requested_amount": parse_amount,
        "term_months": parse_term_months,
        "monthly_installment": parse_amount,
        "existing_monthly_debt": parse_amount,
        "past_loans_total": lambda s: pd.to_numeric(s, errors="coerce"),
        "past_loans_on_time": lambda s: pd.to_numeric(s, errors="coerce"),
        "on_time_rate": parse_percent_ratio,
        "declared_income": parse_amount,
        "default": parse_label,
    }
    for col, fn in parsers.items():
        if col in a:
            a[col] = fn(a[col])
    return a


def evaluate(tx_path, applicants_path=None, model_path=DEFAULT_MODEL_PATH, dayfirst=None,
             customers=None, overrides=None):
    """Trả về (results_df, warnings, model_artifact)."""
    warnings = []
    if applicants_path:
        apps = _prepare_applicants(applicants_path)
        if customers:
            apps = apps[apps["customer_id"].isin(set(normalize_id(pd.Series(customers, dtype=str))))]
            if apps.empty:
                raise KeyError(f"Không tìm thấy khách hàng {customers} trong file hồ sơ.")
        income, w = run_income(tx_path, dayfirst=dayfirst, customers=list(apps["customer_id"]))
    else:
        income, w = run_income(tx_path, dayfirst=dayfirst, customers=customers)
        apps = income[["customer_id"]].copy()
        warnings.append("Không có file hồ sơ (--applicants): thiếu tuổi, kinh nghiệm, lịch sử vay, số tiền vay "
                        "-> dùng giá trị mặc định / tham số dòng lệnh.")
    warnings += w

    for key, val in (overrides or {}).items():
        if val is not None:
            apps[key] = val

    clash = [c for c in apps.columns if c in income.columns and c != "customer_id"]
    if clash:
        warnings.append(f"Bỏ cột trùng tên với kết quả ước tính thu nhập trong file hồ sơ: {clash}")
        apps = apps.drop(columns=clash)
    df = apps.merge(income, on="customer_id", how="left")
    no_tx = df["months_of_data"].isna()
    if no_tx.any():
        warnings.append(f"{no_tx.sum()} hồ sơ không có giao dịch nào -> thu nhập = 0.")
    for c in ("months_of_data", "estimated_monthly_income", "income_stability", "income_confidence",
              "detected_monthly_debt"):
        df[c] = df[c].fillna(0)

    # Nợ hiện tại: ưu tiên số khai báo, nếu thiếu dùng khoản trả nợ phát hiện từ giao dịch
    if "existing_monthly_debt" not in df:
        df["existing_monthly_debt"] = np.nan
    df["existing_debt_source"] = np.where(df["existing_monthly_debt"].notna(), "ho_so", "giao_dich")
    df["existing_monthly_debt"] = df["existing_monthly_debt"].fillna(df["detected_monthly_debt"])

    # Khoản trả hằng tháng của khoản vay mới
    loan_cfg = scoring_config()["loan_defaults"]
    if "requested_amount" not in df:
        df["requested_amount"] = np.nan
    if "term_months" not in df:
        df["term_months"] = np.nan
    df["term_months"] = df["term_months"].fillna(loan_cfg["term_months"])
    computed = [annuity_payment(a, loan_cfg["annual_interest_rate"], t) if a == a else 0.0
                for a, t in zip(df["requested_amount"], df["term_months"])]
    if "monthly_installment" not in df:
        df["monthly_installment"] = np.nan
    df["monthly_installment"] = df["monthly_installment"].fillna(pd.Series(computed, index=df.index))
    if df["requested_amount"].isna().all():
        warnings.append("Không có số tiền vay -> khoản trả mới = 0 (điểm nợ/thu nhập chỉ tính nợ hiện tại). "
                        "Dùng --loan-amount để giả định.")

    # ML PD
    model = load_model(model_path)
    if model is not None:
        feats = build_features(df.assign(monthly_income=df["estimated_monthly_income"]))
        df["pd"] = predict_pd(model, feats)
        missing = [c for c in model["num_features"] + model["cat_features"] if feats[c].isna().all()]
        if missing:
            warnings.append(f"Mô hình ML cần {missing} nhưng hồ sơ không có -> dùng giá trị điền khuyết của mô hình.")
    else:
        df["pd"] = np.nan
        warnings.append(f"Chưa có mô hình ML ({model_path.name}) -> PD trung tính 50%. Chạy `train` trước.")

    rows = []
    for rec in df.to_dict("records"):
        t0 = time.perf_counter()
        s = compute_score(rec)
        d = decide(rec, s)
        elapsed = (time.perf_counter() - t0) * 1000
        rows.append({**s, **d, "processing_ms": round(elapsed, 3)})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["score_notes"] = out["score_notes"].map(" | ".join)
    out["decision_reasons"] = out["decision_reasons"].map(" | ".join)
    return out, warnings, model
