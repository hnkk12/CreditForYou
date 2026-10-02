"""Reproducible evaluation helpers; every reported number is computed here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_DIR, REPORTS_DIR, SYNTHETIC_DATA_DIR
from .datasets import load_synthetic_table, synthetic_paths
from .income import INCOME_TYPES
from .pipeline import run_income


def income_metrics(predictions: pd.DataFrame) -> dict:
    valid = predictions[(predictions["monthly_income"] > 0) & predictions["estimated_monthly_income"].notna()].copy()
    if valid.empty:
        raise ValueError("Không có cặp ground truth/prediction hợp lệ để đánh giá thu nhập.")
    error = valid["estimated_monthly_income"] - valid["monthly_income"]
    abs_error = error.abs()
    ape = abs_error / valid["monthly_income"]
    smape = 2 * abs_error / (valid["estimated_monthly_income"].abs() + valid["monthly_income"].abs())
    return {
        "n_customers": int(len(valid)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(ape.mean()),
        "smape": float(smape.mean()),
        "median_absolute_percentage_error": float(ape.median()),
        "within_10pct": float((ape <= 0.10).mean()),
        "within_20pct": float((ape <= 0.20).mean()),
        "mean_signed_error": float(error.mean()),
    }


def fit_income_calibration(df: pd.DataFrame, min_customers: int = 10, dominant_share: float = 0.95) -> dict:
    """Hệ số gộp -> ròng cho từng loại nguồn thu = trung vị (thu nhập đã xác minh / thu nhập gộp),
    chỉ học trên khách có >= dominant_share thu nhập đến từ đúng 1 loại. Loại nào ít hơn
    min_customers khách -> giữ 1.0 (không đủ bằng chứng để hiệu chỉnh)."""
    factors, counts = {}, {}
    total = df["gross_monthly_income"]
    for itype in INCOME_TYPES:
        gross = df[f"gross_{itype}_income"]
        mask = (gross > 0) & (gross >= dominant_share * total) & (df["monthly_income"] > 0)
        counts[itype] = int(mask.sum())
        factors[itype] = float(np.median(df.loc[mask, "monthly_income"] / gross[mask]))             if mask.sum() >= min_customers else 1.0
    return {"factors": factors, "n_customers": counts}


def apply_calibration(df: pd.DataFrame, factors: dict) -> pd.Series:
    return sum(df[f"gross_{t}_income"] * factors.get(t, 1.0) for t in INCOME_TYPES)


def cross_validated_estimates(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series:
    """Ước tính out-of-fold: hệ số học trên 4/5 khách, áp cho 1/5 còn lại -> số liệu đánh giá trung thực."""
    from sklearn.model_selection import KFold
    oof = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=seed).split(df):
        factors = fit_income_calibration(df.iloc[train_idx])["factors"]
        oof.iloc[test_idx] = apply_calibration(df.iloc[test_idx], factors).values
    return oof


def income_metrics_by_group(predictions: pd.DataFrame, group_col: str) -> dict:
    out = {}
    for name, grp in predictions.groupby(group_col):
        try:
            m = income_metrics(grp)
        except ValueError:
            continue
        out[str(name)] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()
                          if k in ("n_customers", "median_absolute_percentage_error", "within_10pct",
                                   "within_20pct", "mean_signed_error")}
    return out


def evaluate_income_dataset(data_dir: Path = SYNTHETIC_DATA_DIR, reports_dir: Path = REPORTS_DIR,
                            calibration_path: Path = CONFIG_DIR / "income_calibration.json"):
    """1) ước tính thu nhập gộp theo loại nguồn thu CHỈ từ giao dịch;
    2) học hệ số gộp -> ròng trên khách có thu nhập đã xác minh (ở đây: customers.monthly_income);
    3) báo cáo sai số bằng ước tính out-of-fold (5-fold) để không tự chấm bài của chính mình."""
    data_dir, reports_dir = Path(data_dir), Path(reports_dir)
    estimates, warnings = run_income(synthetic_paths(data_dir)["transactions"])
    profile_cols = ["customer_id", "monthly_income", "customer_segment", "profile"]
    header = set(pd.read_csv(synthetic_paths(data_dir)["customers"], nrows=0).columns)
    truth = load_synthetic_table("customers", data_dir, usecols=[c for c in profile_cols if c in header])
    df = estimates.merge(truth, on="customer_id", how="left", validate="one_to_one").reset_index(drop=True)

    calibration = fit_income_calibration(df)
    calibration_path.write_text(json.dumps({
        **calibration,
        "method": "median(verified_income / gross_income_of_type) trên khách có >=95% thu nhập từ 1 loại",
        "source": "customers.monthly_income (đóng vai nhóm khách đã xác minh thu nhập)",
        "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    df["estimated_monthly_income_in_sample"] = apply_calibration(df, calibration["factors"])
    df["estimated_monthly_income"] = cross_validated_estimates(df)
    df["no_income_last_12m"] = df["gross_monthly_income"] <= 0
    df["absolute_error"] = (df["estimated_monthly_income"] - df["monthly_income"]).abs()
    df["absolute_percentage_error"] = df["absolute_error"] / df["monthly_income"].where(df["monthly_income"] > 0)

    metrics = income_metrics(df)
    metrics["evaluation"] = "5-fold out-of-fold (hệ số hiệu chỉnh học trên 4/5, đo trên 1/5 còn lại)"
    metrics["calibration_factors"] = calibration["factors"]
    metrics["excluding_no_income_last_12m"] = income_metrics(df[~df["no_income_last_12m"]])
    metrics["customers_no_income_last_12m"] = int(df["no_income_last_12m"].sum())
    metrics["by_main_income_type"] = income_metrics_by_group(df, "main_income_type")
    if "profile" in df:
        metrics["by_profile"] = income_metrics_by_group(df, "profile")
    metrics["warnings"] = warnings
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / "income_predictions.csv", index=False, encoding="utf-8-sig")
    with open(reports_dir / "income_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return df, metrics
