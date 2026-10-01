"""Reproducible evaluation helpers; every reported number is computed here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from .datasets import load_synthetic_table, synthetic_paths
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


def evaluate_income_dataset(data_dir: Path = SYNTHETIC_DATA_DIR, reports_dir: Path = REPORTS_DIR):
    """Evaluate estimator against ground truth used only after prediction."""
    data_dir, reports_dir = Path(data_dir), Path(reports_dir)
    # run_income receives transactions only; customer monthly_income cannot leak into it.
    estimates, warnings = run_income(synthetic_paths(data_dir)["transactions"])
    truth = load_synthetic_table("customers", data_dir, usecols=["customer_id", "monthly_income"])
    predictions = estimates.merge(truth, on="customer_id", how="left", validate="one_to_one")
    predictions["absolute_error"] = (predictions["estimated_monthly_income"] - predictions["monthly_income"]).abs()
    predictions["absolute_percentage_error"] = predictions["absolute_error"] / predictions["monthly_income"].where(
        predictions["monthly_income"] > 0)
    metrics = income_metrics(predictions)
    metrics["warnings"] = warnings
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(reports_dir / "income_predictions.csv", index=False, encoding="utf-8-sig")
    with open(reports_dir / "income_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return predictions, metrics
