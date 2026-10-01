"""Income estimation and transaction cash-flow features.

This is a prototype heuristic, not a bank-validated income verification model.
Ground-truth ``customers.monthly_income`` is intentionally never accepted here.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import scoring_config
from .io import normalize_id, normalize_name, parse_amount, parse_dates, parse_direction


def _bool_series(s: pd.Series, default=False) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def _keyword_regex(keywords):
    kws = [normalize_name(k) for k in keywords if normalize_name(k)]
    if not kws:
        return None
    return r"(?:^|_)(?:" + "|".join(re.escape(k) for k in kws) + r")(?:_|$)"


def classify_income_transactions(df: pd.DataFrame, rich_schema: bool) -> pd.Series:
    """Classify eligible income without using customer ground truth."""
    eligible = ((df["inflow"] > 0) & ~df["is_excluded_inflow"] & ~df["is_transfer"])
    if not rich_schema:
        return eligible
    income_terms = _keyword_regex([
        "income", "salary", "payroll", "wage", "revenue", "business income",
        "freelance", "gig", "doanh thu", "thu nhap", "luong",
    ])
    text_signal = df["text"].str.contains(income_terms, regex=True) if income_terms else False
    credit_signal = df["transaction_type"].isin({"ach_credit", "credit", "deposit", "cash_deposit"})
    return eligible & (df["is_salary"] | text_signal | credit_signal)


def prepare_transactions(tx: pd.DataFrame, dayfirst=None):
    """Normalize mapped/generic or native ISB transactions for feature building."""
    cfg = scoring_config()["income"]
    warnings = []
    df = pd.DataFrame(index=tx.index)
    df["customer_id"] = normalize_id(tx["customer_id"])
    date_col = "date" if "date" in tx else "timestamp"
    df["date"] = parse_dates(tx[date_col], dayfirst=dayfirst)

    if "inflow" in tx.columns:
        df["inflow"] = parse_amount(tx["inflow"]).abs().fillna(0.0)
        df["outflow"] = parse_amount(tx["outflow"]).abs().fillna(0.0) if "outflow" in tx else 0.0
    else:
        amount = parse_amount(tx["amount"])
        if "direction" in tx.columns:
            direction = parse_direction(tx["direction"])
        elif "transaction_type" in tx.columns:
            direction = parse_direction(tx["transaction_type"])
        else:
            direction = pd.Series(np.nan, index=tx.index)
        signed = amount.where(direction.isna(), amount.abs() * direction)
        if direction.isna().all() and (amount.dropna() >= 0).all():
            warnings.append("KHÔNG có hướng giao dịch nhận diện được và mọi số tiền đều dương; "
                            "toàn bộ bị coi là TIỀN VÀO, cần kiểm tra schema.")
        df["inflow"] = signed.clip(lower=0).fillna(0.0)
        df["outflow"] = (-signed).clip(lower=0).fillna(0.0)

    text = pd.Series("", index=tx.index, dtype=object)
    for col in ("description", "label", "category", "subcategory", "merchant_category", "merchant_name"):
        if col in tx:
            text = text + " " + tx[col].fillna("").astype(str)
    df["text"] = text.map(normalize_name)
    df["transaction_type"] = (tx["transaction_type"].astype(str).map(normalize_name)
                              if "transaction_type" in tx else "")
    df["is_salary"] = _bool_series(tx["is_salary"]) if "is_salary" in tx else False
    df["is_transfer"] = _bool_series(tx["is_transfer"]) if "is_transfer" in tx else False
    df["balance"] = (pd.to_numeric(tx["balance_after_transaction"], errors="coerce")
                     if "balance_after_transaction" in tx else np.nan)
    source = None
    for col in ("merchant_name", "label", "description", "category"):
        if col in tx:
            candidate = tx[col].fillna("").astype(str).map(normalize_name)
            source = candidate if source is None else source.where(source.ne(""), candidate)
    df["income_source"] = source if source is not None else "unknown"

    keep = pd.Series(True, index=tx.index)
    if "status" in tx:
        bad_status = {normalize_name(s) for s in cfg["excluded_status"]}
        keep &= ~tx["status"].astype(str).map(normalize_name).isin(bad_status)
    bad_date = df["date"].isna()
    keep &= ~bad_date
    if bad_date.any():
        warnings.append(f"{bad_date.sum()} dòng không đọc được ngày -> bỏ qua.")
    df = df[keep].copy()

    excl_pattern = _keyword_regex(cfg["exclude_inflow_keywords"])
    df["is_excluded_inflow"] = df["text"].str.contains(excl_pattern, regex=True) if excl_pattern else False
    if "category" in tx:
        excluded_categories = {normalize_name(c) for c in cfg["exclude_inflow_categories"]}
        df["is_excluded_inflow"] |= tx.loc[df.index, "category"].astype(str).map(normalize_name).isin(excluded_categories)
    debt_pattern = _keyword_regex(cfg["debt_payment_keywords"])
    df["is_debt_payment"] = ((df["outflow"] > 0) & df["text"].str.contains(debt_pattern, regex=True)
                             if debt_pattern else False)
    rich_schema = any(c in tx.columns for c in ("is_salary", "is_transfer", "subcategory", "label"))
    df["is_income"] = classify_income_transactions(df, rich_schema=rich_schema)
    return df, warnings


def estimate_income(tx: pd.DataFrame) -> pd.DataFrame:
    """Return income/cash-flow features, one row per customer."""
    cfg = scoring_config()["income"]
    if tx.empty:
        return pd.DataFrame(columns=["customer_id", "active_months", "estimated_monthly_income"])
    tx = tx.copy()
    tx["m"] = tx["date"].dt.year * 12 + tx["date"].dt.month - 1
    span = tx.groupby("customer_id", sort=False)["date"].agg(first="min", last="max")
    start = span["first"].dt.year * 12 + span["first"].dt.month - 1
    end = span["last"].dt.year * 12 + span["last"].dt.month - 1
    if cfg.get("trim_partial_months", True):
        # Avoid treating a few boundary days as a zero/low-income calendar month.
        candidate_start = start + ((span["first"].dt.day > 7) & (end > start)).astype(int)
        candidate_end = end - ((span["last"].dt.day < span["last"].dt.days_in_month - 6)
                               & (end > candidate_start)).astype(int)
        enough = (candidate_end - candidate_start + 1) >= cfg["min_months_required"]
        start = start.where(~enough, candidate_start)
        end = end.where(~enough, candidate_end)
    start = np.maximum(start, end - cfg["lookback_months"] + 1)
    n_months = (end - start + 1).astype(int)

    grid = pd.DataFrame({"customer_id": np.repeat(span.index.values, n_months.values)})
    grid["m"] = np.concatenate([np.arange(s, e + 1) for s, e in zip(start, end)])
    bounds = pd.DataFrame({"customer_id": span.index, "_s": start.values, "_e": end.values})
    win = tx.merge(bounds, on="customer_id")
    win = win[(win["m"] >= win["_s"]) & (win["m"] <= win["_e"])].copy()
    # Salary is explicit income even with short history. Other external credits
    # must recur from the same normalized source before entering the estimate.
    candidates = win[win["is_income"]].copy()
    source_months = (candidates.groupby(["customer_id", "income_source"])["m"].nunique()
                     .rename("source_months").reset_index())
    win = win.merge(source_months, on=["customer_id", "income_source"], how="left")
    win["is_recurring_income"] = (win["is_income"] &
                                   (win["is_salary"] |
                                    (win["source_months"].fillna(0) >= cfg.get("recurring_min_months", 2))))
    win["eligible_income"] = win["inflow"].where(win["is_recurring_income"], 0.0)
    win["debt"] = win["outflow"].where(win["is_debt_payment"], 0.0)
    monthly = (win.groupby(["customer_id", "m"], as_index=False)
               .agg(income=("eligible_income", "sum"), inflow=("inflow", "sum"),
                    expense=("outflow", "sum"), debt=("debt", "sum"),
                    avg_balance=("balance", "mean"), min_balance=("balance", "min")))
    monthly["net"] = monthly["inflow"] - monthly["expense"]
    grid = grid.merge(monthly, on=["customer_id", "m"], how="left")
    for col in ("income", "inflow", "expense", "debt", "net"):
        grid[col] = grid[col].fillna(0.0)

    g = grid.groupby("customer_id", sort=False)
    out = pd.DataFrame(index=span.index)
    out["active_months"] = n_months
    out["months_of_data"] = n_months
    out["first_month"] = [f"{s // 12}-{s % 12 + 1:02d}" for s in start]
    out["last_month"] = [f"{e // 12}-{e % 12 + 1:02d}" for e in end]
    out["n_transactions"] = win.groupby("customer_id").size().reindex(span.index, fill_value=0)
    out["income_transaction_count"] = win.groupby("customer_id")["is_recurring_income"].sum().reindex(span.index, fill_value=0)
    out["avg_monthly_income"] = g["income"].mean()
    out["median_monthly_income"] = g["income"].median()
    out["min_monthly_income"] = g["income"].min()
    out["max_monthly_income"] = g["income"].max()
    out["std_monthly_income"] = g["income"].std(ddof=0)
    positive_income = grid[grid["income"] > 0].groupby("customer_id")["income"]
    percentile = cfg.get("estimated_income_positive_percentile", 10) / 100
    out["conservative_recurring_income"] = positive_income.quantile(percentile).reindex(span.index, fill_value=0.0)
    out["income_cv"] = out["std_monthly_income"] / out["avg_monthly_income"].where(out["avg_monthly_income"] > 0)
    out["income_stability_score"] = (1 - out["income_cv"]).clip(0, 1).fillna(0.0)
    out["income_stability"] = out["income_stability_score"]
    out["months_zero_income"] = g["income"].apply(lambda s: int((s <= 0).sum()))
    out["avg_monthly_expense"] = g["expense"].mean()
    out["avg_monthly_outflow"] = out["avg_monthly_expense"]
    out["expense_income_ratio"] = (out["avg_monthly_expense"] /
                                    out["avg_monthly_income"].where(out["avg_monthly_income"] > 0)).clip(upper=10)
    out["avg_monthly_net_cashflow"] = g["net"].mean()
    out["positive_cashflow_month_ratio"] = g["net"].apply(lambda s: float((s > 0).mean()))
    out["avg_balance"] = g["avg_balance"].mean()
    out["min_balance"] = g["min_balance"].min()
    out["cashflow_volatility"] = (g["net"].std(ddof=0) /
                                  out["avg_monthly_income"].where(out["avg_monthly_income"] > 0)).clip(upper=10)
    out["detected_monthly_debt"] = g["debt"].mean()

    income_tx = win[win["is_recurring_income"]].copy()
    source_stats = (income_tx.groupby(["customer_id", "income_source"])
                    .agg(amount=("eligible_income", "sum"), months=("m", "nunique"))).reset_index()
    recurring = source_stats[source_stats["months"] >= cfg.get("recurring_min_months", 2)]
    total_income = income_tx.groupby("customer_id")["eligible_income"].sum()
    recurring_income = recurring.groupby("customer_id")["amount"].sum()
    out["recurring_income_ratio"] = (recurring_income / total_income).reindex(span.index).fillna(0.0).clip(0, 1)
    out["income_source_count"] = source_stats.groupby("customer_id").size().reindex(span.index, fill_value=0)
    transfer_in = win[win["is_transfer"]].groupby("customer_id")["inflow"].sum()
    total_inflow = win.groupby("customer_id")["inflow"].sum()
    out["transfer_inflow_ratio"] = (transfer_in / total_inflow).reindex(span.index).fillna(0.0).clip(0, 1)
    eligible_total = total_income.reindex(span.index, fill_value=0)
    out["excluded_inflow_share"] = (1 - eligible_total / total_inflow.where(total_inflow > 0)).fillna(0.0).clip(0, 1)

    out["estimated_monthly_income"] = out["conservative_recurring_income"]
    out["estimation_method"] = f"p{int(percentile * 100)}_thu_nhap_dinh_ky_duong"
    out["income_confidence_score"] = 0.0
    for rule in sorted(cfg["confidence_by_months"], key=lambda r: r["min_months"]):
        out.loc[out["active_months"] >= rule["min_months"], "income_confidence_score"] = rule["confidence"]
    density = (out["income_transaction_count"] / out["active_months"].clip(lower=1)).clip(upper=1)
    quality = (1 - cfg.get("transfer_confidence_penalty", 0.5) * out["transfer_inflow_ratio"]).clip(0, 1)
    out["income_confidence_score"] = (out["income_confidence_score"] * density * quality).clip(0, 1)
    out["income_confidence"] = out["income_confidence_score"]
    out["income_confidence_label"] = pd.cut(
        out["income_confidence_score"], bins=[-0.01, 0.49, 0.69, 0.84, 1.0],
        labels=["Low", "Medium", "Good", "High"]).astype(str)
    out["repayment_capacity"] = np.select(
        [(out["estimated_monthly_income"] > 0) & (out["income_stability_score"] >= 0.8) &
         (out["income_confidence_score"] >= 0.85),
         (out["estimated_monthly_income"] > 0) & (out["income_confidence_score"] >= 0.6)],
        ["Cao", "Trung bình"], "Thấp")
    out["avg_monthly_inflow"] = out["avg_monthly_income"]
    out["income_std"] = out["std_monthly_income"]
    out["monthly_income_series"] = g.apply(
        lambda d: ";".join(f"{int(m) // 12}-{int(m) % 12 + 1:02d}:{v:.0f}"
                            for m, v in zip(d["m"], d["income"])), include_groups=False)

    money_cols = ["avg_monthly_income", "median_monthly_income", "min_monthly_income", "max_monthly_income",
                  "std_monthly_income", "estimated_monthly_income", "avg_monthly_expense",
                  "avg_monthly_outflow", "avg_monthly_net_cashflow", "avg_balance", "min_balance",
                  "detected_monthly_debt", "avg_monthly_inflow", "income_std", "conservative_recurring_income"]
    for col in money_cols:
        out[col] = out[col].round(2)
    ratio_cols = ["income_cv", "income_stability_score", "income_stability", "income_confidence_score",
                  "income_confidence", "recurring_income_ratio", "expense_income_ratio",
                  "positive_cashflow_month_ratio", "cashflow_volatility", "transfer_inflow_ratio",
                  "excluded_inflow_share"]
    for col in ratio_cols:
        out[col] = out[col].round(4)
    return out.rename_axis("customer_id").reset_index()
