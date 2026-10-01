"""Lending Club target definition and leakage policy."""

from __future__ import annotations

import pandas as pd

FINAL_STATUS_MAP = {
    "fully paid": 0,
    "charged off": 1,
    "default": 1,
}

# Explicit names plus prefixes checked by ``assert_no_leakage``.
LEAKAGE_COLUMNS = {
    "grade", "sub_grade", "int_rate", "installment", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low",
}
LEAKAGE_PREFIXES = ("hardship_", "debt_settlement_", "settlement_")


def build_default_target(loan_status: pd.Series) -> pd.Series:
    """Map only resolved baseline outcomes; open/late statuses remain NaN."""
    normalized = loan_status.astype(str).str.strip().str.lower()
    return normalized.map(FINAL_STATUS_MAP).astype(float)


def target_counts(loan_status: pd.Series) -> dict:
    target = build_default_target(loan_status)
    return {
        "raw_status_counts": {str(k): int(v) for k, v in
                              loan_status.astype(str).value_counts(dropna=False).items()},
        "mapped_good": int((target == 0).sum()),
        "mapped_default": int((target == 1).sum()),
        "excluded_unresolved": int(target.isna().sum()),
    }


def assert_no_leakage(features) -> None:
    normalized = {str(c).lower() for c in features}
    bad = sorted(c for c in normalized
                 if c in LEAKAGE_COLUMNS or any(c.startswith(prefix) for prefix in LEAKAGE_PREFIXES))
    if bad:
        raise ValueError(f"Post-loan/platform-derived leakage features are forbidden: {bad}")
