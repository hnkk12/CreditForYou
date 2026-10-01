"""Dataset discovery and memory-safe ingestion for the two data populations.

The synthetic banking files and Lending Club are deliberately never joined at
row level.  This module only validates/loads each population independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import LENDING_CLUB_PATH, SYNTHETIC_DATA_DIR

SYNTHETIC_REQUIRED = {
    "customers": {"customer_id", "monthly_income"},
    "accounts": {"account_id", "customer_id"},
    "transactions": {"transaction_id", "account_id", "customer_id", "timestamp", "amount"},
    "loans": {"loan_id", "account_id", "customer_id", "monthly_payment", "status"},
}

TRANSACTION_COLUMNS = [
    "transaction_id", "account_id", "customer_id", "timestamp", "amount",
    "merchant_name", "merchant_category", "transaction_type", "label",
    "category", "subcategory", "is_salary", "is_transfer",
    "balance_after_transaction",
]


def synthetic_paths(data_dir: Path = SYNTHETIC_DATA_DIR) -> dict[str, Path]:
    data_dir = Path(data_dir)
    paths = {name: data_dir / f"{name}.csv" for name in SYNTHETIC_REQUIRED}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file synthetic banking: {missing}")
    return paths


def lending_club_path(path: Path = LENDING_CLUB_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy Lending Club dataset: {path}")
    return path


def load_synthetic_table(name: str, data_dir: Path = SYNTHETIC_DATA_DIR, **kwargs) -> pd.DataFrame:
    if name not in SYNTHETIC_REQUIRED:
        raise KeyError(f"Bảng synthetic không hỗ trợ: {name}")
    path = synthetic_paths(data_dir)[name]
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(SYNTHETIC_REQUIRED[name] - header)
    if missing:
        raise ValueError(f"{path.name} thiếu cột bắt buộc: {missing}")
    return pd.read_csv(path, **kwargs)


def iter_transactions(
    path: Path | None = None,
    customers: Iterable[str] | None = None,
    chunksize: int = 200_000,
    usecols: list[str] | None = None,
):
    """Yield transaction chunks, optionally filtered to selected customers."""
    path = Path(path) if path else synthetic_paths()["transactions"]
    wanted = None if customers is None else {str(v) for v in customers}
    columns = usecols or TRANSACTION_COLUMNS
    header = pd.read_csv(path, nrows=0).columns
    columns = [c for c in columns if c in header]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        if wanted is not None:
            chunk = chunk[chunk["customer_id"].astype(str).isin(wanted)]
        if not chunk.empty:
            yield chunk


def load_transactions(
    path: Path | None = None,
    customers: Iterable[str] | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    chunks = list(iter_transactions(path=path, customers=customers, chunksize=chunksize))
    if not chunks:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def existing_debt_by_customer(data_dir: Path = SYNTHETIC_DATA_DIR) -> pd.DataFrame:
    """Sum contractual payment for active loans; closed loans contribute zero."""
    loans = load_synthetic_table("loans", data_dir)
    active = loans[loans["status"].astype(str).str.lower().isin({"active", "outstanding"})].copy()
    active["monthly_payment"] = pd.to_numeric(active["monthly_payment"], errors="coerce").fillna(0.0)
    return (active.groupby("customer_id", as_index=False)["monthly_payment"].sum()
            .rename(columns={"monthly_payment": "existing_monthly_debt"}))


def validate_synthetic_joins(data_dir: Path = SYNTHETIC_DATA_DIR) -> dict:
    """Validate keys without ever loading raw transactions into SQLite."""
    customers = load_synthetic_table("customers", data_dir, usecols=["customer_id"])
    accounts = load_synthetic_table("accounts", data_dir, usecols=["account_id", "customer_id"])
    loans = load_synthetic_table("loans", data_dir,
                                 usecols=["loan_id", "account_id", "customer_id", "monthly_payment", "status"])
    customer_ids = set(customers["customer_id"].astype(str))
    account_ids = set(accounts["account_id"].astype(str))
    tx_customers, tx_accounts = set(), set()
    rows = duplicate_ids = bad_dates = bad_amounts = 0
    seen = set()
    for chunk in iter_transactions(data_dir / "transactions.csv",
                                   usecols=["transaction_id", "account_id", "customer_id", "timestamp", "amount"]):
        rows += len(chunk)
        ids = chunk["transaction_id"].astype(str)
        duplicate_ids += int(ids.duplicated().sum()) + sum(v in seen for v in ids.unique())
        seen.update(ids)
        tx_customers.update(chunk["customer_id"].astype(str))
        tx_accounts.update(chunk["account_id"].astype(str))
        bad_dates += int(pd.to_datetime(chunk["timestamp"], errors="coerce").isna().sum())
        bad_amounts += int(pd.to_numeric(chunk["amount"], errors="coerce").isna().sum())
    return {
        "transaction_rows": rows,
        "duplicate_transaction_ids": duplicate_ids,
        "unparseable_timestamps": bad_dates,
        "nonnumeric_amounts": bad_amounts,
        "accounts_customer_orphans": len(set(accounts["customer_id"].astype(str)) - customer_ids),
        "transactions_customer_orphans": len(tx_customers - customer_ids),
        "transactions_account_orphans": len(tx_accounts - account_ids),
        "loans_customer_orphans": len(set(loans["customer_id"].astype(str)) - customer_ids),
        "loans_account_orphans": len(set(loans["account_id"].astype(str)) - account_ids),
    }
