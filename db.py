"""SQLite persistence for prototype assessments (raw transactions stay in CSV)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from creditforyou.config import DATA_DIR

DEFAULT_DB_PATH = DATA_DIR / "creditforyou.db"


def connect(path=DEFAULT_DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path=DEFAULT_DB_PATH):
    with connect(path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS applications (
            application_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            requested_loan_amount REAL NOT NULL,
            term_months INTEGER NOT NULL,
            purpose TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS income_assessments (
            application_id TEXT PRIMARY KEY,
            estimated_income REAL NOT NULL,
            income_stability REAL,
            income_confidence REAL,
            cashflow_json TEXT NOT NULL,
            FOREIGN KEY(application_id) REFERENCES applications(application_id)
        );
        CREATE TABLE IF NOT EXISTS credit_predictions (
            application_id TEXT PRIMARY KEY,
            probability_of_default REAL,
            model_version TEXT,
            FOREIGN KEY(application_id) REFERENCES applications(application_id)
        );
        CREATE TABLE IF NOT EXISTS credit_scores (
            application_id TEXT PRIMARY KEY,
            post_loan_debt_ratio REAL,
            credit_score REAL NOT NULL,
            tier TEXT NOT NULL,
            decision TEXT NOT NULL,
            component_json TEXT NOT NULL,
            reason_codes TEXT NOT NULL,
            FOREIGN KEY(application_id) REFERENCES applications(application_id)
        );
        """)
    return Path(path)


def save_assessment(row, profile=None, path=DEFAULT_DB_PATH):
    init_db(path)
    application_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    components = {k: row.get(k) for k in (
        "score_income_stability", "score_debt_to_income", "score_experience",
        "score_repayment_history", "score_ml", "available_weight", "missing_components")}
    cashflow = {k: row.get(k) for k in (
        "avg_monthly_expense", "expense_income_ratio", "avg_monthly_net_cashflow",
        "positive_cashflow_month_ratio", "cashflow_volatility", "recurring_income_ratio")}
    with connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO customers VALUES (?, ?, ?)",
                     (str(row["customer_id"]), json.dumps(profile or {}, default=str), now))
        conn.execute("INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?)",
                     (application_id, str(row["customer_id"]), float(row["requested_amount"]),
                      int(row["term_months"]), str(row["purpose"]), now))
        conn.execute("INSERT INTO income_assessments VALUES (?, ?, ?, ?, ?)",
                     (application_id, float(row["estimated_monthly_income"]), row.get("income_stability_score"),
                      row.get("income_confidence_score"), json.dumps(cashflow, default=str)))
        conn.execute("INSERT INTO credit_predictions VALUES (?, ?, ?)",
                     (application_id, row.get("pd_used"), row.get("model_version")))
        conn.execute("INSERT INTO credit_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (application_id, row.get("post_loan_debt_ratio"), float(row["credit_score"]),
                      str(row["tier"]), str(row["decision"]), json.dumps(components, default=str),
                      str(row.get("reason_codes", ""))))
    return application_id


def assessment_history(customer_id=None, path=DEFAULT_DB_PATH):
    init_db(path)
    query = """SELECT a.*, i.estimated_income, i.income_stability, i.income_confidence,
                      p.probability_of_default, p.model_version, s.post_loan_debt_ratio,
                      s.credit_score, s.tier, s.decision, s.reason_codes
               FROM applications a
               JOIN income_assessments i USING(application_id)
               JOIN credit_predictions p USING(application_id)
               JOIN credit_scores s USING(application_id)"""
    args = []
    if customer_id is not None:
        query += " WHERE a.customer_id = ?"
        args.append(str(customer_id))
    query += " ORDER BY a.created_at DESC"
    with connect(path) as conn:
        return [dict(row) for row in conn.execute(query, args).fetchall()]
