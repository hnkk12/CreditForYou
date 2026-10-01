"""
Database Management Module for CreditForYou.

Provides SQLite connection management, database initialization, user authentication,
role-based access control (Lender vs Applicant), seeding, and persistence.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ml.data_loader import load_combined_dataset


DB_PATH = "creditforyou_legacy.db"


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass



def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Enables dict-like row access.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH, seed_data: bool = True):
    """
    Initializes SQLite tables for CreditForYou and seeds them with sample data and user accounts.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Users Table (Authentication & Role-Based Access Control)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('lender', 'applicant')),
        applicant_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Applicants Table (Demographic Info)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS applicants (
        applicant_id TEXT PRIMARY KEY,
        age INTEGER,
        monthly_income REAL,
        months_at_job INTEGER,
        housing TEXT,
        education TEXT,
        employment_status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 3. Financial Profiles Table (Transactional & Credit Behavior)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_profiles (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id TEXT UNIQUE,
        monthly_spend REAL,
        essential_pct REAL,
        cashflow_volatility REAL,
        savings_days INTEGER,
        on_time_rate REAL,
        dti REAL,
        credit_util REAL,
        delinq_90plus INTEGER,
        delinq_60plus INTEGER,
        delinq_30plus INTEGER,
        positive_habits INTEGER,
        risk_flags INTEGER,
        tx_late_ratio REAL,
        tx_debit_credit_ratio REAL,
        FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id)
    );
    """)

    # 4. Credit Evaluations Audit & Lender Decision Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS credit_evaluations (
        eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id TEXT,
        rule_credit_score INTEGER,
        pd REAL,
        ml_score INTEGER,
        risk_tier TEXT,
        risk_category TEXT,
        lender_status TEXT DEFAULT 'Pending Review',
        requested_loan_amount REAL DEFAULT 5000.0,
        lender_notes TEXT DEFAULT '',
        eligible_products_json TEXT,
        component_scores_json TEXT,
        eval_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id)
    );
    """)

    # 5. Product Catalog Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT,
        type TEXT,
        min_credit_score INTEGER,
        max_credit_score INTEGER,
        interest_rate TEXT,
        max_limit REAL
    );
    """)

    # 6. Payments & Credit Activation Audit Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        txn_id TEXT UNIQUE NOT NULL,
        applicant_id TEXT NOT NULL,
        product_name TEXT NOT NULL,
        product_type TEXT,
        interest_rate TEXT,
        payment_type TEXT NOT NULL,
        account_name TEXT,
        payment_detail TEXT,
        amount REAL DEFAULT 25.0,
        status TEXT DEFAULT 'PAID',
        payment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Auto-migrations for backwards compatibility
    for col_def in [
        "lender_status TEXT DEFAULT 'Pending Review'",
        "requested_loan_amount REAL DEFAULT 5000.0",
        "lender_notes TEXT DEFAULT ''"
    ]:
        try:
            cursor.execute(f"ALTER TABLE credit_evaluations ADD COLUMN {col_def};")
        except sqlite3.OperationalError:
            pass

    conn.commit()

    if seed_data:
        seed_database(conn)

    conn.close()
    safe_print(f"[Database] SQLite DB initialized successfully at '{db_path}'.")


def seed_database(conn: sqlite3.Connection, data_dir: str = "."):
    """
    Seeds SQLite database with raw data and default user authentication accounts.
    """
    cursor = conn.cursor()

    # Check if applicants table is already populated
    cursor.execute("SELECT COUNT(*) FROM applicants;")
    if cursor.fetchone()[0] > 0:
        safe_print("[Database] DB already contains data. Ensuring seed users exist.")
        seed_default_users(cursor)
        conn.commit()
        return

    safe_print("[Database] Seeding database from JSON/CSV files...")

    # Load combined raw dataset
    df_combined = load_combined_dataset(data_dir)

    for _, row in df_combined.iterrows():
        app_id = str(row["applicant_id"])

        cursor.execute("""
        INSERT OR REPLACE INTO applicants (applicant_id, age, monthly_income, months_at_job, housing, education, employment_status)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            app_id,
            int(row.get("age", 30)),
            float(row.get("monthly_income", 3000.0)),
            int(row.get("months_at_job", 12)),
            str(row.get("housing", "rent")),
            str(row.get("education", "bachelor")),
            str(row.get("employment_status", "employed"))
        ))

        cursor.execute("""
        INSERT OR REPLACE INTO financial_profiles (
            applicant_id, monthly_spend, essential_pct, cashflow_volatility, savings_days,
            on_time_rate, dti, credit_util, delinq_90plus, delinq_60plus, delinq_30plus,
            positive_habits, risk_flags, tx_late_ratio, tx_debit_credit_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            app_id,
            float(row.get("monthly_spend", 1500.0)),
            float(row.get("essential_pct", 0.6)),
            float(row.get("cashflow_volatility", 0.1)),
            int(row.get("savings_days", 30)),
            float(row.get("on_time_rate", 0.9)),
            float(row.get("dti", 0.3)),
            float(row.get("credit_util", 0.2)),
            int(row.get("delinq_90plus", 0)),
            int(row.get("delinq_60plus", 0)),
            int(row.get("delinq_30plus", 0)),
            int(row.get("positive_habits", 1)),
            int(row.get("risk_flags", 0)),
            float(row.get("tx_late_ratio", 0.0)),
            float(row.get("tx_debit_credit_ratio", 0.8))
        ))

    # Seed products catalog
    catalog_path = os.path.join(data_dir, "product_catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
            prod_list = catalog if isinstance(catalog, list) else catalog.get("products", [])
            for prod in prod_list:
                cursor.execute("""
                INSERT OR REPLACE INTO products (product_id, product_name, type, min_credit_score, max_credit_score, interest_rate, max_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    str(prod.get("product_id")),
                    str(prod.get("product_name")),
                    str(prod.get("type")),
                    int(prod.get("min_score", prod.get("min_credit_score", 350))),
                    int(prod.get("max_score", prod.get("max_credit_score", 1000))),
                    str(prod.get("interest_rate", "15%")),
                    float(prod.get("max_limit", 5000))
                ))

    seed_default_users(cursor)
    seed_initial_evaluations(conn)
    conn.commit()
    safe_print("[Database] Database successfully seeded!")


def seed_initial_evaluations(conn: sqlite3.Connection):
    """
    Generates initial credit evaluation audit records for any applicants that have not been evaluated yet.
    """
    cursor = conn.cursor()
    cursor.execute("""
    SELECT a.applicant_id, a.age, a.monthly_income, a.months_at_job, a.housing, a.education, a.employment_status,
           fp.monthly_spend, fp.essential_pct, fp.cashflow_volatility, fp.savings_days, fp.on_time_rate, fp.dti,
           fp.credit_util, fp.delinq_90plus, fp.delinq_60plus, fp.delinq_30plus, fp.positive_habits, fp.risk_flags,
           fp.tx_late_ratio, fp.tx_debit_credit_ratio
    FROM applicants a
    JOIN financial_profiles fp ON a.applicant_id = fp.applicant_id
    WHERE a.applicant_id NOT IN (SELECT DISTINCT applicant_id FROM credit_evaluations);
    """)
    rows = cursor.fetchall()
    if not rows:
        return

    from ml.predict import predict_credit_risk

    for r in rows:
        app_dict = dict(r)
        res = predict_credit_risk(app_dict)
        rule_score = int(res.get("rule_credit_score", 0))
        pd_val = float(res.get("pd", 0.0))
        ml_score = int(res.get("ml_score", 0))
        risk_tier = str(res.get("risk_info", {}).get("tier", "Unknown"))
        risk_category = str(res.get("risk_info", {}).get("category", "Unknown"))
        eligible_json = json.dumps(res.get("recommended_products", []))
        comp_json = json.dumps(res.get("component_scores", {}))

        status = "Pending Review"
        if rule_score >= 650:
            status = "Approved"
        elif rule_score < 350:
            status = "Rejected"

        cursor.execute("""
        INSERT INTO credit_evaluations (
            applicant_id, rule_credit_score, pd, ml_score, risk_tier, risk_category,
            lender_status, requested_loan_amount, lender_notes,
            eligible_products_json, component_scores_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            app_dict["applicant_id"], rule_score, pd_val, ml_score, risk_tier, risk_category,
            status, 5000.0, "Initial Seed Evaluation",
            eligible_json, comp_json
        ))
    conn.commit()


def seed_default_users(cursor: sqlite3.Cursor):
    """
    Seeds default Lender and Applicant user accounts for instant sign-in.
    """
    default_users = [
        ("lender@creditforyou.vn", "lender123", "lender", None),
        ("lender@bank.com", "lender123", "lender", None),
        ("user@creditforyou.vn", "user123", "applicant", "8a0d6320-e982-445b-afa8-bb9e0054aca9"),
        ("alex@gmail.com", "user123", "applicant", "111ff86d-d582-4503-baac-78ebb7932c94")
    ]

    for email, password, role, app_id in default_users:
        cursor.execute("""
        INSERT OR IGNORE INTO users (email, password, role, applicant_id)
        VALUES (?, ?, ?, ?);
        """, (email, password, role, app_id))


def authenticate_user(email: str, password: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user by email and password.
    Returns user record dict with role ('lender' or 'applicant') or None if invalid.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM users WHERE LOWER(email) = LOWER(?) AND password = ?;
    """, (email.strip(), password.strip()))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def register_user(email: str, password: str, role: str = "applicant", applicant_id: Optional[str] = None, db_path: str = DB_PATH) -> bool:
    """
    Registers a new user account in SQLite DB.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO users (email, password, role, applicant_id)
        VALUES (?, ?, ?, ?);
        """, (email.strip().lower(), password, role, applicant_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def save_credit_evaluation(
    applicant_id: str,
    result_dict: Dict[str, Any],
    lender_status: str = "Pending Review",
    requested_amount: float = 5000.0,
    lender_notes: str = "",
    db_path: str = DB_PATH
):
    """
    Logs an applicant's credit risk evaluation result into SQLite database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rule_score = int(result_dict.get("rule_credit_score", 0))
    pd_val = float(result_dict.get("pd", 0.0))
    ml_score = int(result_dict.get("ml_score", 0))
    risk_tier = str(result_dict.get("risk_info", {}).get("tier", "Unknown"))
    risk_category = str(result_dict.get("risk_info", {}).get("category", "Unknown"))

    eligible_json = json.dumps(result_dict.get("recommended_products", []))
    comp_json = json.dumps(result_dict.get("component_scores", {}))

    if lender_status == "Pending Review":
        if rule_score < 350:
            lender_status = "Rejected"

    cursor.execute("""
    INSERT INTO credit_evaluations (
        applicant_id, rule_credit_score, pd, ml_score, risk_tier, risk_category,
        lender_status, requested_loan_amount, lender_notes,
        eligible_products_json, component_scores_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        applicant_id, rule_score, pd_val, ml_score, risk_tier, risk_category,
        lender_status, requested_amount, lender_notes,
        eligible_json, comp_json
    ))

    conn.commit()
    conn.close()


def get_latest_evaluation_for_applicant(applicant_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves the most recent credit evaluation record for an applicant.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM credit_evaluations
    WHERE applicant_id = ?
    ORDER BY eval_timestamp DESC, eval_id DESC
    LIMIT 1;
    """, (applicant_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)



def update_lender_decision(
    eval_id: int,
    new_status: str,
    lender_notes: str = "",
    db_path: str = DB_PATH
) -> bool:
    """
    Updates the lender decision (Approved, Disbursed, Rejected, Pending Review) and notes in SQLite DB.
    Guarantees persistence across all evaluation records for the target applicant.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Retrieve target applicant_id
    cursor.execute("SELECT applicant_id FROM credit_evaluations WHERE eval_id = ?;", (eval_id,))
    row = cursor.fetchone()

    if row:
        app_id = row["applicant_id"]
        cursor.execute("""
        UPDATE credit_evaluations
        SET lender_status = ?, lender_notes = ?
        WHERE applicant_id = ?;
        """, (new_status, lender_notes, app_id))
    else:
        cursor.execute("""
        UPDATE credit_evaluations
        SET lender_status = ?, lender_notes = ?
        WHERE eval_id = ?;
        """, (new_status, lender_notes, eval_id))

    conn.commit()
    conn.close()
    return True


def get_applicant_from_db(applicant_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves full applicant demographic & financial profile from SQLite database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT a.*, fp.* 
    FROM applicants a
    JOIN financial_profiles fp ON a.applicant_id = fp.applicant_id
    WHERE a.applicant_id = ?;
    """, (applicant_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_lender_filtered_applications(
    min_score: int = 0,
    max_score: int = 1000,
    min_income: float = 0.0,
    max_dti: float = 1.0,
    risk_tiers: Optional[List[str]] = None,
    lender_statuses: Optional[List[str]] = None,
    db_path: str = DB_PATH
) -> pd.DataFrame:
    """
    Advanced SQL filtering function for Lenders to query applicants & evaluations.
    """
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        ce.eval_id,
        ce.applicant_id,
        ce.rule_credit_score,
        ce.ml_score,
        ce.pd,
        ce.risk_tier,
        ce.lender_status,
        ce.requested_loan_amount,
        a.age,
        a.monthly_income,
        a.employment_status,
        a.housing,
        fp.dti,
        fp.credit_util,
        fp.savings_days,
        fp.on_time_rate,
        ce.lender_notes,
        ce.eval_timestamp
    FROM credit_evaluations ce
    JOIN applicants a ON ce.applicant_id = a.applicant_id
    JOIN financial_profiles fp ON ce.applicant_id = fp.applicant_id
    WHERE ce.rule_credit_score >= ? AND ce.rule_credit_score <= ?
      AND a.monthly_income >= ?
      AND fp.dti <= ?
    """

    params: List[Any] = [min_score, max_score, min_income, max_dti]

    if risk_tiers and len(risk_tiers) > 0:
        placeholders = ",".join(["?"] * len(risk_tiers))
        query += f" AND ce.risk_tier IN ({placeholders})"
        params.extend(risk_tiers)

    if lender_statuses and len(lender_statuses) > 0:
        placeholders = ",".join(["?"] * len(lender_statuses))
        query += f" AND ce.lender_status IN ({placeholders})"
        params.extend(lender_statuses)

    query += " ORDER BY ce.eval_timestamp DESC;"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_evaluation_history(applicant_id: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves evaluation audit log history for a given applicant.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM credit_evaluations
    WHERE applicant_id = ?
    ORDER BY eval_timestamp DESC;
    """, (applicant_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def upsert_applicant_profile(applicant_data: Dict[str, Any], db_path: str = DB_PATH):
    """
    Inserts or updates an applicant's demographic & financial profile in SQLite DB.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    app_id = str(applicant_data.get("applicant_id", "APP_WEB_001"))

    cursor.execute("""
    INSERT OR REPLACE INTO applicants (applicant_id, age, monthly_income, months_at_job, housing, education, employment_status)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (
        app_id,
        int(applicant_data.get("age", 30)),
        float(applicant_data.get("monthly_income", 5000.0)),
        int(applicant_data.get("months_at_job", 36)),
        str(applicant_data.get("housing", "owner")),
        str(applicant_data.get("education", "bachelor")),
        str(applicant_data.get("employment_status", "employed"))
    ))

    cursor.execute("""
    INSERT OR REPLACE INTO financial_profiles (
        applicant_id, monthly_spend, essential_pct, cashflow_volatility, savings_days,
        on_time_rate, dti, credit_util, delinq_90plus, delinq_60plus, delinq_30plus,
        positive_habits, risk_flags, tx_late_ratio, tx_debit_credit_ratio
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        app_id,
        float(applicant_data.get("monthly_spend", 2000.0)),
        float(applicant_data.get("essential_pct", 0.65)),
        float(applicant_data.get("cashflow_volatility", 0.08)),
        int(applicant_data.get("savings_days", 120)),
        float(applicant_data.get("on_time_rate", 0.95)),
        float(applicant_data.get("dti", 0.25)),
        float(applicant_data.get("credit_util", 0.20)),
        int(applicant_data.get("delinq_90plus", 0)),
        int(applicant_data.get("delinq_60plus", 0)),
        int(applicant_data.get("delinq_30plus", 0)),
        int(applicant_data.get("positive_habits", 2)),
        int(applicant_data.get("risk_flags", 0)),
        float(applicant_data.get("tx_late_ratio", 0.02)),
        float(applicant_data.get("tx_debit_credit_ratio", 0.8))
    ))

    conn.commit()
    conn.close()


def save_payment_record(
    txn_id: str,
    applicant_id: str,
    product_name: str,
    product_type: str,
    interest_rate: str,
    payment_type: str,
    account_name: str,
    payment_detail: str,
    amount: float = 25.0,
    status: str = "PAID",
    db_path: str = DB_PATH
) -> bool:
    """
    Saves a completed payment / credit activation transaction to SQLite database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT OR REPLACE INTO payments (
            txn_id, applicant_id, product_name, product_type, interest_rate,
            payment_type, account_name, payment_detail, amount, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            txn_id, applicant_id, product_name, product_type, interest_rate,
            payment_type, account_name, payment_detail, amount, status
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        safe_print(f"[Database Error] Failed to save payment record: {e}")
        conn.close()
        return False


def get_payment_history(applicant_id: Optional[str] = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Retrieves stored payment transactions as a pandas DataFrame.
    """
    conn = get_db_connection(db_path)

    if applicant_id:
        query = "SELECT * FROM payments WHERE applicant_id = ? ORDER BY payment_timestamp DESC;"
        df = pd.read_sql_query(query, conn, params=[applicant_id])
    else:
        query = "SELECT * FROM payments ORDER BY payment_timestamp DESC;"
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


if __name__ == "__main__":
    init_db()
    user = authenticate_user("lender@creditforyou.vn", "lender123")
    safe_print("\nAuthenticated Lender User:", user)
