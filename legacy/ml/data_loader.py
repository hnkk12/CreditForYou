"""
Data Loader Module for CreditForYou ML Pipeline.

Loads raw JSON and CSV datasets, verifies mappings, extracts aggregate
transaction metrics, and combines them into a unified raw dataset.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_raw_files(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets from the specified data directory.
    
    Returns:
        Tuple of DataFrames: (df_demo, df_id_map, df_newage, df_merged, df_tx)
    """
    demo_path = os.path.join(data_dir, "demographic_data.json")
    id_map_path = os.path.join(data_dir, "id_mapping.json")
    newage_path = os.path.join(data_dir, "new_age_sample_data.json")
    merged_path = os.path.join(data_dir, "merged_data.json")
    tx_path = os.path.join(data_dir, "transactional_data.csv")

    with open(demo_path, "r", encoding="utf-8") as f:
        df_demo = pd.DataFrame(json.load(f))

    with open(id_map_path, "r", encoding="utf-8") as f:
        df_id_map = pd.DataFrame(json.load(f))

    with open(newage_path, "r", encoding="utf-8") as f:
        df_newage = pd.DataFrame(json.load(f))

    with open(merged_path, "r", encoding="utf-8") as f:
        df_merged = pd.DataFrame(json.load(f))

    df_tx = pd.read_csv(tx_path)

    return df_demo, df_id_map, df_newage, df_merged, df_tx


def aggregate_transactional_data(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates granular transactional records per applicant_id.
    Derives behavioral metrics such as late transaction ratio, debit/credit ratio,
    and category-specific spending/income totals.
    """
    tx_copy = df_tx.copy()
    
    # Ensure status string matching
    tx_copy["is_late"] = (tx_copy["status"].astype(str).str.strip().str.lower() == "late").astype(int)
    tx_copy["is_debit"] = (tx_copy["type"].astype(str).str.strip().str.upper() == "DEBIT").astype(int)
    tx_copy["is_credit"] = (tx_copy["type"].astype(str).str.strip().str.upper() == "CREDIT").astype(int)

    # Category flags
    tx_copy["is_utility"] = (tx_copy["category"].astype(str).str.strip().str.lower() == "utility bill").astype(int)
    tx_copy["is_rent"] = (tx_copy["category"].astype(str).str.strip().str.lower() == "rent").astype(int)
    tx_copy["is_salary"] = (tx_copy["category"].astype(str).str.strip().str.lower() == "salary").astype(int)
    tx_copy["is_savings"] = (tx_copy["category"].astype(str).str.strip().str.lower() == "savings").astype(int)

    # Amounts by type
    tx_copy["debit_amount"] = tx_copy["amount"] * tx_copy["is_debit"]
    tx_copy["credit_amount"] = tx_copy["amount"] * tx_copy["is_credit"]
    tx_copy["salary_amount"] = tx_copy["amount"] * tx_copy["is_salary"]
    tx_copy["savings_amount"] = tx_copy["amount"] * tx_copy["is_savings"]

    agg_df = tx_copy.groupby("applicant_id").agg(
        tx_total_count=("transaction_id", "count"),
        tx_total_amount=("amount", "sum"),
        tx_avg_amount=("amount", "mean"),
        tx_late_count=("is_late", "sum"),
        tx_debit_amount=("debit_amount", "sum"),
        tx_credit_amount=("credit_amount", "sum"),
        tx_utility_count=("is_utility", "sum"),
        tx_rent_count=("is_rent", "sum"),
        tx_salary_total=("salary_amount", "sum"),
        tx_savings_total=("savings_amount", "sum"),
    ).reset_index()

    # Calculate ratios
    agg_df["tx_late_ratio"] = agg_df["tx_late_count"] / (agg_df["tx_total_count"] + 1e-5)
    agg_df["tx_debit_credit_ratio"] = agg_df["tx_debit_amount"] / (agg_df["tx_credit_amount"] + 1e-5)
    
    return agg_df


def load_combined_dataset(data_dir: str = ".") -> pd.DataFrame:
    """
    Main loader function: Loads and merges demographic, alternative credit, and 
    transactional datasets into a single raw DataFrame.
    """
    df_demo, df_id_map, df_newage, df_merged, df_tx = load_raw_files(data_dir)

    # Start with merged_data.json which contains demographic + new_age attributes
    df_base = df_merged.copy()

    # Aggregate transactional data
    df_tx_agg = aggregate_transactional_data(df_tx)

    # Merge base dataset with aggregated transactional data on applicant_id
    combined_df = pd.merge(df_base, df_tx_agg, on="applicant_id", how="left")

    # Fill any missing transaction metrics for applicants with no transactions
    tx_cols = [c for c in df_tx_agg.columns if c != "applicant_id"]
    for col in tx_cols:
        combined_df[col] = combined_df[col].fillna(0)

    print(f"[DataLoader] Successfully loaded combined dataset with {len(combined_df)} records and {combined_df.shape[1]} raw columns.")
    return combined_df


if __name__ == "__main__":
    df = load_combined_dataset(".")
    print("Columns:", list(df.columns))
    print("Head:\n", df.head(2))
