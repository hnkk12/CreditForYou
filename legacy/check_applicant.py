"""
CreditForYou Simple Credit Eligibility Checker.

Queries applicant profile from SQLite database (or dataset fallback), evaluates:
1. TOTAL CREDIT SCORE (0 - 1000)
2. ELIGIBILITY STATUS (APPROVED / REJECTED)
3. PRE-APPROVED PRODUCT OFFERS
4. SCORE BREAKDOWN BY FACTOR
Persists evaluation audit trail to SQLite DB.
"""

import os
import json
import sys
from ml.predict import predict_credit_risk
from ml.data_loader import load_combined_dataset
from db import init_db, get_applicant_from_db, save_credit_evaluation, get_evaluation_history


def load_applicant(applicant_id: str, data_dir: str = ".") -> dict:
    """
    Looks up applicant attributes from SQLite database (or fallback dataset).
    """
    init_db()
    db_data = get_applicant_from_db(applicant_id)
    if db_data is not None:
        return db_data

    # Fallback to dataset lookup
    df = load_combined_dataset(data_dir)
    matches = df[df["applicant_id"].astype(str) == str(applicant_id)]
    if len(matches) == 0:
        raise KeyError(f"Applicant ID '{applicant_id}' not found in database or dataset.")
    return matches.iloc[0].to_dict()


def check_applicant_eligibility(applicant_data: dict, save_to_db: bool = True) -> dict:
    """
    Evaluates applicant score, risk status, product approval eligibility, and logs to DB.
    """
    result = predict_credit_risk(applicant_data)

    score = result["rule_credit_score"]
    risk_info = result["risk_info"]
    products = result["recommended_products"]

    is_approved = score >= 350
    status = "APPROVED" if is_approved else "REJECTED"

    applicant_id = str(result["applicant_id"])

    # Persist evaluation to database audit log
    if save_to_db:
        save_credit_evaluation(applicant_id, result)

    summary = {
        "applicant_id": applicant_id,
        "total_credit_score": f"{score} / 1000",
        "eligibility_status": status,
        "risk_tier": risk_info["tier"],
        "risk_category": risk_info["category"],
        "eligible_products_count": len(products),
        "pre_approved_products": [
            f"{p['product_name']} ({p['type']} @ {p['interest_rate']})" for p in products
        ] if is_approved else ["None - Score below minimum eligibility threshold (350)"],
        "component_score_summary": result["component_scores"],
        "factor_narratives": [f["narrative"] for f in result["factor_analysis"]]
    }
    return summary


def print_eligibility_report(summary: dict):
    print("\n" + "="*75)
    print(f"      CREDITFORYOU APPLICANT DECISION REPORT - {summary['applicant_id']}")
    print("="*75)
    print(f" TOTAL CREDIT SCORE    : {summary['total_credit_score']}")
    print(f" ELIGIBILITY STATUS    : {summary['eligibility_status']}")
    print(f" RISK TIER             : {summary['risk_tier']} ({summary['risk_category']})")
    print(f" ELIGIBLE PRODUCTS     : {summary['eligible_products_count']} product(s) available")
    print("-" * 75)
    print(" PRE-APPROVED OFFERS:")
    for prod in summary["pre_approved_products"]:
        print(f"   - {prod}")
    print("-" * 75)
    print(" SCORE BREAKDOWN BY COMPONENT:")
    for comp, pts in summary["component_score_summary"].items():
        print(f"   - {comp.replace('_', ' ').title():22s}: {pts} pts")
    print("-" * 75)
    print(" KEY CONTRIBUTING FACTORS (EXPLAINABLE AI):")
    for factor in summary["factor_narratives"]:
        print(f"   - {factor}")
    print("="*75 + "\n")


if __name__ == "__main__":
    init_db()
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        try:
            app_data = load_applicant(target_id)
            report = check_applicant_eligibility(app_data)
            print_eligibility_report(report)
        except KeyError as e:
            print(f"Error: {e}")
    else:
        # Evaluate sample dataset applicant from SQLite DB
        sample_id = "8a0d6320-e982-445b-afa8-bb9e0054aca9"
        print(f"\n[Database Query] Checking applicant ID: {sample_id}...\n")
        
        app_data = load_applicant(sample_id)
        report = check_applicant_eligibility(app_data)
        print_eligibility_report(report)

        # Show DB evaluation audit history
        history = get_evaluation_history(sample_id)
        print(f"[Database Audit Trail] Found {len(history)} past evaluation record(s) for applicant '{sample_id}'.")

