"""
CreditForYou Rule-Based "New-Age" Credit Scoring Engine.

Implements the exact 0-1000 points Rule-Based Framework defined by BNP Paribas:
  Score = Lifestyle + Spending + Repayment Discipline + Bonus/Penalty
Bounded strictly between 0 and 1000 points.

Components:
1. Lifestyle (Max 350 pts):
   - 1.1 Employment Stability (0-150)
   - 1.2 Housing Status (0-80)
   - 1.3 Digital Footprint (0-70)
   - 1.4 Education / Skill Level (0-50)

2. Spending Behavior (Max 350 pts):
   - 2.1 Spend-to-Income Ratio (0-120)
   - 2.2 Expense Diversity (0-80)
   - 2.3 Cash-flow Volatility (0-70)
   - 2.4 Savings / Emergency Fund (0-80)

3. Repayment Discipline (Max 570 pts):
   - 3.1 On-time Payment Rate (0-200)
   - 3.2 Debt-to-Income Ratio (0-120)
   - 3.3 Credit Utilization (0-100)
   - 3.4 Recent Delinquency Severity (0-150)

4. Bonus & Penalty Adjustments:
   - 4.1 Positive Financial Behavior (+20 per habit, max +50)
   - 4.2 Risk Flags (-20 per flag, max -50)
"""

import os
import json
from typing import Dict, Any, List, Union, Tuple


def load_product_catalog(data_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Loads financial product catalog.
    """
    path = os.path.join(data_dir, "product_catalog.json")
    if not os.path.exists(path):
        return [
            {"product_id": "P_01", "product_name": "Starter Credit Card", "min_score": 350, "type": "Credit Card", "interest_rate": "12% APR"},
            {"product_id": "P_02", "product_name": "Micro-Loan Lite", "min_score": 550, "type": "Loan", "interest_rate": "17% PA"},
            {"product_id": "P_03", "product_name": "Standard Credit Card", "min_score": 650, "type": "Credit Card", "interest_rate": "18% APR"},
            {"product_id": "P_04", "product_name": "Premium Personal Loan", "min_score": 750, "type": "Loan", "interest_rate": "11% PA"}
        ]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_recommended_products(score: int, data_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Filters eligible products based on rule credit score.
    """
    catalog = load_product_catalog(data_dir)
    return [p for p in catalog if score >= p.get("min_score", 0)]


def evaluate_lifestyle_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Computes Lifestyle Component score (0-350 points).
    """
    breakdown = {}

    # 1.1 Employment Stability (0-150 pts)
    months = float(data.get("months_at_job", 0))
    if months >= 24:
        emp_pts = 150
        emp_reason = ">= 24 months employment tenure (+150 pts)"
    elif months >= 12:
        emp_pts = 100
        emp_reason = "12-23 months employment tenure (+100 pts)"
    elif months >= 6:
        emp_pts = 50
        emp_reason = "6-11 months employment tenure (+50 pts)"
    else:
        emp_pts = 0
        emp_reason = "< 6 months employment tenure (+0 pts)"
    breakdown["1.1_employment_stability"] = {"score": emp_pts, "max": 150, "reason": emp_reason}

    # 1.2 Housing Status (0-80 pts)
    housing = str(data.get("housing", "")).lower().strip()
    rent_on_time_months = float(data.get("rent_on_time_months", 0))
    if housing in ["owner", "own home", "owned"]:
        house_pts = 80
        house_reason = "Home owner (+80 pts)"
    elif housing in ["rent", "rented"]:
        if rent_on_time_months >= 12 or float(data.get("on_time_rate", 0.9)) >= 0.90:
            house_pts = 60
            house_reason = "Renting with >= 12 mo verified on-time rent payments (+60 pts)"
        else:
            house_pts = 30
            house_reason = "Renting with < 12 mo payment history (+30 pts)"
    else:
        house_pts = 0
        house_reason = "No verified stable address (+0 pts)"
    breakdown["1.2_housing_status"] = {"score": house_pts, "max": 80, "reason": house_reason}

    # 1.3 Digital Footprint (0-70 pts)
    dig_rate = float(data.get("digital_payment_rate", data.get("on_time_rate", 0.80)))
    dig_pct_str = f"{int(round(dig_rate * 100))}%"
    if dig_rate >= 0.95:
        dig_pts = 70
        dig_reason = f">= 95% digital payment regularity ({dig_pct_str}) (+70 pts)"
    elif dig_rate >= 0.80:
        dig_pts = 45
        dig_reason = f"80-94% digital payment regularity ({dig_pct_str}) (+45 pts)"
    elif dig_rate >= 0.60:
        dig_pts = 20
        dig_reason = f"60-79% digital payment regularity ({dig_pct_str}) (+20 pts)"
    else:
        dig_pts = 0
        dig_reason = f"< 60% digital payment regularity ({dig_pct_str}) (+0 pts)"
    breakdown["1.3_digital_footprint"] = {"score": dig_pts, "max": 70, "reason": dig_reason}

    # 1.4 Education / Skill Level (0-50 pts)
    edu = str(data.get("education", "")).lower().strip()
    if edu in ["master", "master's", "phd", "doctorate"]:
        edu_pts = 50
        edu_reason = "Master's degree / PhD (+50 pts)"
    elif edu in ["bachelor", "bachelor's", "undergraduate"]:
        edu_pts = 40
        edu_reason = "Bachelor's degree (+40 pts)"
    elif edu in ["cert", "professional cert", "diploma"]:
        edu_pts = 30
        edu_reason = "Professional certification / Diploma (+30 pts)"
    elif edu in ["highschool", "high school", "secondary"]:
        edu_pts = 20
        edu_reason = "High school graduate (+20 pts)"
    else:
        edu_pts = 0
        edu_reason = "< High School credential (+0 pts)"
    breakdown["1.4_education_skill"] = {"score": edu_pts, "max": 50, "reason": edu_reason}

    total_lifestyle = emp_pts + house_pts + dig_pts + edu_pts
    return total_lifestyle, breakdown


def evaluate_spending_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Computes Spending Behavior Component score (0-350 points).
    """
    breakdown = {}

    # 2.1 Spend-to-Income Ratio (0-120 pts)
    income = float(data.get("monthly_income", 1000))
    spend = float(data.get("monthly_spend", 500))
    spend_income_ratio = spend / (income + 1e-5) if income > 0 else 1.0
    sti_pct_str = f"{int(round(spend_income_ratio * 100))}%"

    if spend_income_ratio <= 0.30:
        sti_pts = 120
        sti_reason = f"Spend-to-income ratio <= 30% ({sti_pct_str}) (+120 pts)"
    elif spend_income_ratio <= 0.50:
        sti_pts = 80
        sti_reason = f"Spend-to-income ratio 30-50% ({sti_pct_str}) (+80 pts)"
    elif spend_income_ratio <= 0.70:
        sti_pts = 40
        sti_reason = f"Spend-to-income ratio 50-70% ({sti_pct_str}) (+40 pts)"
    else:
        sti_pts = 0
        sti_reason = f"Spend-to-income ratio > 70% ({sti_pct_str}) (+0 pts)"
    breakdown["2.1_spend_to_income"] = {"score": sti_pts, "max": 120, "reason": sti_reason}

    # 2.2 Expense Diversity (0-80 pts)
    essential_pct = float(data.get("essential_pct", 0.50))
    ess_pct_str = f"{int(round(essential_pct * 100))}%"
    if essential_pct >= 0.70:
        exp_pts = 80
        exp_reason = f"Essentials expenditure >= 70% ({ess_pct_str}) (+80 pts)"
    elif essential_pct >= 0.55:
        exp_pts = 45
        exp_reason = f"Essentials expenditure 55-69% ({ess_pct_str}) (+45 pts)"
    elif essential_pct >= 0.40:
        exp_pts = 20
        exp_reason = f"Essentials expenditure 40-54% ({ess_pct_str}) (+20 pts)"
    else:
        exp_pts = 0
        exp_reason = f"Essentials expenditure < 40% ({ess_pct_str}) (+0 pts)"
    breakdown["2.2_expense_diversity"] = {"score": exp_pts, "max": 80, "reason": exp_reason}

    # 2.3 Cash-flow Volatility (0-70 pts)
    cf_vol = float(data.get("cashflow_volatility", 0.10))
    vol_pct_str = f"{int(round(cf_vol * 100))}%"
    if cf_vol <= 0.05:
        cf_pts = 70
        cf_reason = f"Cash-flow volatility <= 5% ({vol_pct_str}) (+70 pts)"
    elif cf_vol <= 0.10:
        cf_pts = 40
        cf_reason = f"Cash-flow volatility 5-10% ({vol_pct_str}) (+40 pts)"
    elif cf_vol <= 0.20:
        cf_pts = 15
        cf_reason = f"Cash-flow volatility 10-20% ({vol_pct_str}) (+15 pts)"
    else:
        cf_pts = 0
        cf_reason = f"Cash-flow volatility > 20% ({vol_pct_str}) (+0 pts)"
    breakdown["2.3_cashflow_volatility"] = {"score": cf_pts, "max": 70, "reason": cf_reason}

    # 2.4 Savings / Emergency Fund (0-80 pts)
    savings_days = float(data.get("savings_days", 30))
    if savings_days >= 180:
        sav_pts = 80
        sav_reason = f"Savings reserve >= 180 days ({int(savings_days)} days) (+80 pts)"
    elif savings_days >= 90:
        sav_pts = 50
        sav_reason = f"Savings reserve 90-179 days ({int(savings_days)} days) (+50 pts)"
    elif savings_days >= 30:
        sav_pts = 20
        sav_reason = f"Savings reserve 30-89 days ({int(savings_days)} days) (+20 pts)"
    else:
        sav_pts = 0
        sav_reason = f"Savings reserve < 30 days ({int(savings_days)} days) (+0 pts)"
    breakdown["2.4_savings_emergency_fund"] = {"score": sav_pts, "max": 80, "reason": sav_reason}

    total_spending = sti_pts + exp_pts + cf_pts + sav_pts
    return total_spending, breakdown


def evaluate_repayment_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Computes Repayment Discipline Component score (0-570 points).
    """
    breakdown = {}

    # 3.1 On-time Payment Rate (0-200 pts)
    on_time = float(data.get("on_time_rate", 0.90))
    ontime_pct_str = f"{int(round(on_time * 100))}%"
    if on_time >= 0.98:
        ontime_pts = 200
        ontime_reason = f"On-time payment rate >= 98% ({ontime_pct_str}) (+200 pts)"
    elif on_time >= 0.95:
        ontime_pts = 150
        ontime_reason = f"On-time payment rate 95-97% ({ontime_pct_str}) (+150 pts)"
    elif on_time >= 0.90:
        ontime_pts = 100
        ontime_reason = f"On-time payment rate 90-94% ({ontime_pct_str}) (+100 pts)"
    elif on_time >= 0.80:
        ontime_pts = 50
        ontime_reason = f"On-time payment rate 80-89% ({ontime_pct_str}) (+50 pts)"
    else:
        ontime_pts = 0
        ontime_reason = f"On-time payment rate < 80% ({ontime_pct_str}) (+0 pts)"
    breakdown["3.1_ontime_payment_rate"] = {"score": ontime_pts, "max": 200, "reason": ontime_reason}

    # 3.2 Debt-to-Income Ratio (0-120 pts)
    dti = float(data.get("dti", 0.25))
    dti_pct_str = f"{int(round(dti * 100))}%"
    if dti <= 0.20:
        dti_pts = 120
        dti_reason = f"Debt-to-income ratio <= 20% ({dti_pct_str}) (+120 pts)"
    elif dti <= 0.35:
        dti_pts = 80
        dti_reason = f"Debt-to-income ratio 21-35% ({dti_pct_str}) (+80 pts)"
    elif dti <= 0.50:
        dti_pts = 40
        dti_reason = f"Debt-to-income ratio 36-50% ({dti_pct_str}) (+40 pts)"
    else:
        dti_pts = 0
        dti_reason = f"Debt-to-income ratio > 50% ({dti_pct_str}) (+0 pts)"
    breakdown["3.2_debt_to_income"] = {"score": dti_pts, "max": 120, "reason": dti_reason}

    # 3.3 Credit Utilization (0-100 pts)
    util = float(data.get("credit_util", 0.20))
    util_pct_str = f"{int(round(util * 100))}%"
    if util <= 0.10:
        util_pts = 100
        util_reason = f"Credit utilization <= 10% ({util_pct_str}) (+100 pts)"
    elif util <= 0.30:
        util_pts = 70
        util_reason = f"Credit utilization 11-30% ({util_pct_str}) (+70 pts)"
    elif util <= 0.50:
        util_pts = 30
        util_reason = f"Credit utilization 31-50% ({util_pct_str}) (+30 pts)"
    else:
        util_pts = 0
        util_reason = f"Credit utilization > 50% ({util_pct_str}) (+0 pts)"
    breakdown["3.3_credit_utilization"] = {"score": util_pts, "max": 100, "reason": util_reason}

    # 3.4 Recent Delinquency Severity (0-150 pts)
    delinq_90 = int(data.get("delinq_90plus", 0))
    delinq_60 = int(data.get("delinq_60plus", 0))
    delinq_30 = int(data.get("delinq_30plus", 0))

    if delinq_90 > 0 or (delinq_30 + delinq_60 + delinq_90) > 1:
        delinq_pts = 0
        delinq_reason = "Multiple delinquencies or >= 90-day missed payment (+0 pts)"
    elif delinq_60 > 0:
        delinq_pts = 50
        delinq_reason = "1-60 day missed payment recorded (+50 pts)"
    elif delinq_30 > 0:
        delinq_pts = 100
        delinq_reason = "1-30 day missed payment recorded (+100 pts)"
    else:
        delinq_pts = 150
        delinq_reason = "Zero delinquencies in last 24 months (+150 pts)"
    breakdown["3.4_delinquency_severity"] = {"score": delinq_pts, "max": 150, "reason": delinq_reason}

    total_repayment = ontime_pts + dti_pts + util_pts + delinq_pts
    return total_repayment, breakdown


def evaluate_adjustments_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Computes Bonus (+50 max) and Penalty (-50 max) Adjustments.
    """
    breakdown = {}

    # 4.1 Positive Financial Behavior (+20 per habit, max +50)
    habits = int(data.get("positive_habits", 0))
    bonus_pts = min(50, habits * 20)
    bonus_reason = f"{habits} qualifying positive financial habits (+{bonus_pts} pts, max +50)"
    breakdown["4.1_positive_behavior_bonus"] = {"score": bonus_pts, "max": 50, "reason": bonus_reason}

    # 4.2 Risk Flags (-20 per flag, max -50)
    flags = int(data.get("risk_flags", 0))
    penalty_pts = min(50, flags * 20)
    penalty_reason = f"{flags} risk flags identified (-{penalty_pts} pts, max -50)"
    breakdown["4.2_risk_flags_penalty"] = {"score": -penalty_pts, "max": -50, "reason": penalty_reason}

    net_adj = bonus_pts - penalty_pts
    return net_adj, breakdown


def get_risk_tier(score: int) -> Dict[str, str]:
    """
    Returns risk categorization metadata based on score.
    """
    if score >= 750:
        return {
            "tier": "Excellent",
            "category": "Low Risk",
            "color": "Green",
            "recommendation_level": "Tier 1 Prime",
            "description": "Prime candidate. High creditworthiness profile with minimal risk."
        }
    elif score >= 600:
        return {
            "tier": "Good",
            "category": "Moderate Risk",
            "color": "Blue",
            "recommendation_level": "Tier 2 Standard",
            "description": "Solid credit candidate. Qualified for standard credit cards and micro-loans."
        }
    elif score >= 450:
        return {
            "tier": "Fair",
            "category": "Moderate Risk",
            "color": "Yellow",
            "recommendation_level": "Tier 3 Starter",
            "description": "Fair credit history. Qualified for starter credit lines with monitoring."
        }
    else:
        return {
            "tier": "Bad",
            "category": "High Risk",
            "color": "Red",
            "recommendation_level": "Restricted",
            "description": "Vulnerable financial profile. Requires builder/secured credit intervention."
        }


def compute_rule_based_credit_score(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates exact Rule-Based Credit Score (0-1000 pts) and Factor Analysis.
    """
    app_id = str(applicant_data.get("applicant_id", "UNKNOWN"))

    lifestyle_score, lifestyle_breakdown = evaluate_lifestyle_score(applicant_data)
    spending_score, spending_breakdown = evaluate_spending_score(applicant_data)
    repayment_score, repayment_breakdown = evaluate_repayment_score(applicant_data)
    adj_score, adj_breakdown = evaluate_adjustments_score(applicant_data)

    raw_score = lifestyle_score + spending_score + repayment_score + adj_score
    final_score = max(0, min(1000, int(round(raw_score))))

    subfactor_breakdown = {}
    subfactor_breakdown.update(lifestyle_breakdown)
    subfactor_breakdown.update(spending_breakdown)
    subfactor_breakdown.update(repayment_breakdown)
    subfactor_breakdown.update(adj_breakdown)

    factor_analysis = []
    for sf_key, info in subfactor_breakdown.items():
        factor_analysis.append({
            "subfactor": sf_key,
            "points": info["score"],
            "max_points": info["max"],
            "narrative": info["reason"]
        })

    risk_info = get_risk_tier(final_score)
    recommended_products = get_recommended_products(final_score)

    return {
        "applicant_id": app_id,
        "rule_credit_score": final_score,
        "raw_points": raw_score,
        "risk_info": risk_info,
        "component_scores": {
            "lifestyle": lifestyle_score,
            "spending_behavior": spending_score,
            "repayment_discipline": repayment_score,
            "adjustments": adj_score
        },
        "subfactor_breakdown": subfactor_breakdown,
        "factor_analysis": factor_analysis,
        "recommended_products": recommended_products
    }



def run_what_if_simulation(applicant_data: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """
    Executes What-If Simulations as specified in the hackathon document (Page 5):
    1. 'saving_boost': Save extra Rs 1000/mo for 3 months (+35 pts)
    2. 'autopay_builder': Set up auto-pay for rent & utility bills (+60 pts)
    3. 'discretionary_spike': Spend 75% income on non-essentials (-50 pts)
    4. 'delinquency_spike': Miss internet/utility by 45 days & delay rent by 15 days (-110 pts)
    """
    base_res = compute_rule_based_credit_score(applicant_data)
    base_score = base_res["rule_credit_score"]

    mod_data = applicant_data.copy()

    if scenario == "saving_boost":
        # Increases savings_days and reduces volatility
        mod_data["savings_days"] = float(mod_data.get("savings_days", 30)) + 90
        mod_data["cashflow_volatility"] = max(0.02, float(mod_data.get("cashflow_volatility", 0.15)) * 0.5)
        description = "Save extra Rs 1000/mo for next 3 months"
    elif scenario == "autopay_builder":
        # Maximizes on-time payment rate and rent consistency
        mod_data["on_time_rate"] = 0.99
        mod_data["rent_on_time_months"] = max(12, int(mod_data.get("rent_on_time_months", 6)) + 6)
        mod_data["digital_payment_rate"] = 0.98
        description = "Set up auto-pay for rent and utility bills"
    elif scenario == "discretionary_spike":
        # Lowers expense diversity and worsens spend-to-income
        mod_data["essential_pct"] = 0.25
        income = float(mod_data.get("monthly_income", 3000))
        mod_data["monthly_spend"] = income * 0.75
        description = "Spend 75% of monthly income on non-essentials for 2 consecutive months"
    elif scenario == "delinquency_spike":
        # Triggers delinquency penalty & slashes on-time rate
        mod_data["delinq_30plus"] = int(mod_data.get("delinq_30plus", 0)) + 1
        mod_data["delinq_60plus"] = int(mod_data.get("delinq_60plus", 0)) + 1
        mod_data["on_time_rate"] = 0.65
        description = "Miss internet/utility bill by 45 days and delay rent by 15 days"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    sim_res = compute_rule_based_credit_score(mod_data)
    sim_score = sim_res["rule_credit_score"]
    score_delta = sim_score - base_score

    return {
        "scenario": scenario,
        "description": description,
        "baseline_score": base_score,
        "simulated_score": sim_score,
        "score_delta": score_delta,
        "baseline_risk_tier": base_res["risk_info"]["tier"],
        "simulated_risk_tier": sim_res["risk_info"]["tier"]
    }


if __name__ == "__main__":
    test_applicant = {
        "applicant_id": "TEST_APP_001",
        "months_at_job": 36,
        "housing": "owner",
        "rent_on_time_months": 12,
        "digital_payment_rate": 0.96,
        "education": "master",
        "monthly_income": 6500,
        "monthly_spend": 2100,
        "essential_pct": 0.72,
        "cashflow_volatility": 0.04,
        "savings_days": 200,
        "on_time_rate": 0.99,
        "dti": 0.18,
        "credit_util": 0.08,
        "delinq_90plus": 0,
        "delinq_60plus": 0,
        "delinq_30plus": 0,
        "positive_habits": 3,
        "risk_flags": 0
    }

    result = compute_rule_based_credit_score(test_applicant)
    print("Rule-Based Credit Score Result:\n", json.dumps(result, indent=2))

    print("\n--- What-If Simulations ---")
    for scen in ["saving_boost", "autopay_builder", "discretionary_spike", "delinquency_spike"]:
        sim = run_what_if_simulation(test_applicant, scen)
        print(f"Scenario [{scen}]: Base={sim['baseline_score']} -> Sim={sim['simulated_score']} (Delta={sim['score_delta']:+d})")
