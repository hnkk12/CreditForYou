"""Bài toán kinh tế của danh mục hạn mức trả góp tại điểm bán (POP).

Trả lời câu hỏi của hội đồng tín dụng: cấp các hạn mức này thì
  - giải ngân bao nhiêu / năm, dư nợ bình quân bao nhiêu,
  - thu lãi + phí đối tác bao nhiêu, trừ chi phí vốn, vận hành, lỗ dự kiến còn lãi không,
  - tỷ lệ vỡ nợ tối đa bao nhiêu thì vẫn hoà vốn (break-even PD),
  - lãi trên vốn chủ sở hữu phân bổ (RoC),
dưới nhiều kịch bản vỡ nợ. Mọi tham số nằm trong config/scoring.json > economics và là GIẢ ĐỊNH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import scoring_config
from .limit import installment_for


def average_balance_factor(term_months: int, annual_rate: float) -> float:
    """Dư nợ bình quân (đầu mỗi kỳ) / gốc ban đầu của khoản trả góp đều."""
    r = annual_rate / 12
    pay = installment_for(1.0, term_months, annual_rate)
    bal, total = 1.0, 0.0
    for _ in range(term_months):
        total += bal
        bal = bal * (1 + r) - pay
    return total / term_months


def _traditional_eligible(df: pd.DataFrame) -> pd.Series:
    """Quy trình truyền thống: chỉ khách có lương/lương hưu cố định VÀ đã có lịch sử vay."""
    return df["main_income_type"].isin(["salary", "pension"]) & df["has_credit_history"].astype(bool)


def portfolio_economics(limits: pd.DataFrame) -> dict:
    eco = scoring_config()["economics"]
    pos = scoring_config()["pos_limit"]
    term = pos["term_months"]
    df = limits[limits["pos_limit"] > 0].copy()
    if df.empty:
        return {"so_khach_co_han_muc": 0}

    rate = (df["interest_rate_pct"].astype(float) / 100).fillna(pos["annual_rate"]) \
        if "interest_rate_pct" in df else pd.Series(pos["annual_rate"], index=df.index)
    df["annual_rate"] = rate
    df["financed"] = df["pos_limit"] * eco["annual_utilization"]
    df["avg_balance"] = [f * average_balance_factor(term, r) * term / 12 for f, r in zip(df["financed"], rate)]
    df["interest"] = df["avg_balance"] * df["annual_rate"]
    df["merchant_fee"] = df["financed"] * eco["merchant_discount_rate"]
    df["funding"] = df["avg_balance"] * eco["cost_of_funds"]
    loans_per_year = eco["annual_utilization"] / eco["avg_ticket_share_of_limit"]
    df["opex"] = loans_per_year * eco["cost_per_loan_automated"]
    df["capital"] = df["avg_balance"] * eco["capital_ratio"]
    df["margin_before_loss"] = df["interest"] + df["merchant_fee"] - df["funding"] - df["opex"]
    loss_base = df["financed"] * eco["lgd"] * eco["ead_share"]   # lỗ nếu 100% vỡ nợ

    def summarize(pd_series: pd.Series, mask=None) -> dict:
        d = df if mask is None else df[mask]
        p = pd_series if mask is None else pd_series[mask]
        el = (loss_base.loc[d.index] * p).sum()
        profit = d["margin_before_loss"].sum() - el
        return {
            "so_khach": int(len(d)),
            "giai_ngan_nam": round(d["financed"].sum(), 0),
            "du_no_binh_quan": round(d["avg_balance"].sum(), 0),
            "thu_lai": round(d["interest"].sum(), 0),
            "phi_doi_tac": round(d["merchant_fee"].sum(), 0),
            "chi_phi_von": round(d["funding"].sum(), 0),
            "chi_phi_van_hanh": round(d["opex"].sum(), 0),
            "lo_du_kien": round(el, 0),
            "loi_nhuan": round(profit, 0),
            "pd_binh_quan_theo_du_no": round(float(np.average(p, weights=d["financed"])), 4) if len(d) else 0,
            "RoC": round(profit / d["capital"].sum(), 4) if d["capital"].sum() else None,
            "bien_loi_nhuan_tren_giai_ngan": round(profit / d["financed"].sum(), 4) if d["financed"].sum() else None,
            "ty_le_lo_tren_giai_ngan": round(el / d["financed"].sum(), 4) if d["financed"].sum() else None,
        }

    scenarios = {}
    for name, by_tier in eco["pd_scenarios"].items():
        if name.startswith("_"):
            continue
        p = df["tier"].map(by_tier).astype(float)
        scenarios[name] = {"pd_theo_nhom": by_tier, **summarize(p)}
    if "pd_challenger" in df and df["pd_challenger"].notna().any():
        # PD Lending Club là xác suất vỡ nợ cả đời khoản vay ~36 tháng -> quy về kỳ hạn POP (giả định hazard đều)
        lc_term = eco["lending_club_reference_term_months"]
        p = 1 - (1 - df["pd_challenger"].clip(0, 1)) ** (term / lc_term)
        scenarios["theo_PD_Lending_Club"] = {
            "pd_theo_nhom": df.assign(p=p).groupby("tier")["p"].mean().round(4).to_dict(), **summarize(p)}

    # PD hoà vốn: margin trước lỗ = lỗ dự kiến
    breakeven = {}
    for tier, g in df.groupby("tier"):
        breakeven[tier] = round(float(g["margin_before_loss"].sum() / loss_base.loc[g.index].sum()), 4)
    breakeven["toan_danh_muc"] = round(float(df["margin_before_loss"].sum() / loss_base.sum()), 4)
    # tỷ lệ lỗ hoà vốn trên giải ngân: so sánh trực tiếp được với tỷ lệ xoá nợ (charge-off) của thị trường
    breakeven["ty_le_lo_hoa_von_tren_giai_ngan"] = round(float(df["margin_before_loss"].sum() / df["financed"].sum()), 4)
    # độ nhạy: +1 điểm % lãi suất / phí đối tác làm lợi nhuận tăng bao nhiêu
    sensitivity = {"them_1pp_lai_suat": round(float(df["avg_balance"].sum() * 0.01), 0),
                   "them_1pp_phi_doi_tac": round(float(df["financed"].sum() * 0.01), 0),
                   "them_1pp_chi_phi_von": round(float(-df["avg_balance"].sum() * 0.01), 0)}

    trad = _traditional_eligible(df)
    base_pd = df["tier"].map(eco["pd_scenarios"]["co_so"]).astype(float)
    target = (df.get("profile", pd.Series("", index=df.index)).isin(["freelancer", "entrepreneur"])
              | df["main_income_type"].eq("self_employed"))
    incremental = {
        "khach_mo_them_(truyen_thong_khong_duyet)": summarize(base_pd, ~trad),
        "trong_do_nhom_muc_tieu": summarize(base_pd, target & ~trad),
    }

    # Chi phí vận hành tránh được: hồ sơ trước đây phải người duyệt nay cấp tức thì (dạng giảm)
    avoided = int((limits["limit_status"] == "PREAPPROVED_REDUCED").sum())
    ops = {
        "ho_so_khong_con_phai_nguoi_duyet": avoided,
        "chi_phi_tranh_duoc_moi_lan_xet": round(avoided * (eco["cost_per_manual_review"] - eco["cost_per_loan_automated"]), 0),
    }
    return {
        "gia_dinh": {k: v for k, v in eco.items() if not k.startswith("_")},
        "ky_han_thang": term,
        "so_khach_co_han_muc": int(len(df)),
        "kich_ban": scenarios,
        "pd_hoa_von_theo_nhom": breakeven,
        "do_nhay": sensitivity,
        "kinh_doanh_mo_them": incremental,
        "van_hanh": ops,
    }


def print_economics(e: dict, currency="VND"):
    line = "=" * 100
    print(line)
    print(f" BÀI TOÁN KINH TẾ — danh mục hạn mức POP ({e['so_khach_co_han_muc']} khách có hạn mức, "
          f"kỳ hạn {e['ky_han_thang']} tháng, đơn vị {currency}/năm)")
    print(line)
    rows = {name: {k: v for k, v in s.items() if k != "pd_theo_nhom"} for name, s in e["kich_ban"].items()}
    table = pd.DataFrame(rows)
    fmt = table.copy().astype(object)
    for idx in table.index:
        for col in table.columns:
            v = table.loc[idx, col]
            if idx in ("RoC", "bien_loi_nhuan_tren_giai_ngan", "pd_binh_quan_theo_du_no", "ty_le_lo_tren_giai_ngan"):
                fmt.loc[idx, col] = f"{v:.1%}" if v is not None else "-"
            else:
                fmt.loc[idx, col] = f"{v:,.0f}"
    print(fmt.to_string())
    print("\n PD giả định theo nhóm (mỗi khoản 6 tháng):")
    for name, s in e["kich_ban"].items():
        print(f"   {name:22s}: {s['pd_theo_nhom']}")
    print("\n PD HOÀ VỐN (vỡ nợ tối đa mà danh mục vẫn không lỗ):")
    for tier, v in e["pd_hoa_von_theo_nhom"].items():
        print(f"   {tier:32s}: {v:.1%}")
    print(f"\n ĐỘ NHẠY (lợi nhuận/năm): {e['do_nhay']}")
    print("\n KINH DOANH MỞ THÊM (khách mà quy trình truyền thống không phục vụ được), kịch bản cơ sở:")
    for name, s in e["kinh_doanh_mo_them"].items():
        print(f"   {name:42s}: {s['so_khach']:>4} khách | giải ngân {s['giai_ngan_nam']:>10,.0f} | "
              f"lợi nhuận {s['loi_nhuan']:>9,.0f} | RoC {s['RoC']:.1%}" if s['RoC'] is not None else name)
    o = e["van_hanh"]
    print(f"\n VẬN HÀNH: {o['ho_so_khong_con_phai_nguoi_duyet']} hồ sơ không còn phải người duyệt "
          f"-> tránh được ~{o['chi_phi_tranh_duoc_moi_lan_xet']:,.0f} {currency} mỗi lượt xét")
