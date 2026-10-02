import pandas as pd
import pytest

from creditforyou.config import scoring_config
from creditforyou.economics import average_balance_factor, portfolio_economics


def _limits():
    rows = []
    for i, tier in enumerate(["A", "A", "B", "B", "C"]):
        rows.append({"customer_id": f"C{i}", "tier": tier, "pos_limit": 1000.0 * (i + 1),
                     "interest_rate_pct": {"A": 9.5, "B": 12.0, "C": 15.0}[tier],
                     "main_income_type": "self_employed" if i == 4 else "salary",
                     "has_credit_history": i < 2, "profile": "freelancer" if i == 4 else "manager",
                     "limit_status": "PREAPPROVED_REDUCED" if tier == "C" else "PREAPPROVED",
                     "pd_challenger": 0.2})
    rows.append({**rows[0], "customer_id": "none", "pos_limit": 0.0, "limit_status": "NOT_ELIGIBLE"})
    return pd.DataFrame(rows)


def test_average_balance_factor():
    # trả góp đều 6 kỳ, lãi 0: dư nợ đầu kỳ 6/6, 5/6, ..., 1/6 -> bình quân 7/12
    assert average_balance_factor(6, 0.0) == pytest.approx(7 / 12)
    assert 7 / 12 < average_balance_factor(6, 0.18) < 0.6


def test_portfolio_economics_consistency():
    e = portfolio_economics(_limits())
    assert e["so_khach_co_han_muc"] == 5                      # khách không có hạn mức bị loại
    base, stress = e["kich_ban"]["co_so"], e["kich_ban"]["cang_thang_x2"]
    assert stress["lo_du_kien"] == pytest.approx(2 * base["lo_du_kien"], abs=1)   # số đã làm tròn
    assert base["loi_nhuan"] == pytest.approx(base["thu_lai"] + base["phi_doi_tac"] - base["chi_phi_von"]
                                              - base["chi_phi_van_hanh"] - base["lo_du_kien"], abs=2)
    assert "theo_PD_Lending_Club" in e["kich_ban"]
    # PD hoà vốn: thay PD = break-even cho mọi nhóm -> lợi nhuận ~ 0
    eco = scoring_config()["economics"]
    be = e["pd_hoa_von_theo_nhom"]["toan_danh_muc"]
    margin = base["thu_lai"] + base["phi_doi_tac"] - base["chi_phi_von"] - base["chi_phi_van_hanh"]
    loss_if_all_default = base["giai_ngan_nam"] * eco["lgd"] * eco["ead_share"]
    assert margin - be * loss_if_all_default == pytest.approx(0, abs=3)
    # khách mở thêm = khách quy trình truyền thống (lương + lịch sử vay) không duyệt
    assert e["kinh_doanh_mo_them"]["khach_mo_them_(truyen_thong_khong_duyet)"]["so_khach"] == 3
    assert e["kinh_doanh_mo_them"]["trong_do_nhom_muc_tieu"]["so_khach"] == 1
    assert e["van_hanh"]["ho_so_khong_con_phai_nguoi_duyet"] == 1
