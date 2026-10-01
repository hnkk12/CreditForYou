"""In kết quả ra terminal + tính KPI theo bản ý tưởng (mục 5)."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import scoring_config

LINE = "=" * 78


def vnd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.0f} đ".replace(",", ".")


def print_income_table(income: pd.DataFrame, limit=20):
    cols = ["customer_id", "months_of_data", "estimated_monthly_income", "avg_monthly_inflow",
            "income_stability", "income_confidence", "estimation_method", "repayment_capacity",
            "excluded_inflow_share"]
    view = income[cols].head(limit).copy()
    for c in ("estimated_monthly_income", "avg_monthly_inflow"):
        view[c] = view[c].map(vnd)
    for c in ("income_stability", "income_confidence", "excluded_inflow_share"):
        view[c] = view[c].map(lambda v: f"{v:.0%}")
    print(view.to_string(index=False))
    if len(income) > limit:
        print(f"... ({len(income) - limit} khách hàng nữa trong file output)")


def print_application(r: pd.Series):
    w = scoring_config()["scoring"]["weights"]
    print(LINE)
    print(f" HỒ SƠ {r['customer_id']}")
    print(LINE)
    print(" [BƯỚC 1] Ước tính thu nhập từ giao dịch")
    print(f"   Thu nhập ước tính : {vnd(r['estimated_monthly_income'])}/tháng "
          f"(TB tiền vào {vnd(r.get('avg_monthly_inflow'))}, cách tính: {r.get('estimation_method', '-')})")
    print(f"   Độ ổn định        : {r['income_stability']:.0%}    Mức tin tưởng: {r['income_confidence']:.0%} "
          f"({int(r['months_of_data'])} tháng dữ liệu)")
    print(f"   Khả năng trả nợ   : {r.get('repayment_capacity', '-')}")
    print(" [BƯỚC 2] Chấm điểm tín dụng")
    items = [
        ("Ổn định thu nhập", "score_income_stability", f"{r['income_stability']:.0%} × {w['income_stability']}"),
        ("Tỷ lệ nợ/thu nhập", "score_debt_to_income",
         f"(1 - {vnd(r['monthly_installment'] + r['existing_monthly_debt'])} / {vnd(r['estimated_monthly_income'])}) × {w['debt_to_income']}"),
        ("Kinh nghiệm", "score_experience", f"{r.get('years_experience', 0) or 0:g} năm / 30 × {w['experience']}"),
        ("Lịch sử trả nợ", "score_repayment_history", f"{r['repayment_ratio']:.0%} × {w['repayment_history']}"),
        ("Mô hình ML", "score_ml", f"(1 - PD {r['pd_used']:.1%}) × {w['ml_probability']}"),
    ]
    for label, key, formula in items:
        print(f"   {label:18s}: {r[key]:6.2f}  <- {formula}")
    print(f"   {'TỔNG ĐIỂM':18s}: {r['credit_score']:6.2f} / 100")
    if r.get("score_notes"):
        print(f"   Ghi chú: {r['score_notes']}")
    print(" [BƯỚC 3] Quyết định")
    print(f"   Nhóm {r['tier']} ({r['tier_label']}) -> {r['decision_vn']}")
    if r["decision"] != "REJECT":
        print(f"   Lãi suất đề xuất  : {r['interest_rate_pct']}%/năm")
        if r.get("approved_amount") is not None and r["approved_amount"] == r["approved_amount"]:
            print(f"   Hạn mức           : {vnd(r['approved_amount'])} (đề nghị {vnd(r.get('requested_amount'))})")
    if r.get("decision_reasons"):
        print(f"   Lý do             : {r['decision_reasons']}")
    print(f"   Thời gian xử lý   : {r['processing_ms']:.2f} ms")


def kpi_summary(res: pd.DataFrame, total_seconds: float) -> dict:
    n = len(res)
    k = {"so_ho_so": n}
    k["phan_bo_nhom"] = res["tier"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0).to_dict()
    k["quyet_dinh"] = res["decision_vn"].value_counts().to_dict()
    k["ty_le_phe_duyet_tu_dong"] = float((res["decision"] == "AUTO_APPROVE").mean())
    k["ty_le_nhom_A_B"] = float(res["tier"].isin(["A", "B"]).mean())
    k["thoi_gian_tb_moi_ho_so_ms"] = total_seconds * 1000 / max(n, 1)

    if "declared_income" in res and res["declared_income"].notna().any():
        m = res[res["declared_income"] > 0]
        err = (m["estimated_monthly_income"] - m["declared_income"]).abs() / m["declared_income"]
        k["thu_nhap_ty_le_sai_lech_<=10%"] = float((err <= 0.10).mean())
        k["thu_nhap_ty_le_sai_lech_<=20%"] = float((err <= 0.20).mean())
        k["thu_nhap_MAPE"] = float(err.mean())
        k["thu_nhap_uoc_tinh_thap_hon_thuc_te"] = float((m["estimated_monthly_income"] < m["declared_income"]).mean())
        stable = m["income_stability"] >= scoring_config()["income"]["high_stability_threshold"]
        if stable.any():
            k["thu_nhap_<=10%_nhom_on_dinh"] = float((err[stable] <= 0.10).mean())
        if (~stable).any():
            k["thu_nhap_<=10%_nhom_bat_on_(co_y_bao_thu)"] = float((err[~stable] <= 0.10).mean())

    if "default" in res and res["default"].notna().any():
        m = res[res["default"].notna()]
        y = m["default"].astype(int)
        k["ty_le_vo_no_thuc_te"] = float(y.mean())
        k["ty_le_vo_no_theo_nhom"] = m.groupby("tier")["default"].mean().round(4).to_dict()
        if y.nunique() == 2:
            k["AUC_diem_tin_dung"] = float(roc_auc_score(y, -m["credit_score"]))
            k["Gini_diem_tin_dung"] = 2 * k["AUC_diem_tin_dung"] - 1
        pred_bad = m["tier"].isin(["C", "D"]).astype(int)
        k["do_chinh_xac_(C+D=xau)"] = float((pred_bad == y).mean())
        k["do_chinh_xac_(D=xau)"] = float((m["tier"].eq("D").astype(int) == y).mean())
        k["do_chinh_xac_baseline_(doan_tat_ca_tot)"] = float(1 - y.mean())
    return k


def print_kpis(k: dict):
    targets = {
        "ty_le_nhom_A_B": (">= 70%", lambda v: v >= 0.70),
        "thoi_gian_tb_moi_ho_so_ms": ("< 5000 ms", lambda v: v < 5000),
        "thu_nhap_ty_le_sai_lech_<=10%": (">= 90%", lambda v: v >= 0.90),
        "do_chinh_xac_(C+D=xau)": (">= 85%", lambda v: v >= 0.85),
    }
    print(LINE)
    print(" TỔNG KẾT & KPI")
    print(LINE)
    for key, val in k.items():
        if isinstance(val, float):
            shown = f"{val:.2f}" if key.endswith("_ms") else f"{val:.2%}" if abs(val) <= 1.5 else f"{val:.2f}"
        else:
            shown = str(val)
        mark = ""
        if key in targets:
            t, ok = targets[key]
            mark = f"   [mục tiêu {t}: {'ĐẠT' if ok(val) else 'CHƯA ĐẠT'}]"
        print(f"  {key:42s}: {shown}{mark}")
    tier_rates = k.get("ty_le_vo_no_theo_nhom", {})
    for tier, limit in (("A", 0.02), ("B", 0.05)):
        if tier in tier_rates:
            v = tier_rates[tier]
            print(f"  {'ty_le_vo_no_nhom_' + tier:42s}: {v:.2%}   [mục tiêu < {limit:.0%}: {'ĐẠT' if v < limit else 'CHƯA ĐẠT'}]")
    if "do_chinh_xac_(C+D=xau)" in k:
        print("  * Lưu ý: 'độ chính xác' dễ gây hiểu lầm khi tỷ lệ vỡ nợ thấp (đoán 'tất cả tốt' đã đạt baseline).")
        print("    Nên báo cáo thêm AUC/Gini và tỷ lệ vỡ nợ theo nhóm.")
