"""In kết quả ra terminal + tính KPI theo bản ý tưởng (mục 5)."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import scoring_config

LINE = "=" * 78


def vnd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.0f}".replace(",", ".")


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


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def print_application(r: pd.Series, currency: str = "đ"):
    """In chi tiết 1 hồ sơ. Trần điểm mỗi thành phần là trần SAU chuẩn hoá (khi có thành phần bị thiếu)."""
    cfg = scoring_config()["scoring"]
    caps = r.get("normalized_component_max")
    caps = caps if isinstance(caps, dict) else {}
    cap_key = {"score_income_stability": "income_stability", "score_debt_to_income": "debt_to_income",
               "score_experience": "experience", "score_repayment_history": "repayment_history",
               "score_ml": "ml"}
    money = lambda x: f"{vnd(x)} {currency}"  # noqa: E731
    print(LINE)
    print(f" HỒ SƠ {r['customer_id']}")
    print(LINE)
    print(" [BƯỚC 1] Ước tính thu nhập từ giao dịch")
    print(f"   Thu nhập ước tính : {money(r['estimated_monthly_income'])}/tháng "
          f"(TB thu nhập định kỳ {money(r.get('avg_monthly_inflow'))}, cách tính: {r.get('estimation_method', '-')})")
    if not _missing(r.get("actual_monthly_income")):
        actual = r["actual_monthly_income"]
        err = (r["estimated_monthly_income"] - actual) / actual if actual else float("nan")
        print(f"   [Đánh giá] Thu nhập thực tế: {money(actual)}/tháng -> sai lệch {err:+.1%}")
    print(f"   Độ ổn định        : {r['income_stability']:.0%}    Mức tin tưởng: {r['income_confidence']:.0%} "
          f"({int(r['months_of_data'])} tháng dữ liệu, {r.get('first_month', '?')} -> {r.get('last_month', '?')})")
    if not _missing(r.get("expense_income_ratio")):
        print(f"   Chi tiêu/thu nhập : {r['expense_income_ratio']:.0%}")
    print(f"   Khả năng trả nợ   : {r.get('repayment_capacity', '-')}")

    print(" [BƯỚC 2] Chấm điểm tín dụng")
    tenure = r.get("financial_tenure_years")
    years, years_label = (tenure, "năm là khách hàng") if not _missing(tenure) else (r.get("years_experience"), "năm kinh nghiệm")
    debt_total = (r.get("monthly_installment") or 0) + (r.get("existing_monthly_debt") or 0)
    formulas = {
        "score_income_stability": f"độ ổn định {r['income_stability']:.0%}",
        "score_debt_to_income": "không đủ dữ liệu" if _missing(r.get("dti")) else
            f"1 - ({money(debt_total)} / {money(r['estimated_monthly_income'])}) = 1 - {r['dti']:.0%}",
        "score_experience": "không đủ dữ liệu" if _missing(years) else
            f"min({years:.1f} {years_label} / {cfg['experience_full_years']}, 1)",
        "score_repayment_history": "không đủ dữ liệu -> tắt thành phần" if _missing(r.get("repayment_ratio"))
            else f"tỷ lệ trả đúng hạn {r['repayment_ratio']:.0%}",
        "score_ml": (f"1 - PD {r['pd_used']:.1%}" if not _missing(r.get("pd_used"))
                     else f"tắt (PD challenger {r['pd_challenger']:.1%} chỉ để so sánh)"
                     if not _missing(r.get("pd_challenger")) else "chưa có mô hình -> tắt thành phần"),
    }
    labels = {"score_income_stability": "Ổn định thu nhập", "score_debt_to_income": "Tỷ lệ nợ/thu nhập",
              "score_experience": "Kinh nghiệm/thâm niên", "score_repayment_history": "Lịch sử trả nợ",
              "score_ml": "Mô hình ML (PD)"}
    for key, label in labels.items():
        cap = caps.get(cap_key[key])
        cap_txt = f"/{cap:5.2f}" if cap is not None else "/  -  "
        print(f"   {label:22s}: {r[key]:6.2f}{cap_txt}  <- {formulas[key]}")
    print(f"   {'TỔNG ĐIỂM':22s}: {r['credit_score']:6.2f}/100")
    if r.get("score_notes"):
        print(f"   Ghi chú: {r['score_notes']}")

    print(" [BƯỚC 3] Quyết định")
    print(f"   Nhóm {r['tier']} ({r['tier_label']}) -> {r['decision_vn']}")
    if r["decision"] != "REJECT":
        print(f"   Lãi suất đề xuất  : {r['interest_rate_pct']}%/năm")
        if not _missing(r.get("approved_amount")):
            print(f"   Hạn mức           : {money(r['approved_amount'])} (đề nghị {money(r.get('requested_amount'))})")
    if r.get("decision_reasons"):
        print(f"   Lý do             : {r['decision_reasons']}")
    if r.get("reason_codes"):
        print(f"   Giải thích        : {r['reason_codes']}")
    print(f"   Thời gian chấm điểm: {r['processing_ms']:.2f} ms")


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


# --------------------------------------------------------------------------- hạn mức trả góp tại điểm bán

TARGET_PROFILES = ("freelancer", "entrepreneur")


def _target_mask(res: pd.DataFrame) -> pd.Series:
    by_profile = res["profile"].isin(TARGET_PROFILES) if "profile" in res else pd.Series(False, index=res.index)
    by_income = res["main_income_type"].eq("self_employed") if "main_income_type" in res \
        else pd.Series(False, index=res.index)
    return by_profile | by_income


def print_limit_detail(r: pd.Series, tickets, currency="VND"):
    from .limit import checkout_decision, fx
    k = fx()   # số tiền gốc của dataset -> VND; hạn mức đã ở VND
    money = lambda x: f"{vnd(x)} {currency}"  # noqa: E731
    conv = lambda x: f"{vnd(None if _missing(x) else x * k)} {currency}"  # noqa: E731
    print(LINE)
    print(f" KHÁCH {r['customer_id']}  ({r.get('profile', '-')}, nguồn thu chính: {r.get('main_income_type', '-')})")
    print(LINE)
    print(" [A] CẤP HẠN MỨC TRƯỚC")
    actual = r.get("actual_monthly_income")
    print(f"   Thu nhập ước tính (ròng)      : {conv(r['estimated_monthly_income'])}/tháng"
          + (f"   [thực tế {conv(actual)}]" if not _missing(actual) else ""))
    print(f"   Thu nhập tiền mặt bảo thủ     : {conv(r['conservative_cash_income'])}/tháng "
          f"(gộp {conv(r['gross_monthly_income'])}, ổn định {r['income_stability']:.0%})")
    print(f"   Chi phí bắt buộc / nợ đang trả: {conv(r['avg_monthly_committed_expense'])} / "
          f"{conv(r['debt_service_monthly'])}")
    print(f"   Khả năng trả góp mới/tháng    : {money(r['monthly_capacity'])}  "
          f"(chặn bởi: {r['capacity_binding_constraint']})")
    print(f"   Điểm / nhóm                   : {r['credit_score']:.1f} / {r['tier']}   "
          f"tin cậy thu nhập {r['income_confidence']:.0%}")
    print(f"   => {r['limit_status_vn'].upper()}: hạn mức {money(r['pos_limit'])}  "
          f"(giới hạn bởi: {r['limit_binding_cap']})")
    if r.get("limit_reasons"):
        print(f"   Lý do: {r['limit_reasons']}")
    if not _missing(r.get("pd_challenger")):
        print(f"   [Tham khảo] PD Lending Club (challenger, không dùng để quyết định): {r['pd_challenger']:.1%}")
    print(" [B] TẠI QUẦY (giả lập)")
    for t in tickets:
        d = checkout_decision(r["pos_limit"], t, available_limit_max_term=r["pos_limit_max_term"])
        extra = f", trả góp {money(d['installment'])}/tháng" if d.get("installment") else ""
        print(f"   Mua {money(t):>12s} -> {d['checkout_decision']:13s} {d['message']}{extra}")


def limit_summary(res: pd.DataFrame, tickets) -> dict:
    from .limit import checkout_decision
    has_limit = res["pos_limit"] > 0
    target = _target_mask(res)
    k = {"so_khach": len(res),
         "trang_thai_han_muc": res["limit_status"].value_counts().to_dict(),
         "ty_le_co_han_muc": float(has_limit.mean()),
         "han_muc_trung_vi_(khach_co_han_muc)": float(res.loc[has_limit, "pos_limit"].median()) if has_limit.any() else 0.0,
         "ho_so_can_nguoi_duyet_tai_quay": 0,
         "ho_so_xem_xet_ngoai_gio": int((res["limit_status"] == "OFFLINE_REVIEW").sum()),
         "nhom_muc_tieu_so_khach": int(target.sum()),
         "nhom_muc_tieu_ty_le_co_han_muc": float(has_limit[target].mean()) if target.any() else None}
    sims = {}
    for t in tickets:
        s = pd.Series([checkout_decision(l, t, available_limit_max_term=lm)["checkout_decision"]
                       for l, lm in zip(res["pos_limit"], res["pos_limit_max_term"])], index=res.index)
        sims[t] = {"tat_ca": s.value_counts(normalize=True).round(3).to_dict(),
                   "nhom_muc_tieu": s[target].value_counts(normalize=True).round(3).to_dict() if target.any() else {}}
    k["gia_lap_tai_quay"] = sims
    return k


def print_limit_summary(res: pd.DataFrame, tickets, elapsed: float):
    print(LINE)
    print(" CẤP HẠN MỨC TRẢ GÓP TẠI ĐIỂM BÁN — TỔNG KẾT")
    print(LINE)
    if "profile" in res:
        g = res.groupby("profile")
        table = pd.DataFrame({
            "so_khach": g.size(),
            "co_han_muc_%": (g["pos_limit"].apply(lambda s: (s > 0).mean()) * 100).round(0),
            "han_muc_tv": g["pos_limit"].apply(lambda s: s[s > 0].median() if (s > 0).any() else 0).round(0),
            "cap_du": g["limit_status"].apply(lambda s: int((s == "PREAPPROVED").sum())),
            "cap_giam": g["limit_status"].apply(lambda s: int((s == "PREAPPROVED_REDUCED").sum())),
            "xem_ngoai_gio": g["limit_status"].apply(lambda s: int((s == "OFFLINE_REVIEW").sum())),
            "khong_du_dk": g["limit_status"].apply(lambda s: int((s == "NOT_ELIGIBLE").sum())),
        })
        if "actual_monthly_income" in res:
            ape = (res["estimated_monthly_income"] - res["actual_monthly_income"]).abs() / res["actual_monthly_income"]
            table["thu_nhap_sai<=10%_%"] = (ape.groupby(res["profile"]).apply(lambda s: (s <= 0.1).mean()) * 100).round(0)
        table = table.sort_values("so_khach", ascending=False)
        table.index = [f"* {i}" if i in TARGET_PROFILES else i for i in table.index]
        print(table.to_string())
        print("  (* = nhóm khách mục tiêu của đề bài; han_muc_tv = hạn mức trung vị của khách có hạn mức)\n")
    k = limit_summary(res, tickets)
    for key, val in k.items():
        if key == "gia_lap_tai_quay":
            continue
        if isinstance(val, float):
            shown = f"{val:.1%}" if val <= 1 else f"{val:,.0f}"
        else:
            shown = val
        print(f"  {key:42s}: {shown}")
    print(f"  {'thoi_gian_cap_han_muc_tb_moi_khach_ms':42s}: {elapsed * 1000 / max(len(res), 1):.1f}")
    print("\n  Giả lập tại quầy (100% quyết định tức thì, không chờ người duyệt):")
    for t, v in k["gia_lap_tai_quay"].items():
        print(f"    Đơn {t:>6,.0f} | tất cả: {v['tat_ca']}")
        print(f"    {'':11s}| nhóm mục tiêu: {v['nhom_muc_tieu']}")


# --------------------------------------------------------------------------- 5 phân khúc theo đề (Exhibit 2)

def assign_case_segment(res: pd.DataFrame) -> pd.Series:
    cfg = scoring_config()["case_segments"]
    profile = res.get("profile", pd.Series("", index=res.index)).astype(str)
    has_hist = res["has_credit_history"].astype(bool)
    seg = pd.Series("salaried_no_credit", index=res.index)
    seg[has_hist] = "salaried_with_credit"
    seg[~has_hist & profile.isin(cfg["first_time_profiles"])] = "first_time"
    seg[res["main_income_type"].eq("self_employed") | profile.eq("freelancer")] = "gig_platform"
    seg[profile.eq("entrepreneur")] = "online_merchant"
    return seg


def case_mix_view(res: pd.DataFrame) -> dict:
    """Tỷ lệ có hạn mức theo 5 phân khúc của đề, so với quy tắc truyền thống, và tổng tính lại theo cơ cấu đề."""
    cfg = scoring_config()["case_segments"]
    seg = assign_case_segment(res)
    has = res["pos_limit"] > 0
    trad = res["main_income_type"].isin(["salary", "pension"]) & res["has_credit_history"].astype(bool)
    rows = {}
    for key, w in cfg["weights"].items():
        m = seg == key
        rows[key] = {"label": cfg["labels"][key], "case_weight": w, "n_sample": int(m.sum()),
                     "traditional": float(trad[m].mean()) if m.any() else None,
                     "creditforyou": float(has[m].mean()) if m.any() else None}
    weighted = lambda col: sum(v["case_weight"] * v[col] for v in rows.values() if v[col] is not None)  # noqa: E731
    target = ["gig_platform", "online_merchant", "first_time"]
    tw = sum(cfg["weights"][k] for k in target)
    return {"segments": rows,
            "case_weighted_all": {"traditional": weighted("traditional"), "creditforyou": weighted("creditforyou")},
            "case_weighted_thin_file_segments": {
                "traditional": sum(cfg["weights"][k] * rows[k]["traditional"] for k in target) / tw,
                "creditforyou": sum(cfg["weights"][k] * rows[k]["creditforyou"] for k in target) / tw},
            "sample_mix": {k: round(v["n_sample"] / len(res), 3) for k, v in rows.items()}}
