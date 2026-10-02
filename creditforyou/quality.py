"""Báo cáo chất lượng dữ liệu: phát hiện nhiễu + ghi rõ hệ thống xử lý thế nào.

Mỗi phát hiện gồm: bảng, vấn đề, số lượng, tỷ lệ, cách xử lý, ảnh hưởng tới quyết định.
Dataset gốc KHÔNG bị sửa; mọi xử lý nằm trong pipeline để chạy lại được khi đổi dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, SYNTHETIC_DATA_DIR, scoring_config
from .datasets import load_synthetic_table, load_transactions, synthetic_paths


def _finding(table, issue, count, total, handling, impact="", severity="info"):
    return {"table": table, "issue": issue, "count": int(count), "total": int(total),
            "share": round(float(count) / total, 4) if total else 0.0,
            "severity": severity, "handling": handling, "impact": impact}


def _optional_csv(data_dir: Path, name: str) -> pd.DataFrame | None:
    path = data_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def data_quality_report(data_dir: Path = SYNTHETIC_DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    findings = []
    customers = load_synthetic_table("customers", data_dir)
    accounts = load_synthetic_table("accounts", data_dir)
    loans = load_synthetic_table("loans", data_dir)
    cards = _optional_csv(data_dir, "cards")
    tx = load_transactions(synthetic_paths(data_dir)["transactions"])
    ts = pd.to_datetime(tx["timestamp"], errors="coerce")
    as_of = ts.max().normalize()
    n_tx = len(tx)

    # ---------------------------------------------------------------- toàn vẹn khoá & kiểu dữ liệu
    for name, df, key in (("customers", customers, "customer_id"), ("accounts", accounts, "account_id"),
                          ("loans", loans, "loan_id"), ("transactions", tx, "transaction_id")):
        if key in df:
            findings.append(_finding(name, f"{key} trùng lặp", df[key].duplicated().sum(), len(df),
                                     "Kiểm tra mỗi lần chạy; trùng thì dừng (fail-fast)",
                                     severity="critical" if df[key].duplicated().any() else "ok"))
    findings.append(_finding("transactions", "timestamp không đọc được", ts.isna().sum(), n_tx,
                             "Bỏ dòng, ghi cảnh báo", severity="ok" if ts.notna().all() else "warning"))
    amount = pd.to_numeric(tx["amount"], errors="coerce")
    findings.append(_finding("transactions", "amount rỗng hoặc bằng 0", (amount.isna() | (amount == 0)).sum(), n_tx,
                             "Bỏ dòng", severity="ok" if (amount.notna() & (amount != 0)).all() else "warning"))
    dup = tx.duplicated(["customer_id", "timestamp", "amount", "label"] if "label" in tx else
                        ["customer_id", "timestamp", "amount"]).sum()
    findings.append(_finding("transactions", "giao dịch trùng (khách, thời điểm, số tiền, nội dung)", dup, n_tx,
                             "Cảnh báo; không tự xoá vì có thể là giao dịch lặp hợp lệ",
                             severity="warning" if dup else "ok"))
    orphan = ~tx["customer_id"].astype(str).isin(customers["customer_id"].astype(str))
    findings.append(_finding("transactions", "giao dịch của khách không có trong customers", orphan.sum(), n_tx,
                             "Bỏ qua khi chấm điểm", severity="warning" if orphan.any() else "ok"))

    # ---------------------------------------------------------------- nhiễu có chủ đích / bẫy dữ liệu
    if {"is_salary", "subcategory"}.issubset(tx.columns):
        sal = tx["is_salary"].astype(str).str.lower().isin({"true", "1"})
        wrong = sal & tx["subcategory"].astype(str).str.lower().ne("salary")
        detail = tx.loc[wrong, "subcategory"].value_counts().to_dict()
        findings.append(_finding(
            "transactions", f"is_salary=True nhưng không phải lương {detail}", wrong.sum(), int(sal.sum()),
            "Phân loại nguồn thu theo subcategory/từ khoá TRƯỚC cờ is_salary (income.classify_income_type)",
            "thu nhập freelance bị coi là lương cố định, đánh giá sai độ ổn định",
            severity="high"))
    if "is_transfer" in tx:
        tr = tx["is_transfer"].astype(str).str.lower().isin({"true", "1"})
        findings.append(_finding(
            "transactions", "chuyển khoản giữa các tài khoản của chính khách", tr.sum(), n_tx,
            "Loại khỏi CẢ thu nhập và chi tiêu",
            "chuyển tiền sang tiết kiệm bị tính là chi tiêu (đã đo: tỷ lệ duyệt tự động tụt từ 18% xuống 1,6%)",
            severity="high"))
    if "balance_after_transaction" in tx:
        t = tx.assign(_ts=ts).sort_values(["account_id", "_ts"])
        prev = t.groupby("account_id")["balance_after_transaction"].shift()
        gap = (prev + pd.to_numeric(t["amount"], errors="coerce") - t["balance_after_transaction"]).abs() > 0.01
        findings.append(_finding(
            "transactions", "số dư sau giao dịch không liên tục (số dư trước + số tiền ≠ số dư sau)",
            gap[prev.notna()].sum(), int(prev.notna().sum()),
            "KHÔNG dùng số dư (avg/min balance) để ra quyết định; chỉ dùng dòng tiền",
            "Các đặc trưng dựa trên số dư không đáng tin", severity="high"))
    text = (tx.get("subcategory", pd.Series("", index=tx.index)).astype(str) + " " +
            tx.get("label", pd.Series("", index=tx.index)).astype(str)).str.lower()
    fee = text.str.contains("overdraft")
    if fee.any():
        months = tx.assign(m=ts.dt.to_period("M"), fee=fee).groupby(["customer_id", "m"])["fee"].any()
        share_months = months.groupby(level=0).mean()
        findings.append(_finding(
            "transactions", f"phí thấu chi xuất hiện ở trung vị {share_months.median():.0%} số tháng của mỗi khách",
            fee.sum(), n_tx, "Chỉ giữ làm chỉ số tham khảo, KHÔNG làm quy tắc",
            "Gần như ai cũng có -> không phân biệt được rủi ro trong dataset này", severity="medium"))

    # ---------------------------------------------------------------- thu nhập trong hồ sơ vs dòng tiền
    lookback = scoring_config()["income"]["lookback_months"]
    start = (as_of.to_period("M") - lookback).to_timestamp()
    income_cat = tx.get("category", pd.Series("", index=tx.index)).astype(str).str.lower().eq("income")
    recent_income = set(tx.loc[income_cat & (ts >= start) & (amount > 0), "customer_id"].astype(str))
    stale = customers["customer_id"].astype(str).map(lambda c: c not in recent_income) & (customers["monthly_income"] > 0)
    findings.append(_finding(
        "customers", f"monthly_income > 0 nhưng KHÔNG có khoản thu nhập nào trong {lookback} tháng gần nhất",
        stale.sum(), len(customers),
        "Thu nhập ước tính = 0 -> không cấp hạn mức (không tin thu nhập tự khai đã cũ)",
        "Đúng về rủi ro; làm giảm chỉ số 'độ chính xác thu nhập' khi so với hồ sơ", severity="medium"))
    if "birth_date" in customers and "age" in customers:
        calc = np.floor((as_of - pd.to_datetime(customers["birth_date"], errors="coerce")).dt.days / 365.25)
        bad_age = (calc - customers["age"]).abs() > 1
        findings.append(_finding("customers", "age lệch birth_date > 1 năm", bad_age.sum(), len(customers),
                                 "Dùng cột age", severity="warning" if bad_age.any() else "ok"))

    # ---------------------------------------------------------------- accounts / cards / loans
    if "opened_at" in accounts:
        opened = pd.to_datetime(accounts["opened_at"], errors="coerce")
        placeholder = opened.dt.normalize() >= as_of - pd.Timedelta(days=1)
        findings.append(_finding(
            "accounts", f"opened_at = ngày xuất dữ liệu ({as_of.date()}) -> nghi là giá trị mặc định",
            placeholder.sum(), len(accounts),
            "Thâm niên lấy theo customers.customer_since, không theo ngày mở tài khoản", severity="medium"))
    if cards is not None and {"status", "expires_at"}.issubset(cards.columns):
        expired = cards["status"].astype(str).str.lower().eq("active") & (pd.to_datetime(cards["expires_at"], errors="coerce") < as_of)
        findings.append(_finding("cards", "thẻ 'active' nhưng đã hết hạn", expired.sum(), len(cards),
                                 "Không dùng bảng cards cho quyết định", severity="low"))
    excluded = {t.lower() for t in scoring_config()["model"]["dti_exclude_loan_types"]}
    mortgage = loans["loan_type"].astype(str).str.lower().isin(excluded)
    findings.append(_finding(
        "loans", "khoản vay mua nhà trong nợ đang có", mortgage.sum(), len(loans),
        "Tính đủ trong khả năng chi trả; BỎ khỏi dti đưa vào mô hình PD (giống định nghĩa dti Lending Club)",
        "dti của C0000001 thành 54% (ngoài phân phối Lending Club, 99% < 40%) và PD bị đẩy từ 11,5% lên 25,3%", severity="high"))
    r = loans["annual_rate"] / 12
    formula = loans["principal"] * r / (1 - (1 + r) ** (-loans["term_months"]))
    off = (formula - loans["monthly_payment"]).abs() / formula > 0.05
    findings.append(_finding("loans", "monthly_payment lệch > 5% so với công thức trả góp", off.sum(), len(loans),
                             "Dùng monthly_payment như hợp đồng", severity="warning" if off.any() else "ok"))
    findings.append(_finding("loans", "không có lịch sử trả đúng hạn / trễ hạn", len(loans), len(loans),
                             "Tắt thành phần 'lịch sử trả nợ' và chuẩn hoá trọng số còn lại",
                             "Điểm thực tế chỉ gồm 4 thành phần", severity="medium"))

    # ---------------------------------------------------------------- Lending Club (đọc từ báo cáo train)
    lc_report = REPORTS_DIR / "credit_model_metrics.json"
    if lc_report.exists():
        info = json.loads(lc_report.read_text(encoding="utf-8")).get("train_info", {})
        tm = info.get("target_mapping", {})
        total = sum(tm.get("raw_status_counts", {}).values())
        if total:
            findings.append(_finding(
                "lending_club", "khoản vay chưa có kết cục (Current, Late, In Grace...) hoặc dòng tổng kết cuối file",
                tm.get("excluded_unresolved", 0), total,
                "Loại khỏi tập train; chỉ giữ Fully Paid (0) và Charged Off/Default (1)", severity="medium"))
            findings.append(_finding(
                "lending_club", "cột phát sinh sau giải ngân / do Lending Club định giá (rò rỉ dữ liệu)",
                1, 1, "Chặn bằng assert_no_leakage; tính lại trả góp bằng lãi suất cố định",
                "AUC ảo, mô hình học lại quyết định của chính Lending Club", severity="high"))

    severities = pd.Series([f["severity"] for f in findings]).value_counts().to_dict()
    return {"as_of": str(as_of.date()), "rows": {"customers": len(customers), "accounts": len(accounts),
                                                 "loans": len(loans), "transactions": n_tx},
            "summary_by_severity": severities, "findings": findings}


def print_quality_report(report: dict):
    print("=" * 100)
    print(f" BÁO CÁO CHẤT LƯỢNG DỮ LIỆU (dữ liệu tới {report['as_of']}) — {report['rows']}")
    print("=" * 100)
    order = {"critical": 0, "high": 1, "medium": 2, "warning": 3, "low": 4, "info": 5, "ok": 6}
    for f in sorted(report["findings"], key=lambda x: order.get(x["severity"], 9)):
        mark = {"ok": "OK ", "critical": "!!!", "high": "!! ", "medium": "!  "}.get(f["severity"], ".  ")
        print(f" {mark} [{f['table']}] {f['issue']}: {f['count']:,}/{f['total']:,} ({f['share']:.1%})")
        if f["severity"] != "ok":
            print(f"       -> Xử lý: {f['handling']}")
            if f["impact"]:
                print(f"       -> Nếu không xử lý: {f['impact']}")
    print(f"\n Tổng hợp: {report['summary_by_severity']}")
