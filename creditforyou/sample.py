"""Sinh dữ liệu GIẢ LẬP để chạy thử toàn bộ hệ thống trước khi có dataset thật.

Tạo 3 file trong data/sample/:
  - transactions.csv   : sao kê giao dịch (header tiếng Việt, ngày dd/mm/yyyy, cột Loại Ghi có/Ghi nợ)
  - applicants.csv     : hồ sơ xin vay + thu nhập thực tế + nhãn vỡ nợ (để đo KPI)
  - loans_training.csv : dữ liệu kiểu Lending Club (tên cột y hệt) để huấn luyện mô hình ML
Dữ liệu giả lập CHỈ để kiểm tra code chạy đúng, KHÔNG dùng để báo cáo kết quả.
"""

import numpy as np
import pandas as pd

from .config import SAMPLE_DIR

PERSONAS = {
    # tên: (thu nhập TB/tháng, độ biến động, số lần nhận tiền/tháng, nội dung)
    "tai_xe_cong_nghe": (8_300_000, 0.04, 4, "GRAB THANH TOAN DOANH THU TUAN"),
    "nhan_vien_van_phong": (15_000_000, 0.02, 1, "LUONG THANG CONG TY TNHH ABC"),
    "chu_shop_online": (18_000_000, 0.30, 6, "SHOPEE CHUYEN DOANH THU DON HANG"),
    "freelancer": (12_000_000, 0.60, 2, "CK TU KHACH HANG THANH TOAN DU AN"),
    "lao_dong_thoi_vu": (5_500_000, 0.45, 2, "TIEN CONG LAM THEM"),
}
NOISE_INFLOWS = ["CHUYEN KHOAN NOI BO TU TK TIET KIEM", "GIAI NGAN KHOAN VAY TIEU DUNG", "HOAN TIEN DON HANG SHOPEE"]
SPEND = ["THANH TOAN DIEN NUOC", "MUA SAM CIRCLE K", "CHUYEN TIEN NHA", "THANH TOAN VINMART", "NAP DIEN THOAI",
         "AN UONG GRABFOOD", "XANG XE PETROLIMEX"]


def generate(n_customers=300, n_loans=20000, seed=7, out_dir=SAMPLE_DIR):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp("2026-08-31")
    tx_rows, app_rows = [], []

    for i in range(n_customers):
        cid = f"KH{i + 1:05d}"
        persona = rng.choice(list(PERSONAS))
        base, vol, k, desc = PERSONAS[persona]
        base *= rng.uniform(0.7, 1.4)
        n_months = int(rng.choice([12, 12, 12, 9, 6, 4, 2], p=[.3, .2, .15, .12, .1, .08, .05]))
        months = pd.period_range(end=end.to_period("M"), periods=n_months, freq="M")
        true_monthly = []
        for m in months:
            seasonal = 1 + (0.35 if persona == "chu_shop_online" and m.month in (11, 12, 1) else 0)
            month_income = max(base * seasonal * rng.normal(1, vol), 0)
            if persona in ("freelancer", "lao_dong_thoi_vu") and rng.random() < 0.12:
                month_income = 0.0                                   # tháng không có việc
            true_monthly.append(month_income)
            parts = rng.dirichlet(np.ones(k)) * month_income
            for p in parts:
                if p > 0:
                    day = int(rng.integers(1, m.days_in_month + 1))
                    tx_rows.append((cid, _d(m, day), round(p, -3), "Ghi có", desc))
            if rng.random() < 0.15:
                tx_rows.append((cid, _d(m, int(rng.integers(1, 28))), round(rng.uniform(2e6, 3e7), -3), "Ghi có",
                                rng.choice(NOISE_INFLOWS)))
            spend_total = month_income * rng.uniform(0.55, 0.95) + 1_500_000
            for p in rng.dirichlet(np.ones(8)) * spend_total:
                tx_rows.append((cid, _d(m, int(rng.integers(1, 29))), round(p, -3), "Ghi nợ", rng.choice(SPEND)))

        has_debt = rng.random() < 0.35
        existing_debt = round(base * rng.uniform(0.05, 0.3), -3) if has_debt else 0
        if has_debt:
            for m in months:
                tx_rows.append((cid, _d(m, 15), existing_debt, "Ghi nợ", "TRA GOP KHOAN VAY THANG"))

        true_income = float(np.mean(true_monthly))
        age = int(rng.integers(19, 60))
        years = float(min(max(age - 18 - rng.integers(0, 6), 0), 30))
        past = int(rng.choice([0, 0, 1, 2, 3, 5]))
        on_time = int(rng.binomial(past, rng.uniform(0.6, 1.0))) if past else 0
        amount = round(base * rng.uniform(1, 8), -6)
        term = int(rng.choice([12, 24, 36]))

        installment = amount * 0.01 / (1 - 1.01 ** -term)
        dti = (installment + existing_debt) / max(true_income, 1)
        stab = 1 - min(np.std(true_monthly) / max(true_income, 1), 1)
        logit = (-3.0 + 3.0 * max(dti - 0.3, 0) + 2.0 * (1 - stab) - 0.03 * years
                 + 1.5 * ((past - on_time) / past if past else 0.3) + rng.normal(0, 0.6))
        default = int(rng.random() < 1 / (1 + np.exp(-logit)))
        app_rows.append({
            "customer_id": cid, "age": age, "years_experience": years, "loan_amount": amount,
            "term_months": term, "past_loans": past, "on_time_loans": on_time,
            "existing_debt_payment": existing_debt if rng.random() < 0.7 else None,
            "home_ownership": rng.choice(["rent", "own", "family"]), "purpose": rng.choice(
                ["kinh_doanh", "tieu_dung", "mua_xe", "sua_nha"]),
            "nghe_nghiep_mo_phong": persona, "declared_income": round(true_income, -3), "default": default,
        })

    tx = pd.DataFrame(tx_rows, columns=["Mã KH", "Ngày giao dịch", "Số tiền", "Loại", "Nội dung"])
    tx = tx.sample(frac=1, random_state=seed).reset_index(drop=True)
    tx.to_csv(out_dir / "transactions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(app_rows).to_csv(out_dir / "applicants.csv", index=False, encoding="utf-8-sig")
    _lending_club_like(rng, n_loans).to_csv(out_dir / "loans_training.csv", index=False)
    return {"transactions": len(tx), "applicants": len(app_rows), "loans": n_loans, "dir": out_dir}


def _d(period, day):
    return f"{min(day, period.days_in_month):02d}/{period.month:02d}/{period.year}"


def _lending_club_like(rng, n):
    annual_inc = np.exp(rng.normal(11.0, 0.5, n)).round(-2)
    loan = np.clip(annual_inc * rng.uniform(0.05, 0.5, n), 1000, 40000).round(-2)
    term = rng.choice([36, 60], n, p=[0.75, 0.25])
    emp = rng.integers(0, 12, n)
    dti = np.clip(rng.gamma(4, 4.5, n), 0, 45).round(2)
    revol = np.clip(rng.normal(50, 25, n), 0, 120).round(1)
    delinq = rng.poisson(0.3, n)
    inq = rng.poisson(0.8, n)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.4, 0.5, 0.1])
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "car"], n)
    risk = (-2.4 + 2.2 * (loan / annual_inc) + 0.03 * dti + 0.5 * (term == 60) - 0.04 * emp + 0.25 * delinq
            + 0.15 * inq + 0.008 * revol + 0.5 * (purpose == "small_business") - 0.2 * (home == "MORTGAGE")
            + rng.normal(0, 0.5, n))
    pd_true = 1 / (1 + np.exp(-risk))
    bad = rng.random(n) < pd_true
    status = np.where(bad, "Charged Off", "Fully Paid")
    status[rng.random(n) < 0.1] = "Current"                       # chưa có kết cục -> bị loại khi train
    rate = np.clip(6 + 20 * pd_true + rng.normal(0, 1, n), 5, 30)
    r = rate / 1200
    installment = loan * r / (1 - (1 + r) ** -term)
    grade = np.select([rate < 9, rate < 13, rate < 17, rate < 21], list("ABCD"), "E")
    emp_txt = np.where(emp >= 10, "10+ years", np.where(emp == 0, "< 1 year", [f"{e} years" for e in emp]))
    return pd.DataFrame({
        "loan_amnt": loan, "term": [f" {t} months" for t in term], "int_rate": [f"{x:.2f}%" for x in rate],
        "installment": installment.round(2), "grade": grade, "emp_length": emp_txt, "home_ownership": home,
        "annual_inc": annual_inc, "purpose": purpose, "dti": dti, "delinq_2yrs": delinq, "inq_last_6mths": inq,
        "open_acc": rng.poisson(10, n), "pub_rec": rng.poisson(0.15, n), "revol_util": [f"{x}%" for x in revol],
        "loan_status": status, "total_pymnt": (installment * term * np.where(bad, 0.4, 1.0)).round(2),
    })
