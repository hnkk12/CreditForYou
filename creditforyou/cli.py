"""Giao diện dòng lệnh CreditForYou.

    python -m creditforyou inspect  <file> [--as transactions|applicants|loans]
    python -m creditforyou sample
    python -m creditforyou income   --transactions <file> [--customer ID]
    python -m creditforyou train    --loans <file> [--features core|full] [--max-rows N]
    python -m creditforyou score    --transactions <file> [--applicants <file>] [--customer ID]
    python -m creditforyou demo

  Dataset ISB ("dataset isb/"):
    python -m creditforyou cache                     # tạo cache parquet (1 lần, ~20 giây)
    python -m creditforyou isb --customer C0000001 [--amount 10000 --term 36 ...]
    python -m creditforyou isb --all [--with-truth]  # chấm 1 khoản vay cụ thể cho 500 khách + KPI

  Giải pháp trả góp tại điểm bán (POP):
    python -m creditforyou limits --all --with-truth          # [A] cấp hạn mức trước + báo cáo theo nhóm
    python -m creditforyou limits -c C0000013                 # chi tiết 1 khách + giả lập tại quầy
    python -m creditforyou checkout -c C0000013 --amount 800  # [B] quyết định tức thì tại quầy
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import DEFAULT_MODEL_PATH, MODELS_DIR, OUTPUTS_DIR, SAMPLE_DIR, schema_config

# demo train trên dữ liệu giả lập -> file riêng, không ghi đè model Lending Club
DEMO_MODEL_PATH = MODELS_DIR / "pd_model_demo.joblib"


def _setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)


def _warn(warnings):
    for w in warnings:
        print(f"  [!] {w}")


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig: Excel mở tiếng Việt không lỗi font
    print(f"\n-> Đã lưu: {path}")


def _dayfirst(args):
    return True if args.dayfirst else False if args.monthfirst else None


# --------------------------------------------------------------------------- commands

def cmd_inspect(args):
    from .io import map_columns, read_table
    df = read_table(args.file, nrows=args.rows)
    print(f"File: {args.file}")
    print(f"Số cột: {df.shape[1]} | đọc thử {len(df)} dòng đầu\n")
    print(df.head(5).to_string(max_colwidth=30))
    datasets = [args.as_] if args.as_ else ["transactions", "applicants", "loans"]
    for ds in datasets:
        cm = map_columns(list(df.columns), ds)
        print(f"\n--- Map theo dạng '{ds}': {'ĐỦ CỘT BẮT BUỘC' if cm.ok else 'THIẾU ' + str(cm.missing_required)}")
        for canon in schema_config()[ds]["columns"]:
            print(f"   {canon:24s} <- {cm.mapping.get(canon, '(không có)')}")
        if cm.unmapped_columns:
            shown = cm.unmapped_columns[:25]
            more = f" ... (+{len(cm.unmapped_columns) - 25})" if len(cm.unmapped_columns) > 25 else ""
            print(f"   Cột không dùng: {shown}{more}")
    print("\nNếu cột bị map sai/thiếu: sửa 'overrides' hoặc thêm alias trong config/schema.json.")


def cmd_sample(args):
    from .sample import generate
    info = generate(n_customers=args.customers, n_loans=args.loans, seed=args.seed)
    print(f"Đã sinh dữ liệu giả lập vào {info['dir']}:")
    print(f"  transactions.csv   : {info['transactions']:,} giao dịch")
    print(f"  applicants.csv     : {info['applicants']:,} hồ sơ")
    print(f"  loans_training.csv : {info['loans']:,} khoản vay kiểu Lending Club")


def cmd_income(args):
    from .pipeline import run_income
    from .report import print_income_table
    t0 = time.perf_counter()
    income, warnings = run_income(args.transactions, dayfirst=_dayfirst(args), customers=args.customer)
    elapsed = time.perf_counter() - t0
    _warn(warnings)
    print(f"\nƯỚC TÍNH THU NHẬP — {len(income)} khách hàng ({elapsed:.2f}s)\n")
    print_income_table(income, limit=args.limit)
    if args.customer:
        for _, r in income.iterrows():
            print(f"\n  {r['customer_id']} - thu nhập theo tháng: {r['monthly_income_series']}")
    _save(income, Path(args.out))


def cmd_train(args):
    from .model import train
    print(f"HUẤN LUYỆN MÔ HÌNH XÁC SUẤT VỠ NỢ — {args.loans}")
    art = train(args.loans, feature_set=args.features, max_rows=args.max_rows, model_path=Path(args.model))
    print("\nKết quả trên tập kiểm tra (20%):")
    print(pd.DataFrame(art["metrics_test"]).to_string())
    print(f"\nĐặc trưng: {art['num_features'] + art['cat_features']}")
    print("* accuracy_baseline_all_good = độ chính xác khi đoán mọi khoản vay đều tốt -> so sánh với accuracy@0.5")
    metrics_path = Path(args.model).with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in art.items() if k != "model"}, f, ensure_ascii=False, indent=2)
    print(f"-> Metrics: {metrics_path}")


def cmd_score(args):
    from .pipeline import evaluate
    from .report import kpi_summary, print_application, print_kpis
    overrides = {"requested_amount": args.loan_amount, "term_months": args.term,
                 "years_experience": args.years_exp, "age": args.age}
    t0 = time.perf_counter()
    res, warnings, model = evaluate(args.transactions, args.applicants, model_path=Path(args.model),
                                    dayfirst=_dayfirst(args), customers=args.customer, overrides=overrides)
    elapsed = time.perf_counter() - t0
    _warn(warnings)
    if model is not None:
        print(f"  Mô hình ML: {model['model_name']} (train {model['trained_at']} trên {Path(model['trained_on']).name})")

    if args.customer or len(res) <= 3:
        for _, r in res.iterrows():
            print_application(r)
    else:
        cols = ["customer_id", "estimated_monthly_income", "income_stability", "dti", "pd_used",
                "credit_score", "tier", "decision_vn", "interest_rate_pct"]
        print(f"\nKẾT QUẢ {len(res)} HỒ SƠ (hiển thị {min(args.limit, len(res))} dòng đầu)\n")
        print(res[cols].head(args.limit).to_string(index=False))
    print_kpis(kpi_summary(res, elapsed))
    _save(res, Path(args.out))


def cmd_demo(args):
    print(">>> 1/3 Sinh dữ liệu giả lập")
    cmd_sample(argparse.Namespace(customers=300, loans=20000, seed=7))
    print("\n>>> 2/3 Huấn luyện mô hình ML")
    cmd_train(argparse.Namespace(loans=SAMPLE_DIR / "loans_training.csv", features="core", max_rows=None,
                                 model=str(DEMO_MODEL_PATH)))
    print("\n>>> 3/3 Chấm điểm toàn bộ hồ sơ")
    cmd_score(argparse.Namespace(
        transactions=SAMPLE_DIR / "transactions.csv", applicants=SAMPLE_DIR / "applicants.csv",
        model=str(DEMO_MODEL_PATH), dayfirst=False, monthfirst=False, customer=None, loan_amount=None,
        term=None, years_exp=None, age=None, limit=15, out=str(OUTPUTS_DIR / "decisions.csv")))
    print("\nXem chi tiết 1 hồ sơ:  python -m creditforyou score --transactions data/sample/transactions.csv "
          "--applicants data/sample/applicants.csv --customer KH00001")


def cmd_cache(args):
    from .datasets import build_transaction_cache, synthetic_paths
    build_transaction_cache(Path(args.transactions) if args.transactions else synthetic_paths()["transactions"])


def cmd_isb(args):
    from .datasets import (build_transaction_cache, load_synthetic_table, synthetic_paths,
                           transaction_cache_is_fresh)
    from .evaluation import income_metrics
    from .pipeline import evaluate_synthetic_customer
    from .report import kpi_summary, print_application, print_kpis

    tx_path = synthetic_paths()["transactions"]
    if not transaction_cache_is_fresh(tx_path):
        build_transaction_cache(tx_path)
    customers = load_synthetic_table("customers", usecols=["customer_id"])["customer_id"].astype(str).tolist()
    ids = customers if args.all else args.customer
    unknown = sorted(set(ids) - set(customers))
    if unknown:
        raise KeyError(f"Không có khách hàng {unknown} trong customers.csv")

    t0 = time.perf_counter()
    res, warnings, model = evaluate_synthetic_customer(
        ids, args.amount, args.term, args.purpose, args.years_exp, args.home, model_path=Path(args.model))
    elapsed = time.perf_counter() - t0
    _warn(warnings)
    if model is not None:
        print(f"  Mô hình PD: {model.get('model_version', model['model_name'])} ({model['model_name']}, "
              f"train {model['trained_at']})")
    print(f"  Khoản vay giả định: {args.amount:,.0f} USD, {args.term} tháng, mục đích {args.purpose}")

    if args.with_truth:
        # Ground truth chỉ được ghép SAU khi đã ước tính xong, để đánh giá.
        truth = load_synthetic_table("customers", usecols=["customer_id", "monthly_income"])
        truth = truth.rename(columns={"monthly_income": "actual_monthly_income"})
        res = res.merge(truth, on="customer_id", how="left")

    if not args.all:
        for _, r in res.iterrows():
            print_application(r, currency="USD")
    else:
        cols = ["customer_id", "estimated_monthly_income", "income_stability", "income_confidence",
                "post_loan_debt_ratio", "pd_used", "credit_score", "tier", "decision"]
        print(f"\nKẾT QUẢ {len(res)} KHÁCH (hiển thị {min(args.limit, len(res))} dòng đầu)\n")
        print(res[cols].head(args.limit).to_string(index=False))
    print_kpis(kpi_summary(res, elapsed))
    if args.with_truth:
        m = income_metrics(res.rename(columns={"actual_monthly_income": "monthly_income"}))
        print("  Ước tính thu nhập so với thực tế:")
        print(f"    MAE {m['mae']:,.0f} USD | MAPE {m['mape']:.1%} | trung vị APE "
              f"{m['median_absolute_percentage_error']:.1%} | ±10%: {m['within_10pct']:.1%} | ±20%: {m['within_20pct']:.1%}")
    _save(res, Path(args.out))


def _ensure_cache():
    from .datasets import build_transaction_cache, synthetic_paths, transaction_cache_is_fresh
    tx_path = synthetic_paths()["transactions"]
    if not transaction_cache_is_fresh(tx_path):
        build_transaction_cache(tx_path)


def _attach_truth(res):
    from .datasets import load_synthetic_table
    truth = load_synthetic_table("customers", usecols=["customer_id", "monthly_income"])
    truth["customer_id"] = truth["customer_id"].astype(str)
    return res.merge(truth.rename(columns={"monthly_income": "actual_monthly_income"}), on="customer_id", how="left")


def cmd_limits(args):
    from .pipeline import preapprove_limits
    from .report import print_limit_detail, print_limit_summary
    _ensure_cache()
    t0 = time.perf_counter()
    res, warnings, model = preapprove_limits(None if args.all else args.customer, model_path=Path(args.model))
    elapsed = time.perf_counter() - t0
    _warn(warnings)
    if args.with_truth:
        # Ground truth chỉ ghép SAU khi đã ước tính xong, để đánh giá.
        res = _attach_truth(res)
    if not args.all:
        for _, r in res.iterrows():
            print_limit_detail(r, args.tickets)
    print_limit_summary(res, args.tickets, elapsed)
    if not args.all and Path(args.out) == OUTPUTS_DIR / "isb_limits.csv":
        args.out = str(OUTPUTS_DIR / "isb_limits_selected.csv")   # không ghi đè kết quả toàn bộ khách
    if args.all:
        from .config import REPORTS_DIR
        from .report import limit_summary
        summary = limit_summary(res, args.tickets)
        summary["gia_lap_tai_quay"] = {str(k): v for k, v in summary["gia_lap_tai_quay"].items()}
        if "profile" in res:
            summary["theo_nhom_khach"] = {
                str(name): {"so_khach": int(len(g)), "ty_le_co_han_muc": round(float((g["pos_limit"] > 0).mean()), 4),
                            "han_muc_trung_vi": float(g.loc[g["pos_limit"] > 0, "pos_limit"].median())
                            if (g["pos_limit"] > 0).any() else 0.0,
                            "trang_thai": g["limit_status"].value_counts().to_dict()}
                for name, g in res.groupby("profile")}
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "pos_limit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"-> Tổng kết: {REPORTS_DIR / 'pos_limit_summary.json'}")
    _save(res, Path(args.out))


def cmd_checkout(args):
    from .limit import checkout_decision
    from .pipeline import preapprove_limits
    _ensure_cache()
    res, _, _ = preapprove_limits([args.customer], model_path=Path(args.model))
    r = res.iloc[0]
    t0 = time.perf_counter()
    d = checkout_decision(r["pos_limit"], args.amount, args.term, available_limit_max_term=r["pos_limit_max_term"])
    ms = (time.perf_counter() - t0) * 1000
    print(f"Khách {r['customer_id']} | hạn mức {r['pos_limit']:,.0f} ({r['limit_status_vn']}) | "
          f"khả năng trả {r['monthly_capacity']:,.0f}/tháng")
    print(f"Đơn {args.amount:,.0f} -> {d['checkout_decision']}: {d['message']}"
          + (f" | trả góp {d['installment']:,.2f}/tháng" if d.get("installment") else "")
          + f"  (quyết định trong {ms:.2f} ms)")


def cmd_quality(args):
    from .config import REPORTS_DIR
    from .quality import data_quality_report, print_quality_report
    _ensure_cache()
    report = data_quality_report()
    print_quality_report(report)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_quality.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> Đã lưu: {path}")


def cmd_economics(args):
    from .config import REPORTS_DIR
    from .economics import portfolio_economics, print_economics
    from .pipeline import preapprove_limits
    _ensure_cache()
    res, warnings, _ = preapprove_limits(None, model_path=Path(args.model))
    e = portfolio_economics(res)
    print_economics(e)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "pos_economics.json"
    path.write_text(json.dumps(e, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> Đã lưu: {path}")


# --------------------------------------------------------------------------- parser

def build_parser():
    p = argparse.ArgumentParser(prog="python -m creditforyou",
                                description="CreditForYou — Ước tính thu nhập & chấm điểm tín dụng")
    p.add_argument("--version", action="version", version=f"CreditForYou {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def date_opts(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--dayfirst", action="store_true", help="Ngày dạng dd/mm/yyyy (mặc định: tự đoán)")
        g.add_argument("--monthfirst", action="store_true", help="Ngày dạng mm/dd/yyyy")

    s = sub.add_parser("inspect", help="Xem cấu trúc dataset & cách map cột")
    s.add_argument("file")
    s.add_argument("--as", dest="as_", choices=["transactions", "applicants", "loans"])
    s.add_argument("--rows", type=int, default=1000)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("sample", help="Sinh dữ liệu giả lập vào data/sample/")
    s.add_argument("--customers", type=int, default=300)
    s.add_argument("--loans", type=int, default=20000)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("income", help="Bước #2: ước tính thu nhập từ giao dịch")
    s.add_argument("--transactions", "-t", required=True)
    s.add_argument("--customer", "-c", nargs="+")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--out", default=str(OUTPUTS_DIR / "income_estimates.csv"))
    date_opts(s)
    s.set_defaults(func=cmd_income)

    s = sub.add_parser("train", help="Huấn luyện mô hình ML xác suất vỡ nợ")
    s.add_argument("--loans", "-l", required=True)
    s.add_argument("--features", choices=["core", "full"], default="core",
                   help="core: chỉ đặc trưng khách VN cũng có (khuyến nghị); full: thêm dữ liệu bureau")
    s.add_argument("--max-rows", type=int, help="Lấy mẫu ngẫu nhiên N dòng (dataset lớn)")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("score", help="Quy trình đầy đủ #2 + #5 + quyết định")
    s.add_argument("--transactions", "-t", required=True)
    s.add_argument("--applicants", "-a")
    s.add_argument("--customer", "-c", nargs="+")
    s.add_argument("--loan-amount", type=float, help="Ghi đè số tiền vay (VND)")
    s.add_argument("--term", type=float, help="Ghi đè kỳ hạn (tháng)")
    s.add_argument("--years-exp", type=float, help="Ghi đè số năm kinh nghiệm")
    s.add_argument("--age", type=float, help="Ghi đè tuổi")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--out", default=str(OUTPUTS_DIR / "decisions.csv"))
    date_opts(s)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("cache", help="Tạo cache parquet cho file giao dịch ISB (tăng tốc ~20 lần)")
    s.add_argument("--transactions", "-t", help="Mặc định: dataset isb/data/transactions.csv")
    s.set_defaults(func=cmd_cache)

    s = sub.add_parser("isb", help="Chấm điểm khách trong dataset ISB (giao dịch + khoản vay đang có)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--customer", "-c", nargs="+", help="VD: C0000001 C0000002")
    g.add_argument("--all", action="store_true", help="Chấm toàn bộ khách trong customers.csv")
    s.add_argument("--amount", type=float, default=10_000, help="Số tiền vay (USD), mặc định 10000")
    s.add_argument("--term", type=int, default=36, help="Kỳ hạn (tháng), mặc định 36")
    s.add_argument("--purpose", default="debt_consolidation",
                   help="Mục đích (theo Lending Club: debt_consolidation, credit_card, home_improvement, ...)")
    s.add_argument("--years-exp", type=float, default=5, help="Số năm làm việc khai báo, mặc định 5")
    s.add_argument("--home", default="rent", choices=["rent", "mortgage", "own", "family", "other"])
    s.add_argument("--with-truth", action="store_true",
                   help="Ghép thu nhập thực tế (customers.monthly_income) SAU khi ước tính để đánh giá")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--out", default=str(OUTPUTS_DIR / "isb_decisions.csv"))
    s.set_defaults(func=cmd_isb)

    s = sub.add_parser("quality", help="Báo cáo chất lượng dữ liệu ISB: nhiễu phát hiện được & cách xử lý")
    s.set_defaults(func=cmd_quality)

    s = sub.add_parser("economics", help="[Giải pháp POP] Bài toán kinh tế: lợi nhuận, lỗ dự kiến, PD hoà vốn, RoC")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.set_defaults(func=cmd_economics)

    s = sub.add_parser("limits", help="[Giải pháp POP] Cấp trước hạn mức trả góp tại điểm bán cho khách ISB")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--customer", "-c", nargs="+")
    g.add_argument("--all", action="store_true")
    s.add_argument("--tickets", type=float, nargs="+", default=[5_000_000, 15_000_000, 40_000_000],
                   help="Giá trị đơn hàng (VND) để giả lập quyết định tại quầy (mặc định 5tr 15tr 40tr)")
    s.add_argument("--with-truth", action="store_true", help="Ghép thu nhập thực tế sau khi ước tính để đánh giá")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.add_argument("--out", default=str(OUTPUTS_DIR / "isb_limits.csv"))
    s.set_defaults(func=cmd_limits)

    s = sub.add_parser("checkout", help="[Giải pháp POP] Quyết định tức thì tại quầy cho 1 đơn hàng")
    s.add_argument("--customer", "-c", required=True)
    s.add_argument("--amount", type=float, required=True, help="Giá trị đơn hàng (VND)")
    s.add_argument("--term", type=int, help="Kỳ hạn mong muốn (tháng), mặc định theo config")
    s.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    s.set_defaults(func=cmd_checkout)

    s = sub.add_parser("demo", help="Chạy thử toàn bộ trên dữ liệu giả lập")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    _setup_console()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nLỖI: {e}", file=sys.stderr)
        sys.exit(1)
