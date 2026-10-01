"""Giao diện dòng lệnh CreditForYou.

    python -m creditforyou inspect  <file> [--as transactions|applicants|loans]
    python -m creditforyou sample
    python -m creditforyou income   --transactions <file> [--customer ID]
    python -m creditforyou train    --loans <file> [--features core|full] [--max-rows N]
    python -m creditforyou score    --transactions <file> [--applicants <file>] [--customer ID]
    python -m creditforyou demo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import DEFAULT_MODEL_PATH, OUTPUTS_DIR, SAMPLE_DIR, schema_config


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
                                 model=str(DEFAULT_MODEL_PATH)))
    print("\n>>> 3/3 Chấm điểm toàn bộ hồ sơ")
    cmd_score(argparse.Namespace(
        transactions=SAMPLE_DIR / "transactions.csv", applicants=SAMPLE_DIR / "applicants.csv",
        model=str(DEFAULT_MODEL_PATH), dayfirst=False, monthfirst=False, customer=None, loan_amount=None,
        term=None, years_exp=None, age=None, limit=15, out=str(OUTPUTS_DIR / "decisions.csv")))
    print("\nXem chi tiết 1 hồ sơ:  python -m creditforyou score --transactions data/sample/transactions.csv "
          "--applicants data/sample/applicants.csv --customer KH00001")


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
