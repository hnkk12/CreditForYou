# CreditForYou — Ước tính thu nhập & Chấm điểm tín dụng

Hệ thống cho khách **không có giấy tờ chứng minh thu nhập** (tài xế công nghệ, chủ shop, freelancer...):

```
Hồ sơ xin vay
   ↓
[BƯỚC 1] Ước tính thu nhập từ lịch sử giao dịch      (#2)  -> creditforyou/income.py
   ↓  VD: 8.28 triệu/tháng, ổn định 98%, tin tưởng 95%
[BƯỚC 2] Chấm điểm tín dụng thang 100                 (#5)  -> creditforyou/scoring.py + model.py
   ↓  VD: 63.85 điểm
[BƯỚC 3] Quy tắc nghiệp vụ + xếp nhóm A/B/C/D + lãi suất     -> creditforyou/decision.py
   ↓
Phê duyệt tự động / Review thủ công / Từ chối   (< 5 giây)
```

## Cấu trúc thư mục

```
CreditForYou/
├── config/
│   ├── scoring.json        # TẤT CẢ tham số nghiệp vụ: trọng số, ngưỡng, nhóm, lãi suất, quy tắc chặn
│   └── schema.json         # map tên cột dataset -> tên chuẩn (thêm alias khi dataset có tên lạ)
├── creditforyou/           # code chính
│   ├── cli.py              # lệnh terminal
│   ├── io.py               # đọc file, tự nhận diện cột, parse tiền/ngày kiểu Việt Nam
│   ├── income.py           # BƯỚC #2 ước tính thu nhập
│   ├── scoring.py          # BƯỚC #5 công thức 5 thành phần
│   ├── model.py            # mô hình ML xác suất vỡ nợ (PD)
│   ├── decision.py         # quy tắc nghiệp vụ + nhóm A/B/C/D
│   ├── pipeline.py         # ghép #2 + #5 + quyết định
│   ├── report.py           # in kết quả + KPI
│   └── sample.py           # sinh dữ liệu giả lập để chạy thử
├── data/
│   ├── raw/                # <- BỎ DATASET THẬT VÀO ĐÂY
│   └── sample/             # dữ liệu giả lập (tạo bằng lệnh `sample`)
├── models/                 # mô hình đã train (pd_model.joblib + metrics)
├── outputs/                # kết quả CSV (mở bằng Excel được)
├── tests/                  # pytest
└── legacy/                 # code prototype cũ (Streamlit, thang 1000) — chỉ để tham khảo
```

## Cài đặt (1 lần)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Đã tạo sẵn `.venv`. Mỗi lần mở terminal mới chỉ cần `.venv\Scripts\activate`.)

## Chạy thử ngay (dữ liệu giả lập)

```bash
python -m creditforyou demo
```

## Khi có dataset thật

```bash
# 1. Xem hệ thống đọc cột thế nào (cột nào khớp, cột nào thiếu)
python -m creditforyou inspect data/raw/giao_dich.csv --as transactions
python -m creditforyou inspect data/raw/ho_so.csv --as applicants
python -m creditforyou inspect data/raw/lending_club.csv --as loans
#    -> nếu map sai/thiếu: sửa "overrides" trong config/schema.json, VD: "date": "Posting Date"

# 2. Huấn luyện mô hình ML (dataset lớn thì lấy mẫu)
python -m creditforyou train --loans data/raw/lending_club.csv --max-rows 300000

# 3. Chỉ ước tính thu nhập
python -m creditforyou income --transactions data/raw/giao_dich.csv

# 4. Chạy toàn bộ quy trình + KPI
python -m creditforyou score --transactions data/raw/giao_dich.csv --applicants data/raw/ho_so.csv

# 5. Xem chi tiết từng hồ sơ
python -m creditforyou score -t data/raw/giao_dich.csv -a data/raw/ho_so.csv --customer KH001 KH002

# Không có file hồ sơ: giả định khoản vay cho 1 khách
python -m creditforyou score -t data/raw/giao_dich.csv -c KH001 --loan-amount 50000000 --term 24 --years-exp 5 --age 30
```

Tuỳ chọn ngày: mặc định tự đoán dd/mm hay mm/dd; ép bằng `--dayfirst` / `--monthfirst`.
Kết quả lưu ở `outputs/income_estimates.csv` và `outputs/decisions.csv`.

Chạy test: `python -m pytest -q`

## Phương pháp

### Bước #2 — Ước tính thu nhập
1. Lấy giao dịch **tiền vào**, **loại** tiền vào không phải thu nhập: chuyển khoản nội bộ, giải ngân khoản vay, hoàn tiền, tất toán tiết kiệm (từ khoá trong `scoring.json > income.exclude_inflow_keywords`).
2. Gom theo tháng, tối đa 12 tháng gần nhất. Tháng không có tiền vào = 0. Bỏ tháng đầu/cuối nếu dữ liệu không trọn tháng.
3. Độ ổn định = 1 − CV (CV = độ lệch chuẩn ÷ trung bình), kẹp trong [0, 1].
4. Ổn định ≥ 80% → lấy trung bình. Ổn định thấp → TB × (0.5 + 0.5 × độ ổn định) (bảo thủ, giảm tối đa 50%).
5. Mức tin tưởng: ≥6 tháng → 95%, 3–5 tháng → 70%, <3 tháng → 40% (và đẩy sang review thủ công).
6. Phụ: phát hiện khoản **trả nợ đang có** từ giao dịch (trả góp, thanh toán thẻ) để dùng khi hồ sơ không khai báo.

### Bước #5 — Điểm tín dụng (100 điểm)

| Thành phần | Điểm | Công thức |
|---|---|---|
| Ổn định thu nhập | 25 | độ ổn định × 25 |
| Tỷ lệ nợ/thu nhập | 25 | (1 − (trả góp khoản mới + nợ đang trả hằng tháng) ÷ thu nhập tháng) × 25 |
| Kinh nghiệm | 15 | min(năm kinh nghiệm ÷ 30, 1) × 15 |
| Lịch sử trả nợ | 20 | (số lần trả đúng hạn ÷ tổng khoản vay) × 20 — chưa từng vay: 50% (trung tính) |
| Mô hình ML | 15 | (1 − xác suất vỡ nợ) × 15 |

Mô hình ML: so sánh Logistic Regression và Gradient Boosting, chọn AUC cao hơn. Chỉ dùng đặc trưng dạng **tỷ lệ** (khoản vay/thu nhập, trả góp/thu nhập, DTI, kinh nghiệm, kỳ hạn, nhà ở, mục đích) để áp được từ dữ liệu Mỹ (USD/năm) sang khách Việt Nam (VND/tháng).

### Bước 3 — Quyết định

| Nhóm | Điểm | Quyết định | Lãi suất | Hạn mức |
|---|---|---|---|---|
| A | 80–100 | Phê duyệt tự động | 9–10% | theo đề nghị |
| B | 60–79 | Phê duyệt tự động | 11–13% | theo đề nghị |
| C | 40–59 | Review thủ công | 14–16% | 80% số tiền đề nghị |
| D | <40 | Từ chối | – | – |

Lãi suất nội suy trong khoảng của nhóm (điểm cao → lãi thấp). Quy tắc chặn chạy trước: tuổi ngoài 18–65, thu nhập < 3 triệu, tỷ lệ trả nợ/thu nhập > 70% → từ chối; < 3 tháng dữ liệu, tin tưởng < 70%, PD > 35% → không duyệt tự động. Tất cả chỉnh trong `config/scoring.json`.

## Những điểm đã kiểm tra & điều chỉnh so với bản ý tưởng

| # | Vấn đề trong bản ý tưởng | Xử lý |
|---|---|---|
| 1 | **Ví dụ tính sai**: (1 − 5M/8.28M) × 25 = **9.90**, không phải 15.0 → tổng đúng là **63.85**, không phải 68.95 (vẫn nhóm B) | Có test `test_doc_example_score` khoá lại con số đúng |
| 2 | "Vay/Thu nhập" không rõ là tổng khoản vay hay khoản trả hằng tháng. Nếu là tổng khoản vay thì gần như mọi hồ sơ đều > 1 → 0 điểm | Dùng **khoản trả hằng tháng** (trả góp khoản mới + nợ đang có) ÷ thu nhập tháng; kẹp [0, 1] |
| 3 | "Lấy tất cả tiền vào" sẽ tính cả chuyển khoản nội bộ, giải ngân vay, hoàn tiền → thổi phồng thu nhập, dễ gian lận (tự chuyển tiền vòng) | Loại theo từ khoá/danh mục |
| 4 | "Chia 12 tháng" sai khi khách chỉ có 3–6 tháng dữ liệu | Chia cho số tháng thực có |
| 5 | Mốc tin tưởng 6 tháng nằm ở cả 2 khoảng (6–12 và 3–6); chưa có mức cho < 3 tháng | ≥6: 95%, 3–5: 70%, <3: 40% + review |
| 6 | "Ổn định thấp → lấy giá trị thấp hơn" chưa có công thức. Thử dùng phân vị 25 thì freelancer có vài tháng không việc bị ước tính **0 đồng** | TB × (0.5 + 0.5 × ổn định); vẫn có tuỳ chọn phân vị trong config |
| 7 | Khách chưa từng vay (đúng nhóm khách mục tiêu) sẽ bị 0/20 điểm lịch sử trả nợ | Cho mức trung tính 50% (chỉnh được) |
| 8 | KPI "độ chính xác chấm điểm ≥ 85%" gây hiểu lầm: tỷ lệ vỡ nợ ~12–20% thì đoán "ai cũng tốt" đã đạt 80–88% | Báo cáo thêm **AUC/Gini/KS** và **tỷ lệ vỡ nợ theo nhóm** (KPI A < 2%, B < 5%) |
| 9 | KPI "độ chính xác thu nhập ≥ 90%" mâu thuẫn với quy tắc bảo thủ (cố ý ước tính thấp cho người thu nhập bất ổn) | KPI tách 2 nhóm: ổn định (trên dữ liệu giả lập đạt 98%) và bất ổn |
| 10 | Dữ liệu Lending Club có cột **rò rỉ** (total_pymnt, recoveries...) và grade/int_rate (là kết quả chấm điểm của LC) → AUC ảo | Chỉ đọc cột cần thiết; không dùng installment gốc (ngầm chứa lãi suất ↔ grade), tính lại bằng lãi suất cố định |
| 11 | Lending Club là dữ liệu Mỹ (USD/năm), khách là VND/tháng | Chỉ dùng đặc trưng dạng tỷ lệ |
| 12 | Nhóm A/B "không hạn chế" số tiền vay là rủi ro | Giữ đúng ý tưởng nhưng có trần DTI 70% chặn chung; có thể đặt `max_amount_ratio` cho A/B trong config |
| 13 | Code cũ dùng `class_weight='balanced'` → PD bị thổi phồng, không còn là xác suất thật | Bỏ, để PD đúng nghĩa xác suất |
| 14 | Code cũ tự sinh nhãn vỡ nợ bằng luật từ chính các đặc trưng → mô hình chỉ học lại luật (AUC 0.99 ảo) | Train trên nhãn thật (Lending Club / dataset có nhãn) |

> Dữ liệu trong `data/sample/` là **giả lập** để kiểm tra code — không dùng số liệu này trong báo cáo.
