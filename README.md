# CreditForYou

CreditForYou là nguyên mẫu hệ thống đánh giá tín dụng cá nhân dựa trên hai nguồn dữ liệu độc lập: lịch sử giao dịch ngân hàng tổng hợp và dữ liệu khoản vay Lending Club. Hệ thống ước tính thu nhập từ dòng tiền thực tế, dự báo xác suất vỡ nợ, sau đó kết hợp các tín hiệu tài chính thành điểm tín dụng, nhóm rủi ro và quyết định sơ bộ.

> Đây là dự án nghiên cứu/học thuật. Kết quả không thay thế quy trình thẩm định, chính sách tín dụng hoặc quyết định cho vay của tổ chức tài chính.

## Mục tiêu dự án

- Ước tính thu nhập hàng tháng mà không sử dụng `customers.monthly_income` làm đặc trưng đầu vào.
- Đánh giá độ ổn định và mức độ tin cậy của thu nhập từ lịch sử giao dịch 6–12 tháng.
- Huấn luyện mô hình xác suất vỡ nợ trên dữ liệu Lending Club bằng các đặc trưng chỉ có tại thời điểm cấp khoản vay.
- Tổng hợp thu nhập, khả năng chi trả và xác suất vỡ nợ thành điểm tín dụng có thể giải thích.
- Áp dụng các guardrail để điểm số cao không thể bỏ qua rủi ro hoặc khả năng chi trả nghiêm trọng.
- Cung cấp giao diện Streamlit, CLI, báo cáo đánh giá và bộ kiểm thử tự động.

## Kiến trúc tổng quan

```text
Dữ liệu giao dịch tổng hợp                 Dữ liệu Lending Club
            │                                      │
            ▼                                      ▼
  Làm sạch và phân loại dòng tiền       Lọc kết quả khoản vay đã xác định
            │                                      │
            ▼                                      ▼
 Ước tính thu nhập + độ ổn định          Tiền xử lý và mô hình PD
            │                                      │
            └──────────────────┬───────────────────┘
                               ▼
                  Scoring + affordability
                               │
                               ▼
             Điểm tín dụng, risk tier, quyết định
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Streamlit UI          CLI / báo cáo
```

Hai tập dữ liệu không được ghép theo khách hàng:

| Nguồn dữ liệu | Vai trò |
| --- | --- |
| Synthetic banking data | Ước tính thu nhập, chi phí, độ ổn định và khả năng chi trả |
| Lending Club | Huấn luyện và đánh giá mô hình Probability of Default (PD) |

## Các thành phần chính

### 1. Ước tính thu nhập

Module thu nhập phân tích tối đa 12 tháng giao dịch gần nhất và ưu tiên có ít nhất 6 tháng dữ liệu khi khả dụng. Quy trình hiện tại:

- Nhận diện giao dịch lương từ category, subcategory và mô tả giao dịch.
- Phát hiện các khoản thu nhập ngoài lương có tính lặp lại.
- Loại chuyển khoản nội bộ, hoàn tiền, giải ngân khoản vay và các khoản tiền vào bất thường hoặc chỉ xuất hiện một lần.
- Loại các tháng quan sát không đầy đủ ở biên thời gian.
- Tổng hợp theo từng khách hàng và từng tháng.
- Áp dụng xử lý ngoại lệ thận trọng, sau đó lấy phân vị 10% của thu nhập định kỳ hàng tháng để tránh đánh giá quá cao khả năng trả nợ.
- Tính `income_stability_score` và `income_confidence` từ độ dài lịch sử, mức độ lặp lại và biến động dòng tiền.

`customers.monthly_income` chỉ được dùng làm ground truth để đánh giá, không tham gia vào logic ước tính hoặc đặc trưng mô hình.

### 2. Mô hình rủi ro tín dụng

Mô hình cuối cùng là `HistGradientBoostingClassifier`, được chọn bằng tập validation. Ngưỡng phân loại cũng được tối ưu trên validation; test set chỉ được sử dụng một lần cho đánh giá cuối cùng.

Các đặc trưng an toàn trước thời điểm cấp khoản vay gồm:

- `loan_to_income`
- `installment_to_income`
- `dti`
- `years_experience`
- `term_months`
- `log_annual_income`
- `log_loan_amount`
- `home_ownership`
- `purpose`
- `verification_status`

Ánh xạ nhãn:

- `Fully Paid` → 0
- `Charged Off` → 1
- `Default` → 1
- Các trạng thái chưa có kết quả cuối cùng bị loại khỏi tập huấn luyện.

Những trường phát sinh sau khi giải ngân không được dùng, bao gồm dư nợ còn lại, tổng thanh toán, tiền gốc/lãi đã thu, recoveries, lịch sử thanh toán sau giải ngân, last-FICO, hardship và settlement. `grade`, `sub_grade`, `int_rate` và installment gốc của Lending Club cũng được loại để tránh dùng tín hiệu quyết định hoặc định giá của bên cho vay.

### 3. Chấm điểm và quyết định

Điểm tín dụng là tổng có trọng số của năm nhóm tín hiệu:

| Thành phần | Trọng số |
| --- | ---: |
| Thu nhập | 25% |
| Độ ổn định thu nhập | 25% |
| Hành vi chi tiêu | 20% |
| Khả năng chi trả | 15% |
| Rủi ro vỡ nợ | 15% |

Khi thiếu một thành phần, các trọng số còn lại được chuẩn hóa. Điểm cuối cùng được ánh xạ sang nhóm rủi ro:

| Điểm | Risk tier |
| --- | --- |
| 80–100 | A |
| 60–79.99 | B |
| 40–59.99 | C |
| Dưới 40 | D |

Các guardrail có thể cấu hình tại `config/scoring.json`:

- Tỷ lệ installment/thu nhập trên 30%: không tự động duyệt.
- Tổng nghĩa vụ nợ/thu nhập trên 45%: không tự động duyệt; trên 70%: từ chối.
- Chi phí/thu nhập trên 90%: không tự động duyệt; trên 120%: từ chối.
- PD trên 25%: không tự động duyệt.
- Độ tin cậy thu nhập dưới 70%: không tự động duyệt.

Guardrail được áp dụng sau bước chấm điểm, vì vậy điểm hoặc risk tier tốt không thể ghi đè một điều kiện rủi ro nghiêm trọng.

## Kết quả đánh giá hiện tại

### Income estimation

Đánh giá trên dữ liệu giao dịch với cửa sổ lịch sử tối đa 12 tháng:

| Chỉ số | Kết quả |
| --- | ---: |
| MAE | 877.00 USD |
| RMSE | 1,724.37 USD |
| MAPE | 22.89% |
| Median APE | 8.83% |
| Trong ±10% | 61.60% |
| Trong ±20% | 72.40% |

### Credit risk model

Kết quả trên test set độc lập gồm 60.000 khoản vay:

| Chỉ số | Kết quả |
| --- | ---: |
| ROC-AUC | 0.6667 |
| PR-AUC | 0.3349 |
| Precision | 0.3010 |
| Recall | 0.5745 |
| F1 | 0.3951 |
| Brier score | 0.1512 |
| Classification threshold | 0.2067 |

Confusion matrix:

```text
[[31798, 16118],
 [ 5142,  6942]]
```

Việc chia dữ liệu được thực hiện theo tỷ lệ 180.000 train / 60.000 validation / 60.000 test trên 300.000 khoản vay có kết quả xác định. Toàn bộ preprocessing, encoding và lựa chọn đặc trưng được fit bằng train data; model và threshold được chọn bằng validation data.

## Ví dụ đầu ra end-to-end

Kết quả hiện tại cho khách hàng `C0000001`:

| Trường | Giá trị |
| --- | ---: |
| Actual monthly income | 5,673.94 |
| Estimated monthly income | 6,210.63 |
| Income stability | 0.9517 |
| Income confidence | 0.9500 |
| Expense/income ratio | 1.0077 |
| Affordability ratio | 0.5433 |
| Probability of Default | 0.2457 |
| Credit score | 61.15 |
| Risk tier | B |
| Decision | `MANUAL_REVIEW` |

Mặc dù điểm thuộc tier B, tỷ lệ chi phí/thu nhập vượt ngưỡng guardrail nên hệ thống không đưa ra `AUTO_APPROVE`.

## Cấu trúc thư mục

```text
CreditForYou/
├── app.py                         # Giao diện Streamlit
├── db.py                          # Lớp truy cập SQLite
├── creditforyou/
│   ├── datasets.py                # Nạp và kiểm tra dữ liệu
│   ├── income.py                  # Ước tính thu nhập
│   ├── lendingclub.py             # Chuẩn bị dữ liệu Lending Club
│   ├── model.py                   # Huấn luyện và suy luận PD
│   ├── scoring.py                 # Tính điểm tín dụng
│   ├── decision.py                # Guardrail và quyết định
│   ├── pipeline.py                # Pipeline end-to-end
│   ├── evaluation.py              # Metrics và đánh giá
│   ├── explain.py                 # Giải thích kết quả
│   ├── io.py                      # Đọc/ghi artifact
│   └── report.py                  # Tạo báo cáo
├── config/
│   ├── schema.yaml                # Schema dữ liệu
│   └── scoring.json               # Trọng số và ngưỡng quyết định
├── scripts/
│   ├── build_db.py                # Tạo SQLite database
│   ├── train_pd_model.py          # Huấn luyện mô hình PD
│   ├── evaluate_income.py         # Đánh giá income estimator
│   └── evaluate_end_to_end.py     # Đánh giá toàn pipeline
├── tests/                         # Bộ kiểm thử tự động
├── reports/                       # Kết quả đánh giá sinh ra
└── legacy/                        # Mã cũ được giữ để tham khảo
```

## Chuẩn bị dữ liệu

Dữ liệu thô không được đưa lên Git do dung lượng lớn. Tạo thư mục `dataset isb` ở thư mục gốc với cấu trúc:

```text
dataset isb/
├── accepted_2007_to_2018Q4.csv
└── data/
    ├── customers.csv
    ├── accounts.csv
    ├── transactions.csv
    ├── loans.csv
    ├── cards.csv             # tùy chọn
    ├── merchants.csv         # tùy chọn
    └── subscriptions.csv     # tùy chọn
```

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chạy dự án

Tạo database và artifact cần thiết:

```powershell
python scripts/build_db.py
python scripts/evaluate_income.py
python scripts/train_pd_model.py
python scripts/evaluate_end_to_end.py
```

Khởi động giao diện:

```powershell
streamlit run app.py
```

Chạy đánh giá cho một khách hàng cụ thể:

```powershell
python scripts/evaluate_end_to_end.py --customer-id C0000001
```

## Kiểm thử

```powershell
pytest -q
```

Bộ kiểm thử hiện tại gồm 21 test cho logic thu nhập, dữ liệu Lending Club, chống leakage, scoring, guardrail và pipeline tích hợp.

## Giới hạn

- Dữ liệu giao dịch khách hàng là dữ liệu tổng hợp nên chưa phản ánh đầy đủ hành vi trong môi trường thực tế.
- Lending Club và dữ liệu giao dịch không đại diện cho cùng một quần thể khách hàng.
- Hiệu năng PD còn ở mức mô hình nghiên cứu và cần kiểm định thêm trước khi áp dụng thực tế.
- Các ngưỡng quyết định cần được hiệu chỉnh theo khẩu vị rủi ro, quy định pháp lý và dữ liệu của từng tổ chức.
- Hệ thống chưa thay thế các bước KYC, kiểm tra gian lận, xác minh hồ sơ hoặc thẩm định thủ công.
