# Kế hoạch hoàn thiện project AltCredict theo hướng Income Estimation + Credit Risk

## 0. Mục tiêu của task

Hãy dùng repository AltCredict hiện tại làm **code base** và refactor thành prototype cho hệ thống:

```text
Transaction data
    -> Income Estimation
    -> Income Stability / Confidence / Cash-flow features

Lending Club loan data
    -> Credit Risk Model
    -> Probability of Default (PD)

Income/Cash-flow outputs + PD + applicant/loan information
    -> Scoring Engine
    -> Credit Score 0-100
    -> Tier A/B/C/D
    -> Approval / Manual Review / Reject
```

Mục tiêu là giữ lại những phần có ích của AltCredict như UI, database, dashboard, prediction/explainability flow, nhưng thay phần dữ liệu, income logic và credit-risk training để phù hợp với project này.

Đây là prototype học thuật. Không được mô tả model như hệ thống tín dụng production-ready hoặc đại diện cho khách hàng Việt Nam nếu chưa có dữ liệu thực tế tương ứng.

---

## 1. Việc đầu tiên cần làm trước khi sửa code

Trước khi thay đổi bất kỳ file nào:

1. Đọc toàn bộ cấu trúc repository hiện tại.
2. Xác định các file đang phụ trách:
   - Streamlit/UI.
   - database.
   - data loading.
   - preprocessing.
   - feature engineering.
   - target generation.
   - model training.
   - prediction.
   - evaluation.
   - SHAP/explainability.
   - scoring/rule engine.
3. Chạy project gốc để xác nhận baseline hiện tại hoạt động.
4. Ghi lại các file sẽ giữ, sửa, thay hoặc bỏ.
5. Không xóa UI hoặc chức năng đang chạy được chỉ vì muốn viết lại cho đẹp.

Sau khi audit xong mới tiến hành refactor.

---

## 2. Dataset mới được thêm vào repository

Tôi sẽ thêm một folder dataset vào project. Không sửa trực tiếp raw dataset.

Claude cần tự detect path thực tế của folder, nhưng schema dự kiến gồm hai nhóm dữ liệu.

### 2.1. Lending Club

File chính:

```text
accepted_2007_to_2018Q4.csv
```

File có 151 cột. Một số cột đáng chú ý:

```text
loan_amnt
term
int_rate
installment
emp_title
emp_length
home_ownership
annual_inc
verification_status
issue_d
loan_status
purpose
dti
delinq_2yrs
earliest_cr_line
fico_range_low
fico_range_high
inq_last_6mths
open_acc
pub_rec
revol_bal
revol_util
total_acc
collections_12_mths_ex_med
acc_now_delinq
tot_cur_bal
mort_acc
pub_rec_bankruptcies
tax_liens
...
```

Dataset này chỉ dùng cho **Credit Risk Model / Probability of Default**.

### 2.2. Synthetic banking dataset

Các file hiện có:

```text
output/customers.csv
output/accounts.csv
output/cards.csv
output/loans.csv
output/merchants.csv
output/subscriptions.csv
output/transactions.csv
```

Các file chính cho MVP:

```text
customers.csv
accounts.csv
transactions.csv
loans.csv
```

Các file optional:

```text
cards.csv
subscriptions.csv
merchants.csv
```

#### customers.csv

Các cột:

```text
customer_id
first_name
last_name
sex
age
birth_date
city
postal_code
country
locale_code
family_situation
num_children
profession
customer_segment
profile
bank_preset
monthly_income
estimated_wealth
financial_score
risk_appetite
customer_since
```

`monthly_income` được dùng làm **ground truth để đánh giá income estimator**.

QUAN TRỌNG:

```text
customers.monthly_income
```

không được dùng làm input để tính `estimated_income`, nếu không sẽ tạo leakage và việc đánh giá income estimation mất ý nghĩa.

#### accounts.csv

```text
account_id
customer_id
iban
bic
bank_name
account_type
role
opened_at
initial_balance
current_balance
overdraft_limit
status
```

#### loans.csv

```text
loan_id
customer_id
account_id
loan_type
principal
outstanding_balance
annual_rate
term_months
monthly_payment
start_date
end_date
payment_day
status
```

File này dùng để tạo current debt / monthly debt obligation / affordability features cho khách hàng synthetic. Không mặc định dùng file này làm training target cho default model.

#### transactions.csv

Các cột:

```text
transaction_id
account_id
customer_id
timestamp
amount
currency
merchant_name
merchant_category
merchant_mcc
transaction_type
payment_channel
city
country
counterparty_iban
label
category
subcategory
is_subscription
is_salary
is_transfer
is_cash_withdrawal
balance_after_transaction
is_foreign
original_amount
original_currency
fx_rate
foreign_fee
```

Đây là file chính cho Income Estimation và cash-flow analysis.

---

## 3. Nguyên tắc data architecture

Không merge Lending Club và synthetic banking data theo `customer_id` hoặc `account_id`.

Hai dataset thuộc hai population khác nhau.

Luồng đúng:

```text
Synthetic transactions
    -> income/cash-flow module

Lending Club
    -> train credit-risk model

Khi chạy inference cho một applicant mới:
    income/cash-flow outputs
    + applicant information
    + requested loan information
    + những credit fields thực sự có ở thời điểm inference
    -> credit-risk/scoring pipeline
```

Nếu một feature chỉ tồn tại trong Lending Club nhưng hệ thống thực tế không thể cung cấp feature đó khi user mới xin vay, không được âm thầm tạo số giả hoặc điền constant để model chạy.

Giải quyết bằng một trong hai cách:

1. thêm field tương ứng vào form/applicant input; hoặc
2. loại feature đó khỏi model và train lại bằng tập feature có thể cung cấp thật lúc inference.

Ưu tiên phương án 2 cho MVP nếu form sẽ trở nên quá phức tạp.

---

# PHẦN A. Income Estimation Module

## 4. Tạo module mới cho transaction analysis

Tạo module rõ ràng, ví dụ:

```text
ml/income_estimation.py
ml/transaction_features.py
```

Tên file có thể thay đổi để phù hợp với structure hiện tại, nhưng logic phải tách khỏi credit model.

### 4.1. Xác định income transactions

Không được coi mọi CREDIT hoặc mọi amount dương là income.

Một transaction có thể được coi là income candidate nếu thỏa điều kiện phù hợp với schema, ví dụ:

```text
amount > 0
AND is_transfer != true
AND không phải loan disbursement
AND không phải refund/reversal nếu dataset có label tương ứng
AND thuộc salary / income / business income / recurring external income
```

Ưu tiên các signal:

```text
is_salary
category
subcategory
label
transaction_type
merchant_category
```

Phải hỗ trợ cả khách hàng không có salary cố định. Với freelancer/gig/self-employed, income có thể là nhiều khoản credit nhỏ hoặc recurring external credits.

Nếu phải dùng heuristic để nhận diện non-salary income, viết heuristic thành function riêng và document rõ. Không hard-code lẫn trong UI.

### 4.2. Monthly aggregation

Với mỗi `customer_id`:

1. parse `timestamp`.
2. group theo calendar month.
3. tính eligible income của từng tháng.
4. tính expense của từng tháng.
5. tính net cash flow.
6. tính balance behavior nếu dữ liệu cho phép.

Tạo ít nhất các feature:

```text
active_months
income_transaction_count
avg_monthly_income
median_monthly_income
min_monthly_income
max_monthly_income
std_monthly_income
income_cv
income_stability_score
income_confidence
recurring_income_ratio
income_source_count
avg_monthly_expense
expense_income_ratio
avg_monthly_net_cashflow
positive_cashflow_month_ratio
avg_balance
min_balance
cashflow_volatility
```

### 4.3. Công thức mặc định cho prototype

Tách công thức ra config/constants để có thể chỉnh sau, không rải magic numbers trong code.

#### Income CV

```text
income_cv = std_monthly_income / avg_monthly_income
```

Nếu mean bằng 0 thì xử lý an toàn, không chia 0.

#### Stability score

Có thể dùng baseline:

```text
income_stability_score = clip(1 - income_cv, 0, 1)
```

Đây là heuristic prototype, phải ghi rõ trong code/docs rằng chưa được ngân hàng validate.

#### Estimated monthly income

Ưu tiên estimator robust trước:

```text
base_income = median(monthly_eligible_income)
```

Sau đó có thể áp dụng conservative adjustment dựa trên stability, ví dụ:

```text
estimated_monthly_income = base_income * (0.85 + 0.15 * income_stability_score)
```

Nếu repo hiện tại đã có cách tính tốt hơn và có lý do rõ ràng, có thể giữ nhưng phải so sánh bằng metrics.

#### Confidence

Confidence phải phản ánh lượng lịch sử giao dịch và data coverage, không phải random percentage.

Baseline gợi ý:

```text
< 3 active months  -> low confidence
3-5 months         -> medium
6-8 months         -> good
9-12 months        -> high
> 12 months        -> very high
```

Nên trả về cả:

```text
income_confidence_score  # 0-1
income_confidence_label  # Low/Medium/High
```

Có thể giảm confidence nếu income transaction quá thưa hoặc tỷ lệ transfer quá cao.

---

## 5. Đánh giá Income Estimation

Ground truth:

```text
customers.monthly_income
```

Prediction:

```text
estimated_monthly_income
```

Tính tối thiểu:

```text
MAE
RMSE
MAPE hoặc sMAPE
median absolute percentage error
% customers có error <= 10%
% customers có error <= 20%
```

Không tự gọi kết quả là "90% accuracy" nếu metric chưa được định nghĩa.

Nếu project muốn giữ KPI "income estimation accuracy >= 90%", định nghĩa rõ KPI, ví dụ:

```text
percentage of customers whose estimated income is within +/-10% of ground truth
```

Chỉ báo cáo con số thực tế sau khi chạy evaluation.

Tạo script/report riêng, ví dụ:

```text
scripts/evaluate_income_estimation.py
```

Output ít nhất:

```text
reports/income_metrics.json
reports/income_predictions.csv
```

---

# PHẦN B. Lending Club Credit Risk Model

## 6. Bỏ synthetic default target của AltCredict

Nếu repo hiện tại có logic kiểu:

```text
target_generation.py
```

và target default được tự tạo bằng rule từ DTI, delinquency, utilization, transaction behavior..., không dùng target đó cho final model.

Lý do: model sẽ học lại chính rule dùng để tạo label và performance có thể bị thổi phồng.

Có thể giữ file cũ trong archive/reference nếu cần, nhưng production pipeline của prototype phải dùng outcome thực từ Lending Club.

---

## 7. Tạo target Lending Club

Dùng `loan_status`.

Baseline mapping:

```text
Fully Paid  -> 0
Charged Off -> 1
Default     -> 1   # nếu status này tồn tại đủ rõ trong file
```

Các trạng thái chưa có outcome cuối cùng như:

```text
Current
Issued
In Grace Period
Late (16-30 days)
Late (31-120 days)
```

không mặc định ép thành 0/1 trong model baseline. Exclude khỏi training dataset trừ khi có quyết định khác được document rõ.

Tạo function riêng:

```text
build_default_target()
```

và log số lượng từng class sau mapping.

---

## 8. Feature selection cho Lending Club

Chỉ dùng feature có thể biết tại hoặc trước thời điểm underwriting.

### 8.1. Candidate features

Bắt đầu từ nhóm này rồi kiểm tra missingness và khả năng cung cấp tại inference:

```text
loan_amnt
term
emp_length
home_ownership
annual_inc
verification_status
purpose
dti
delinq_2yrs
earliest_cr_line
fico_range_low
fico_range_high
inq_last_6mths
open_acc
pub_rec
revol_bal
revol_util
total_acc
collections_12_mths_ex_med
acc_now_delinq
tot_cur_bal
mort_acc
pub_rec_bankruptcies
tax_liens
```

### 8.2. Cẩn thận với platform-derived features

Các cột như:

```text
grade
sub_grade
int_rate
installment
```

có thể đã chứa thông tin từ underwriting/risk assessment của Lending Club.

Không mặc định dùng chúng trong model chính nếu mục tiêu là xây scoring độc lập. Có thể test ở experiment riêng, nhưng baseline nên ưu tiên applicant/credit features có thể tái tạo.

### 8.3. Cấm leakage từ thông tin sau giải ngân

Không dùng các cột dạng outcome/post-loan, ví dụ:

```text
out_prncp
out_prncp_inv
total_pymnt
total_pymnt_inv
total_rec_prncp
total_rec_int
total_rec_late_fee
recoveries
collection_recovery_fee
last_pymnt_d
last_pymnt_amnt
next_pymnt_d
last_credit_pull_d
last_fico_range_high
last_fico_range_low
hardship_*
debt_settlement_*
settlement_*
```

Tạo một `LEAKAGE_COLUMNS` list hoặc rule rõ ràng trong code.

Model training phải fail/warn nếu leakage columns vô tình xuất hiện trong feature set.

---

## 9. Training pipeline phải sửa leakage kỹ thuật

Không được làm:

```text
fit preprocessing trên full dataset
-> feature selection
-> train/test split
```

Pipeline đúng:

```text
raw labeled data
-> train/test split
-> fit preprocessing chỉ trên TRAIN
-> fit imputer/encoder/scaler chỉ trên TRAIN
-> fit feature selection chỉ trên TRAIN
-> transform TEST bằng các object đã fit trên TRAIN
-> train model
-> evaluate
```

Ưu tiên dùng `sklearn.pipeline.Pipeline` + `ColumnTransformer` để tránh leakage.

Nếu dùng cross-validation, preprocessing phải nằm bên trong pipeline CV.

---

## 10. Xử lý file Lending Club lớn

Không load 151 cột full file vào RAM nếu không cần.

Làm theo hướng:

1. đọc header.
2. xác định `usecols`.
3. load chỉ các cột cần.
4. dùng `chunksize` nếu RAM không đủ.
5. sau preprocessing cơ bản có thể cache dataset sạch sang Parquet.

Ví dụ structure:

```text
data/raw/                # raw, read-only
<data folder hiện tại>

data/processed/
    lendingclub_model_data.parquet
    transaction_features.parquet
    income_predictions.csv
```

Không commit file processed quá lớn nếu repo không phù hợp.

---

## 11. Model baseline

Train ít nhất:

```text
Logistic Regression
```

Có thể thêm model tree-based nếu dependency sẵn có và repo support tốt:

```text
Random Forest
XGBoost/LightGBM
```

Không ép phải thêm XGBoost nếu làm dependency phức tạp.

Mục tiêu của MVP là pipeline đúng và reproducible trước khi chạy đua metric.

Vì output cần dùng như Probability of Default, đánh giá thêm calibration nếu có thể.

Metrics tối thiểu:

```text
ROC-AUC
PR-AUC
Precision
Recall
F1
Confusion Matrix
Brier Score hoặc calibration curve nếu triển khai được
```

Không dùng Accuracy làm metric duy nhất vì default thường mất cân bằng class.

Lưu:

```text
model artifact
preprocessor/pipeline artifact
feature list
metrics JSON
training metadata
```

Không dùng model `.pkl` cũ của AltCredict làm final model.

---

# PHẦN C. Nối Income Module với Credit Model

## 12. Mapping income vào credit model

Income module trả:

```text
estimated_monthly_income
```

Lending Club dùng:

```text
annual_inc
```

Khi inference:

```text
annual_inc_for_model = estimated_monthly_income * 12
```

Đây là mapping giữa hai module. Không lấy `customers.monthly_income` ground truth để thay cho estimated income trong luồng demo chính.

Có thể có chế độ debug/evaluation dùng ground truth để so sánh, nhưng UI phải phân biệt rõ.

---

## 13. DTI / affordability

Không dùng công thức `total loan amount / monthly income` làm chỉ số affordability chính.

Ưu tiên:

```text
installment_income_ratio = requested_monthly_installment / estimated_monthly_income
```

Nếu có existing loans trong `output/loans.csv`:

```text
existing_monthly_debt = sum(monthly_payment của active/outstanding loans)
```

Tạo proxy DTI:

```text
dti_proxy = existing_monthly_debt / estimated_monthly_income
```

Nếu muốn tính thêm requested installment:

```text
post_loan_debt_ratio = (existing_monthly_debt + requested_monthly_installment)
                       / estimated_monthly_income
```

Phải document đơn vị: ratio 0-1 hay percent 0-100. Không trộn hai kiểu.

---

## 14. Không đưa feature transaction mới vào Lending Club model nếu Lending Club không có feature tương ứng

Các output như:

```text
income_stability_score
income_confidence
recurring_income_ratio
cashflow_volatility
positive_cashflow_month_ratio
```

không có trực tiếp trong Lending Club training data.

Do đó trong MVP:

```text
Không train Lending Club model bằng các feature này.
```

Chúng sẽ đi vào **final scoring/rule layer**, tách khỏi PD model.

Sau này chỉ nên train end-to-end bằng transaction features nếu có dataset thực sự chứa cả transaction history và default outcome cho cùng một borrower population.

---

# PHẦN D. Final Scoring Engine

## 15. Refactor scoring thành module riêng

Ví dụ:

```text
ml/credit_scoring.py
```

Không hard-code scoring logic trong UI.

Scoring engine nhận tối thiểu:

```text
income_stability_score
income_confidence
installment_income_ratio hoặc post_loan_debt_ratio
repayment/credit-history signal nếu có thật
probability_of_default
```

### 15.1. Default weight configuration

Giữ tinh thần của proposal hiện tại nhưng điều chỉnh affordability cho đúng đơn vị.

Có thể bắt đầu bằng config:

```yaml
income_stability: 25
affordability: 25
credit_or_repayment_history: 20
financial_tenure_or_experience: 15
ml_pd: 15
```

Tổng = 100.

Nếu một component không có dữ liệu thật ở inference, không tự tạo giá trị giả. Phải:

- yêu cầu input; hoặc
- disable component; hoặc
- normalize trọng số còn lại theo rule rõ ràng.

Ưu tiên hiển thị "insufficient data" hơn là bịa điểm.

### 15.2. Age / sex

Không dùng `sex` trong credit score hoặc model.

Không dùng `age` làm scoring feature mặc định cho MVP. Có thể giữ để phân tích dữ liệu hoặc hiển thị hồ sơ nếu cần.

Nếu code cũ đang dùng age trực tiếp để quyết định credit score, tách khỏi default config và document lại.

### 15.3. PD component

Nếu PD nằm trong [0,1]:

```text
pd_component_score = (1 - PD) * 15
```

Nếu model trả probability chưa calibrated, UI phải gọi đúng là `model predicted default probability`, không trình bày như xác suất ngân hàng đã được validate.

---

## 16. Tiering và decision

Có thể giữ rule hiện tại theo proposal:

```text
A: 80-100
B: 60-79
C: 40-59
D: <40
```

Decision:

```text
A/B -> Auto approve trong prototype
C   -> Manual review
D   -> Reject
```

Nhưng wording trong UI/docs phải ghi đây là **prototype rule**, không phải policy chính thức của ngân hàng.

Đưa thresholds vào config để chỉnh được mà không sửa code.

---

# PHẦN E. UI / Dashboard

## 17. Giữ UI AltCredict nhưng đổi flow

Không cần viết lại Streamlit UI từ đầu nếu app hiện tại chạy ổn.

UI mới nên có các khu vực sau.

### Applicant / customer selection

Cho phép chọn một synthetic `customer_id` để demo transaction history.

Hiển thị:

```text
customer profile
number of accounts
transaction coverage period
number of transactions
existing loans
```

### Income Analysis

Hiển thị:

```text
Estimated Monthly Income
Ground Truth Monthly Income  # chỉ trong evaluation/demo mode
Estimation Error
Income Stability
Income Confidence
Active Months
Average Expense
Expense/Income Ratio
Average Net Cash Flow
```

Nên có monthly income chart và monthly cash-flow chart nếu UI hiện tại hỗ trợ dễ dàng.

### Credit application

Form cho:

```text
requested loan amount
term
purpose
```

và các credit/applicant fields model thật sự cần nhưng không suy ra được từ synthetic banking dataset.

Không tạo random default values chỉ để model chạy.

### Credit Risk

Hiển thị:

```text
Predicted Default Probability
model version
main explanatory factors
```

Nếu SHAP hiện tại hoạt động, giữ và refactor để dùng model mới.

### Final Decision

Hiển thị:

```text
Credit Score
Tier
Decision
component breakdown
reason codes
```

Reason code nên là mô tả cụ thể, ví dụ:

```text
High income volatility
High post-loan debt ratio
Low predicted default risk
Strong recurring income pattern
```

Tránh wording kiểu tuyệt đối như "customer will default".

---

# PHẦN F. Database

## 18. Giữ SQLite nếu đang chạy ổn

Không cần đổi database technology cho MVP.

Nếu schema cũ không đủ, thêm bảng hoặc field cho:

```text
customers
applications
income_assessments
credit_predictions
credit_scores
```

Một application nên lưu:

```text
application_id
customer_id
requested_loan_amount
term
estimated_income
income_stability
income_confidence
dti/post_loan_debt_ratio
probability_of_default
credit_score
tier
decision
model_version
created_at
```

Không cần copy toàn bộ hàng triệu raw transactions vào SQLite nếu app chỉ đọc chúng từ dataset.

---

# PHẦN G. Những phần của AltCredict cần giữ / sửa / bỏ

## 19. Giữ nếu đang hoạt động

```text
Streamlit app structure
SQLite/database layer
navigation/dashboard
prediction presentation
SHAP/explainability utilities nếu tương thích
existing CSS/layout
```

## 20. Sửa mạnh

```text
data_loader.py
preprocessing.py
feature_engineering.py
train.py
evaluate.py
predict.py
rule/scoring engine
```

## 21. Bỏ khỏi final pipeline

```text
synthetic default target generation từ rule
old pretrained model artifacts
old reported performance nếu không reproduce được
transaction dataset cũ làm data chính
```

Có thể giữ code cũ trong `legacy/` nếu cần tham khảo, nhưng app final không được gọi nó.

---

# PHẦN H. Folder structure gợi ý

Không bắt buộc đổi đúng 100% nếu repo hiện tại đã có structure hợp lý. Mục tiêu là separation of concerns.

```text
project/
|
|-- app.py
|-- db.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- ml/
|   |-- data_loader.py
|   |-- transaction_features.py
|   |-- income_estimation.py
|   |-- lendingclub_target.py
|   |-- credit_preprocessing.py
|   |-- credit_model.py
|   |-- credit_scoring.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- explain.py
|
|-- config/
|   |-- scoring.yaml
|   `-- model_features.yaml
|
|-- scripts/
|   |-- prepare_lendingclub.py
|   |-- build_transaction_features.py
|   |-- evaluate_income_estimation.py
|   `-- train_credit_model.py
|
|-- models/
|-- reports/
|-- tests/
|-- requirements.txt
`-- README.md
```

Nếu repository hiện tại có tên/path khác, giữ structure cũ khi hợp lý. Không refactor folder chỉ để giống sơ đồ này.

---

# PHẦN I. Tests bắt buộc

## 22. Data tests

Kiểm tra:

```text
customer_id join được customers <-> accounts <-> transactions <-> loans
transaction timestamp parse được
amount numeric
không duplicate transaction_id bất thường
monthly_income ground truth không lọt vào income estimator input
loan_status target mapping đúng
leakage columns không lọt vào model features
```

## 23. Pipeline tests

Ít nhất có test cho:

```text
income estimator chạy với customer có 12 tháng data
income estimator chạy với customer có ít tháng data
zero income month
no eligible income transaction
high transfer customer
credit model predict_proba trả [0,1]
scoring output nằm trong [0,100]
tier mapping đúng boundary
```

---

# PHẦN J. README cần viết lại

## 24. README final phải giải thích đúng data provenance

README phải nêu rõ:

### Synthetic banking data

Dùng cho:

```text
transaction analysis
income estimation
income stability
income confidence
cash-flow behavior
```

Dữ liệu là synthetic.

### Lending Club

Dùng cho:

```text
historical loan outcome
credit-risk training
probability of default model
```

Dataset là US Lending Club historical data. Không được tuyên bố đại diện trực tiếp cho Vietnam banking population.

### Model limitation

Nêu rõ:

```text
PD model và transaction-income module được xây từ hai dataset khác nhau.
Transaction behavior features chưa được train end-to-end với actual default outcome.
Final score là prototype hybrid scoring framework.
```

Điểm này phải minh bạch trong README và report.

---

# PHẦN K. Thứ tự thực hiện

Claude hãy triển khai theo đúng thứ tự dưới đây, không sửa tất cả cùng lúc.

## Phase 1 - Audit và chạy baseline

```text
- inspect repo
- run current app
- identify reusable modules
- report planned file changes
```

## Phase 2 - Data ingestion

```text
- load synthetic banking files
- validate joins
- load Lending Club bằng usecols/chunking
- create processed data folder/cache
```

## Phase 3 - Income module

```text
- transaction classification
- monthly aggregation
- income features
- estimated income
- stability
- confidence
- evaluation vs customers.monthly_income
```

Chỉ chuyển phase khi evaluation script chạy được.

## Phase 4 - Credit-risk model

```text
- clean Lending Club
- map loan_status -> default_flag
- remove unresolved statuses
- prevent leakage
- split before preprocessing
- train baseline model
- evaluate
- save pipeline/model artifacts
```

## Phase 5 - Integration

```text
- estimated_monthly_income -> annualized income
- derive affordability/DTI proxy
- gather applicant inputs required by model
- predict PD
- final scoring engine
- tier/decision
```

## Phase 6 - UI

```text
- connect new pipeline to existing Streamlit UI
- income analysis page/section
- credit application inputs
- PD/explainability
- final score breakdown
```

## Phase 7 - Tests + docs

```text
- tests
- README
- requirements
- run instructions
- model/data limitations
```

---

# PHẦN L. Definition of Done

Project chỉ được coi là hoàn thiện MVP khi tất cả điều sau chạy được:

1. Có thể chọn một customer từ synthetic banking dataset.
2. App đọc lịch sử transaction của customer đó.
3. App tính được estimated monthly income mà không dùng `customers.monthly_income` làm input.
4. App tính income stability và confidence.
5. Có report so sánh estimated income với ground-truth monthly income.
6. Lending Club model được train bằng actual loan outcome, không dùng synthetic rule target.
7. Không có post-loan leakage columns trong model.
8. Train/test split xảy ra trước khi fit preprocessing/feature selection.
9. Model trả `predict_proba` hoặc equivalent PD output.
10. Income output được map sang input credit pipeline theo rule rõ ràng.
11. App tính final credit score 0-100.
12. App trả Tier A/B/C/D và decision.
13. UI hiển thị component breakdown/reasons.
14. README nói rõ synthetic transaction data, Lending Club data và giới hạn của việc dùng hai population khác nhau.
15. Project chạy lại từ đầu theo README trên một môi trường sạch.

---

# PHẦN M. Quy tắc làm việc khi Claude sửa repository

- Không rewrite toàn bộ project nếu chưa cần.
- Giữ những phần đang chạy ổn.
- Thực hiện từng phase và test sau mỗi phase.
- Không invent cột dataset không tồn tại.
- Không tạo dữ liệu giả để lấp field thiếu mà không báo.
- Không dùng `customers.monthly_income` làm feature của income estimator.
- Không merge synthetic users với Lending Club borrowers theo ID.
- Không dùng post-loan outcome columns làm feature dự đoán default.
- Không fit preprocessing trên toàn bộ dataset trước train/test split.
- Không tái sử dụng model cũ và metric cũ nếu chưa reproduce.
- Không thay đổi scoring thresholds/weights âm thầm. Đưa vào config và document.
- Khi gặp feature mismatch giữa training và inference, dừng và giải quyết ở schema/model design. Không điền random/constant chỉ để tránh lỗi.

---

# PHẦN N. Output mong muốn từ Claude sau khi hoàn thiện

Sau khi code xong, trả lại:

1. Danh sách file đã tạo/sửa/xóa.
2. Architecture final của project.
3. Dataset nào đi vào module nào.
4. Feature list cuối của income module.
5. Feature list cuối của credit-risk model.
6. Target mapping của `loan_status`.
7. Danh sách leakage columns đã loại.
8. Income estimation metrics thực tế.
9. Credit-risk metrics thực tế.
10. Cách chạy project từ môi trường sạch.
11. Các limitation còn lại.
12. Các task chưa làm được và lý do, nếu có.

Không tự tuyên bố project đạt KPI nếu chưa chạy ra metric tương ứng.
