# CreditForYou

CreditForYou là nguyên mẫu hệ thống **cấp hạn mức trả góp tại điểm bán (point-of-purchase, POP)** cho khách hàng thiếu lịch sử tín dụng (freelancer, gig worker, chủ shop online). Thay vì yêu cầu bảng lương hay lịch sử CIC, hệ thống đọc **dòng tiền trong tài khoản** để ước tính thu nhập, đo khả năng chi trả, rồi **cấp trước** một hạn mức. Tại quầy thanh toán, quyết định là **tức thì**, không có bước chờ người duyệt.

> Đây là dự án nghiên cứu/học thuật. Kết quả không thay thế quy trình thẩm định, chính sách tín dụng hoặc quyết định cho vay của tổ chức tài chính. Dataset hiện tại là dữ liệu tổng hợp **thay thế** (USD, quy đổi sang VND) cho vòng tự do; dataset chính thức của đề chỉ mở ở vòng Strategy, khi có chỉ cần thay vào `dataset isb/` và đặt `fx_dataset_to_vnd = 1`.

## Bài toán (theo đề, Exhibit 2)

**Nút thắt:** khách mua hàng tại điểm bán bị chuyển sang quy trình thẩm định thủ công tập trung (nộp giấy tờ, tra CIC, chuyên viên tín dụng duyệt). Quy trình này mất **1,8–4,6 ngày làm việc** và tốn **380.000 VND/hồ sơ** bất kể khoản vay lớn hay nhỏ. Khi có kết quả thì thời điểm mua hàng đã qua.

**Khoản vay nhỏ không đủ trả chi phí thẩm định:** một khoản 5 triệu VND, 6 tháng, lãi 12% chỉ mang lại khoảng **276.000 VND** (lãi + 2% phí đối tác), thấp hơn chi phí thẩm định 380.000 VND. Thẩm định thủ công chỉ hoà vốn với khoản vay từ khoảng **6,9 triệu VND** trở lên, trong khi khoảng vay của đề bắt đầu từ 1,6 triệu.

**Nguyên nhân gốc:** quy trình dựa vào giấy tờ và hồ sơ CIC, thứ mà gig worker, chủ shop online và khách vay lần đầu không có. Trong khi đó, dòng tiền của họ đã nằm sẵn trong tài khoản tại ngân hàng nhưng chưa được dùng.

**Bằng chứng (mô phỏng):** với quy tắc thay thế cho quy trình truyền thống (lương cố định + đã có lịch sử vay), chỉ **16,3%** gig worker và chủ shop online đủ điều kiện.

## Giải pháp: 3 bước tách rời

```text
 [A] CẤP HẠN MỨC TRƯỚC (chạy theo lô)  [B] TẠI QUẦY (tức thì)              [C] SAU KHI CHO VAY
 Giao dịch 12 tháng                    Đơn hàng ≤ hạn mức?                 3 kỳ đúng hạn -> nâng hạn mức +25%
  -> thu nhập theo loại nguồn thu       ├─ có  -> DUYỆT                    Trễ ≥ 30 ngày -> khoá hạn mức
  -> chi phí bắt buộc, nợ đang trả      ├─ vượt -> ĐỀ XUẤT KHÁC            Nhãn trả nợ thật -> train ML sau này
  -> khả năng trả hằng tháng            │   (kéo dài kỳ hạn / trả trước)
  -> điểm + nhóm A/B/C/D                └─ chưa có hạn mức -> TỪ CHỐI
  -> HẠN MỨC + trạng thái               Không có bước người duyệt tại quầy
```

| Trạng thái hạn mức | Khi nào | Tại quầy |
| --- | --- | --- |
| `PREAPPROVED` | Nhóm A/B, không vướng guardrail | Dùng toàn bộ hạn mức |
| `PREAPPROVED_REDUCED` | Nhóm C hoặc vướng guardrail mềm (chi tiêu cao, nợ cao...) | Vẫn tức thì, hạn mức × 0,5 |
| `OFFLINE_REVIEW` | Dữ liệu chưa đủ, hoặc khách thuộc nhóm dễ tổn thương (`VULNERABLE`) | Chuyên viên xem xét **ngoài giờ mua hàng** |
| `NOT_ELIGIBLE` | Quy tắc cứng (tuổi, nợ > 70%, chi tiêu > 120%), nhóm D, hoặc không còn dư để trả | Từ chối |

## Phương pháp

### 1. Ước tính thu nhập — `creditforyou/income.py`

- **Phân loại nguồn thu** theo nhóm giao dịch: lương / lương hưu / tự doanh (freelance, hoá đơn, doanh thu) / khác. Ưu tiên `subcategory`, sau đó từ khoá, cuối cùng mới đến cờ `is_salary`. Dataset ISB gắn `is_salary = True` cho cả 7.730 giao dịch freelance và lương hưu; nếu tin cờ này, thu nhập tự doanh sẽ bị coi như lương cố định.
- Loại chuyển khoản giữa các tài khoản của chính khách, hoàn tiền, giải ngân vay. Tiền vào "khác" phải lặp lại ≥ 2 tháng mới được tính.
- **Mức thu nhập gộp:** lương/lương hưu lấy trung vị các tháng có nhận; tự doanh lấy trung bình cả 12 tháng, tính cả tháng không có thu.
- **Hiệu chỉnh gộp → ròng theo loại nguồn thu** (`config/income_calibration.json`): hệ số = trung vị (thu nhập đã xác minh ÷ thu nhập gộp), học trên khách có ≥ 95% thu nhập từ một loại. Hệ số hiện tại: lương 0,887; lương hưu 0,886; tự doanh 0,533. Trong vận hành thật, nhóm "đã xác minh" là khách từng nộp bảng lương hoặc tờ khai thuế.
- **Hai con số thu nhập:**
  - `estimated_monthly_income`: ước tính tốt nhất (ròng), dùng cho chấm điểm.
  - `conservative_cash_income`: thu nhập gộp × (0,5 + 0,5 × độ ổn định), dùng cho khả năng chi trả, vì chi tiêu trong sao kê được trả bằng tiền gộp.
- **Chi phí:** tách chi phí bắt buộc (nhà, đi chợ, điện nước, viễn thông, y tế, bảo hiểm, xăng xe, trông trẻ), trả nợ (`mortgage`, `auto_loan`...) và chi tiêu tuỳ ý.

### 2. Điểm tín dụng — `creditforyou/scoring.py`

| Thành phần | Trọng số gốc | Công thức |
| --- | ---: | --- |
| Ổn định thu nhập | 25 | 1 − CV của thu nhập theo tháng |
| Tỷ lệ nợ/thu nhập | 25 | 1 − (trả góp mới + nợ đang trả) ÷ thu nhập |
| Thâm niên | 15 | min(số năm là khách hàng ÷ 30, 1) |
| Lịch sử trả nợ | 20 | số kỳ đúng hạn ÷ tổng; **dataset ISB không có dữ liệu này nên thành phần bị tắt** |
| Mô hình ML (PD) | 15 | 1 − PD; **mặc định tắt** (xem mục 4) |

Thành phần nào thiếu dữ liệu sẽ bị tắt, các thành phần còn lại được chuẩn hoá về 100. Nhóm: A ≥ 80, B ≥ 60, C ≥ 40, D < 40.

### 3. Hạn mức — `creditforyou/limit.py`, cấu hình `pos_limit` trong `config/scoring.json`

```text
Khả năng trả góp mới/tháng = min( 50% × (thu nhập tiền mặt bảo thủ − chi phí bắt buộc − nợ đang trả),
                                  20% × thu nhập ròng,
                                  50% × thu nhập ròng − nợ đang trả )
Hạn mức (VND) = min( khả năng trả × hệ số niên kim (6 tháng, 18%/năm),
                     trần nhóm (A 90 triệu / B 60 triệu / C 30 triệu; trần A = mức vay tối đa của đề),
                     bội số thu nhập (A 1,5× / B 1× / C 0,5×),
                     trần khách mới chưa có lịch sử vay 30 triệu ),  tối thiểu 1,6 triệu (mức vay nhỏ nhất của đề)
Kỳ hạn: 6 tháng chuẩn, tối đa 12 tháng (đề cho phép 3/6/9/12/18/24; giới hạn 12 cho khách thiếu lịch sử tín dụng)
```

Chi tiêu tuỳ ý **không** bị trừ, vì đó là phần khách có thể cắt bớt (và cũng là thứ được trả góp). Tại quầy, phương án "kéo dài kỳ hạn" chỉ được đưa ra khi hạn mức đang bị chặn bởi khả năng trả, **không bao giờ vượt trần cứng**.

### 4. Mô hình PD Lending Club — chỉ để so sánh (challenger)

Mô hình `HistGradientBoostingClassifier` vẫn được train trên 300.000 khoản vay Lending Club đã có kết cục (test AUC 0,667, chống rò rỉ dữ liệu, `dti` không gồm vay mua nhà). Tuy vậy nó **không dùng để quyết định** (`scoring.use_ml_pd = false`), vì Lending Club là vay tiêu dùng ở Mỹ 2007–2018, khác sản phẩm và khác quần thể khách. PD được in ra ở cột `pd_challenger` để so sánh song song. Bật lại khi có nhãn trả nợ thật của sản phẩm POP (bước [C]).

## Kết quả (mô phỏng minh hoạ, 500 khách)

> **Đọc kỹ:** đây là dataset **thay thế** (dữ liệu tổng hợp, tính bằng USD, quy đổi 25.000 VND/USD); dataset chính thức của đề chỉ mở ở vòng Strategy. Vì là dữ liệu kiểu Mỹ nên thu nhập quy đổi cao bất thường so với Việt Nam (khoảng 100–200 triệu VND/tháng), do đó hạn mức phần lớn chạm trần. Các con số dưới đây minh hoạ cơ chế, **không phải** dự báo cho HLBVN. Cơ cấu khách của dataset (63% chưa có lịch sử vay) khác đề, nên mọi tỷ lệ "toàn bộ" được tính lại theo cơ cấu Exhibit 2.

### Theo 5 phân khúc của đề

| Phân khúc (Exhibit 2) | Tỷ trọng theo đề | n mẫu | Quy tắc truyền thống (thay thế) | CreditForYou có hạn mức |
| --- | ---: | ---: | ---: | ---: |
| Salaried, with credit | 35,5% | 168 | 98,8% | 45,2%* |
| Salaried, no credit | 20,5% | 234 | 0% (theo định nghĩa) | 62,0% |
| Gig / platform workers | 15% | 26 | 7,7% | 57,7% |
| Online merchants (MSME) | 14,5% | 17 | 29,4% | 41,2% |
| First-time borrowers | 14,5% | 55 | 0% (theo định nghĩa) | 87,3% |
| **Tính lại theo cơ cấu đề** | 100% | 500 | 40,5% | **56,0%** |

\* Nhóm *Salaried with credit* **giữ nguyên điều kiện dựa trên CIC** (Exhibit 1: STP *"maintains terms while automating execution"*). Dòng tiền chỉ thêm cờ khả năng trả nợ: trong dữ liệu mô phỏng, nhiều khách nhóm này nợ vay mua nhà nặng và chi tiêu gần hết thu nhập.

Mốc "0%" của *Salaried no credit* và *First-time* đúng theo định nghĩa (quy tắc truyền thống đòi lịch sử vay), nên **không dùng làm bằng chứng**. Bằng chứng so sánh được là **gig worker + chủ shop online: 16,3% → 51,2%** (n = 43, khoảng tin cậy 95%: 37–65%; cần kiểm chứng lại ở giai đoạn backtest).

**Dòng khách so với quy tắc truyền thống:** +211 khách mới được phục vụ, −93 khách bị loại vì nợ quá cao, ròng +118 (173 → 291).

Trạng thái: 92 cấp đủ, 199 cấp giảm, 32 xem xét ngoài giờ (30 thuộc nhóm `VULNERABLE`), 177 không đủ điều kiện. Hạn mức trung vị **15 triệu VND**. Không hồ sơ nào phải chờ người duyệt tại quầy.

### Giả lập tại quầy (đơn hàng 5 / 15 / 40 triệu VND)

| Đơn | Duyệt | Đề xuất khác | Từ chối |
| ---: | ---: | ---: | ---: |
| 5 triệu | 57,2% | 1,0% | 41,8% |
| 15 triệu | 47,4% | 10,8% | 41,8% |
| 40 triệu | 9,2% | 49,0% | 41,8% |

### Ước tính thu nhập (đánh giá out-of-fold 5 lần, so với `customers.monthly_income`)

| Chỉ số | Trước (phân vị 10%) | Hiện tại |
| --- | ---: | ---: |
| Sai lệch trung vị | 8,8% | **1,7%** |
| Sai lệch ≤ 10% | 61,6% | **69,8%** |
| Sai lệch ≤ 20% | 72,4% | **75,2%** |
| Sai lệch ≤ 10%, không tính 31 khách không có thu nhập 12 tháng | – | **74,4%** |
| Freelancer: sai lệch ≤ 10% | 23% | **38%** |
| Entrepreneur: sai lệch ≤ 10% | 18% | 6% |

Các giới hạn đã biết:
- **31 khách không có khoản thu nhập nào trong 12 tháng** nhưng hồ sơ vẫn ghi thu nhập. Hệ thống ước tính 0 và không cấp hạn mức; đó là hành vi đúng về rủi ro.
- **Entrepreneur:** chủ doanh nghiệp chỉ tự trả lương nhỏ vào tài khoản cá nhân; thu nhập thật nằm ở tài khoản doanh nghiệp. Cần thêm dữ liệu tài khoản doanh nghiệp hoặc dữ liệu thanh toán từ đối tác bán hàng.
- **Freelancer** tiêu gần hết doanh thu gộp (số dư giảm trung vị khoảng 3,7 triệu VND trong 12 tháng). Hệ thống chỉ cấp hạn mức cho người còn dư và chặn người thâm hụt.

### Ví dụ: freelancer `C0000013`

`python -m creditforyou limits -c C0000013 --with-truth`

| Trường | Giá trị |
| --- | ---: |
| Ổn định thu nhập | 19% (3 tháng không có thu) |
| Khả năng trả góp mới | 9,8 triệu VND/tháng (bị chặn bởi phần dư khả dụng) |
| Điểm / nhóm | 46,4 / C |
| Hạn mức | **15 triệu VND** (`PREAPPROVED_REDUCED`: trần nhóm C 30 triệu × 0,5, vì chi tiêu 95% > 90%) |
| Đơn 5 triệu | Duyệt, 877.626 VND/tháng × 6 |
| Đơn 40 triệu | Trả trước 25 triệu, góp 15 triệu |

Theo quy trình hiện tại, khách này phải qua thẩm định thủ công 1,8–4,6 ngày; giờ được cấp hạn mức nhỏ ngay.

## Chất lượng dữ liệu

`python -m creditforyou quality` → `reports/data_quality.json`. Dataset gốc **không bị sửa**; mọi xử lý nằm trong pipeline nên chạy lại được khi đổi sang dataset chính thức.

| Mức | Phát hiện | Số lượng | Cách xử lý | Nếu không xử lý |
| --- | --- | ---: | --- | --- |
| Cao | `is_salary = True` cho cả tiền freelance (4.962) và lương hưu (2.768) | 7.730 / 28.655 (27%) | Phân loại nguồn thu theo `subcategory`/từ khoá trước cờ `is_salary` | Freelancer bị coi như có lương cố định |
| Cao | Chuyển khoản giữa các tài khoản của chính khách | 31.324 | Loại khỏi cả thu nhập và chi tiêu | Tiết kiệm bị tính là chi tiêu; duyệt tự động tụt từ 18% xuống 1,6% |
| Cao | Số dư sau giao dịch không liên tục | 76% dòng | Không dùng số dư để quyết định, chỉ dùng dòng tiền | Đặc trưng số dư sai |
| Cao | Khoản vay mua nhà trong nợ đang có | 133 / 233 | Tính trong khả năng chi trả; bỏ khỏi `dti` của mô hình PD | `dti` 54% nằm ngoài phân phối Lending Club, PD của C0000001 từ 11,5% thành 25,3% |
| Cao | Cột rò rỉ của Lending Club (dư nợ, đã trả, grade, int_rate...) | – | `assert_no_leakage` chặn tự động | AUC ảo |
| TB | Phí thấu chi ở trung vị 93% số tháng của mỗi khách | 111.363 | Chỉ tham khảo, không làm quy tắc | Không phân biệt được rủi ro |
| TB | Hồ sơ ghi thu nhập nhưng không có khoản thu nào trong 12 tháng | 31 / 500 | Thu nhập = 0, không cấp hạn mức | Tin thu nhập tự khai đã cũ |
| TB | `opened_at` bằng ngày xuất dữ liệu (giá trị mặc định) | 138 / 1.426 | Thâm niên lấy theo `customer_since` | Thâm niên sai |
| TB | `loans.csv` không có lịch sử trả đúng hạn | 233 / 233 | Tắt thành phần lịch sử trả nợ, chuẩn hoá trọng số | – |
| TB | Lending Club: khoản vay chưa có kết cục và dòng tổng kết cuối file | 305.087 | Loại khỏi tập train | Nhãn sai |
| Thấp | Thẻ `active` nhưng đã hết hạn | 717 / 901 | Không dùng bảng thẻ | – |
| OK | Trùng ID, ngày lỗi, số tiền rỗng, giao dịch trùng, khách mồ côi, tuổi lệch ngày sinh, trả góp sai công thức | 0 | Kiểm tra mỗi lần chạy | – |

## Bài toán kinh tế

`python -m creditforyou economics` → `reports/pos_economics.json`. Đơn vị **VND/năm** cho 291 khách có hạn mức (mô phỏng). Giả định trong `config/scoring.json > economics`, kèm nguồn neo:

| Giả định | Giá trị | Nguồn neo |
| --- | --- | --- |
| Chi phí thẩm định thủ công | 380.000 VND/hồ sơ | **Đề, Exhibit 2** |
| Chi phí quyết định tự động | 50.000 VND | Giả định của nhóm (eKYC, tra cứu, hạ tầng), cần xác nhận với HLBVN |
| Tổn thất khi vỡ nợ (LGD) | 91% | **Theo dữ liệu**: Lending Club, 90.150 khoản vỡ nợ |
| Dư nợ còn lại khi vỡ nợ (EAD) | 72% | **Theo dữ liệu**: Lending Club |
| PD 6 tháng, kịch bản cơ sở | A 2% / B 5% / C 10% | **Thận trọng**: ≈ 2× Lending Club hạng A/B/C quy về 6 tháng (1,0 / 2,3 / 3,8%); khớp KPI bản ý tưởng |
| Phí đối tác bán hàng | 2% | **Thận trọng**: phí BNPL bình quân 2,49% năm 2021 ([CFPB](https://files.consumerfinance.gov/f/documents/cfpb_buy-now-pay-later-market-trends-and-consumer-impacts_presentation_2022-11.pdf)) |
| Chi phí vốn | 6%/năm | **Thận trọng**: lãi tiền gửi 12 tháng VN 2025 khoảng 4,6–5,8% ([MBS](https://mbs.com.vn/?p=19561)) |
| Lãi suất theo nhóm | A 9–10% / B 11–13% / C 14–16% | Bản ý tưởng của nhóm |
| Vốn chủ phân bổ | 10% dư nợ | Tỷ lệ an toàn vốn tối thiểu 8% + đệm |
| Tần suất dùng hạn mức · đơn trung bình | 1,5 lần/năm · 60% hạn mức | Giả định, đo khi thí điểm |

| (triệu VND/năm) | Cơ sở | Căng thẳng ×2 | Nghiêm trọng ×3 | Theo PD Lending Club |
| --- | ---: | ---: | ---: | ---: |
| Giải ngân | 11.343 | 11.343 | 11.343 | 11.343 |
| Thu lãi + phí đối tác | 578 | 578 | 578 | 578 |
| Chi phí vốn + vận hành | 236 | 236 | 236 | 236 |
| Lỗ dự kiến | 246 | 493 | 739 | 123 |
| Tỷ lệ lỗ / giải ngân | 2,2% | 4,3% | 6,5% | 1,1% |
| **Lợi nhuận** | **95** | −151 | −398 | 219 |
| **RoC** | **28,5%** | −45,5% | −119,4% | 65,6% |

**Ngưỡng hoà vốn:**
- PD 6 tháng: A 4,3% · B 5,0% · C 6,0% · **toàn danh mục 4,6%**. Đây là **quy tắc dừng**.
- Tỷ lệ lỗ trên giải ngân: **3,0%**. Để đối chiếu, tỷ lệ xoá nợ BNPL ở Mỹ năm 2021 là 3,8% (CFPB).

**Độ nhạy (triệu VND/năm):** +1 điểm % phí đối tác = **+113** · +1 điểm % lãi suất = +33 · +1 điểm % chi phí vốn = −33.

**Vận hành:** 199 hồ sơ trước đây phải thẩm định nay được cấp tức thì (hạn mức giảm), tránh khoảng 65,7 triệu VND (199 × 330.000 VND) mỗi lượt xét. **211 khách mới** được phục vụ (lợi nhuận cơ sở 45,6 triệu, RoC 27%). Riêng 18 khách mới thuộc nhóm mục tiêu thì chỉ hoà vốn.

Đọc kết quả:
1. **Biên lợi nhuận mỏng** (0,8% giải ngân), và ngưỡng lỗ hoà vốn 3,0% thấp hơn mức xoá nợ BNPL Mỹ (3,8%). Đòn bẩy chính là phí đối tác và lãi suất.
2. **Vỡ nợ ×2 / ×3 đều lỗ.** Cơ chế chặn lỗ:
   - quy tắc dừng ở PD 4,6%;
   - trần khởi đầu 30 triệu VND;
   - cấp giảm ×0,5 cho hồ sơ vướng guardrail;
   - kỳ hạn ngắn;
   - ngân sách lỗ cố định khi thí điểm.
3. **PD Lending Club xếp sai thứ tự giữa các nhóm**, nên chỉ dùng làm tham chiếu so sánh.

## Lộ trình triển khai 6–12 tháng

| Giai đoạn | Việc chính | Điều kiện đi tiếp |
| --- | --- | --- |
| **0–3 tháng: chạy thử ngược (backtest) & chính sách** | Chạy pipeline (chỉ đọc, chưa cấp) trên dữ liệu tài khoản của khách **hiện hữu** của HLBVN. Học lại hệ số gộp → ròng trên khách đã có hồ sơ lương/thuế. Chạy `quality` trên dữ liệu thật. Hội đồng tín dụng duyệt `pos_limit` và `business_rules`. Hoàn tất cơ chế xin đồng ý xử lý dữ liệu cá nhân của khách | Sai lệch thu nhập trung vị ≤ 10%; kết quả nhóm mục tiêu giữ được trên mẫu lớn hơn (hiện n = 43, khoảng tin cậy 95% của tỷ lệ có hạn mức là 37–65%); chính sách được duyệt |
| **3–6 tháng: thí điểm có kiểm soát** | 1–2 đối tác bán hàng (điện máy / thương mại điện tử). Chỉ khách hiện hữu. Trần hạn mức thấp, **ngân sách lỗ cố định**. Theo dõi trễ hạn hằng tuần | Vỡ nợ 6 tháng ≤ 3% (dưới mức hoà vốn 4,6% với biên an toàn); quá hạn 30+ ngày nhóm A < 2%, B < 5% |
| **6–12 tháng: mở rộng + ML** | Thêm đối tác. Bật nâng hạn mức (3 kỳ đúng hạn → +25%). Dùng nhãn trả nợ thật để huấn luyện mô hình ML dự báo vỡ nợ, chạy song song với bảng điểm (champion–challenger) để tinh chỉnh hạn mức và giá. Bổ sung dữ liệu tài khoản doanh nghiệp cho chủ kinh doanh | **Dừng nếu vỡ nợ > 4,6%**; ML chỉ thay bảng điểm khi tốt hơn rõ rệt trên dữ liệu thật |

Trong giai đoạn đầu, giải pháp **không cần hạ tầng mới**: hạn mức được tính sẵn theo lô trên dữ liệu tài khoản ngân hàng đã có, còn tại quầy chỉ là **tra cứu hạn mức**. Bản thân phép so sánh mất dưới 1 ms; thời gian thực tế do API/kết nối POS quyết định, mục tiêu < 1 giây.

### KPI đo trực tiếp nút thắt

| KPI | Mục tiêu (giả định) | Hiện tại (dataset ISB) | Đo ở đâu |
| --- | --- | --- | --- |
| Thời gian ra quyết định | 1,8–4,6 ngày làm việc (đề) → < 1 giây | Tra cứu hạn mức tính sẵn | Thí điểm |
| Chi phí mỗi quyết định | 380.000 VND (đề) → ≤ 50.000 VND | Giả định | Thí điểm |
| Tỷ lệ chuyển đổi tại quầy (duyệt + đề xuất khác) | ≥ 60% | Giả lập đơn 5 triệu VND: 58% | `limits --all`; đo thật khi thí điểm |
| Tỷ lệ có hạn mức — gig worker + chủ shop online | ≥ 50% | 51,2% mô phỏng (khoảng tin cậy 95%: 37–65%, n = 43; truyền thống 16,3%) | `limits --all` |
| Hồ sơ phải xem xét ngoài giờ | ≤ 10% | 6,4% | `limits --all` |
| Sai lệch thu nhập trung vị (nhóm đã xác minh) | ≤ 10% | 1,7% | `scripts/evaluate_income_estimation.py` |
| Tỷ lệ vỡ nợ 6 tháng toàn danh mục | ≤ 3%; **dừng nếu > 4,6%** | Đo khi thí điểm | Theo dõi sau giải ngân |
| Quá hạn 30+ ngày theo nhóm | A < 2%, B < 5% | Đo khi thí điểm | Theo dõi sau giải ngân |
| RoC danh mục | ≥ ngưỡng ngân hàng | 28,5% (kịch bản cơ sở, mô phỏng) | `economics` |

## Cấu trúc thư mục

```text
CreditForYou/
├── app.py                         # Giao diện Streamlit (chấm 1 khoản vay cụ thể)
├── db.py                          # Lưu kết quả đánh giá vào SQLite
├── creditforyou/
│   ├── cli.py                     # Lệnh terminal: python -m creditforyou ...
│   ├── datasets.py                # Đọc dataset ISB, cache parquet, nợ đang có
│   ├── income.py                  # Ước tính thu nhập theo loại nguồn thu + đặc trưng dòng tiền
│   ├── limit.py                   # [A] cấp hạn mức, [B] quyết định tại quầy, [C] nâng/khoá hạn mức
│   ├── economics.py               # Bài toán kinh tế: lợi nhuận, lỗ dự kiến, PD hoà vốn, RoC
│   ├── quality.py                 # Báo cáo chất lượng dữ liệu
│   ├── scoring.py                 # Điểm tín dụng
│   ├── decision.py                # Guardrail, nhóm, lãi suất
│   ├── pipeline.py                # Pipeline end-to-end (preapprove_limits, evaluate_synthetic_customer)
│   ├── evaluation.py              # Hiệu chỉnh thu nhập + đánh giá out-of-fold
│   ├── model.py, lendingclub.py   # Mô hình PD challenger + chống rò rỉ
│   ├── explain.py, report.py, io.py, sample.py
├── config/
│   ├── scoring.json               # Mọi trọng số, ngưỡng, guardrail, tham số hạn mức
│   ├── income_calibration.json    # Hệ số gộp -> ròng (sinh bởi evaluate_income_estimation.py)
│   └── schema.json                # Map tên cột cho dataset ngoài ISB
├── scripts/                       # validate_data, evaluate_income_estimation, train_credit_model, ...
├── tests/                         # pytest (35 test)
├── reports/                       # data_quality, income_metrics, pos_limit_summary, pos_economics, credit_model_metrics
├── data/processed/                # Cache parquet (tự tạo)
└── legacy/                        # Mã cũ, chỉ để tham khảo
```

## Chuẩn bị dữ liệu

```text
dataset isb/
├── accepted_2007_to_2018Q4.csv    # Lending Club (chỉ cho PD challenger)
└── data/
    ├── customers.csv, accounts.csv, transactions.csv, loans.csv
    └── cards.csv, merchants.csv, subscriptions.csv   # chưa dùng
```

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy dự án

Lần đầu (theo thứ tự):

```powershell
python -m creditforyou cache                   # 1. cache parquet cho giao dịch (~20 giây, 1 lần)
python -m creditforyou quality                 # 2. báo cáo chất lượng dữ liệu -> reports/data_quality.json
python scripts/evaluate_income_estimation.py   # 3. hiệu chỉnh + đánh giá thu nhập -> config/income_calibration.json
python scripts/train_credit_model.py           # 4. (tuỳ chọn) PD challenger trên Lending Club (~2 phút)
```

Giải pháp POP:

```powershell
python -m creditforyou limits --all --with-truth             # cấp hạn mức cho 500 khách + báo cáo theo nhóm
python -m creditforyou limits -c C0000013 --with-truth       # chi tiết 1 khách + giả lập tại quầy
python -m creditforyou checkout -c C0000013 --amount 15000000  # quyết định tức thì tại quầy (VND)
python -m creditforyou limits --all --tickets 3000000 10000000 30000000   # đổi giá trị đơn hàng giả lập (VND)
python -m creditforyou economics                             # lợi nhuận, lỗ dự kiến, PD hoà vốn, RoC theo kịch bản
```

Chấm 1 khoản vay cụ thể (luồng cũ, dùng cho app):

```powershell
python -m creditforyou isb -c C0000001 --amount 10000 --term 36 --with-truth
streamlit run app.py
```

Tham số chính sách (trần hạn mức, tỷ lệ khả năng trả, kỳ hạn, guardrail, nhóm dễ tổn thương...) đều nằm trong `config/scoring.json`, mục `pos_limit` và `business_rules`.

Khi có dataset chính thức: đặt vào `dataset isb/`, chạy lại các bước "Lần đầu". Cache tự tạo lại khi file giao dịch thay đổi.

## Kiểm thử

```powershell
pytest -q
```

35 test: chất lượng dữ liệu, bài toán kinh tế, phân loại nguồn thu (kể cả cờ `is_salary` sai), hiệu chỉnh và đánh giá out-of-fold, khả năng trả và hạn mức, quyết định tại quầy (không vượt trần, không có bước người duyệt), nâng/khoá hạn mức, chống rò rỉ, `dti` không gồm vay mua nhà, chuyển khoản nội bộ không tính là chi tiêu, cache parquet, pipeline tích hợp.

## Giả định & giới hạn

- Mọi ngưỡng (20% thu nhập cho trả góp mới, trần nhóm, hệ số giảm 0,5, kỳ hạn 6 tháng, lãi 18%/năm...) là **giả định nghiên cứu**, cần hiệu chỉnh khi thử nghiệm thực tế.
- Hệ số gộp → ròng học từ `customers.monthly_income` của dataset tổng hợp; với dữ liệu thật phải học lại trên khách đã xác minh thu nhập.
- Chưa có nhãn trả nợ của sản phẩm POP, nên tỷ lệ vỡ nợ trong bài toán kinh tế là **kịch bản**, chưa phải số đo. Đây là KPI quan trọng nhất khi thí điểm (quy tắc dừng: hoà vốn 4,6%).
- Nhóm mục tiêu chỉ có 43 khách (khoảng tin cậy 95% của tỷ lệ có hạn mức: 37–65%); phải kiểm chứng lại trên mẫu lớn ở giai đoạn backtest.
- Phí thấu chi xuất hiện gần như mọi tháng ở mọi khách trong dataset này, nên không phân biệt được rủi ro; giữ làm chỉ số tham khảo.
- Không thay thế KYC, chống gian lận, hay quy định bảo vệ dữ liệu cá nhân (cần sự đồng ý của khách khi dùng dữ liệu giao dịch).
