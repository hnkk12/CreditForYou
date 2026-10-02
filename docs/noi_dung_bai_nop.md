# CreditForYou — Nội dung bài nộp (HLBVN × UEH.ISB Business Challenge 2026, vòng 2)

> Tài liệu tổng hợp toàn bộ nội dung để dựng trang PDF 1920×1080, sắp theo 5 tiêu chí của rubric.
> Mọi con số lấy từ code trong repo, chạy trên dataset **thay thế** cho vòng tự do (`dataset isb/`: 500 khách, 1.669.697 giao dịch; gốc USD, quy đổi sang VND 25.000 VND/USD). Số liệu nền (thời gian, chi phí, khoảng vay, kỳ hạn, cơ cấu khách) lấy từ đề, Exhibit 2.
> Các ô **[FIGURE]** là biểu đồ sẽ vẽ sau; dữ liệu để vẽ có sẵn ngay bên dưới.
> Cách chạy lại số liệu: xem Phụ lục C.

---

## 0. Thông điệp chính (đặt ở đầu trang)

**Tiêu đề phụ:** Cash-flow rules decide today · ML learns from real repayments to sharpen limits tomorrow.

**Một câu:** Cấp **trước** hạn mức trả góp dựa trên **dòng tiền tài khoản** để tại quầy thanh toán, quyết định cho khách freelancer / gig worker / chủ shop online là **tức thì**, không cần giấy tờ, không cần người duyệt, và lỗ được khống chế bằng hạn mức.

**4 con số đầu trang:**

| Chỉ số | Giá trị |
| --- | --- |
| Thời gian ra quyết định | **1,8–4,6 ngày làm việc (đề, Exhibit 2) → tức thì** (tra cứu hạn mức tính sẵn, mục tiêu < 1 giây gồm API/POS) |
| Chi phí mỗi quyết định | **380.000 VND (đề) → ~50.000 VND** (giả định của nhóm) |
| Gig worker + chủ shop online có hạn mức | **16% → 51%** so với quy tắc thay thế (*mô phỏng*, n = 43) |
| Ngưỡng hoà vốn = quy tắc dừng | **Vỡ nợ 6 tháng 4,6%**; RoC 28,5% kịch bản cơ sở (*mô phỏng*) |

> Mọi số "mô phỏng" chạy trên dataset **thay thế** (tổng hợp, USD quy đổi 25.000 VND/USD), tính lại theo cơ cấu khách của đề. Dataset chính thức chỉ mở ở vòng Strategy.

---

## 1. Chẩn đoán vấn đề (Problem Diagnosis — 20%)

### 1.1 Nút thắt cốt lõi (điểm quyết định có đòn bẩy vận hành lớn nhất)

**Quyết định cho vay tại quầy thanh toán.** Tại điểm bán, ngân hàng chỉ có vài giây để trả lời "có/không". Với khách thiếu lịch sử tín dụng, ngân hàng **không có dữ liệu để quyết định trong vài giây đó**, nên chỉ còn 2 lựa chọn tệ:

| Lựa chọn | Hậu quả |
| --- | --- |
| Từ chối | Mất doanh số cho ngân hàng và đối tác bán hàng; mất khách vào tay đối thủ hoặc tín dụng phi chính thức |
| Thẩm định thủ công tập trung: **1,8–4,6 ngày làm việc, 380.000 VND/hồ sơ** (đề, Exhibit 2) | Thời điểm mua hàng đã qua → khách bỏ đơn hoặc tìm tín dụng phi chính thức; chi phí cố định vượt doanh thu của khoản vay nhỏ |

### 1.2 Nguyên nhân gốc và cơ chế

- **Nguyên nhân gốc:** quy trình đánh giá dựa trên **giấy tờ chứng minh** (bảng lương, hợp đồng lao động, lịch sử CIC). Đây đúng là thứ freelancer, gig worker, chủ shop online **không có**.
- **Cơ chế gây nghẽn:**

```text
Thu nhập không qua bảng lương ─┐
Chưa từng vay (thin-file)      ├─> Không có "bằng chứng" theo quy trình ─> Không quyết định được trong vài giây
Thu nhập biến động             ┘                                              ├─> Từ chối (mất doanh số)
                                                                              └─> Review thủ công (khách bỏ đi)
Trong khi đó: dòng tiền thật của khách đã nằm sẵn trong tài khoản tại ngân hàng nhưng chưa được dùng.
```

- **Điểm mấu chốt:** quyết định tại quầy đang bị **gắn chặt** với bước thẩm định. Tách 2 việc này ra (thẩm định trước, ngoài giờ mua hàng) là gỡ được nút thắt.

### 1.3 Bằng chứng

**Từ số liệu của đề (Exhibit 2):** chi phí thẩm định 380.000 VND/hồ sơ là **cố định**, bất kể khoản vay lớn hay nhỏ.

| Khoản vay 6 tháng | Doanh thu (lãi 12% + phí đối tác 2%) | So với chi phí thẩm định 380.000 VND |
| ---: | ---: | --- |
| 1,6 triệu (mức tối thiểu của đề) | ≈ 88.000 VND | Lỗ ngay từ khâu thẩm định |
| 5 triệu | ≈ 276.000 VND | Lỗ ngay từ khâu thẩm định |
| ≈ 6,9 triệu | ≈ 380.000 VND | Hoà vốn (chưa tính chi phí vốn và vỡ nợ) |

→ Thẩm định thủ công **không thể** dùng cho khoản vay nhỏ tại điểm bán. Đây là lý do nút thắt có đòn bẩy cao nhất: giải được nó thì cải thiện **cả tốc độ lẫn chi phí đơn vị** cùng lúc.

**Từ dữ liệu mô phỏng** (quy tắc thay thế cho quy trình truyền thống = lương cố định + đã có lịch sử vay):

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

Đặc điểm thu nhập (mô phỏng):

| Đặc điểm | Freelancer (26) | Entrepreneur (17) | Manager (60), để so sánh |
| --- | ---: | ---: | ---: |
| Độ ổn định thu nhập (1 − hệ số biến thiên), trung vị | 0,40 | 0,61 | 0,96 |
| Doanh thu gộp ÷ thu nhập thực (trung vị) | ~1,85 (thu nhập ròng ≈ 54% tiền nhận) | – | ~1,12 |

→ Nhóm này **có thu nhập thật** nhưng **biến động** và **không ở dạng lương**, nên quy trình giấy tờ loại họ.

**[FIGURE 1 — Nút thắt]** Bảng chi phí thẩm định so với doanh thu khoản vay (ở trên), cùng cột tỷ lệ đủ điều kiện theo phân khúc.

---

## 2. Lập luận giải pháp (Intervention Justification — 30%)

### 2.1 Cơ chế cốt lõi: tách "thẩm định" khỏi "quyết định tại quầy"

```text
 [A] CẤP HẠN MỨC TRƯỚC (chạy theo lô)   [B] TẠI QUẦY (tức thì)               [C] SAU KHI CHO VAY
 Giao dịch 12 tháng                      Đơn hàng ≤ hạn mức?                  3 kỳ đúng hạn -> nâng hạn mức +25%
  -> thu nhập theo loại nguồn thu         ├─ có   -> DUYỆT                    Trễ ≥ 30 ngày -> khoá hạn mức
  -> chi phí bắt buộc, nợ đang trả        ├─ vượt -> ĐỀ XUẤT KHÁC              Nhãn trả nợ thật -> train ML sau này
  -> khả năng trả hằng tháng              │   (kéo dài kỳ hạn / trả trước)
  -> điểm + nhóm A/B/C/D                  └─ chưa có hạn mức -> TỪ CHỐI
  -> HẠN MỨC + trạng thái                 KHÔNG có bước người duyệt tại quầy
```

**Vì sao giải quyết trực tiếp nút thắt:** toàn bộ phần "khó" (đọc dòng tiền, ước tính thu nhập, đo khả năng trả) được làm **trước**, theo lô. Tại quầy chỉ còn phép so sánh "đơn hàng ≤ hạn mức?", nên quyết định luôn tức thì. Rủi ro được kiểm soát bằng **quy mô hạn mức** chứ không phải bằng cách từ chối cả nhóm khách.

### 2.2 Lựa chọn: KHÔNG dùng AI để ra quyết định (rule-based redesign)

| Lý do | Giải thích trước hội đồng tín dụng |
| --- | --- |
| Chưa có nhãn vỡ nợ của sản phẩm POP cho nhóm khách này | Mô hình AI cần dữ liệu "ai đã vỡ nợ"; HLBVN chưa có cho sản phẩm và phân khúc này |
| Dữ liệu thay thế không phù hợp | Đã thử mô hình PD trên 300.000 khoản vay Lending Club (Mỹ, 2007–2018): AUC test chỉ **0,667**, và **xếp sai thứ tự giữa các nhóm** (PD nhóm C 2,0% < nhóm B 2,2%) |
| Minh bạch, giải thích được | Mỗi hạn mức có lý do cụ thể: ràng buộc nào chặn, guardrail nào kích hoạt |
| Triển khai nhanh, rẻ | Chạy theo lô trên dữ liệu tài khoản sẵn có, không cần hạ tầng AI |
| ML vẫn có lộ trình | PD Lending Club chạy **song song để so sánh** (challenger). Sau thí điểm sẽ train lại trên nhãn thật và chỉ thay bảng điểm khi tốt hơn rõ rệt |

### 2.3 Lợi ích đo được

| Lợi ích | Trước | Sau | Nguồn |
| --- | --- | --- | --- |
| **Tốc độ:** thời gian ra quyết định | 1,8–4,6 ngày làm việc (đề) | **Tức thì**: tra cứu hạn mức tính sẵn (phép so sánh < 1 ms; mục tiêu < 1 giây gồm API/POS); cấp hạn mức theo lô ~50–90 ms/khách | Exhibit 2; `checkout` |
| **Chi phí:** mỗi quyết định | 380.000 VND (đề) | ~50.000 VND (giả định), tương xứng khoản vay nhỏ | Exhibit 2; `economics` |
| **Tốc độ:** tỷ lệ quyết định tức thì | – | **100%** | thiết kế; `test_limit.py` |
| **Hiệu quả vận hành:** hồ sơ phải người duyệt tại quầy | Mọi hồ sơ nhóm C hoặc vướng guardrail | **0**; 199 hồ sơ chuyển thành cấp hạn mức giảm tức thì | `economics` |
| **Hiệu quả vận hành:** chi phí tránh được | – | ~**65,7 triệu VND** mỗi lượt xét (199 × 330.000 VND) | `economics` |
| **Trải nghiệm khách:** không giấy tờ, biết trước hạn mức | Nộp giấy tờ, chờ | Khách biết hạn mức trước khi mua; vượt hạn mức nhận đề xuất thay vì bị từ chối | `limits`, `checkout` |
| **Mở rộng khách hàng:** gig worker + chủ shop online có hạn mức | 16,3% | **51,2%** mô phỏng (khoảng tin cậy 95%: 37–65%, n = 43) | `limits --all` |
| **Mở rộng khách hàng:** dòng khách | 173 khách theo quy tắc truyền thống | **+211 mới, −93 nợ quá cao bị loại, ròng +118** (291) | `limits --all` |
| **Mở rộng khách hàng:** tính lại theo cơ cấu đề | 40,5% | **56,0%** hồ sơ đến có hạn mức tức thì (mô phỏng) | `case_mix_view` |

### 2.4 Tính khả thi

- **Dữ liệu:** chỉ cần lịch sử giao dịch tài khoản (khách hiện hữu của HLBVN đã có sẵn) + danh sách khoản vay đang có.
- **Hạ tầng:** batch job hằng đêm; tại quầy là một phép so sánh. Giai đoạn đầu không cần hệ thống mới.
- **Vận hành:** mọi ngưỡng nằm trong file cấu hình (`config/scoring.json`). Hội đồng tín dụng chỉnh chính sách mà **không cần sửa code**.
- **Đã kiểm chứng:** 35 kiểm thử tự động, trong đó có kiểm thử "tại quầy không bao giờ trả về REVIEW" và "đề xuất kéo dài kỳ hạn không bao giờ vượt trần hạn mức".

---

## 3. Kiến trúc giải pháp (Solution Concept — 35%)

### 3.1 Từ dữ liệu đến quyết định

| Lớp | Nội dung |
| --- | --- |
| **Dữ liệu vào** | Giao dịch 12 tháng gần nhất (số tiền, ngày, loại giao dịch, nhóm, mô tả, cờ chuyển khoản); khoản vay đang có (khoản trả hằng tháng, loại vay); hồ sơ khách (tuổi, ngày trở thành khách hàng, phân khúc). **Không** dùng thu nhập tự khai |
| **Bước 1. Làm sạch** | Loại chuyển khoản giữa các tài khoản của chính khách, hoàn tiền, giải ngân vay; phân loại nguồn thu theo nhóm giao dịch (sửa cờ `is_salary` sai); bỏ tháng đầu/cuối không trọn vẹn |
| **Bước 2. Ước tính thu nhập** | Theo loại nguồn thu: lương/lương hưu lấy trung vị tháng có nhận; tự doanh lấy trung bình cả 12 tháng. Hiệu chỉnh gộp → ròng theo loại (lương 0,887; lương hưu 0,886; tự doanh 0,533) |
| **Bước 3. Đo khả năng trả** | Thu nhập tiền mặt bảo thủ = thu nhập gộp × (0,5 + 0,5 × độ ổn định); trừ chi phí bắt buộc (nhà, ăn uống, điện nước, y tế, bảo hiểm, đi lại) và nợ đang trả |
| **Bước 4. Chấm điểm** | Ổn định thu nhập 25, nợ/thu nhập 25, thâm niên 15, lịch sử trả nợ 20 (tắt khi không có dữ liệu), ML 15 (tắt, chỉ để so sánh). Chuẩn hoá về 100 → nhóm A ≥ 80, B ≥ 60, C ≥ 40, D < 40 |
| **Bước 5. Tính hạn mức** | Khả năng trả/tháng = min(50% phần dư khả dụng; 20% thu nhập ròng; 50% thu nhập ròng − nợ). Hạn mức (VND) = min(khả năng trả × hệ số 6 tháng; trần nhóm A 90 / B 60 / C 30 triệu; bội số thu nhập A 1,5× / B 1× / C 0,5×; trần khách mới 30 triệu), tối thiểu 1,6 triệu. Kỳ hạn 6 tháng, tối đa 12 (đề cho phép tới 24) |
| **Đầu ra** | Hạn mức + trạng thái + lý do + ràng buộc đang chặn |
| **Hành động kinh doanh** | `PREAPPROVED`: dùng toàn bộ hạn mức tại quầy · `PREAPPROVED_REDUCED`: hạn mức × 0,5, vẫn tức thì · `OFFLINE_REVIEW`: chuyên viên tư vấn **ngoài giờ** · `NOT_ELIGIBLE`: từ chối, gợi ý sản phẩm khác |

**Guardrail (chỉ làm quyết định chặt hơn, không bao giờ nới):**

| Điều kiện | Kết quả |
| --- | --- |
| Tổng trả nợ sau vay / thu nhập > 70%; chi tiêu / thu nhập > 120%; tuổi ngoài 18–65 | Không đủ điều kiện |
| Trả góp mới / thu nhập > 30%; tổng nợ > 45%; chi tiêu > 90%; tin cậy thu nhập < 70% | Cấp hạn mức giảm |
| Ít hơn 3 tháng dữ liệu, tin cậy < 50%, khách thuộc nhóm `VULNERABLE` | Xem xét ngoài giờ |

**Quyết định tại quầy:**

| Tình huống | Kết quả |
| --- | --- |
| Đơn ≤ hạn mức | Duyệt, trả góp 6 tháng |
| Đơn > hạn mức, hạn mức đang bị chặn bởi khả năng trả | Đề xuất kéo dài kỳ hạn (tối đa 12 tháng, không vượt trần) |
| Đơn > hạn mức, hạn mức chạm trần | Đề xuất trả trước phần vượt |
| Chưa có hạn mức | Từ chối |

**[FIGURE 2 — Sơ đồ kiến trúc]** Sơ đồ 3 bước [A] → [B] → [C] ở mục 2.1 + bảng 3.1.

### 3.2 Kết quả (mô phỏng minh hoạ, 500 khách)

> **Đọc kỹ:** đây là dataset **thay thế** (dữ liệu tổng hợp, tính bằng USD, quy đổi 25.000 VND/USD); dataset chính thức của đề chỉ mở ở vòng Strategy. Vì là dữ liệu kiểu Mỹ nên thu nhập quy đổi cao bất thường so với Việt Nam (khoảng 100–200 triệu VND/tháng), do đó hạn mức phần lớn chạm trần. Các con số dưới đây minh hoạ cơ chế, **không phải** dự báo cho HLBVN. Cơ cấu khách của dataset (63% chưa có lịch sử vay) khác đề, nên mọi tỷ lệ "toàn bộ" được tính lại theo cơ cấu Exhibit 2.

**Theo 5 phân khúc của đề** — [FIGURE 3]

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

**Giả lập tại quầy (đơn hàng 5 / 15 / 40 triệu VND)** — [FIGURE 5]

| Đơn | Duyệt | Đề xuất khác | Từ chối |
| ---: | ---: | ---: | ---: |
| 5 triệu | 57,2% | 1,0% | 41,8% |
| 15 triệu | 47,4% | 10,8% | 41,8% |
| 40 triệu | 9,2% | 49,0% | 41,8% |

**Ví dụ minh hoạ: freelancer `C0000013`**

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

### 3.3 Bài toán kinh tế (VND/năm, mô phỏng)

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

**[FIGURE 6 — Kịch bản lợi nhuận]** Cột lợi nhuận theo 4 kịch bản + mốc tỷ lệ lỗ hoà vốn 3,0% và BNPL Mỹ 3,8%.

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

### 3.4 Tác động cộng đồng và tính bền vững

- **Tài chính toàn diện:** đưa người lao động tự do và khách chưa từng vay vào hệ thống tín dụng chính thức (trong mô phỏng, 87% khách vay lần đầu có hạn mức khởi đầu). Có lộ trình nâng hạn mức để khách xây dựng lịch sử tín dụng.
- **Cho vay có trách nhiệm:**
  - hạn mức theo khả năng trả thực tế (chỉ tính phần dư sau chi phí bắt buộc);
  - khách dễ tổn thương (`VULNERABLE`) không được cấp tự động mà được tư vấn;
  - khách thâm hụt dòng tiền không được cấp.
- **Bền vững tài chính:** có lãi khi vỡ nợ < 4,6% (RoC 28,5% kịch bản cơ sở, mô phỏng); quy tắc dừng, ngân sách lỗ, cơ chế nâng/khoá hạn mức tự điều chỉnh danh mục theo hành vi thật.
- **Bền vững vận hành:** quy tắc minh bạch, cấu hình hoá, hội đồng chỉnh được; không phụ thuộc mô hình "hộp đen".
- **Môi trường:** không giấy tờ (bỏ hồ sơ chứng minh thu nhập giấy); tính toán nhẹ (chạy theo lô trên CPU, không cần hạ tầng AI/GPU).

### 3.5 KPI đo trực tiếp nút thắt

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

### 3.6 Lộ trình triển khai 6–12 tháng

| Giai đoạn | Việc chính | Điều kiện đi tiếp |
| --- | --- | --- |
| **0–3 tháng: chạy thử ngược (backtest) & chính sách** | Chạy pipeline (chỉ đọc, chưa cấp) trên dữ liệu tài khoản của khách **hiện hữu** của HLBVN. Học lại hệ số gộp → ròng trên khách đã có hồ sơ lương/thuế. Chạy `quality` trên dữ liệu thật. Hội đồng tín dụng duyệt `pos_limit` và `business_rules`. Hoàn tất cơ chế xin đồng ý xử lý dữ liệu cá nhân của khách | Sai lệch thu nhập trung vị ≤ 10%; kết quả nhóm mục tiêu giữ được trên mẫu lớn hơn (hiện n = 43, khoảng tin cậy 95% của tỷ lệ có hạn mức là 37–65%); chính sách được duyệt |
| **3–6 tháng: thí điểm có kiểm soát** | 1–2 đối tác bán hàng (điện máy / thương mại điện tử). Chỉ khách hiện hữu. Trần hạn mức thấp, **ngân sách lỗ cố định**. Theo dõi trễ hạn hằng tuần | Vỡ nợ 6 tháng ≤ 3% (dưới mức hoà vốn 4,6% với biên an toàn); quá hạn 30+ ngày nhóm A < 2%, B < 5% |
| **6–12 tháng: mở rộng + ML** | Thêm đối tác. Bật nâng hạn mức (3 kỳ đúng hạn → +25%). Dùng nhãn trả nợ thật để huấn luyện mô hình ML dự báo vỡ nợ, chạy song song với bảng điểm (champion–challenger) để tinh chỉnh hạn mức và giá. Bổ sung dữ liệu tài khoản doanh nghiệp cho chủ kinh doanh | **Dừng nếu vỡ nợ > 4,6%**; ML chỉ thay bảng điểm khi tốt hơn rõ rệt trên dữ liệu thật |

Trong giai đoạn đầu, giải pháp **không cần hạ tầng mới**: hạn mức được tính sẵn theo lô trên dữ liệu tài khoản ngân hàng đã có, còn tại quầy chỉ là **tra cứu hạn mức**. Bản thân phép so sánh mất dưới 1 ms; thời gian thực tế do API/kết nối POS quyết định, mục tiêu < 1 giây.

**[FIGURE 7 — Lộ trình]** Timeline 3 giai đoạn + điều kiện đi tiếp.

---

## 4. Dữ liệu & giả định (Data and Assumptions — 10%)

### 4.1 Nguồn dữ liệu

| Nguồn | Quy mô | Vai trò |
| --- | --- | --- |
| `transactions.csv` | 1.669.697 giao dịch, 10/2021 – 10/2026 | Ước tính thu nhập, chi phí, khả năng trả |
| `customers.csv` | 500 khách | Tuổi, ngày trở thành khách hàng, phân khúc. `monthly_income` **chỉ dùng để đánh giá và hiệu chỉnh**, không phải đầu vào |
| `loans.csv` | 233 khoản vay | Nợ đang trả hằng tháng |
| `accounts.csv` | 1.426 tài khoản | Kiểm tra toàn vẹn khoá |
| Lending Club `accepted_2007_to_2018Q4.csv` | 753.566 dòng; 300.000 khoản vay đã có kết cục được dùng | Chỉ cho mô hình PD so sánh (challenger) |

Hai nguồn **không ghép theo khách**: giao dịch dùng cho thu nhập và khả năng trả; Lending Club chỉ để tham chiếu PD.

### 4.2 Làm sạch dữ liệu (21 hạng mục kiểm tra; dataset gốc không bị sửa)

| Mức | Phát hiện | Số lượng | Xử lý | Nếu không xử lý |
| --- | --- | ---: | --- | --- |
| Cao | `is_salary = True` cho cả tiền freelance (4.962) và lương hưu (2.768) | 7.730 / 28.655 | Phân loại nguồn thu theo nhóm giao dịch trước cờ `is_salary` | Freelancer bị coi như có lương cố định |
| Cao | Chuyển khoản giữa các tài khoản của chính khách | 31.324 | Loại khỏi cả thu nhập và chi tiêu | Tiết kiệm bị tính là chi tiêu (duyệt tự động tụt 18% → 1,6%) |
| Cao | Số dư sau giao dịch không liên tục | 76% dòng | Không dùng số dư để quyết định | Đặc trưng số dư sai |
| Cao | Khoản vay mua nhà trong nợ đang có | 133 / 233 | Tính trong khả năng trả; bỏ khỏi `dti` của mô hình PD | `dti` 54% ngoài phân phối Lending Club; PD của C0000001 11,5% → 25,3% |
| Cao | Cột rò rỉ Lending Club (dư nợ, đã trả, grade, int_rate...) | – | Chặn tự động | AUC ảo |
| TB | Phí thấu chi ở trung vị 93% số tháng mỗi khách | 111.363 | Chỉ tham khảo | Không phân biệt rủi ro |
| TB | Hồ sơ có thu nhập nhưng 12 tháng không có khoản thu nào | 31 / 500 | Thu nhập = 0, không cấp | Tin thu nhập tự khai đã cũ |
| TB | `opened_at` = ngày xuất dữ liệu | 138 / 1.426 | Thâm niên theo `customer_since` | Thâm niên sai |
| TB | Không có lịch sử trả đúng hạn | 233 / 233 | Tắt thành phần, chuẩn hoá trọng số | – |
| TB | Lending Club: khoản chưa có kết cục và dòng tổng kết cuối file | 305.087 | Loại khỏi tập train | Nhãn sai |
| Thấp | Thẻ `active` nhưng đã hết hạn | 717 / 901 | Không dùng bảng thẻ | – |
| OK | Trùng ID, ngày lỗi, số tiền rỗng, giao dịch trùng, khách mồ côi, tuổi lệch ngày sinh, trả góp sai công thức | 0 | Kiểm tra mỗi lần chạy | – |

**[FIGURE 8 — Chất lượng dữ liệu]** (phụ) Bảng/thanh 5 lỗi mức cao.

### 4.3 Độ chính xác ước tính thu nhập (kiểm định chéo 5 lần, so với `monthly_income`)

| Chỉ số | Cách cũ (phân vị 10%) | CreditForYou |
| --- | ---: | ---: |
| Sai lệch trung vị | 8,8% | **1,7%** |
| Sai lệch ≤ 10% | 61,6% | **69,8%** |
| Sai lệch ≤ 20% | 72,4% | **75,2%** |
| Sai lệch ≤ 10%, bỏ 31 khách không có thu 12 tháng | – | **74,4%** |

| Nhóm khách | n | Sai lệch trung vị | ≤ 10% | ≤ 20% |
| --- | ---: | ---: | ---: | ---: |
| investor | 24 | 1,3% | 79,2% | 83,3% |
| retiree | 54 | 1,3% | 77,8% | 77,8% |
| student | 62 | 1,6% | 77,4% | 77,4% |
| manager | 60 | 1,5% | 76,7% | 81,7% |
| young_professional | 102 | 1,4% | 75,5% | 77,5% |
| high_earner | 52 | 1,9% | 71,2% | 75,0% |
| family | 73 | 1,7% | 69,9% | 78,1% |
| vulnerable | 30 | 5,6% | 60,0% | 73,3% |
| **freelancer** | 26 | 18,4% | **38,5%** (cách cũ 23%) | 53,8% |
| **entrepreneur** | 17 | 28,4% | **5,9%** | 35,3% |

Giải thích 2 nhóm yếu:
- **Freelancer:** thu nhập biến động mạnh, tỷ lệ ròng/gộp dao động 0,39–1,42. Hệ thống bù bằng thu nhập bảo thủ và trần hạn mức.
- **Entrepreneur:** chủ doanh nghiệp chỉ tự trả lương nhỏ vào tài khoản cá nhân, thu nhập thật nằm ở tài khoản doanh nghiệp (ví dụ C0000043: lương nhận 4.514 USD/tháng, thu nhập thực 9.728 USD). Cần thêm dữ liệu tài khoản doanh nghiệp (giai đoạn 6–12 tháng).

### 4.4 Danh sách giả định

| Giả định | Giá trị | Neo vào | Cần xác nhận khi |
| --- | --- | --- | --- |
| Hệ số gộp → ròng theo loại nguồn thu | lương 0,887; lương hưu 0,886; tự doanh 0,533 | Học từ dữ liệu (khách có ≥ 95% thu nhập từ 1 loại), kiểm định chéo | Học lại trên khách đã xác minh lương/thuế của HLBVN |
| Thu nhập tiền mặt bảo thủ | gộp × (0,5 + 0,5 × độ ổn định) | Thiết kế bảo thủ: giảm tối đa 50% | Backtest |
| Trả góp mới tối đa | 20% thu nhập ròng | Giả định chính sách | Hội đồng tín dụng |
| Tổng nợ tối đa (tính hạn mức) | 50% thu nhập ròng | Giả định chính sách | Hội đồng tín dụng |
| Phần dư khả dụng dùng để trả góp | 50% | Giả định bảo thủ | Thí điểm |
| Kỳ hạn chuẩn / tối đa | 6 / 12 tháng | Đặc thù trả góp tại điểm bán | Đối tác bán hàng |
| Lãi suất tính hạn mức | 18%/năm | Giả định bảo thủ (lãi cao → hạn mức nhỏ hơn) | Biểu phí sản phẩm |
| Lãi suất theo nhóm (kinh tế) | A 9–10%, B 11–13%, C 14–16% | Bản ý tưởng của nhóm | Biểu phí sản phẩm |
| Trần nhóm A / B / C | 90 / 60 / 30 triệu VND | Trần A = mức vay tối đa của đề (90,1 triệu, Exhibit 2) | Hội đồng tín dụng |
| Trần khách mới / hạn mức tối thiểu | 30 triệu / 1,6 triệu VND | "Bắt đầu nhỏ, nâng dần"; tối thiểu = mức vay nhỏ nhất của đề | Thí điểm |
| Kỳ hạn | 6 tháng chuẩn, tối đa 12 | Đề cho phép 3/6/9/12/18/24 tháng | Chính sách |
| Quy đổi tiền tệ dataset thay thế | 25.000 VND/USD | Chỉ cho dữ liệu mô phỏng; dataset chính thức bằng VND | – |
| Hệ số cấp giảm | × 0,5 | Giả định chính sách | Thí điểm |
| Nâng hạn mức | 3 kỳ đúng hạn → +25% | Giả định | Thí điểm |
| Khoá hạn mức | trễ ≥ 30 ngày | Mốc quá hạn thông dụng | Chính sách |
| Ngưỡng guardrail | trả góp mới 30%, tổng nợ 45% / 70%, chi tiêu 90% / 120%, tin cậy 70% | Giả định nghiên cứu | Hội đồng tín dụng |
| Tuổi | 18–65 | Giả định chính sách | Chính sách |
| Tần suất dùng hạn mức | 1,5 lần/năm | Giả định | Dữ liệu thí điểm |
| Đơn hàng trung bình | 60% hạn mức | Giả định | Dữ liệu đối tác |
| Phí đối tác bán hàng | 2% giá trị trả góp | **Thận trọng**: CFPB, phí BNPL bình quân 2,49% (2021) | Hợp đồng đối tác |
| Chi phí vốn | 6%/năm | **Thận trọng**: lãi tiền gửi 12 tháng VN 2025 khoảng 4,6–5,8% | Bộ phận nguồn vốn |
| Tổn thất khi vỡ nợ (LGD) | 91% | **Theo dữ liệu Lending Club**: 90.150 khoản vỡ nợ | Dữ liệu thu hồi nợ của HLBVN |
| Dư nợ còn lại khi vỡ nợ | 72% gốc | **Theo dữ liệu Lending Club** (khoản 36–60 tháng, thận trọng cho khoản 6 tháng) | Dữ liệu thí điểm |
| Vốn chủ phân bổ | 10% dư nợ | Tỷ lệ an toàn vốn tối thiểu 8% + đệm | Bộ phận quản trị rủi ro |
| Chi phí xử lý hồ sơ | 380.000 VND thủ công / 50.000 VND tự động | Thủ công: **đề, Exhibit 2**; tự động: giả định của nhóm | Dữ liệu vận hành HLBVN |
| PD 6 tháng kịch bản cơ sở | A 2%, B 5%, C 10% | **Thận trọng**: ≈ 2× Lending Club hạng A/B/C quy về 6 tháng (1,0 / 2,3 / 3,8%); khớp KPI bản ý tưởng | **Đo khi thí điểm** |
| Quy đổi PD Lending Club | 1 − (1 − PD)^(6/36) | Hazard đều trên kỳ hạn 36 tháng | – |

> Nguồn: [CFPB 2022, BNPL market trends](https://files.consumerfinance.gov/f/documents/cfpb_buy-now-pay-later-market-trends-and-consumer-impacts_presentation_2022-11.pdf) (phí đối tác 2,49%, xoá nợ 3,8% năm 2021); [MBS, thị trường tiền tệ 7/2025](https://mbs.com.vn/?p=19561) (lãi tiền gửi 12 tháng). Còn thiếu nguồn: chi phí thẩm định hồ sơ, tần suất dùng hạn mức (đo khi thí điểm).

### 4.5 Mô hình PD Lending Club (challenger — chỉ tham khảo)

| Chỉ số (test set 60.000 khoản) | Giá trị |
| --- | ---: |
| Mô hình | HistGradientBoostingClassifier, chọn trên validation |
| Chia dữ liệu | 180.000 train / 60.000 validation / 60.000 test |
| ROC-AUC | 0,667 |
| PR-AUC | 0,335 |
| Brier score | 0,151 |
| Tỷ lệ vỡ nợ thực tế | 20,1% |
| Đặc trưng | loan_to_income, installment_to_income, dti (không gồm vay mua nhà), years_experience, term_months, log_annual_income, log_loan_amount, home_ownership, purpose, verification_status |
| Chống rò rỉ | Loại cột phát sinh sau giải ngân, grade, sub_grade, int_rate, installment gốc |

---

## 5. Trình bày (Visualization — 5%) — gợi ý bố cục trang PDF 1920×1080

| Vùng | Cấp độ | Nội dung |
| --- | --- | --- |
| Dải trên cùng | **Chính** | Tiêu đề + thông điệp 1 câu + 4 con số đầu trang (mục 0) |
| Cột trái (~25%) | **Chính** | Nút thắt & nguyên nhân gốc (1.1, 1.2) + FIGURE 1 hoặc FIGURE 4 (trước/sau) |
| Cột giữa (~45%) | **Chính** | FIGURE 2: sơ đồ [A] → [B] → [C] + bảng trạng thái hạn mức + guardrail rút gọn |
| Cột phải (~30%) | **Chính** | FIGURE 6: kinh tế (lợi nhuận/RoC 4 kịch bản, ngưỡng hoà vốn 4,6%) + khung rủi ro và cơ chế chặn lỗ |
| Dải dưới | **Phụ** | FIGURE 7 lộ trình 3 giai đoạn · làm sạch dữ liệu (5 lỗi cao) · giả định chính · giới hạn |

Danh sách figure: **1** nút thắt · **2** kiến trúc · **3** hạn mức theo nhóm · **4** trước/sau · **5** giả lập tại quầy · **6** kịch bản kinh tế · **7** lộ trình · **8** chất lượng dữ liệu. Với 1 trang, ưu tiên **2, 4, 6, 7**; các figure còn lại để phụ lục hoặc Q&A.

---

## Phụ lục A — Câu hỏi hội đồng tín dụng có thể hỏi

| Câu hỏi | Trả lời |
| --- | --- |
| Không có bảng lương, sao tin được thu nhập? | Dùng dòng tiền 12 tháng thật trong tài khoản, phân loại theo nguồn thu, loại chuyển khoản nội bộ/hoàn tiền/giải ngân. Sai lệch trung vị 1,7% so với thu nhập thực (kiểm định chéo) |
| Freelancer thu nhập thất thường, cấp hạn mức có liều không? | Thu nhập càng bất ổn càng bị giảm (tối đa 50%); chỉ tính phần dư sau chi phí bắt buộc và nợ; trần khách mới 30 triệu VND; người thâm hụt không được cấp. Rủi ro khống chế bằng **quy mô** khoản vay |
| Vì sao không dùng AI? | Chưa có nhãn vỡ nợ cho sản phẩm này. Mô hình trên Lending Club có AUC 0,667 và xếp sai thứ tự giữa các nhóm. Quy tắc minh bạch hơn và triển khai nhanh hơn. AI được đưa vào sau thí điểm, khi có nhãn thật |
| Lỗ bao nhiêu thì dừng? | Hoà vốn ở vỡ nợ 6 tháng 4,6% (tỷ lệ lỗ 3,0% giải ngân); mục tiêu thí điểm ≤ 3%; vượt 4,6% thì dừng mở rộng. Thí điểm có ngân sách lỗ cố định |
| Khách dễ tổn thương thì sao? | Không cấp tự động, chuyển chuyên viên tư vấn ngoài giờ (30 khách `VULNERABLE`) |
| Khách vượt hạn mức tại quầy? | Không bị từ chối ngay: được đề xuất kéo dài kỳ hạn (nếu khả năng trả cho phép, không vượt trần) hoặc trả trước phần vượt |
| Dữ liệu lấy từ đâu, có hợp pháp không? | Giai đoạn đầu chỉ dùng khách hiện hữu (dữ liệu tài khoản HLBVN đã có), kèm xin đồng ý xử lý dữ liệu cá nhân. Dữ liệu từ ngân hàng khác chỉ cân nhắc ở giai đoạn sau |
| Chủ doanh nghiệp bị ước tính thấp? | Đúng (6% hồ sơ sai lệch ≤ 10%): thu nhập thật nằm ở tài khoản doanh nghiệp. Bổ sung dữ liệu tài khoản doanh nghiệp ở giai đoạn 6–12 tháng; trước đó, ước tính thấp nghĩa là cấp hạn mức nhỏ (an toàn) |
| Biên lợi nhuận mỏng? | Đúng: 0,8% giải ngân, ngưỡng lỗ hoà vốn 3,0% thấp hơn mức xoá nợ BNPL Mỹ 3,8%. Đòn bẩy: +1 điểm % phí đối tác = +113 triệu VND/năm; lãi suất theo nhóm chỉnh qua cấu hình |
| "211 khách mở thêm" mâu thuẫn với 35% → 58%? | Không: +211 khách mới, đồng thời −93 khách truyền thống bị loại vì nợ quá cao (chủ yếu vay mua nhà), ròng +118 (173 → 291) |
| Vỡ nợ ×2 thì sao? | Lỗ (RoC −45%). Vì vậy rủi ro được chặn bằng: quy tắc dừng 4,6%, trần khởi đầu 30 triệu VND, cấp giảm ×0,5, kỳ hạn 6 tháng, ngân sách lỗ thí điểm |
| ML dùng để làm gì? | Sau thí điểm, nhãn trả nợ thật huấn luyện mô hình dự báo vỡ nợ để tinh chỉnh hạn mức và giá; chạy song song (champion–challenger), chỉ thay quy tắc khi tốt hơn rõ rệt |

## Phụ lục B — Giới hạn

- Tỷ lệ vỡ nợ trong bài toán kinh tế là **kịch bản**, chưa phải số đo (dataset không có dữ liệu trả nợ của sản phẩm POP).
- Dataset là dữ liệu tổng hợp (USD, khách Mỹ); hệ số hiệu chỉnh và ngưỡng phải học/duyệt lại trên dữ liệu thật của HLBVN.
- Thành phần lịch sử trả nợ bị tắt vì không có dữ liệu.
- Mọi ngưỡng chính sách là giả định nghiên cứu.
- Không thay thế KYC, chống gian lận hay các quy định bảo vệ dữ liệu cá nhân.

## Phụ lục C — Tái tạo số liệu

```powershell
python -m creditforyou cache                     # cache giao dịch
python -m creditforyou quality                   # -> reports/data_quality.json      (mục 4.2)
python scripts/evaluate_income_estimation.py     # -> reports/income_metrics.json    (mục 4.3)
python scripts/train_credit_model.py             # -> reports/credit_model_metrics.json (mục 4.5)
python -m creditforyou limits --all --with-truth # -> reports/pos_limit_summary.json (mục 1.3, 3.2)
python -m creditforyou economics                 # -> reports/pos_economics.json     (mục 3.3)
python -m creditforyou limits -c C0000013 --with-truth   # ví dụ minh hoạ
```

Tham số chính sách: `config/scoring.json` (mục `pos_limit`, `business_rules`, `economics`). Hệ số hiệu chỉnh thu nhập: `config/income_calibration.json`.
