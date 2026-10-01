# Bỏ dataset thật vào đây

Hệ thống cần tối đa 3 file (CSV / XLSX / Parquet / JSON đều được, tên cột tiếng Việt hay tiếng Anh đều được):

| File | Bắt buộc? | Dùng cho | Cột tối thiểu |
|---|---|---|---|
| Giao dịch (sao kê) | **Có** | Bước #2 ước tính thu nhập | mã KH, ngày, số tiền + (hướng giao dịch HOẶC tiền ra mang dấu âm HOẶC 2 cột tiền vào/tiền ra) |
| Hồ sơ xin vay | Nên có | Bước #5 (tuổi, kinh nghiệm, lịch sử vay, số tiền vay) + đo KPI | mã KH (+ các cột còn lại nếu có) |
| Khoản vay có nhãn (VD: Lending Club) | Nên có | Huấn luyện mô hình ML xác suất vỡ nợ | nhãn (loan_status/default), số tiền vay, thu nhập |

Trước khi chạy, kiểm tra cách hệ thống đọc cột:

```bash
python -m creditforyou inspect data/raw/<ten_file>
```
