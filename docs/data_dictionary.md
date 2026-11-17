# Data dictionary — M02 / Phú

**Trạng thái: chưa xác nhận dữ liệu thực tế.** Phú bổ sung nội dung này ở T04–T07 trước khi Hiển dùng schema để xây dựng pipeline.

| Trường | Giá trị |
|---|---|
| Nguồn dự kiến | CDC Diabetes Health Indicators, UCI #891 |
| URL phiên bản/file thực tế | TO_CONFIRM |
| Giấy phép | TO_CONFIRM |
| Ngày tải | TO_CONFIRM |
| Tên file gốc / biến thể | TO_CONFIRM |
| Đường dẫn cục bộ chuẩn | `data/raw/diabetes.csv` |
| SHA-256 | TO_CONFIRM |
| Kích thước trước / sau cleaning | TO_CONFIRM |
| Target dự kiến | `Diabetes_binary` — kiểm tra lại trên CSV |
| Ý nghĩa nhãn 0 và 1 | TO_CONFIRM — xác minh cách xử lý tiền tiểu đường |
| Đặc trưng hợp lệ / cột loại bỏ | TO_CONFIRM |

## Mô tả từng cột

| Cột | Kiểu dữ liệu | Đơn vị / mã giá trị | Khoảng hợp lệ | Missing | Quy tắc cleaning |
|---|---|---|---|---|---|
| TO_CONFIRM | | | | | |

## Nhật ký khảo sát và làm sạch

Ghi phân phối nhãn, dữ liệu thiếu, trùng lặp, bất thường; số mẫu thay đổi theo từng quy tắc và lý do. Bàn giao code `src/cleaning.py` cùng lệnh tái chạy. Không fit imputer/scaler/encoder trên toàn dataset tại M02.
