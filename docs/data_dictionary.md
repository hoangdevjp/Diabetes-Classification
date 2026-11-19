# Data dictionary — M02 / Phú

**Trạng thái: đã xác nhận dữ liệu thực tế** (survey 2026-11-19).

| Trường | Giá trị |
|---|---|
| Nguồn | CDC Diabetes Health Indicators, UCI #891 (bản gốc: BRFSS 2015, Alex Teboul / Kaggle) |
| URL file thực tế | https://archive.ics.uci.edu/static/public/891/data.csv |
| Giấy phép | Không ghi rõ trên trang UCI; dataset public dùng cho học tập, cite DOI 10.24432/C53919 |
| Ngày tải | 2026-11-19 |
| Tên file gốc / biến thể | `data.csv` tại mirror UCI — biến thể **binary imbalanced** (tương đương `diabetes_binary_health_indicators_BRFSS2015.csv` trên Kaggle) + cột `ID` do UCI thêm |
| Đường dẫn cục bộ chuẩn | `data/raw/diabetes.csv` |
| SHA-256 | `9F71FDA9D4AE5F4878C99B9233B6A16ACCFA9A17C194116A6B78100540934964` |
| Kích thước trước / sau cleaning | 253,680 × 23 / 229,474 × 22 |
| Target | `Diabetes_binary` (đã kiểm tra trên CSV) |
| Ý nghĩa nhãn 0 và 1 | 0 = không tiểu đường, 1 = tiền tiểu đường hoặc tiểu đường (nhãn nhị phân đã gộp sẵn ở biến thể nguồn) |
| Đặc trưng hợp lệ | 21 cột bên dưới; loại bỏ `ID` (index do UCI thêm, không phải đặc trưng) |

## Mô tả từng cột

| Cột | Kiểu | Đơn vị / mã giá trị | Khoảng hợp lệ | Missing | Quy tắc cleaning |
|---|---|---|---|---|---|
| ID | int64 | Index 0..253679 do mirror UCI thêm | — | 0 | R1: drop |
| Diabetes_binary | int64 | 0=không, 1=tiểu đường/tiền tiểu đường | {0,1} | 0 | target, giữ nguyên |
| HighBP | int64 | 0/1 huyết áp cao | {0,1} | 0 | — |
| HighChol | int64 | 0/1 cholesterol cao | {0,1} | 0 | — |
| CholCheck | int64 | 0/1 đo cholesterol trong 5 năm | {0,1} | 0 | — |
| BMI | int64 | chỉ số khối cơ thể | [12,98] | 0 | giữ giá trị cực trị (hợp lệ lâm sàng) |
| Smoker | int64 | 0/1 đã hút ≥100 điếu thuốc | {0,1} | 0 | — |
| Stroke | int64 | 0/1 từng đột quỵ | {0,1} | 0 | — |
| HeartDiseaseorAttack | int64 | 0/1 bệnh tim mạch vành / nhồi máu cơ tim | {0,1} | 0 | — |
| PhysActivity | int64 | 0/1 hoạt động thể lực 30 ngày qua | {0,1} | 0 | — |
| Fruits | int64 | 0/1 ăn trái cây ≥1 lần/ngày | {0,1} | 0 | — |
| Veggies | int64 | 0/1 ăn rau ≥1 lần/ngày | {0,1} | 0 | — |
| HvyAlcoholConsump | int64 | 0/1 uống rượu nặng (nam>14, nữ>7 ly/tuần) | {0,1} | 0 | — |
| AnyHealthcare | int64 | 0/1 có bảo hiểm y tế | {0,1} | 0 | — |
| NoDocbcCost | int64 | 0/1 không khám được vì chi phí 12 tháng qua | {0,1} | 0 | — |
| GenHlth | int64 | tự đánh giá sức khỏe 1=excellent..5=poor | [1,5] | 0 | — |
| MentHlth | int64 | số ngày sức khỏe tinh thần kém /30 ngày | [0,30] | 0 | — |
| PhysHlth | int64 | số ngày sức khỏe thể chất kém /30 ngày | [0,30] | 0 | — |
| DiffWalk | int64 | 0/1 khó đi lại / leo cầu thang | {0,1} | 0 | — |
| Sex | int64 | 0=nữ, 1=nam | {0,1} | 0 | — |
| Age | int64 | 13 bucket tuổi: 1=18-24 .. 13=80+ | [1,13] | 0 | — |
| Education | int64 | 1=chưa đi học .. 6=tốt nghiệp đại học+ | [1,6] | 0 | — |
| Income | int64 | 1=<$10k .. 8=≥$75k | [1,8] | 0 | — |

## Nhật ký khảo sát và làm sạch

Kết quả khảo sát trên file raw (253,680 × 23):

- **Thiếu dữ liệu**: 0 ô missing toàn bộ; không có mã sentinel BRFSS (7/9/77/99 — dữ liệu đã được tác giả Kaggle lọc sẵn).
- **Kiểu dữ liệu**: toàn bộ int64; giá trị nằm trong miền hợp lệ cho mọi cột.
- **Trùng lặp**: 0 trùng theo toàn bộ 23 cột (do `ID` duy nhất), nhưng **24,206 dòng (9.54%) trùng tuyệt đối khi bỏ `ID`** — 35,575 dòng nằm trong các nhóm trùng. Đa số bản ghi trùng thuộc lớp 0 (35,096/35,575).
- **Phân phối nhãn**: 0=218,334 (86.07%), 1=35,346 (13.93%) — mất cân bằng; xử lý ở M04 bằng metric/class_weight, không resample ở M02.
- **Bất thường**: BMI min=12, max=98 (873 dòng ngoài [15,60]) — cực trị nhưng hợp lệ lâm sàng, giữ lại.

Quy tắc làm sạch trong `src/cleaning.py` (độc lập tham số học, chạy trước split):

| Rule | Hành động | Số mẫu thay đổi | Lý do |
|---|---|---|---|
| R5 | Báo cáo missing; dropna chỉ trên target | 0 | Không impute ở M02 |
| R1 | Drop cột `ID` | 0 dòng, −1 cột | Index nhân tạo, không phải đặc trưng |
| R3 | Drop dòng ngoài miền giá trị hợp lệ | 0 | Defensive check |
| R2 | Drop duplicate tuyệt đối (feature+label) | −24,206 dòng | Bản ghi lặp của survey |
| R4 | Ép kiểu int64 | — | Chuẩn hóa dtype |

Kết quả: **229,474 × 22**; nhãn sau cleaning: 0=194,377 (84.71%), 1=35,097 (15.29%).

Tái chạy: `python -m src.cleaning` → `data/processed/diabetes_clean.csv`.
Không fit imputer/scaler/encoder trên toàn dataset tại M02 — các bước đó thuộc pipeline M03.
