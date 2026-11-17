# Quy ước dữ liệu

**Chủ trì M02: Phú.** Nguồn dự kiến và đường dẫn chuẩn nằm trong `configs/experiment.yaml`. Chưa chốt biến thể CSV; không đổi nhãn hoặc gộp tiền tiểu đường vào một lớp khi chưa ghi quyết định.

- `raw/diabetes.csv`: dữ liệu gốc đã được nhóm xác nhận, giữ nguyên để đối chiếu.
- `processed/`: đầu ra làm sạch tái lập bởi code của Phú.
- `splits/`: split/chỉ mục do Hiển tạo, cùng seed 42 và phân tầng 80/20.

Các thư mục này bị `.gitignore` loại trừ. Không lưu dữ liệu dưới vị trí khác để đưa nhầm lên Git. Ghi nguồn, ngày tải, giấy phép, biến thể, tên file gốc, số dòng/cột, ý nghĩa nhãn và SHA-256 vào `docs/data_dictionary.md`.

Tính checksum trên PowerShell sau khi đã xác nhận file:

```powershell
Get-FileHash -Algorithm SHA256 data/raw/diabetes.csv
```

Quy tắc cleaning độc lập với phép fit có thể thực hiện trước split. Các bước học tham số như imputation/scaling/encoding phải đặt trong pipeline của M03 và fit trên training fold. Test chỉ mở ở M06.
