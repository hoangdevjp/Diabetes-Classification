# Kiểm thử theo module

M01 kiểm tra môi trường bằng `scripts/check_environment.py`, chưa có unit test cho chức năng chưa triển khai.

Khi làm M02/M03, bổ sung kiểm thử cho schema/nhãn, cleaning tái lập, không chồng lấn mẫu train/test và preprocessing chỉ fit trên training fold. Khi đó thêm công cụ test và lệnh chạy vào requirements/CI qua PR.

## Chạy kiểm thử đơn vị (M03)

Sử dụng thư viện chuẩn `unittest` (không cần cài thêm gói ngoài):

```powershell
python -m unittest discover -s tests -v
```
