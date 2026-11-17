# Nhật ký quyết định

| Ngày | Nội dung | Căn cứ / trạng thái |
|---|---|---|
| 2026-11-17 | Khởi tạo khung M01; môi trường tham chiếu Python 3.11; dữ liệu và model ở local | Triển khai kỹ thuật theo yêu cầu của Hoàng; chưa xác nhận G1 trên GitHub |
| 2026-11-17 | Giữ split 80/20, seed 42, CV 5 fold, metric chính AP | Theo Project Convention trong bản phân công |
| 2026-11-17 | Môi trường Windows/Python 3.11.9 đã vượt qua pip check, kiểm tra cú pháp, script môi trường và thực thi notebook M01 | Ghi dependency đầy đủ trong requirements-lock.txt; chưa xác minh CI trên GitHub hoặc máy thành viên |

Quyền GitHub chỉ được coi là đã thiết lập sau khi Hoàng áp dụng và kiểm thử theo `docs/github_setup.md`. M02 ghi tại đây quyết định dataset/nhãn/đặc trưng; M03 ghi schema/API và split; M06 ghi model/params/threshold đã khóa **trước** khi mở test. Nếu thay đổi quy ước, ghi người xác nhận, lý do và ảnh hưởng tới kết quả cũ.
