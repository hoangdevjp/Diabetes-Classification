# Dự đoán bệnh tiểu đường

Dự án bài tập lớn môn Machine Learning, hướng đến xây dựng và so sánh các mô hình phân loại nhị phân để dự đoán nguy cơ tiểu đường từ các chỉ báo sức khỏe.

## Mô tả dự án

Đây là hệ thống thử nghiệm học máy dành cho bài toán phân loại nguy cơ tiểu đường. Dự án bao gồm các bước khảo sát dữ liệu, tiền xử lý, huấn luyện nhiều mô hình, đánh giá bằng các chỉ số thống nhất và minh họa kết quả qua giao diện Streamlit.

## Mục tiêu

- Khảo sát và làm sạch dữ liệu sức khỏe.
- Xây dựng quy trình tiền xử lý và huấn luyện mô hình.
- So sánh hiệu quả phân loại và trình bày kết quả bằng bảng, biểu đồ.
- Xây dựng giao diện minh họa dự đoán bằng Streamlit.

## Các mô hình

- Gaussian Naive Bayes.
- K-Nearest Neighbors (KNN).
- Random Forest.
- Support Vector Machine (SVM), sử dụng kernel Linear và RBF.

## Dữ liệu và đánh giá

Nguồn dữ liệu dự kiến là **CDC Diabetes Health Indicators**. Phiên bản dữ liệu, các thuộc tính đầu vào và ý nghĩa nhãn chưa được xác nhận chính thức.

Cấu hình thí nghiệm dự kiến sử dụng 80% dữ liệu để huấn luyện, 20% để kiểm tra, seed 42 và kiểm định chéo StratifiedKFold 5 fold trên tập huấn luyện. Các bước tiền xử lý được fit trong từng fold để tránh rò rỉ dữ liệu; tập test chỉ dùng cho đánh giá cuối cùng.

Các chỉ số đánh giá gồm Accuracy, Precision, Recall, F1, ROC-AUC và Average Precision (chỉ số chính).

## Công nghệ

Python 3.11, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Seaborn, JupyterLab và Streamlit.

## Cấu trúc chính

| Thư mục | Nội dung |
|---|---|
| `data/` | Dữ liệu gốc, dữ liệu đã xử lý và các tập dữ liệu đã chia |
| `notebooks/` | Kiểm tra môi trường, khảo sát dữ liệu và thử nghiệm mô hình |
| `src/` | Mã nguồn xử lý dữ liệu, tiền xử lý, mô hình và đánh giá |
| `configs/` | Cấu hình thí nghiệm |
| `results/` | Chỉ số đánh giá và biểu đồ |
| `artifacts/` | Pipeline và mô hình đã huấn luyện |
| `app/` | Giao diện Streamlit |
| `scripts/` | Công cụ kiểm tra môi trường |
| `tests/` | Các kiểm thử của dự án |
| `docs/` | Mô tả dữ liệu, quyết định kỹ thuật và báo cáo |

## Trạng thái

Dự án hiện có cấu trúc thư mục, cấu hình thí nghiệm, công cụ kiểm tra môi trường và các khung mã nguồn/notebook. Chưa chốt dữ liệu, chưa có mô hình đã huấn luyện hoặc kết quả đánh giá thực tế. Giao diện Streamlit hiện chỉ hiển thị trạng thái dự án.

Dự án phục vụ học tập, không thay thế chẩn đoán y tế.
