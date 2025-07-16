# RoBERTa Sentiment Analysis

## Tổng quan
Dự án này triển khai một ứng dụng web phân tích cảm xúc (sentiment analysis) sử dụng mô hình RoBERTa tùy chỉnh với cơ chế attention và contrastive learning. Ứng dụng được xây dựng bằng Streamlit, cung cấp giao diện tương tác để phân tích cảm xúc của văn bản liên quan đến trạng thái sức khỏe tinh thần.

## Tính năng
- Phân tích cảm xúc đa lớp (7 lớp: normal, bipolar, personality disorder, anxiety, depression, stress, suicidal)
- Mô hình RoBERTa tùy chỉnh với multi-head attention pooling
- Giao diện web tương tác với Streamlit
- Hiển thị trực quan với biểu đồ pie và bar
- Xuất kết quả sang CSV
- Các ví dụ mẫu để thử nghiệm nhanh
- Thiết kế responsive với giao diện tùy chỉnh

## Yêu cầu trước
- Python 3.10 hoặc cao hơn
- GPU hỗ trợ CUDA (tùy chọn, để xử lý nhanh hơn)
- Các thành phần mô hình đã được huấn luyện trước (Custom RoBERTa model, tokenizer, label encoder)

## Cài đặt
Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

## Yêu cầu hệ thống
Các phụ thuộc sau được yêu cầu (liệt kê trong requirements.txt):
```
torch==2.6.0
transformers==4.52.4
scikit-learn==1.6.1
pandas==2.2.2
numpy==2.0.2
joblib==1.4.2
streamlit==1.39.0
plotly==5.24.1
```

## Thiết lập
1. Đảm bảo các thành phần mô hình đã được huấn luyện trước nằm trong thư mục `E:\sentiment_model_components`:
   - last_roberta_model.pt
   - roberta_tokenizer
   - label_encoder.joblib

2. Cập nhật đường dẫn `output_dir` trong `app.py` nếu các thành phần mô hình được lưu ở vị trí khác.

## Hướng dẫn sử dụng
1. Chạy ứng dụng Streamlit:
```bash
streamlit run app.py
```

2. Mở trình duyệt web và truy cập `http://localhost:8501`

3. Sử dụng giao diện để:
   - Nhập văn bản cần phân tích (hỗ trợ nhiều dòng)
   - Chọn các ví dụ mẫu từ sidebar
   - Xem kết quả với điểm confidence
   - Hiển thị phân bố cảm xúc qua biểu đồ
   - Xuất kết quả sang CSV

## Cấu trúc dự án
```
├── app.py                    # Ứng dụng Streamlit chính
├── requirements.txt          # Các phụ thuộc của dự án
├── README.md                 # Tài liệu dự án
└── E:\sentiment_model_components\    # Thư mục chứa thành phần mô hình
    ├── last_roberta_model.pt
    ├── roberta_tokenizer
    └── label_encoder.joblib
```

## Ghi chú phát triển
- Mô hình sử dụng kiến trúc RoBERTa tùy chỉnh với multi-head attention pooling và contrastive learning
- Hỗ trợ xử lý trên cả CPU và GPU (CUDA)
- Văn bản đầu vào bị giới hạn ở 256 token (có thể cấu hình trong app.py)
- Sử dụng Plotly để tạo biểu đồ tương tác
- CSS được nhúng trong ứng dụng Streamlit để cải thiện UI/UX

## Tác giả
- Nhóm 5 (Hỗ trợ)
- Grok AI