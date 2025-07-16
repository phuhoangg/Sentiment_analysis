# RoBERTa Sentiment Analysis UI

## Tổng quan
Dự án này triển khai một ứng dụng web phân tích cảm xúc (sentiment analysis) sử dụng mô hình RoBERTa mở rộng. Ứng dụng được xây dựng bằng Streamlit, cung cấp giao diện tương tác để phân tích cảm xúc của văn bản liên quan đến trạng thái sức khỏe tinh thần.

## Tính năng
- Phân tích cảm xúc đa lớp (7 lớp: normal, bipolar, personality disorder, anxiety, depression, stress, suicidal)
- Mô hình RoBERTa tùy chỉnh
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
1. Đảm bảo các thành phần mô hình đã được huấn luyện trước nằm trong thư mục `.\sentiment_model_components`:
   - last_roberta_model.pt
   - roberta_tokenizer
   - label_encoder.joblib

2. Cập nhật đường dẫn `output_dir` trong `app.py` nếu các thành phần mô Phú
