# Sentiment Analysis với RoBERTa

Chào mừng bạn đến với kho lưu trữ này! Dự án chứa hai Jupyter Notebook triển khai nhiệm vụ sentiment analysis sử dụng mô hình RoBERTa-base từ Hugging Face Transformers. Hai notebook bao gồm:

- **`sentiment_roberta_base.ipynb`**: Phiên bản cơ bản sử dụng `RobertaForSequenceClassification`.
- **`sentiment_roberta_custom.ipynb`**: Phiên bản cải tiến với custom architecture để tăng performance.

Dự án sử dụng tập dữ liệu [Sentiment Data Splited](https://www.kaggle.com/datasets/luilailayda123/sentiment-data-splited) từ Kaggle để train và evaluate mô hình.

## Tổng Quan

Cả hai notebook được thiết kế để:
- Preprocess dữ liệu văn bản và nhãn cảm xúc.
- Train mô hình với các kỹ thuật như mixed precision training, cosine learning rate scheduling, và xử lý imbalanced data.
- Evaluate performance thông qua metric chính là weighted F1-score, cùng với error analysis.
- Save mô hình, tokenizer, và các thành phần liên quan để reuse.

### Cấu Hình Train Chung
Cả hai notebook sử dụng cấu hình train sau:
- Max sequence length: 512 (`MAX_LENGTH=512`).
- Batch size: 16 (`BATCH_SIZE=16`).
- Epochs: 5 (`EPOCHS=5`).
- Learning rate: 2e-5 (`LEARNING_RATE=2e-5`).
- Warmup steps: 100 (`WARMUP_STEPS=100`).
- Mixed precision training với `GradScaler` và `autocast` để optimize GPU performance.
- Cosine learning rate scheduler với warmup.
- Xử lý imbalanced data bằng class weights.
- Visualize loss, accuracy, và F1-score cho train/validation sets.
- Save `state_dict`, `LabelEncoder`, và tokenizer vào thư mục `sentiment_model_components`, nén thành `sentiment_model_components.zip`.

### 1. `sentiment_roberta_base.ipynb`
Notebook này triển khai một mô hình sentiment analysis cơ bản sử dụng `RobertaForSequenceClassification`.

#### Key Features
- **Model**: `RobertaForSequenceClassification` từ Hugging Face, một mô hình tiêu chuẩn với linear classifier trên top của RoBERTa-base.
- **Loss Function**: `LabelSmoothingCrossEntropy` (smoothing=0.1) để cải thiện generalization bằng cách giảm confidence trên nhãn đúng.
- **Dropout Rate**: 0.1 (`DROPOUT_RATE=0.1`).
- **Features**:
  - Sử dụng RoBERTa-base pre-trained weights.
  - Error analysis trên test set, lưu kết quả (accuracy, predictions, labels, probabilities).
- **Performance**: Weighted F1-score trên test set đạt khoảng **82.6%**.

#### Use Cases
Phù hợp cho các dự án cần triển khai nhanh một mô hình sentiment analysis với cấu hình đơn giản và performance tốt.

---

### 2. `sentiment_roberta_custom.ipynb`
Notebook này mở rộng phiên bản cơ bản với `CustomRobertaWithAttentionContrastive`, một kiến trúc tùy chỉnh để cải thiện performance và stability.

#### Model Architecture
Kiến trúc `CustomRobertaWithAttentionContrastive` được thiết kế với các thành phần nâng cao:
- **RoBERTa Base**: Sử dụng `RobertaModel` từ Hugging Face làm backbone để extract contextual embeddings từ input text.
- **MultiHeadAttentionPooling**:
  - Tích hợp ba chiến lược pooling: CLS token, mean pooling, và max pooling.
  - Sử dụng multi-head attention mechanism (12 heads, hidden_size=768) để weigh token representations, kết hợp với layer normalization để stabilize training.
  - Output là một vector kết hợp từ các pooling strategies, được chiếu qua một linear layer để giảm dimension.
- **Multi-Layer Classifier**:
  - Gồm ba dense layers (768→512, 512→256, 256→128) với GELU activation và batch normalization để giảm internal covariate shift.
  - Residual connections giữa các layer (768→512, 512→256) để preserve information và mitigate vanishing gradients.
  - Feature attention mechanism (8 heads, embed_dim=256) để focus vào discriminative features, đặc biệt hữu ích cho similar classes.
- **Metric Learning**: Feature projection layer (256→128) với L2 normalization để hỗ trợ contrastive learning, cải thiện separation giữa các class.
- **Weight Initialization**: Sử dụng Xavier/Glorot initialization cho tất cả linear layers để đảm bảo stable training.

#### Key Features
- **Loss Function**: `FocalLoss` (gamma=1.25) để focus vào hard samples và xử lý imbalanced data hiệu quả hơn `LabelSmoothingCrossEntropy`.
- **Dropout Rate**: 0.3 (`DROPOUT_RATE=0.3`) để tăng regularization, giảm overfitting trên complex datasets.
- **Features**:
  - Enhanced feature extraction thông qua attention-based pooling.
  - Improved stability với batch normalization và residual connections.
  - Error analysis trên test set với detailed metrics.
- **Performance**: Weighted F1-score trên test set đạt khoảng **83.1%**, cải thiện so với phiên bản cơ bản nhờ kiến trúc phức tạp hơn và `FocalLoss`.

#### Use Cases
Phù hợp cho các trường hợp cần high-performance model, xử lý complex data hoặc yêu cầu F1-score cao hơn, đặc biệt khi dataset có nhiều class hoặc imbalanced.

---

## Dataset

Dự án sử dụng tập dữ liệu [Sentiment Data Splited](https://www.kaggle.com/datasets/luilailayda123/sentiment-data-splited) từ Kaggle (Dataset ID: 7725403).

### Dataset Details
- **Columns**:
  - `processed_text`: Văn bản đã preprocess, sẵn sàng để tokenize.
  - `status`: Nhãn cảm xúc (string hoặc numeric, được encode bằng `LabelEncoder`).
- **Structure**: Đã split thành train, validation, và test sets.
- **Usage**: Thiết kế cho sentiment analysis, phù hợp với deep learning models như RoBERTa.
- **Access**: Tải từ Kaggle và đặt trong thư mục phù hợp trước khi chạy notebook.

## Cài Đặt

Cài đặt các dependencies sau:

```bash
pip install torch==2.6.0+cu124 transformers==4.52.4 scikit-learn==1.6.1 pandas==2.2.2 numpy==2.0.2 matplotlib seaborn tqdm joblib
```

### Hardware Requirements
- **GPU**: Khuyến nghị Tesla T4 hoặc tương đương để train nhanh.
- **CPU**: Có thể sử dụng, nhưng training time sẽ lâu hơn.
- **RAM**: Tối thiểu 16GB để xử lý large datasets và models.

### Environment
- Python: 3.11.11
- Jupyter Notebook hoặc Google Colab với GPU support.

## Hướng Dẫn Sử Dụng

1. **Chuẩn bị Dataset**:
   - Tải [Sentiment Data Splited](https://www.kaggle.com/datasets/luilailayda123/sentiment-data-splited) từ Kaggle.
   - Đặt các file (train, validation, test) vào thư mục phù hợp và update đường dẫn trong notebook nếu cần.

2. **Cài Đặt Environment**:
   - Chạy lệnh `pip` ở trên để cài đặt dependencies.
   - Đảm bảo GPU availability nếu muốn optimize training.

3. **Chạy Notebook**:
   - Mở `sentiment_roberta_base.ipynb` hoặc `sentiment_roberta_custom.ipynb` trong Jupyter/Colab.
   - Run cells theo thứ tự để:
     - Preprocess data.
     - Train model.
     - Evaluate performance và save results.
   - Check visualizations (loss, accuracy, F1-score) trong notebook.

4. **Reuse Model**:
   - Unzip `sentiment_model_components.zip` để lấy thư mục `sentiment_model_components`.
   - Sử dụng hàm `load_model_and_tokenizer` để load model và tokenizer.
   - Apply model cho inference trên new data.

5. **Customization**:
   - Adjust hyperparameters (`MAX_LENGTH`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `DROPOUT_RATE`) để optimize cho specific dataset.
   - Experiment với `gamma` trong `FocalLoss` (custom notebook) hoặc smoothing trong `LabelSmoothingCrossEntropy` (base notebook) để improve performance.

## Repository Structure

```plaintext
├── sentiment_roberta_base.ipynb        # Base notebook
├── sentiment_roberta_custom.ipynb      # Custom notebook
├── sentiment_model_components/         # Directory chứa model components
│   ├── last_roberta_model.pt          # Model state_dict
│   ├── label_encoder.joblib           # LabelEncoder
│   ├── roberta_tokenizer/             # Tokenizer directory
├── sentiment_model_components.zip      # Zipped model components
└── README.md                          # Project description
```

## Performance Comparison

| Notebook                          | Test Weighted F1-score | Key Features                              |
|-----------------------------------|------------------------|-------------------------------------------|
| `sentiment_roberta_base.ipynb`    | ~82.6%                | `RobertaForSequenceClassification`, `LabelSmoothingCrossEntropy`, simple architecture. |
| `sentiment_roberta_custom.ipynb`  | ~83.1%                | `CustomRobertaWithAttentionContrastive`, `FocalLoss`, `MultiHeadAttentionPooling`, residual connections, batch normalization, feature attention, metric learning. |

## Contributing

- Để báo lỗi hoặc đề xuất cải tiến, tạo [issue](https://github.com/phuhoangg/Sentiment_analysis/issues) trên GitHub.
- Để đóng góp code, submit [pull request](https://github.com/phuhoangg/Sentiment_analysis/pulls) với detailed description.

## License

Dự án được cấp phép theo [MIT License](https://opensource.org/licenses/MIT). Xem chi tiết trong file [LICENSE](LICENSE).

---
