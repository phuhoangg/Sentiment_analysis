# Mental Health Sentiment Analysis with RoBERTa

## Overview

This project implements a comprehensive sentiment analysis system for mental health text classification using two variants of the RoBERTa model. The core of this project consists of detailed Jupyter Notebook implementations that demonstrate the training, evaluation, and comparison of different RoBERTa-based architectures for mental health sentiment analysis.

The system classifies text into 7 mental health categories:
- Normal
- Bipolar
- Personality Disorder
- Anxiety
- Depression
- Stress
- Suicidal

## Project Structure

```
├── sentiment_roberta_base.ipynb      # Base RoBERTa model implementation
├── sentiment_roberta_custom.ipynb    # Custom RoBERTa model implementation
├── Visualization.ipynb               # Data exploration and visualization
├── App/                              # Streamlit web application (additional component)
│   ├── app.py                       # Main application
│   ├── model.py                     # Model implementations
│   ├── sentiment_analyzer.py        # Analysis functions
│   ├── ui_components.py             # UI components
│   ├── requirements.txt             # Dependencies
│   ├── README.md                    # Application documentation
│   └── sentiment_model_components/
│       ├── roberta_tokenizer/       # Tokenizer files
│       ├── label_encoder.joblib     # Label encoder
│       └── training_metrics_with_f1.png  # Training visualization
├── README.md                         # This file
└── LICENSE                           # MIT License
```

## Core Implementation (Jupyter Notebooks)

### 1. Base RoBERTa Implementation (`sentiment_roberta_base.ipynb`)

This notebook implements a standard sentiment analysis model using `RobertaForSequenceClassification` from Hugging Face Transformers.

**Key Features:**
- Uses the pre-trained `roberta-base` model with a linear classification head
- Implements Label Smoothing Cross Entropy loss function for improved generalization
- Mixed precision training with `GradScaler` and `autocast` for performance optimization
- Cosine learning rate scheduling with warmup
- Class weighting to handle imbalanced dataset
- Comprehensive error analysis with confusion matrices and misclassification examples
- Model persistence with tokenizer and label encoder

**Performance:** Achieved approximately **82.6%** weighted F1-score on the test set.

### 2. Custom RoBERTa Implementation (`sentiment_roberta_custom.ipynb`)

This notebook implements an enhanced architecture with multiple custom components designed to improve performance.

**Key Features:**
- Custom model architecture with `MultiHeadAttentionPooling`
- Combines three pooling strategies: CLS token, mean pooling, and max pooling
- Multi-layer classifier with batch normalization and residual connections
- Feature attention mechanism for discriminative learning
- Metric learning with feature projection and L2 normalization
- Focal Loss with gamma=1.25 for handling imbalanced data
- Enhanced error analysis with detailed metrics and visualizations

**Performance:** Achieved approximately **83.1%** weighted F1-score on the test set.

### 3. Data Visualization and Analysis (`Visualization.ipynb`)

This notebook provides comprehensive data exploration and visualization.

**Key Features:**
- Dataset distribution analysis across all 7 sentiment categories
- Text length analysis with visualizations
- Word clouds for each sentiment category
- Identification of lexical patterns specific to each mental health state
- Analysis of data imbalance and preprocessing strategies

## Dataset

The project uses the [Sentiment Data Splited](https://www.kaggle.com/datasets/luilailayda123/sentiment-data-splited) dataset from Kaggle, which contains text samples labeled with mental health sentiment categories.

Key characteristics:
- 7 sentiment categories with imbalanced distribution
- Preprocessed text ready for tokenization
- Split into train, validation, and test sets
- Original dataset augmented with oversampling to handle class imbalance

## Model Comparison

| Aspect | Base RoBERTa | Custom RoBERTa |
|--------|--------------|----------------|
| Architecture | Standard `RobertaForSequenceClassification` | Enhanced with attention pooling and residual connections |
| Loss Function | Label Smoothing Cross Entropy | Focal Loss (gamma=1.25) |
| Dropout Rate | 0.1 | 0.3 |
| Key Enhancements | None | Multi-head attention pooling, batch normalization, residual connections |
| Test Weighted F1-Score | ~82.6% | ~83.1% |

## Training Configuration

Both models were trained with the following configuration:
- Maximum sequence length: 512 tokens
- Batch size: 16
- Number of epochs: 5
- Learning rate: 2e-5
- Warmup steps: 100
- Mixed precision training enabled
- Cosine learning rate scheduler with warmup
- Early stopping with patience of 2 epochs

## Results and Findings

### Performance Analysis
- The custom RoBERTa model achieved a modest improvement of 0.5% in weighted F1-score
- Both models performed well on majority classes but struggled with minority classes
- "Depression" and "Suicidal" categories showed significant overlap, making them challenging to distinguish
- "Normal" category achieved the highest classification accuracy across both models

### Error Analysis
- Most misclassifications occurred between semantically similar categories
- High confidence wrong predictions often involved texts with mixed emotional signals
- The custom model showed improved discrimination between similar classes due to its enhanced architecture

### Visualization Insights
- Each mental health category exhibited distinct lexical patterns
- Text length varied significantly across categories
- The dataset showed significant class imbalance that was addressed through oversampling

## Additional Component: Streamlit Web Application

A Streamlit web application is included as an additional component to demonstrate real-time sentiment analysis using the trained models. This application allows users to:

- Switch between base and custom RoBERTa models
- Analyze single or multiple text inputs
- View results with confidence scores
- Download analysis results as CSV
- Visualize sentiment distribution with interactive charts

For details on the web application, see `App/README.md`.

## Requirements

For running the Jupyter Notebooks:
- Python 3.8+
- PyTorch 2.6.0+
- Transformers 4.52.4
- Scikit-learn 1.6.1
- Pandas 2.2.2
- NumPy 2.0.2
- Matplotlib and Seaborn
- tqdm and joblib

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/luilailayda123/sentiment-data-splited)
2. Place the train, validation, and test CSV files in the project directory
3. Run the Jupyter notebooks:
   - Start with `sentiment_roberta_base.ipynb` for the base implementation
   - Run `sentiment_roberta_custom.ipynb` for the enhanced model
   - Use `Visualization.ipynb` for data exploration

## Future Improvements

1. Implement a two-stage classification approach to better distinguish between "Depression" and "Suicidal" categories
2. Experiment with ensemble methods combining both model variants
3. Explore data augmentation techniques beyond simple oversampling
4. Investigate domain-specific pre-trained models for mental health text

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Kaggle for the dataset
- PyTorch for the deep learning framework