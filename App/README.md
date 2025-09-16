# RoBERTa Sentiment Analysis Web Application

## Overview

This is an additional web application component built with Streamlit that demonstrates real-time sentiment analysis using the trained RoBERTa models from the main project. This application provides an interactive interface for analyzing mental health-related text.

**Note:** This is an optional component that complements the core Jupyter Notebook implementations. Please refer to the main project README.md for details about the core analysis and model training.

## Features

- **Dual Model Support**: Switch between base and custom RoBERTa models
- **Multi-class Sentiment Analysis**: Classifies text into 7 mental health categories:
  - Normal
  - Bipolar
  - Personality Disorder
  - Anxiety
  - Depression
  - Stress
  - Suicidal
- **Interactive Web Interface**: Built with Streamlit for ease of use
- **Visual Analytics**: Pie charts and bar graphs for sentiment distribution
- **Batch Processing**: Analyze multiple texts simultaneously
- **Export Results**: Download analysis results as CSV
- **Sample Examples**: Preloaded text examples for quick testing
- **Responsive Design**: Mobile-friendly interface with custom styling

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Pre-trained model components (RoBERTa model weights, tokenizer, label encoder)

## Installation

1. Ensure you have the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The application requires the following model components in the `sentiment_model_components` directory:
   - `base_roberta_model.pt` (Base RoBERTa model weights)
   - `last_roberta_model.pt` (Custom RoBERTa model weights)
   - `roberta_tokenizer/` (Tokenizer files)
   - `label_encoder.joblib` (Label encoder for sentiment classes)

## Dependencies

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

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the provided URL (typically `http://localhost:8501`)

3. Select your preferred model type (Base or Custom RoBERTa)

4. Enter text to analyze in the text area:
   - Each line is treated as a separate text for analysis
   - You can analyze multiple texts at once

5. Click "Analyze" to process the text

6. View results in:
   - Individual sentiment predictions with confidence scores
   - Pie chart showing sentiment distribution
   - Bar chart showing confidence per text
   - Detailed results table

7. Download results as CSV using the "Download CSV" button

## Project Structure

```
App/
├── app.py                 # Main Streamlit application
├── model.py               # Model implementations and loading functions
├── sentiment_analyzer.py   # Sentiment analysis functions
├── ui_components.py        # UI components and styling
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── sentiment_model_components/
    ├── base_roberta_model.pt      # Base RoBERTa model weights
    ├── last_roberta_model.pt      # Custom RoBERTa model weights
    ├── roberta_tokenizer/         # Tokenizer files
    ├── label_encoder.joblib       # Label encoder
    └── training_metrics_with_f1.png  # Training metrics visualization
```

## Model Architecture

### Base RoBERTa Model
- Uses `RobertaForSequenceClassification` from Hugging Face Transformers
- Standard linear classifier on top of RoBERTa-base
- Suitable for quick deployment with good performance

### Custom RoBERTa Model
- Enhanced architecture with multiple components:
  - **Multi-Head Attention Pooling**: Combines CLS token, mean pooling, and max pooling
  - **Multi-Layer Classifier**: Three dense layers with GELU activation and batch normalization
  - **Residual Connections**: Between layers to preserve information flow
  - **Feature Attention**: Additional attention mechanism for discriminative feature focus
  - **Metric Learning**: Feature projection layer with L2 normalization
- Provides improved performance for complex sentiment analysis tasks

## Development

### Code Structure

The application is modularized into several components:

1. **`app.py`**: Main application logic and UI orchestration
2. **`model.py`**: Model definitions and loading functions
3. **`sentiment_analyzer.py`**: Core sentiment analysis functions
4. **`ui_components.py`**: UI components and styling

### Extending the Application

To add new features:
1. Add new model variants in `model.py`
2. Update the model selection UI in `app.py`
3. Modify analysis functions in `sentiment_analyzer.py` if needed
4. Add new UI components in `ui_components.py`

## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure all model components are in the `sentiment_model_components` directory
2. **CUDA out of memory**: Reduce batch size or use CPU instead of GPU
3. **Import errors**: Verify all dependencies are installed with correct versions

### Performance Optimization

1. Use GPU for faster inference
2. For large batches, consider processing in smaller chunks
3. Clear browser cache if UI becomes unresponsive

## Contributing

This is an additional component of the main project. For contributions, please refer to the main project repository.

## License

This project is licensed under the MIT License - see the main project [LICENSE](../LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Streamlit for the web application framework
- PyTorch for the deep learning framework