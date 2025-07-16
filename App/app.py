import torch
import os
from transformers import RobertaTokenizerFast, RobertaModel
import joblib
import math
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Streamlit page configuration
st.set_page_config(
    page_title="RoBERTa Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .normal { border-left-color: #28a745; background: #d4edda; }
    .bipolar { border-left-color: #dc3545; background: #f8d7da; }
    .personality-disorder { border-left-color: #17a2b8; background: #d1ecf1; }
    .anxiety { border-left-color: #ffc107; background: #fff3cd; }
    .depression { border-left-color: #6c757d; background: #e2e3e5; }
    .stress { border-left-color: #fd7e14; background: #ffe8d6; }
    .suicidal { border-left-color: #dc3545; background: #f8d7da; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model components
@st.cache_resource
def load_model_components():
    output_dir = "./sentiment_model_components"
    model_save_path = os.path.join(output_dir, "last_roberta_model.pt")
    tokenizer_save_path = os.path.join(output_dir, "roberta_tokenizer")
    label_encoder_save_path = os.path.join(output_dir, "label_encoder.joblib")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class MultiHeadAttentionPooling(torch.nn.Module):
        def __init__(self, hidden_size=768, num_heads=12, dropout=0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

            self.query = torch.nn.Linear(hidden_size, hidden_size)
            self.key = torch.nn.Linear(hidden_size, hidden_size)
            self.value = torch.nn.Linear(hidden_size, hidden_size)
            self.dropout = torch.nn.Dropout(dropout)
            self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.scale = math.sqrt(self.head_dim)
            self.layer_norm = torch.nn.LayerNorm(hidden_size)
            self.pooling_dense = torch.nn.Linear(hidden_size * 3, hidden_size)

        def forward(self, hidden_states, attention_mask=None):
            batch_size, seq_length, hidden_size = hidden_states.size()
            normalized_hidden = self.layer_norm(hidden_states)

            q = self.query(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(attention_mask == 0, -1e4)

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
            context = context.view(batch_size, seq_length, hidden_size)
            output = self.out_proj(context)

            cls_output = output[:, 0, :]
            if attention_mask is not None:
                mask_expanded = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1).expand(output.size()).float()
                sum_embeddings = torch.sum(output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_output = sum_embeddings / sum_mask
            else:
                mean_output = torch.mean(output, dim=1)
            
            max_output, _ = torch.max(output, dim=1)
            combined_pooling = torch.cat([cls_output, mean_output, max_output], dim=-1)
            final_pooled = self.pooling_dense(combined_pooling)
            return final_pooled, attn_weights

    class CustomRobertaWithAttentionContrastive(torch.nn.Module):
        def __init__(self, model_name, num_labels, dropout_rate=0.3):
            super().__init__()
            self.roberta = RobertaModel.from_pretrained(model_name)
            self.attention_pooling = MultiHeadAttentionPooling(
                hidden_size=768,
                num_heads=12,
                dropout=dropout_rate
            )
            self.dropout1 = torch.nn.Dropout(dropout_rate)
            self.dropout2 = torch.nn.Dropout(dropout_rate * 0.7)
            self.dropout3 = torch.nn.Dropout(dropout_rate * 0.5)
            self.dense1 = torch.nn.Linear(768, 512)
            self.dense2 = torch.nn.Linear(512, 256)
            self.dense3 = torch.nn.Linear(256, 128)
            self.bn1 = torch.nn.BatchNorm1d(512)
            self.bn2 = torch.nn.BatchNorm1d(256)
            self.bn3 = torch.nn.BatchNorm1d(128)
            self.feature_attention = torch.nn.MultiheadAttention(
                embed_dim=256,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.residual_proj1 = torch.nn.Linear(768, 512)
            self.residual_proj2 = torch.nn.Linear(512, 256)
            self.classifier = torch.nn.Linear(128, num_labels)
            self.feature_proj = torch.nn.Linear(256, 128)
            self._init_weights()

        def _init_weights(self):
            for module in [self.dense1, self.dense2, self.dense3, self.classifier,
                         self.feature_proj, self.residual_proj1, self.residual_proj2]:
                if hasattr(module, 'weight'):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0]
            pooled, attention_weights = self.attention_pooling(hidden_states, attention_mask)
            x = self.dropout1(pooled)
            dense1_out = F.gelu(self.bn1(self.dense1(x)))
            residual1 = F.gelu(self.residual_proj1(pooled))
            x = dense1_out + residual1
            x = self.dropout2(x)
            dense2_out = F.gelu(self.bn2(self.dense2(x)))
            residual2 = F.gelu(self.residual_proj2(x))
            features_pre_attention = dense2_out + residual2
            features_unsqueezed = features_pre_attention.unsqueeze(1)
            attended_features, feature_attn_weights = self.feature_attention(
                features_unsqueezed, features_unsqueezed, features_unsqueezed
            )
            attended_features = attended_features.squeeze(1)
            enhanced_features = features_pre_attention + attended_features
            projected_features = F.normalize(self.feature_proj(enhanced_features), p=2, dim=1)
            x = self.dropout3(enhanced_features)
            final_features = F.gelu(self.bn3(self.dense3(x)))
            logits = self.classifier(final_features)
            return logits, projected_features

    NUM_LABELS = 7
    model = CustomRobertaWithAttentionContrastive("roberta-base", NUM_LABELS).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_save_path)
    label_encoder = joblib.load(label_encoder_save_path)
    
    return model, tokenizer, label_encoder, device

# Sentiment analysis function
def analyze_sentiment(texts, model, tokenizer, label_encoder, device):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs[0], dim=1).cpu().numpy()
        confidences = torch.softmax(outputs[0], dim=1).cpu().numpy()
        
    predicted_labels = label_encoder.inverse_transform(predictions)
    max_confidences = [confidences[i][predictions[i]] for i in range(len(predictions))]
    
    return predicted_labels, max_confidences

# Color mapping for sentiments (updated to match label encoder classes)
color_mapping = {
    'normal': '#28a745',
    'bipolar': '#dc3545',
    'personality disorder': '#17a2b8',
    'anxiety': '#ffc107',
    'depression': '#6c757d',
    'stress': '#fd7e14',
    'suicidal': '#dc3545'
}

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 RoBERTa Sentiment Analysis</h1>
        <p>Analyze text sentiment with custom model</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading model..."):
        model, tokenizer, label_encoder, device = load_model_components()
    
    # Get label classes from label encoder
    label_classes = label_encoder.classes_
    
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        st.subheader("📊 Model Information")
        st.info(f"🔧 Device: {device}")
        st.info(f"🏷️ Classes: {', '.join(label_classes)}")
        st.info(f"🤖 Model: Custom RoBERTa")
        
        st.subheader("📝 Sample Examples")
        examples = [
            "Today is wonderful! I feel very happy.",
            "I feel very sad and tired.",
            "It feels like my mind has 100 tabs open at once.",
            "The world feels clouded, everything takes so much effort.",
            "I'm exhausted from fighting every day."
        ]
        
        selected_example = st.selectbox("Select example:", [""] + examples)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        default_text = selected_example if selected_example else ""
        text_input = st.text_area(
            "Enter text to analyze (each line is a separate text):",
            value=default_text,
            height=200,
            placeholder="Enter your text here...\nYou can enter multiple sentences, each on a new line."
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            demo_btn = st.button("🎲 Random Example", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if demo_btn:
            import random
            demo_text = random.choice(examples)
            st.text_area("Demo text:", value=demo_text, height=100, disabled=True)
    
    with col2:
        st.subheader("📈 Quick Statistics")
        if text_input:
            texts = [t.strip() for t in text_input.split('\n') if t.strip()]
            st.metric("Number of Texts", len(texts))
    
    if analyze_btn and text_input.strip():
        texts = [t.strip() for t in text_input.split('\n') if t.strip()]
        
        if texts:
            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                predicted_labels, confidences = analyze_sentiment(texts, model, tokenizer, label_encoder, device)
            
            st.subheader("🎯 Analysis Results")
            
            for i, (text, label, confidence) in enumerate(zip(texts, predicted_labels, confidences)):
                # Ensure label is valid and exists in color_mapping
                label_key = label.replace(' ', '-')  # Convert label to CSS-compatible class name
                label_color = color_mapping.get(label, '#000')
                st.markdown(f"""
                <div class="result-box {label_key}">
                    <h4>📄 Text {i+1}:</h4>
                    <p style="font-style: italic; margin: 10px 0;">"{text}"</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; color: {label_color};">
                            🏷️ {label}
                        </span>
                        <span style="background: rgba(0,0,0,0.1); padding: 5px 10px; border-radius: 15px;">
                            📊 {confidence:.2%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📊 Summary Statistics")
            df_results = pd.DataFrame({
                'Label': predicted_labels,
                'Confidence': confidences,
                'Text': texts
            })
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                label_counts = df_results['Label'].value_counts()
                fig_pie = px.pie(
                    values=label_counts.values, 
                    names=label_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map=color_mapping
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_bar = px.bar(
                    x=df_results['Label'], 
                    y=df_results['Confidence'],
                    title="Confidence per Text",
                    color=df_results['Label'],
                    color_discrete_map=color_mapping
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("📋 Detailed Results Table")
            df_display = df_results.copy()
            df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_display, use_container_width=True)
            
            st.session_state.df_results = df_results
            csv = st.session_state.df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    elif analyze_btn:
        st.warning("⚠️ Please enter text to analyze!")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 Powered by RoBERTa & Streamlit | 🧠 Sentiment Analysis</p>
        <p>📧 Developed by Group 5 + Grok AI | 🚀 Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()