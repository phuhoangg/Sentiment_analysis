import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

# Import our modular components
from model import load_model_components
from sentiment_analyzer import analyze_sentiment
from ui_components import apply_custom_css, display_main_header, display_sidebar_info, display_result_box

# Streamlit page configuration
st.set_page_config(
    page_title="RoBERTa Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Apply custom CSS styling
    apply_custom_css()
    
    # Display main header
    display_main_header()
    
    # Model selection in sidebar
    with st.sidebar:
        st.header("⚙️ Model Selection")
        model_type = st.radio(
            "Select Model Type:",
            options=["custom", "base"],
            format_func=lambda x: "Custom RoBERTa (Advanced)" if x == "custom" else "Base RoBERTa (Standard)",
            index=0
        )
        
        # Load model components based on selection
        with st.spinner("Loading model..."):
            model, tokenizer, label_encoder, device, model_name = load_model_components(model_type)
        
        st.info(f"🔧 Device: {device}")
        st.info(f"🤖 Model: {model_name}")
    
    # Get label classes from label encoder
    label_classes = label_encoder.classes_
    
    # Display sidebar and get selected example
    selected_example = display_sidebar_info(device, label_classes)
    
    # Main content columns
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
            examples = [
                "Today is wonderful! I feel very happy.",
                "I feel very sad and tired.",
                "It feels like my mind has 100 tabs open at once.",
                "The world feels clouded, everything takes so much effort.",
                "I'm exhausted from fighting every day."
            ]
            demo_text = random.choice(examples)
            st.text_area("Demo text:", value=demo_text, height=100, disabled=True)
    
    with col2:
        st.subheader("📈 Quick Statistics")
        if text_input:
            texts = [t.strip() for t in text_input.split('\n') if t.strip()]
            st.metric("Number of Texts", len(texts))
    
    # Process analysis when button is clicked
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
            
            # Display results for each text
            for i, (text, label, confidence) in enumerate(zip(texts, predicted_labels, confidences)):
                display_result_box(text, label, confidence, i)
            
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
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_bar = px.bar(
                    x=df_results['Label'], 
                    y=df_results['Confidence'],
                    title="Confidence per Text",
                    color=df_results['Label']
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
        <p>📧 Developed by Grok AI | 🚀 Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()