import streamlit as st

# Color mapping for sentiments
COLOR_MAPPING = {
    'normal': '#28a745',
    'bipolar': '#dc3545',
    'personality disorder': '#17a2b8',
    'anxiety': '#ffc107',
    'depression': '#6c757d',
    'stress': '#fd7e14',
    'suicidal': '#dc3545'
}

# Custom CSS for styling
def apply_custom_css():
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

def display_main_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 RoBERTa Sentiment Analysis</h1>
        <p>Analyze text sentiment with custom model</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info(device, label_classes):
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("📝 Sample Examples")
        examples = [
            "Today is wonderful! I feel very happy.",
            "I feel very sad and tired.",
            "It feels like my mind has 100 tabs open at once.",
            "The world feels clouded, everything takes so much effort.",
            "I'm exhausted from fighting every day."
        ]
        
        selected_example = st.selectbox("Select example:", [""] + examples)
        st.markdown('</div>', unsafe_allow_html=True)
        return selected_example

def display_result_box(text, label, confidence, i):
    # Ensure label is valid and exists in color mapping
    label_key = label.replace(' ', '-')  # Convert label to CSS-compatible class name
    label_color = COLOR_MAPPING.get(label, '#000')
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