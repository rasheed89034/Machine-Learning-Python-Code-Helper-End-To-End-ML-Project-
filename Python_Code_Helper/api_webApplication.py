import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os 
import nltk 
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import time 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Python Code Helper", 
    page_icon="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer() 

base_path = os.path.dirname(__file__)
clean_data = os.path.join(base_path, "Models", "cleand_dataframe.pkl")
matrix = os.path.join(base_path, "Models", "matrix.pkl")
model = os.path.join(base_path, "Models", "tfidf_vectorizer.pkl") 

# --- HIGH VISIBILITY CSS OVERHAUL ---
st.markdown("""
<style>
.stApp { background-color: #0f172a; }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stMarkdown { color: #ffffff !important; }

[data-testid="stSidebar"] {
    background-color: #1e293b; 
    border-right: 1px solid #334155;
    box-shadow: 2px 0 10px rgba(0,0,0,0.3);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important; 
}

div[data-baseweb="input"] > div {
    background-color: #1e293b !important;
    border: 2px solid #475569 !important;
    border-radius: 8px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: #38bdf8 !important; 
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.4), inset 0 2px 4px rgba(0,0,0,0.2) !important;
}
div[data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important; 
}
input::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8); 
    color: white !important;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    width: 100%; 
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.6);
    background: linear-gradient(135deg, #3b82f6, #2563eb);
}

.custom-code-box {
    height: 300px; 
    overflow-y: auto; 
    overflow-x: hidden; 
    white-space: pre-wrap; 
    background-color: #020617; 
    color: #38bdf8; 
    padding: 20px; 
    border-radius: 12px; 
    font-family: 'Courier New', Courier, monospace; 
    font-size: 16px;
    font-weight: 500;
    border: 1px solid #334155;
    border-left: 5px solid #10b981; 
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0,0,0,0.5);
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0f172a; border-radius: 8px; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# --- UPGRADED GLOWING SPEEDOMETER ---
def create_gauge_chart(score):
    score_pct = score * 100
    fig = go.Figure()
    
    theta_bg = np.linspace(np.pi, 0, 150)
    x_bg = np.cos(theta_bg)
    y_bg = np.sin(theta_bg)
    
    fig.add_trace(go.Scatter(
        x=x_bg, y=y_bg, mode='lines', 
        line=dict(color='rgba(255,255,255,0.15)', width=2, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    
    if score > 0:
        render_score = max(score, 0.01) 
        theta_fill = np.linspace(np.pi, np.pi * (1 - render_score), 150)
        x_fill = np.cos(theta_fill)
        y_fill = np.sin(theta_fill)
        
        color_vals = np.linspace(0, render_score, 150)
        neon_scale = [[0, '#00f2fe'], [0.33, '#4facfe'], [0.66, '#7f00ff'], [1, '#e100ff']]
        
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill, mode='markers',
            marker=dict(size=25, color=color_vals, colorscale=neon_scale, cmin=0, cmax=1, opacity=0.3, showscale=False),
            hoverinfo='skip', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill, mode='markers',
            marker=dict(size=10, color=color_vals, colorscale=neon_scale, cmin=0, cmax=1, showscale=False),
            hoverinfo='skip', showlegend=False
        ))
        
        if score >= 0.75: tip_color = "#e100ff"
        elif score >= 0.5: tip_color = "#7f00ff"
        elif score >= 0.25: tip_color = "#4facfe"
        else: tip_color = "#00f2fe"
        
        fig.add_trace(go.Scatter(
            x=[x_fill[-1]], y=[y_fill[-1]], mode='markers',
            marker=dict(size=18, color='white', line=dict(width=4, color=tip_color)),
            hoverinfo='skip', showlegend=False
        ))

    for i in range(0, 110, 10):
        angle = np.pi * (1 - i/100)
        x_tick = np.cos(angle) * 0.82
        y_tick = np.sin(angle) * 0.82
        fig.add_annotation(x=x_tick, y=y_tick, text=str(i), font=dict(size=13, color='#64748b', family='Arial'), showarrow=False)

    fig.add_annotation(x=0, y=0.35, text=f"{score_pct:.1f}%", font=dict(size=60, color='white', family='Courier New, monospace'), showarrow=False)
    fig.add_annotation(x=0, y=0.10, text="CONFIDENCE SCORE", font=dict(size=14, color='#94a3b8', family='Arial, sans-serif'), showarrow=False)
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=0, r=0, t=10, b=10), dragmode=False 
    )
    
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.title("👨🏻‍💻 Developer Profile")
    st.markdown("**Name:** Rasheed Ahmad")
    st.markdown("**Role:** Robotic Engineer & ML Dev")
    st.divider() 
    st.title("🚀 About Project")
    st.info("This is an AI-powered Python Helper. It uses Natural Language Processing to match your queries against a massive database of Python solutions.")
    st.title("📊 Model Details")
    st.markdown("🔹 **Algorithm:** TF-IDF")
    st.markdown("🔹 **Vocabulary Size:** 7,352 Words")
    st.markdown("🔹 **Dataset:** 13,836 Q&A Pairs")
    st.divider()
    st.title("💡 Pro Tip")
    st.warning("For the best results, use clear Python terminology. (e.g., 'How to reverse a list').")

# --- MAIN APP UI ---

# THIS REPLACES THE st.title() TO SHOW THE OFFICIAL PYTHON LOGO
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="60" style="margin-right: 15px;">
        <h1 style="margin: 0; padding: 0;">Python Code Helper</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("### 🔍 Search the Knowledge Base")

@st.cache_resource
def load_ml_components():
    vec = joblib.load(model)
    tfidf = joblib.load(matrix)
    dataframe = pd.read_pickle(clean_data)
    return vec, tfidf, dataframe

vectorizer, X, df = load_ml_components()

user_question = st.text_input("", placeholder="e.g., How do I read a CSV file using pandas?")

if st.button("Search Database"):
    if user_question:
        cleaned_q = user_question.lower().split()
        cleaned_q = [lemmatizer.lemmatize(word) for word in cleaned_q]
        cleaned_q = ' '.join(cleaned_q)

        user_vector = vectorizer.transform([cleaned_q])
        similarities = cosine_similarity(user_vector, X)

        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        
        if best_score > 0.15: 
            final_answer = df['Answer'].iloc[best_match_index]
            
            st.success(f"✅ Match found in database!")
            
            gauge_placeholder = st.empty()
            
            target_pct = int(best_score * 100)
            step_size = max(1, target_pct // 15) 
            
            for current_val in range(0, target_pct, step_size):
                gauge_placeholder.plotly_chart(create_gauge_chart(current_val / 100), use_container_width=True)
                time.sleep(0.02) 
                
            gauge_placeholder.plotly_chart(create_gauge_chart(best_score), use_container_width=True)
            
            st.markdown(f"<div class='custom-code-box'>{final_answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ I'm sorry, I couldn't find a matching Python answer for that in my database.")
    else:
        st.error("❌ Please enter a question before searching!")
