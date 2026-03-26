# import streamlit as st 
# import pandas as pd 
# import numpy as np 
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import os 
# import nltk 
# from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet',quiet=True)
# lemmatizer = WordNetLemmatizer() 

# base_path = os.path.dirname(__file__)

# clean_data = os.path.join(base_path,"Models","cleand_dataframe.pkl")
# matrix = os.path.join(base_path,"Models","matrix.pkl")
# model = os.path.join(base_path,"Models","tfidf_vectorizer.pkl") 

# with st.sidebar:
#     st.title("👨🏻‍💻Developer Info")
#     st.write("**Name:** Rasheed Ahmad")
#     st.write("**Role:** ML Engineer")

# st.title("👋🏻Welcome To Python Code 👨🏻‍💻 Helper")
# st.markdown("🔍 Enter Your Query")

# @st.cache_resource
# def load_ml_components():
#     vec = joblib.load(model)
#     tfidf = joblib.load(matrix)
#     dataframe = pd.read_pickle(clean_data)

#     return vec, tfidf, dataframe

# vertorizer , X, df = load_ml_components()

# user_question = st.text_input("How can I help you with Python today?")

# if st.button("Search"):
#     if user_question:
#         cleaned_q = user_question.lower().split()
#         cleaned_q = [lemmatizer.lemmatize(word) for word in cleaned_q]
#         cleaned_q = ' '.join(cleaned_q)

#         user_vector = vertorizer.transform([cleaned_q]).toarray()

#         similarities = cosine_similarity(user_vector, X)

#         best_match_index = np.argmax(similarities)
#         best_score = similarities[0][best_match_index]
        
#         # Step E: Display the Results
#         if best_score > 0.15: # Threshold to ensure it's a relevant match
#             final_answer = df['Answer'].iloc[best_match_index]
            
#             st.success(f"Match found! (Confidence Score: {best_score:.2f})")
            
#             # Displaying the code beautifully in Streamlit
#             # st.code(final_answer, language="python",wrap_lines=True) 
#             st.markdown(
#                 f"""
#                 <div style='height: 300px; overflow-y: auto; overflow-x: hidden; white-space: pre-wrap; background-color: #1E1E1E; color: #D4D4D4; padding: 15px; border-radius: 8px; font-family: monospace; border: 1px solid #444;'>
#                     {final_answer}
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
#         else:
#             st.warning("I'm sorry, I couldn't find a matching Python answer for that in my database.")
#     else:
#         st.error("Please enter a question before searching!")





# import streamlit as st 
# import pandas as pd 
# import numpy as np 
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import os 
# import nltk 
# from nltk.stem import WordNetLemmatizer

# # --- PAGE CONFIGURATION ---
# st.set_page_config(page_title="Python Helper", page_icon="🐍", layout="centered", initial_sidebar_state="expanded")

# nltk.download('wordnet', quiet=True)
# lemmatizer = WordNetLemmatizer() 

# base_path = os.path.dirname(__file__)
# clean_data = os.path.join(base_path, "Models", "cleand_dataframe.pkl")
# matrix = os.path.join(base_path, "Models", "matrix.pkl")
# model = os.path.join(base_path, "Models", "tfidf_vectorizer.pkl") 

# # --- HIGH VISIBILITY CSS OVERHAUL ---
# st.markdown("""
# <style>
# /* 1. Global App Background & Bright Text */
# .stApp {
#     background-color: #0f172a; /* Deep Slate Background */
#     color: #f8fafc; /* Crisp White Text for High Visibility */
# }
# .stApp h1, .stApp h2, .stApp h3, .stApp p, .stMarkdown {
#     color: #ffffff !important; 
# }

# /* 2. Sidebar Styling */
# [data-testid="stSidebar"] {
#     background-color: #1e293b; /* Slightly lighter slate for contrast */
#     border-right: 1px solid #334155;
# }
# /* Force all text in the sidebar to be bright */
# [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
#     color: #f1f5f9 !important; 
# }

# /* 3. Styling the Text Input Field */
# div[data-baseweb="input"] > div {
#     background-color: #1e293b !important;
#     border: 2px solid #475569 !important;
#     border-radius: 8px !important;
#     color: #ffffff !important;
# }
# div[data-baseweb="input"] > div:focus-within {
#     border-color: #3b82f6 !important; /* Bright blue focus */
#     box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
# }
# div[data-baseweb="input"] input {
#     color: #ffffff !important;
#     -webkit-text-fill-color: #ffffff !important; /* Ensures it works on Safari/Chrome */
# }
# input::placeholder {
#     color: #94a3b8 !important;
#     -webkit-text-fill-color: #94a3b8 !important;
# }

# /* 4. Styling the Search Button */
# div.stButton > button {
#     background: linear-gradient(90deg, #2563eb, #1d4ed8); /* Bright Blue Button */
#     color: white !important;
#     border: none;
#     border-radius: 8px;
#     padding: 10px 24px;
#     font-weight: bold;
#     font-size: 16px;
#     transition: all 0.3s ease;
#     box-shadow: 0 4px 6px rgba(0,0,0,0.3);
#     width: 100%; 
# }
# div.stButton > button:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 6px 15px rgba(37, 99, 235, 0.5);
#     background: linear-gradient(90deg, #3b82f6, #2563eb);
# }

# /* 5. The Output Box */
# .custom-code-box {
#     height: 300px; 
#     overflow-y: auto; 
#     overflow-x: hidden; 
#     white-space: pre-wrap; 
#     background-color: #020617; /* Very dark background for code */
#     color: #38bdf8; /* Bright neon blue for code text */
#     padding: 20px; 
#     border-radius: 12px; 
#     font-family: 'Courier New', Courier, monospace; 
#     font-size: 16px;
#     font-weight: 500;
#     border: 1px solid #334155;
#     border-left: 5px solid #10b981; /* Emerald green accent line */
#     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
# }

# /* Custom Scrollbar */
# ::-webkit-scrollbar { width: 8px; }
# ::-webkit-scrollbar-track { background: #0f172a; border-radius: 8px; }
# ::-webkit-scrollbar-thumb { background: #475569; border-radius: 8px; }
# ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
# </style>
# """, unsafe_allow_html=True)

# # --- ENHANCED SIDEBAR ---
# with st.sidebar:
#     st.title("👤 Developer Profile")
#     st.markdown("**Name:** Rasheed Ahmad")
#     st.markdown("**Role:** Robotic Engineer & ML Dev")
    
    
#     st.divider() # Adds a clean horizontal line
    
#     st.title("🚀 About Project")
#     st.info("This is an AI-powered Python Helper. It uses Natural Language Processing to match your queries against a massive database of Python solutions.")
    
#     st.title("📊 Model Details")
#     st.markdown("🔹 **Algorithm:** TF-IDF")
#     st.markdown("🔹 **Vocabulary Size:** 7,352 Words")
#     st.markdown("🔹 **Dataset:** 13,836 Q&A Pairs")
    
#     st.divider()
    
#     st.title("💡 Pro Tip")
#     st.warning("For the best results, use clear Python terminology. (e.g., 'How to reverse a list' instead of 'how to flip words').")

# # --- MAIN APP UI ---
# st.title("🐍 Python Code Helper")
# st.markdown("### 🔍 Search the Knowledge Base")

# @st.cache_resource
# def load_ml_components():
#     vec = joblib.load(model)
#     tfidf = joblib.load(matrix)
#     dataframe = pd.read_pickle(clean_data)
#     return vec, tfidf, dataframe

# vectorizer, X, df = load_ml_components()

# user_question = st.text_input("", placeholder="e.g., How do I read a CSV file using pandas?")

# if st.button("Search Database"):
#     if user_question:
#         cleaned_q = user_question.lower().split()
#         cleaned_q = [lemmatizer.lemmatize(word) for word in cleaned_q]
#         cleaned_q = ' '.join(cleaned_q)

#         user_vector = vectorizer.transform([cleaned_q]).toarray()
#         similarities = cosine_similarity(user_vector, X)

#         best_match_index = np.argmax(similarities)
#         best_score = similarities[0][best_match_index]
        
#         if best_score > 0.15: 
#             final_answer = df['Answer'].iloc[best_match_index]
#             st.success(f"✅ Match found! (Confidence Score: {best_score:.2f})")
#             st.markdown(f"<div class='custom-code-box'>{final_answer}</div>", unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ I'm sorry, I couldn't find a matching Python answer for that in my database.")
#     else:
#         st.error("❌ Please enter a question before searching!")


# import streamlit as st 
# import pandas as pd 
# import numpy as np 
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import os 
# import nltk 
# from nltk.stem import WordNetLemmatizer
# import plotly.graph_objects as go # NEW: Imported Plotly for the speedometer

# # --- PAGE CONFIGURATION ---
# st.set_page_config(page_title="Python Helper", page_icon="🐍", layout="centered", initial_sidebar_state="expanded")

# nltk.download('wordnet', quiet=True)
# lemmatizer = WordNetLemmatizer() 

# base_path = os.path.dirname(__file__)
# clean_data = os.path.join(base_path, "Models", "cleand_dataframe.pkl")
# matrix = os.path.join(base_path, "Models", "matrix.pkl")
# model = os.path.join(base_path, "Models", "tfidf_vectorizer.pkl") 

# # --- HIGH VISIBILITY CSS OVERHAUL ---
# st.markdown("""
# <style>
# /* 1. Global App Background */
# .stApp { background-color: #0f172a; }

# /* 2. FORCE WHITE TEXT FOR ALL HEADERS AND MARKDOWN */
# .stApp h1, .stApp h2, .stApp h3, .stApp p, .stMarkdown { color: #ffffff !important; }

# /* 3. Sidebar Styling */
# [data-testid="stSidebar"] {
#     background-color: #1e293b; 
#     border-right: 1px solid #334155;
# }
# [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
#     color: #f1f5f9 !important; 
# }

# /* 4. Styling the Text Input Field Container */
# div[data-baseweb="input"] > div {
#     background-color: #1e293b !important;
#     border: 2px solid #475569 !important;
#     border-radius: 8px !important;
# }
# div[data-baseweb="input"] > div:focus-within {
#     border-color: #3b82f6 !important; 
#     box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
# }
# div[data-baseweb="input"] input {
#     color: #ffffff !important;
#     -webkit-text-fill-color: #ffffff !important; 
# }
# input::placeholder {
#     color: #94a3b8 !important;
#     -webkit-text-fill-color: #94a3b8 !important;
# }

# /* 5. Styling the Search Button */
# div.stButton > button {
#     background: linear-gradient(90deg, #2563eb, #1d4ed8); 
#     color: white !important;
#     border: none;
#     border-radius: 8px;
#     padding: 10px 24px;
#     font-weight: bold;
#     font-size: 16px;
#     transition: all 0.3s ease;
#     box-shadow: 0 4px 6px rgba(0,0,0,0.3);
#     width: 100%; 
# }
# div.stButton > button:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 6px 15px rgba(37, 99, 235, 0.5);
#     background: linear-gradient(90deg, #3b82f6, #2563eb);
# }

# /* 6. The Output Box */
# .custom-code-box {
#     height: 300px; 
#     overflow-y: auto; 
#     overflow-x: hidden; 
#     white-space: pre-wrap; 
#     background-color: #020617; 
#     color: #38bdf8; 
#     padding: 20px; 
#     border-radius: 12px; 
#     font-family: 'Courier New', Courier, monospace; 
#     font-size: 16px;
#     font-weight: 500;
#     border: 1px solid #334155;
#     border-left: 5px solid #10b981; 
#     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
# }

# ::-webkit-scrollbar { width: 8px; }
# ::-webkit-scrollbar-track { background: #0f172a; border-radius: 8px; }
# ::-webkit-scrollbar-thumb { background: #475569; border-radius: 8px; }
# ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
# </style>
# """, unsafe_allow_html=True)

# # --- NEW: SPEEDOMETER FUNCTION ---
# def create_gauge_chart(score):
#     """Creates a Plotly gauge chart for the confidence score."""
#     score_pct = score * 100 # Convert decimal to percentage
    
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number",
#         value = score_pct,
#         number = {'suffix': "%", 'font': {'color': 'white', 'size': 40}},
#         title = {'text': "AI Confidence Level", 'font': {'color': '#94a3b8', 'size': 18}},
#         gauge = {
#             'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
#             'bar': {'color': "#ffffff", 'thickness': 0.15}, # The needle
#             'bgcolor': "rgba(0,0,0,0)",
#             'borderwidth': 2,
#             'bordercolor': "#334155",
#             'steps': [
#                 {'range': [0, 40], 'color': "#ef4444"},   # Red (Low)
#                 {'range': [40, 75], 'color': "#f59e0b"},  # Yellow (Medium)
#                 {'range': [75, 100], 'color': "#10b981"}  # Green (High)
#             ]
#         }
#     ))
    
#     # Make the background perfectly transparent to match your app
#     fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         height=250,
#         margin=dict(l=10, r=10, t=40, b=10)
#     )
#     return fig

# # --- ENHANCED SIDEBAR ---
# with st.sidebar:
#     st.title("👤 Developer Profile")
#     st.markdown("**Name:** Muhammad Bilal")
#     st.markdown("**Role:** Robotic Engineer & ML Dev")
#     st.divider() 
#     st.title("🚀 About Project")
#     st.info("This is an AI-powered Python Helper. It uses Natural Language Processing to match your queries against a massive database of Python solutions.")
#     st.title("📊 Model Details")
#     st.markdown("🔹 **Algorithm:** TF-IDF")
#     st.markdown("🔹 **Vocabulary Size:** 7,352 Words")
#     st.markdown("🔹 **Dataset:** 13,836 Q&A Pairs")
#     st.divider()
#     st.title("💡 Pro Tip")
#     st.warning("For the best results, use clear Python terminology. (e.g., 'How to reverse a list' instead of 'how to flip words').")

# # --- MAIN APP UI ---
# st.title("🐍 Python Code Helper")
# st.markdown("### 🔍 Search the Knowledge Base")

# @st.cache_resource
# def load_ml_components():
#     vec = joblib.load(model)
#     tfidf = joblib.load(matrix)
#     dataframe = pd.read_pickle(clean_data)
#     return vec, tfidf, dataframe

# vectorizer, X, df = load_ml_components()

# user_question = st.text_input("", placeholder="e.g., How do I read a CSV file using pandas?")

# if st.button("Search Database"):
#     if user_question:
#         cleaned_q = user_question.lower().split()
#         cleaned_q = [lemmatizer.lemmatize(word) for word in cleaned_q]
#         cleaned_q = ' '.join(cleaned_q)

#         user_vector = vectorizer.transform([cleaned_q]).toarray()
#         similarities = cosine_similarity(user_vector, X)

#         best_match_index = np.argmax(similarities)
#         best_score = similarities[0][best_match_index]
        
#         if best_score > 0.15: 
#             final_answer = df['Answer'].iloc[best_match_index]
            
#             # --- RENDER THE SPEEDOMETER ---
#             st.plotly_chart(create_gauge_chart(best_score), use_container_width=True)
            
#             # --- RENDER THE CODE BOX ---
#             st.markdown(f"<div class='custom-code-box'>{final_answer}</div>", unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ I'm sorry, I couldn't find a matching Python answer for that in my database.")
#     else:
#         st.error("❌ Please enter a question before searching!")


import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os 
import nltk 
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import time # NEW: Needed for the needle animation

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Python Helper", page_icon="🐍", layout="centered", initial_sidebar_state="expanded")

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
    """Creates a custom, pixel-perfect clone of the neon dashboard gauge using polar mathematics."""
    score_pct = score * 100
    
    fig = go.Figure()
    
    # 1. Background Track (Faint, dotted arc line)
    theta_bg = np.linspace(np.pi, 0, 150)
    x_bg = np.cos(theta_bg)
    y_bg = np.sin(theta_bg)
    
    fig.add_trace(go.Scatter(
        x=x_bg, y=y_bg, 
        mode='lines', 
        line=dict(color='rgba(255,255,255,0.15)', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # 2. Glowing Gradient Fill
    if score > 0:
        render_score = max(score, 0.01) 
        theta_fill = np.linspace(np.pi, np.pi * (1 - render_score), 150)
        x_fill = np.cos(theta_fill)
        y_fill = np.sin(theta_fill)
        
        # Lock the colorscale to absolute 0-100% so the gradient unrolls properly
        color_vals = np.linspace(0, render_score, 150)
        neon_scale = [[0, '#00f2fe'], [0.33, '#4facfe'], [0.66, '#7f00ff'], [1, '#e100ff']]
        
        # Outer Glow (Wide, highly transparent trace)
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            mode='markers',
            marker=dict(
                size=25, 
                color=color_vals,
                colorscale=neon_scale,
                cmin=0, cmax=1, 
                opacity=0.3,
                showscale=False
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Inner Solid Tube
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            mode='markers',
            marker=dict(
                size=10, 
                color=color_vals,
                colorscale=neon_scale,
                cmin=0, cmax=1,
                showscale=False
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Glowing Tip (The Needle)
        if score >= 0.75: tip_color = "#e100ff"
        elif score >= 0.5: tip_color = "#7f00ff"
        elif score >= 0.25: tip_color = "#4facfe"
        else: tip_color = "#00f2fe"
        
        fig.add_trace(go.Scatter(
            x=[x_fill[-1]], y=[y_fill[-1]],
            mode='markers',
            marker=dict(size=18, color='white', line=dict(width=4, color=tip_color)),
            hoverinfo='skip',
            showlegend=False
        ))

    # 3. Inner Tick Marks (0, 10, 20... 100) exactly like the image
    for i in range(0, 110, 10):
        angle = np.pi * (1 - i/100)
        x_tick = np.cos(angle) * 0.82
        y_tick = np.sin(angle) * 0.82
        fig.add_annotation(
            x=x_tick, y=y_tick,
            text=str(i),
            font=dict(size=13, color='#64748b', family='Arial'),
            showarrow=False
        )

    # 4. Central Digital Text
    fig.add_annotation(
        x=0, y=0.35, 
        text=f"{score_pct:.1f}%", 
        font=dict(size=60, color='white', family='Courier New, monospace'), 
        showarrow=False
    )
    fig.add_annotation(
        x=0, y=0.10, 
        text="CONFIDENCE SCORE", 
        font=dict(size=14, color='#94a3b8', family='Arial, sans-serif'), 
        showarrow=False
    )
    
    # 5. Hide axes and lock aspect ratio perfectly
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=0, r=0, t=10, b=10),
        dragmode=False 
    )
    
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.title("👨🏻‍💻 Developer Profile")
    st.markdown("**Name:** Rasheed Ahmad")
    st.markdown("**Role:** ML Engineer")
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
st.title("🐍 Python Code Helper")
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
            
            # --- ANIMATION LOGIC ---
            st.success(f"✅ Match found in database!")
            
            # 1. Create an empty container for the chart
            gauge_placeholder = st.empty()
            
            # 2. Loop to animate the needle rising
            target_pct = int(best_score * 100)
            step_size = max(1, target_pct // 15) # Breaks animation into smooth ~15 frames
            
            for current_val in range(0, target_pct, step_size):
                gauge_placeholder.plotly_chart(create_gauge_chart(current_val / 100), use_container_width=True)
                time.sleep(0.02) # Speed of the animation
                
            # 3. Ensure it lands exactly on the final decimal score
            gauge_placeholder.plotly_chart(create_gauge_chart(best_score), use_container_width=True)
            
            # --- RENDER THE CODE BOX ---
            st.markdown(f"<div class='custom-code-box'>{final_answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ I'm sorry, I couldn't find a matching Python answer for that in my database.")
    else:
        st.error("❌ Please enter a question before searching!")