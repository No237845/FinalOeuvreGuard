import streamlit as st
import datetime
from conn import load_data


st.set_page_config(page_title="Oeuvre Guard", layout="wide")
st.markdown("""
    <style>
    /* Augmenter la taille de la police uniquement dans la sidebar */
    .stSidebar {
        font-size: 18px !important;
    }
    /* Tu peux aussi cibler les labels des widgets dans la sidebar */
    .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label,
    .stSidebar .stTextInput label {
        font-size: 17px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Oeuvre Guard")

df = load_data()
if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

st.markdown("""
Bienvenue dans **Oeuvre Guard**  
""")

# --- Résumé global ---
st.subheader("Résumé rapide")

col1, col2, col3 = st.columns(3)
col1.metric("Total œuvres", len(df))
col2.metric("Détections IA", len(df[df["ia_detecte"].notnull()]))

today = datetime.date.today()
today_count = len(df[df["date_enregistrement"].dt.date == today])
col3.metric("Aujourd'hui", today_count)
