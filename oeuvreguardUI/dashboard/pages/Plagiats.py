import streamlit as st
from conn import load_data


st.set_page_config(page_title="Oeuvre Guard - Statistiques", layout="wide")
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

st.title("Œuvres détectées par l'IA")

df = load_data()
if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

suspect_df = df[df["ia_detecte"].notnull()]
if not suspect_df.empty:
    page_size = 5
    total_pages = (len(suspect_df) - 1) // page_size + 1
    page = st.number_input("Page IA", min_value=1, max_value=total_pages, value=1)

    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(
        suspect_df.iloc[start:end][["id", "titre", "auteur", "ia_detecte", "score_ia"]],
        use_container_width=True,
        height=300
    )
    st.caption(f"Page {page}/{total_pages}")
else:
    st.info("Aucune détection IA pour le moment.")
