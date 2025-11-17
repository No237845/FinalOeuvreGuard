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

st.title("Liste des œuvres")

df = load_data()
if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

# Filtres
col1, col2 = st.columns(2)
with col1:
    type_filter = st.selectbox("Filtrer par type", ["Tous"] + sorted(df["type_oeuvre"].dropna().unique().tolist()))
with col2:
    auteur_filter = st.selectbox("Filtrer par auteur", ["Tous"] + sorted(df["auteur"].dropna().unique().tolist()))

filtered_df = df.copy()
if type_filter != "Tous":
    filtered_df = filtered_df[filtered_df["type_oeuvre"] == type_filter]
if auteur_filter != "Tous":
    filtered_df = filtered_df[filtered_df["auteur"] == auteur_filter]

# Pagination
page_size = 10
total_pages = (len(filtered_df) - 1) // page_size + 1
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

start = (page - 1) * page_size
end = start + page_size
st.dataframe(filtered_df.iloc[start:end], use_container_width=True, height=390)
st.caption(f"Page {page}/{total_pages}")
