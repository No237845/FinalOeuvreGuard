import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import locale
from conn import load_data

# --- Config ---
st.set_page_config(page_title="Oeuvre Guard - Statistiques", layout="wide")
st.markdown("""
    <style>
    .stSidebar {
        font-size: 18px !important;
    }
    .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label,
    .stSidebar .stTextInput label {
        font-size: 17px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Statistiques & Graphiques")

df = load_data()
if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

# --- Fonction utilitaire pour afficher une card ---
def card(title, value, color="#00C853"):
    st.markdown(
        f"""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            margin: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 5px solid {color};
            height: 140px;   /* plus haut */
            width: 100%;     /* occupe toute la colonne */
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <h4 style="margin:0; font-size:15px; color: #555;">{title}</h4>
            <p style="margin:0; font-size:24px; font-weight:bold; color:{color};">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Première ligne de cards (espacées sur toute la largeur) ---
col1, spacer1, col2, spacer2 = st.columns([1,0.2,1,0.2])
with col1: card("Total œuvres", len(df))
with col2: card("Détections IA", len(df[df["ia_detecte"].notnull()]))

# --- Deuxième ligne de cards (espacées sur toute la largeur) ---
today = pd.Timestamp(datetime.date.today())
today_count = len(df[df["date_enregistrement"].dt.date == today.date()])

col3, spacer3, col4, spacer4 = st.columns([1,0.2,1,0.2])
with col3: card("Uploadé aujourd'hui", today_count, "#2196F3")
with col4:
    if not df[df["ia_detecte"].notnull()].empty:
        fraud_counts = df[df["ia_detecte"].notnull()].groupby(df["date_enregistrement"].dt.date).size()
        max_day = fraud_counts.idxmax()
        max_count = fraud_counts.max()

        # --- Formatage en français ---
        try:
            locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
            date_str = max_day.strftime("%d %B")
        except:
            mois_fr = {
                1: "janvier", 2: "février", 3: "mars", 4: "avril",
                5: "mai", 6: "juin", 7: "juillet", 8: "août",
                9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
            }
            date_str = f"{max_day.day} {mois_fr[max_day.month]}"

        periode = f"{date_str} dernier"
        card("Période max détection", f"{periode} ({max_count})", "#E91E63")

# --- Les 2 restants en chiffres simples ---
st.subheader("Autres chiffres")
col1, col2 = st.columns(2)
with col1:
    st.metric("Œuvres musicales", len(df[df["type_oeuvre"] == "musique"]))
with col2:
    st.metric("Œuvres textuelles", len(df[df["type_oeuvre"] == "texte"]))

# --- Graphiques côte à côte et carrés ---
st.subheader("Visualisations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Répartition par type**")
    plt.rcParams.update({'font.size': 9})
    type_counts = df["type_oeuvre"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5,5))  # carré
    type_counts.plot(kind="barh", color="#00C853", ax=ax1) 
    ax1.set_xlabel("Nombre d'œuvres")
    ax1.set_ylabel("Type")
    st.pyplot(fig1)

with col2:
    st.markdown("**Répartition par langue**")
    lang_counts = df["langue"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4,4))  # cercle plus petit
    ax2.pie(lang_counts, labels=lang_counts.index, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 6})
    ax2.axis("equal")
    st.pyplot(fig2)
