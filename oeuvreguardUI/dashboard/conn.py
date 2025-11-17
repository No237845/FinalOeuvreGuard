import psycopg2
import pandas as pd
import streamlit as st

def get_conn():
    try:
        return psycopg2.connect(
            dbname="oeuvre_guard_db", # ⚠️
            user="postgres",
            password="qwerty123456",
            host="localhost",
            port="5434" 
        )
    except Exception as e:
        st.error(f"Erreur de connexion à la base : {e}")
        return None

def load_data():
    conn = get_conn()
    if conn:
        try:
            query = """
            SELECT id, type_oeuvre, titre, auteur, genre, langue,
                   fichier_nom, date_creation, date_enregistrement,
                   ia_detecte, score_ia
            FROM oeuvre
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if not pd.api.types.is_datetime64_any_dtype(df["date_enregistrement"]):
                df["date_enregistrement"] = pd.to_datetime(df["date_enregistrement"])
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return pd.DataFrame()
    return pd.DataFrame()
