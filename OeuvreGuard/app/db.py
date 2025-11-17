# app/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Définition de Base ici (pas dans models.py)
Base = declarative_base()

# URL de connexion à ta base Postgres
DATABASE_URL = "postgresql://postgres:qwerty123456@localhost:5434/oeuvre_guard_db"

# Création du moteur
engine = create_engine(DATABASE_URL)

# Session locale
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Fonction utilitaire pour obtenir une session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()