from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Oeuvre(Base):
    __tablename__ = "oeuvre"

    id = Column(String, primary_key=True, index=True)
    type_oeuvre = Column(String, nullable=True)
    titre = Column(String, nullable=True)
    auteur = Column(String, nullable=True)
    date_creation = Column(DateTime, nullable=True)
    date_enregistrement = Column(DateTime, nullable=True)
    genre = Column(String, nullable=True)
    langue = Column(String, nullable=True)
    empreinte_hash = Column(String, nullable=True)
    certificat_url = Column(String, nullable=True)
    ia_detecte = Column(String, nullable=True)
    score_ia = Column(Float, nullable=True)

    # ✅ Ajout des colonnes manquantes utilisées dans main.py
    fichier_nom = Column(String, nullable=True)
    ipfs_cid = Column(String, nullable=True)
    fichier_url = Column(String, nullable=True)

    # Relation vers les analyses de plagiat
    analyses = relationship("AnalysePlagiat", back_populates="oeuvre")


class AnalysePlagiat(Base):
    __tablename__ = "analyse_plagiat"

    id = Column(Integer, primary_key=True, index=True)
    oeuvre_compare = Column(String, ForeignKey("oeuvre.id"), nullable=True)
    score_similaire = Column(Float, nullable=True)
    conclusion = Column(String, nullable=True)

    # Relation inverse vers Oeuvre
    oeuvre = relationship("Oeuvre", back_populates="analyses")

    # Relation vers les segments
    segments = relationship("SegmentMatch", back_populates="analyse")


class SegmentMatch(Base):
    __tablename__ = "segment_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analyse_id = Column(Integer, ForeignKey("analyse_plagiat.id"), nullable=True)
    debut = Column(Float, nullable=True)
    fin = Column(Float, nullable=True)
    score = Column(Float, nullable=True)

    # Relation inverse vers AnalysePlagiat
    analyse = relationship("AnalysePlagiat", back_populates="segments")