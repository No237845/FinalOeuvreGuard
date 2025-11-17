# ingest_index.py
"""
Script: ingérer un dossier contenant fichiers texte/audio, stocker métadonnées dans DB,
calculer embeddings (texte ou audio) et construire les index FAISS.
"""
import os
import numpy as np
from sqlalchemy.orm import Session
from app.db import Base, engine, SessionLocal
from app.models import Oeuvre
from app import processing, embeddings_index
from app.utils import make_uid

Base.metadata.create_all(bind=engine)

DATA_DIR = "data_to_ingest"  # dossier contenant fichiers à ingérer

def ingest():
    db: Session = SessionLocal()
    try:
        text_embs, text_metas = [], []
        audio_embs, audio_metas = [], []

        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                filepath = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                uid = make_uid()

                # Créer entrée Oeuvre
                o = Oeuvre(
                    id=uid,
                    titre=f,
                    auteur="",
                    type_oeuvre="audio" if ext in [".mp3", ".wav", ".flac"] else "texte",
                    fichier_nom=os.path.basename(filepath),
                    # si tu as ajouté fichier_path dans ton modèle :
                    # fichier_path=filepath
                )
                db.add(o)
                db.commit()
                db.refresh(o)

                # Calculer embeddings
                if o.type_oeuvre == "texte":
                    text = processing.extract_text(filepath)
                    emb = processing.compute_text_embedding(text)
                    np.save(f"storage/emb_{o.id}.npy", emb)
                    text_embs.append(emb)
                    text_metas.append({"oeuvre_id": o.id, "type": "texte", "path": filepath})
                else:
                    emb = processing.compute_audio_embedding(filepath)
                    np.save(f"storage/emb_{o.id}.npy", emb)
                    audio_embs.append(emb)
                    audio_metas.append({"oeuvre_id": o.id, "type": "audio", "path": filepath})

        # Construire index texte
        if text_embs:
            text_embs = np.vstack(text_embs).astype("float32")
            text_index = processing.build_faiss_index(text_embs, use_inner_prod=True)
            embeddings_index.TEXT_INDEX = text_index
            embeddings_index.TEXT_METAS = text_metas

        # Construire index audio
        if audio_embs:
            audio_embs = np.vstack(audio_embs).astype("float32")
            audio_index = processing.build_faiss_index(audio_embs, use_inner_prod=True)
            embeddings_index.AUDIO_INDEX = audio_index
            embeddings_index.AUDIO_METAS = audio_metas

        # Sauvegarder tout
        embeddings_index.save_all()
        print("Index FAISS texte et audio construits et sauvegardés.")

    finally:
        db.close()

if __name__ == "__main__":
    ingest()