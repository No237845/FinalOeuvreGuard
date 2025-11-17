# text_analysis.py
"""
Usage:
python text_analysis.py --file path/to/file.txt --k 5

Ce script :
- calcule simhash + embedding pour le fichier
- recherche les k meilleurs candidats dans FAISS
- pour chaque candidat récupéré, effectue l'alignement phrase->phrase et
  produit un rapport PDF détaillé.
"""
import argparse
import numpy as np
from app import processing, embeddings_index, report_utils
from app.db import SessionLocal
from app.models import Oeuvre
import os

def main(filepath, k=5):
    idx, embs, metas = embeddings_index.load_index()
    if idx is None:
        print("Index FAISS introuvable. Lance ingest_index.py d'abord.")
        return
    text = processing.extract_text(filepath)
    simhash_val = processing.compute_simhash(text)
    emb = processing.compute_text_embedding(text)
    D, I = processing.search_faiss_index(idx, emb, k=k)
    candidates = []
    segments_all = []
    db = SessionLocal()
    for score, pos in zip(D, I):
        if pos == -1:
            continue
        meta = metas[pos]
        oeuvre_id = meta["oeuvre_id"]
        oeuvre = db.query(Oeuvre).filter(Oeuvre.id == oeuvre_id).first()
        if oeuvre and oeuvre.type_oeuvre == "texte":
            candidate_text = processing.extract_text(oeuvre.fichier_path)
            matches = processing.sentence_align(text, candidate_text, threshold=0.75)
            candidates.append({"oeuvre_id": oeuvre_id, "score": float(score), "matches": matches[:5]})
            # flatten segments for report
            for m in matches:
                segments_all.append(f"Candidate {oeuvre_id} - ratio {m['ratio']:.2f} - A: {m['a_sentence'][:120]} ...")
    db.close()
    # build report
    metadata = {"file": os.path.basename(filepath), "simhash": simhash_val}
    report_path = f"storage/reports/text_report_{os.path.basename(filepath)}.pdf"
    report_utils.generate_pdf_report(report_path, metadata, [str(c) for c in candidates], segments_all)
    print("Report généré:", report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.file, args.k)
