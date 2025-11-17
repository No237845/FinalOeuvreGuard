# audio_analysis.py
"""
Usage:
python audio_analysis.py --file path/to/file.mp3 --k 5

- calcule fingerprint via fpcalc, embedding audio
- recherche candidats via FAISS
- pour chaque candidat audio -> DTW alignment, extraction segments
- génère rapport PDF
"""
import argparse, os
import numpy as np
from app import processing, embeddings_index, report_utils
from app.db import SessionLocal
from app.models import Oeuvre

def main(filepath, k=5):
    idx, embs, metas = embeddings_index.load_index()
    if idx is None:
        print("Index FAISS introuvable. Lance ingest_index.py d'abord.")
        return
    fp = processing.compute_audio_fingerprint(filepath)
    emb = processing.compute_audio_embedding(filepath)
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
        if oeuvre and oeuvre.type_oeuvre == "audio":
            # run DTW alignment
            segs = processing.audio_dtw_segments(filepath, oeuvre.fichier_path, top_matches=3)
            candidates.append({"oeuvre_id": oeuvre_id, "score": float(score), "segments": segs})
            for s in segs:
                segments_all.append(f"Candidate {oeuvre_id} - {s}")
    db.close()
    metadata = {"file": os.path.basename(filepath), "fingerprint": fp}
    report_path = f"storage/reports/audio_report_{os.path.basename(filepath)}.pdf"
    report_utils.generate_pdf_report(report_path, metadata, [str(c) for c in candidates], segments_all)
    print("Report généré:", report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.file, args.k)
