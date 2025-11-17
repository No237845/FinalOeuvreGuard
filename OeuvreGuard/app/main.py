from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
import torch
from sqlalchemy.orm import Session
import psycopg2, requests, mimetypes, datetime, os
from .db import get_db, engine
from .models import Base, Oeuvre, AnalysePlagiat
from . import processing, embeddings_index, report_utils
from .certificat_utils import generate_certificat
#from bert_detector import predict_bert
from .fingerprints import compute_emp_hash_for_upload
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import tempfile
import io
import uuid
import soundfile as sf
import librosa




import joblib
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================
#  Chargement MODEL AUDIO IA
# ============================

AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "audio_model_final.pth")

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        
        self.fc1 = nn.Linear(20, 64)   # MATCH: weight is [64,20]
        self.fc2 = nn.Linear(64, 32)   # MATCH: weight is [32,64]
        self.fc3 = nn.Linear(32, 2)    # MATCH: weight is [2,32]
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Chargement
try:
    audio_model = AudioClassifier()
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location="cpu"))
    audio_model.eval()
except Exception as e:
    print(f"[AUDIO MODEL] Impossible de charger le modèle : {e}")
    audio_model = None


def extract_mfcc(audio_np, sr):
    # Convertir en float32
    audio_np = audio_np.astype(np.float32)

    # Extraire 20 MFCC
    mfcc = librosa.feature.mfcc(
        y=audio_np,
        sr=sr,
        n_mfcc=20
    )

    # Moyenne temporelle → vecteur (20,)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


def predict_audio_ia(mfcc_vector):
    if audio_model is None:
        return {"error": "Modèle audio introuvable"}

    x = torch.tensor(mfcc_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = audio_model(x)
        probs = torch.softmax(logits, dim=1)[0]

    prob_ia = float(probs[0])
    prob_human = float(probs[1])

    # Gestion incertitude comme BERT
    label, _, _ = prediction_with_uncertainty(prob_ia)

    return {
        "prediction": label,
        "proba_IA": prob_ia,
        "proba_Humain": prob_human
    }

# ============================
#  FONCTIONS D'EXTRACTION ALTERNATIVES
# ============================

def decode_audio_bytes(file_bytes: bytes, filename: str):
    """
    Décode n'importe quel fichier audio (wav, mp3, m4a, aac, ogg, 3gp…)
    en un tableau numpy + sample rate.
    """

    try:
        # Sauvegarde dans un fichier temp afin de permettre à pydub/ffmpeg d’analyser
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        # Charger via pydub/ffmpeg
        audio = AudioSegment.from_file(tmp_path)

        # Convertir en waveform numpy (mono)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            samples = samples.mean(axis=1)

        sr = audio.frame_rate

        return samples, sr

    except Exception as e:
        print("❌ Erreur décodage audio:", e)
        return None, None


def extract_text_with_pymupdf(file_bytes: bytes) -> str:
    """Extraction texte avec PyMuPDF (plus robuste pour PDF)"""
    try:
        import fitz  # PyMuPDF
        
        pdf_stream = io.BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text.strip()
        
    except Exception as e:
        print(f"❌ Erreur PyMuPDF: {e}")
        return ""

def extract_text_with_pypdf2(file_bytes: bytes) -> str:
    """Extraction texte avec PyPDF2"""
    try:
        import PyPDF2
        
        pdf_stream = io.BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return text.strip()
    except Exception as e:
        print(f"❌ Erreur PyPDF2: {e}")
        return ""


# Initialisation DB
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Autorise toutes les URLs (simple pour dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Connexion psycopg2 directe (si besoin)
def get_conn():
    return psycopg2.connect(
        dbname="oeuvre_guard_db",
        user="postgres",
        password="qwerty123456",
        host="localhost",
        port="5434"
    )
##Bert
# ============================
#  Gestion de l'incertitude
# ============================

def prediction_with_uncertainty(prob_ia, threshold=0.15):
    """
    Retourne IA, Humain ou Incertain selon la différence de probabilité.
    """
    prob_human = 1 - prob_ia
    diff = abs(prob_ia - prob_human)

    if diff < threshold:
        return 'incertain', prob_ia, prob_human

    return ('IA', prob_ia, prob_human) if prob_ia > prob_human else ('Humain', prob_ia, prob_human)


# ============================
#  Chargement du modèle BERT
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_DIR = os.path.join(BASE_DIR, "bert_model_directory")

try:
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.eval()
except Exception as e:
    print(f"[BERT] Erreur chargement modèle/tokenizer: {e}")
    tokenizer = None
    bert_model = None


# ============================
#  Fonction principale
# ============================

def predict_bert(text: str, threshold: float = 0.30):

    if bert_model is None or tokenizer is None:
        return {'error': 'Modèle BERT ou tokenizer introuvable'}

    if not text or len(text.strip()) < 5:
        return {"error": "Texte vide ou trop court"}

    print("=== TEXTE REÇU ===")
    print(text[:300], "...\n")

    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits

    print("LOGITS:", logits)

    probs = torch.softmax(logits, dim=1)[0]
    print("PROBS:", probs)

    prob_ia = float(probs[0])
    prob_human = float(probs[1])

    label, final_ia, final_hum = prediction_with_uncertainty(prob_ia, threshold)

    return {
        "prediction": label,
        "proba_IA": final_ia,
        "proba_Humain": final_hum
    }
def is_valid_wav(file_bytes: bytes) -> bool:
    # Vérifie header RIFF / WAVE
    return len(file_bytes) > 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WAVE"

# Upload vers IPFS local
def upload_to_ipfs(file_bytes: bytes, filename: str) -> str:
    try:
        url = "http://127.0.0.1:5001/api/v0/add"
        files = {"file": (filename, file_bytes)}
        response = requests.post(url, files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Erreur IPFS local: {response.text}")

        cid = response.json()["Hash"]
        return cid
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IPFS local: {str(e)}")

# Détection du type
def detect_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if mime.startswith("audio/"):
            return "musique"
        elif mime in (
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain"
        ):
            return "texte"
    return "autre"

# Génération ID
def generate_id(file_type: str) -> str:
    year = datetime.datetime.now().year
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    prefix = "MUS" if file_type == "musique" else "BOOK" if file_type == "texte" else "GEN"
    return f"{prefix}-{year}-{timestamp}"
def sanitize_for_json(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

# Charger FAISS index si dispo

(text_index, text_metas), (audio_index, audio_metas) = embeddings_index.load_index()



@app.post("/upload")
async def upload(
    titre: str = Form(...),
    auteur: str = Form(...),
    genre: str = Form(...),
    langue: str = Form(...),
    fichier: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Lire fichier
    file_bytes = await fichier.read()

    # Upload vers IPFS
    cid = upload_to_ipfs(file_bytes, fichier.filename)
    fichier_url = f"http://127.0.0.1:8080/ipfs/{cid}"

    # Détecter type
    type_oeuvre = detect_type(fichier.filename)

    # Générer ID
    oeuvre_id = generate_id(type_oeuvre)

    # Valeurs IA par défaut
    ia_label = None
    ia_score = None

    # ------------------------------------------------------------------
    # 🔥🔥 MODULE 2 : Empreinte numérique (texte ou audio)
    # ------------------------------------------------------------------
    try:
        empreinte_hash, empreinte_details = compute_emp_hash_for_upload(
            file_bytes,
            fichier.filename,
            type_oeuvre
        )
    except Exception as e:
        empreinte_hash = None
        empreinte_details = {"error": str(e)}

    # ------------------------------------------------------------------
    # 🔥 Pipeline TEXTE
    # ------------------------------------------------------------------
    if type_oeuvre == "texte":

        # ------------------------------
        # 1) Vérifier si déjà enregistré
        # ------------------------------
        existe = db.query(Oeuvre).filter(Oeuvre.empreinte_hash == empreinte_hash).first()
        if existe:
            raise HTTPException(
                status_code=400,
                detail=f"L'œuvre existe déjà (ID : {existe.id})"
            )

        # ------------------------------
        # 2) Sauvegarde définitive du fichier
        # ------------------------------
        os.makedirs("storage/files", exist_ok=True)
        save_path = f"storage/files/{oeuvre_id}_{fichier.filename}"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # ------------------------------
        # 3) Extraction texte stable
        # ------------------------------
        text = processing.extract_text(save_path)

        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Impossible d'extraire du texte utilisable.")

        # ------------------------------
        # 4) Embedding + ajout FAISS
        # ------------------------------
        emb = processing.compute_text_embedding(text)

        if text_index is not None:
            
            
            embeddings_index.add_text(
                embedding=emb,
                meta={
                    "oeuvre_id": oeuvre_id,
                    "titre": titre,
                    "auteur": auteur,
                    "genre": genre,
                    "langue": langue
                }
            )



        # ------------------------------
        # 5) Analyse IA via BERT
        # ------------------------------
        ia_result = predict_bert(text, threshold=0.30)

        if "error" not in ia_result:
            ia_label = ia_result["prediction"]
            ia_score = ia_result["proba_IA"]
        else:
            ia_label = "analyse_impossible"
            ia_score = 0.0

        # ------------------------------
        # 6) Recherche FAISS (similarité)
        # ------------------------------
        candidates = []
        if text_index is not None:
            D, I = text_index.search(emb.reshape(1, -1).astype("float32"), 10)
            for score, idx in zip(D[0], I[0]):
                if idx != -1:
                    candidates.append({
                        "meta": text_metas[idx],
                        "score": float(score)
                    })

        # ------------------------------
        # 7) Génération du rapport PDF
        # ------------------------------
        metadata = {
            "uid": oeuvre_id,
            "titre": titre,
            "auteur": auteur,
            "type": "texte",
            "ia_detecte": ia_label,
            "score_ia": ia_score,
            "empreinte_hash": empreinte_hash
        }
        segments_matches = []  
        conclusion = (
            "Rapport généré à partir de l’empreinte numérique et de la similarité globale. "
            "Analyse détaillée phrase par phrase non exécutée dans ce mode."
        )
        report_path = f"storage/reports/{oeuvre_id}.pdf"
        report_utils.generate_legal_pdf_report(
        report_path,
        metadata,
        candidates,
        segments_matches,
        conclusion
    )

        # ------------------------------
        # 8) Enregistrement DB
        # ------------------------------
        oeuvre = Oeuvre(
            id=oeuvre_id,
            type_oeuvre="texte",
            titre=titre,
            auteur=auteur,
            genre=genre,
            langue=langue,
            fichier_nom=f"{oeuvre_id}_{fichier.filename}",
            ipfs_cid=cid,
            fichier_url=fichier_url,
            date_creation=datetime.datetime.now(),
            date_enregistrement=datetime.datetime.now(),
            ia_detecte=ia_label,
            score_ia=ia_score,
            empreinte_hash=empreinte_hash
        )
        db.add(oeuvre)
        db.commit()
        db.refresh(oeuvre)

        # ------------------------------
        # 9) Génération certificat PDF
        # ------------------------------
        cert_path = f"storage/certificats/{oeuvre_id}.pdf"
        generate_certificat(cert_path, {
            "id": oeuvre_id,
            "titre": titre,
            "auteur": auteur,
            "genre": genre,
            "langue": langue,
            "type": type_oeuvre,
            "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "hash": empreinte_hash
        })

        oeuvre.certificat_url = cert_path
        db.commit()

        # ------------------------------
        # 10) Sauvegarde empreinte détaillée
        # ------------------------------
        os.makedirs("storage/empreintes", exist_ok=True)
        detail_path = f"storage/empreintes/{oeuvre_id}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            import json
            json.dump(sanitize_for_json(empreinte_details), f, ensure_ascii=False, indent=2)

        # ------------------------------
        # 11) Réponse API
        # ------------------------------
        return {
            "oeuvre": oeuvre_id,
            "empreinte_hash": empreinte_hash,
            "ia_prediction": ia_label,
            "score_ia": ia_score,
            "candidates": candidates,
            "rapport_pdf": report_path,
            "certificat_url": f"http://127.0.0.1:8000/certificat/{oeuvre_id}"

        }


    # ------------------------------------------------------------------
    # 🔥 Pipeline MUSIQUE
    # ------------------------------------------------------------------
    elif type_oeuvre == "musique":
        audio_np, sr = decode_audio_bytes(file_bytes, fichier.filename)

        # ---- Analyse IA AUDIO ----
        mfcc_vec = extract_mfcc(audio_np, sr)
        audio_ia_result = predict_audio_ia(mfcc_vec)

        ia_label = audio_ia_result.get("prediction", "analyse_impossible")
        ia_score = audio_ia_result.get("proba_IA", None)
        if not is_valid_wav(file_bytes):
            raise HTTPException(
                status_code=400,
                detail="Le fichier n'est pas un WAV valide (RIFF/WAVE header manquant)."
            )
        # 1) Sauvegarde locale
        os.makedirs("storage/files", exist_ok=True)
        save_path = f"storage/files/{oeuvre_id}_{fichier.filename}"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # 2) Décodage audio (CORRECTION ICI 🔥)
        audio_np, sr = decode_audio_bytes(file_bytes, fichier.filename)

        if audio_np is None:
            raise HTTPException(status_code=400, detail="Impossible de lire le fichier audio (format invalide ou corrompu).")

        # 3) Fingerprint + Embedding
        # Sauvegarde audio dans un fichier temporaire
        temp_audio_path = f"temp_{uuid.uuid4().hex}.wav"
        sf.write(temp_audio_path, audio_np, sr)

        # Audio fingerprint
        fp = processing.compute_audio_fingerprint(temp_audio_path)

        # Audio embedding
        emb = processing.compute_audio_embedding(temp_audio_path)

        # Nettoyage
        os.remove(temp_audio_path)

        # 4) Ajout FAISS audio
        if audio_index is not None:
            embeddings_index.add_audio(
                embedding=emb,
                meta={
                    "oeuvre_id": oeuvre_id,
                    "titre": titre,
                    "auteur": auteur,
                    "genre": genre,
                    "langue": langue
                }
            )
        # 4) Recherche voisins similaires
        candidates = []
        if audio_index is not None:
            D, I = audio_index.search(emb.reshape(1, -1).astype("float32"), 10)
            for score, idx in zip(D[0], I[0]):
                if idx != -1:
                    candidates.append({
                        "meta": audio_metas[idx],
                        "score": float(score)
                    })

        # 5) Rapport légal audio
        segments_matches = []
        conclusion = "Analyse basée sur fingerprint audio & similarité d’embeddings. Aucune comparaison textuelle possible."

        metadata = {
            "uid": oeuvre_id,
            "titre": titre,
            "auteur": auteur,
            "type": "audio",
            "empreinte_hash": empreinte_hash
        }

        report_path = f"storage/reports/{oeuvre_id}.pdf"
        report_utils.generate_legal_pdf_report(
            report_path,
            metadata,
            candidates,
            segments_matches,
            conclusion
        )

        # 6) Enregistrement dans la base
        oeuvre = Oeuvre(
            id=oeuvre_id,
            type_oeuvre="musique",
            titre=titre,
            auteur=auteur,
            genre=genre,
            langue=langue,
            fichier_nom=f"{oeuvre_id}_{fichier.filename}",
            ipfs_cid=cid,
            fichier_url=fichier_url,
            date_creation=datetime.datetime.now(),
            date_enregistrement=datetime.datetime.now(),
            ia_detecte=ia_label,
            score_ia=ia_score,
            empreinte_hash=empreinte_hash
        )
        db.add(oeuvre)
        db.commit()
        db.refresh(oeuvre)

        # 7) Certificat
        cert_path = f"storage/certificats/{oeuvre_id}.pdf"
        generate_certificat(cert_path, {
            "id": oeuvre_id,
            "titre": titre,
            "auteur": auteur,
            "genre": genre,
            "langue": langue,
            "type": "musique",
            "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "hash": empreinte_hash
        })

        oeuvre.certificat_url = cert_path
        db.commit()

        # 8) Détails empreinte
        os.makedirs("storage/empreintes", exist_ok=True)
        detail_path = f"storage/empreintes/{oeuvre_id}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            import json
            json.dump(sanitize_for_json(empreinte_details), f, ensure_ascii=False, indent=2)

        return {
            "oeuvre": oeuvre_id,
            "fingerprint": fp,
            "empreinte_hash": empreinte_hash,
            "candidates": candidates,
            "ia_prediction": ia_label,
            "score_ia": ia_score,
            "rapport_pdf": report_path,
            "certificat_url": f"http://127.0.0.1:8000/certificat/{oeuvre_id}"


        }


    # ------------------------------------------------------------------
    # 🔥 Type non supporté
    # ------------------------------------------------------------------
    else:

        oeuvre = Oeuvre(
            id=oeuvre_id,
            type_oeuvre=type_oeuvre,
            titre=titre,
            auteur=auteur,
            genre=genre,
            langue=langue,
            fichier_nom=fichier.filename,
            ipfs_cid=cid,
            fichier_url=fichier_url,
            date_creation=datetime.datetime.now(),
            date_enregistrement=datetime.datetime.now(),
            ia_detecte="non_applicable",
            score_ia=None,
            empreinte_hash=empreinte_hash
        )

        db.add(oeuvre)
        db.commit()
        db.refresh(oeuvre)

        return {
            "oeuvre": oeuvre_id,
            "type": type_oeuvre,
            "empreinte_hash": empreinte_hash,
            "fichier_url": fichier_url
        }



@app.get("/report/{uid}")
def get_report(uid: str):
    path = f"storage/reports/{uid}.pdf"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{uid}.pdf")



# Load both indexes at startup
(TEXT_IDX, TEXT_METAS), (AUDIO_IDX, AUDIO_METAS) = embeddings_index.load_index()

def ensure_storage_paths():
    os.makedirs("storage/tmp", exist_ok=True)
    os.makedirs("storage/files", exist_ok=True)
    os.makedirs("storage/reports", exist_ok=True)

ensure_storage_paths()

@app.post("/plagiat")
async def analyse_plagiat(
    fichier: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Analyse de plagiat :
    - Extraction texte
    - Embedding
    - Recherche FAISS
    - Alignement de phrases
    - Rapport PDF
    """

    # --- Vérifications format ---
    ext = os.path.splitext(fichier.filename)[1].lower()
    if ext not in [".txt", ".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers texte (.txt, .pdf, .docx) sont supportés."
        )

    # --- Sauvegarde fichier temporaire ---
    tmp_path = f"storage/tmp/{datetime.datetime.now().timestamp()}-{fichier.filename}"
    os.makedirs("storage/tmp", exist_ok=True)

    file_bytes = await fichier.read()
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    # --- ID analyse ---
    analyse_id = f"PLAG-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # --- Charger index texte ---
    (TEXT_IDX, TEXT_METAS), _ = embeddings_index.load_index()
    if TEXT_IDX is None:
        raise HTTPException(
            status_code=500,
            detail="Index FAISS vide ou introuvable. Exécute ingest_index.py."
        )

    # --- Extraction texte ---
    text = processing.extract_text(tmp_path)
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Texte trop court.")

    # --- SimHash ---
    simhash_val = processing.compute_simhash(text)

    # --- Embedding ---
    emb = processing.compute_text_embedding(text)

    # --- Recherche FAISS ---
    D, I = processing.search_faiss_index(TEXT_IDX, emb, k=5)

    candidates = []
    segments_matches = []

    # Parcours résultats
    for score, pos in zip(D, I):
        if pos == -1:
            continue

        meta = TEXT_METAS[pos]
        oeuvre_id_ref = meta.get("oeuvre_id")

        # Vérification DB
        ref = db.query(Oeuvre).filter(Oeuvre.id == oeuvre_id_ref).first()
        if not ref or ref.type_oeuvre != "texte":
            continue

        # Trouver chemin correct du fichier
        if ref.fichier_nom:
            ref_path = f"storage/files/{ref.fichier_nom}"
        else:
            continue

        if not os.path.exists(ref_path):
            continue

        # Charger texte original
        candidate_text = processing.extract_text(ref_path)

        # Alignement avancé
        matches = processing.sentence_align(
            text,
            candidate_text,
            threshold=0.70  # seuil ajusté
        )

        # Ajouter dans résultats
        candidates.append({
            "oeuvre_id": ref.id,
            "titre": ref.titre,
            "score": float(score),
            "match_count": len(matches),
        })

        # Ajouter extraits
        for m in matches[:8]:  # max 8 extraits
            segments_matches.append(
                f"→ {ref.id} (ratio {m['ratio']:.2f})\n"
                f"Votre texte : {m['a_sentence'][:150]}...\n"
                f"Référence : {m['b_sentence'][:150]}...\n"
            )

    # --- Génération PDF ---
    metadata = {
        "uid": analyse_id,
        "fichier": fichier.filename,
        "simhash": simhash_val,
        "date": str(datetime.datetime.now())
    }

    report_path = f"storage/reports/{analyse_id}.pdf"
    os.makedirs("storage/reports", exist_ok=True)

    # Génération conclusion juridique
    similarity_score = sum(c["score"] for c in candidates) / max(len(candidates), 1)

    if similarity_score > 0.25:
        conclusion = (
            "L'analyse révèle une similarité élevée dépassant les standards WIPO.\n"
            "Ce document présente un risque majeur de plagiat.\n"
            "Une expertise humaine BBDA est recommandée."
        )
    else:
        conclusion = (
            "L'analyse ne montre pas de similarité significative.\n"
            "Aucun risque majeur de plagiat n'a été détecté."
        )

    # Appel du rapport juridique
    report_utils.generate_legal_pdf_report(
        report_path,
        metadata,
        candidates,
        segments_matches,
        conclusion
    )

    # --- Retour API ---
    return {
        "analyse_id": analyse_id,  # ← obligatoire
        "simhash": simhash_val,
        "candidats": candidates,
        "extraits": segments_matches[:10],
        "rapport_pdf": f"/plagiat/report/{analyse_id}"
    }

@app.get("/certificat/{uid}")
def get_certificat(uid: str):
    path = f"storage/certificats/{uid}.pdf"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Certificat introuvable")
    return FileResponse(path, media_type="application/pdf")

@app.get("/plagiat/report/{analyse_id}")
def download_plagiat_report(analyse_id: str):
    path = f"storage/reports/{analyse_id}.pdf"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Rapport introuvable")

    return FileResponse(path, media_type="application/pdf", filename=f"{analyse_id}.pdf")


