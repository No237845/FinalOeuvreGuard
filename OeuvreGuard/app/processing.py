# app/processing.py
import os
import subprocess
import numpy as np
from simhash import Simhash
from sentence_transformers import SentenceTransformer
import faiss
import librosa
from difflib import SequenceMatcher

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMB_MODEL = SentenceTransformer(MODEL_NAME)

# ----- TEXT -----
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return open(filepath, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".docx":
        import docx
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    if ext == ".pdf":
        from pdfminer.high_level import extract_text
        return extract_text(filepath)
    return ""

def compute_simhash(text: str) -> int:
    return Simhash(text).value

def compute_text_embedding(text: str) -> np.ndarray:
    emb = EMB_MODEL.encode(text, convert_to_numpy=True)
    return emb.astype("float32")

# ----- AUDIO -----
def compute_audio_fingerprint(filepath: str):
    # fpcalc must be installed (chromaprint)
    try:
        p = subprocess.run(["fpcalc", "-json", filepath], capture_output=True, text=True, check=True)
        import json
        j = json.loads(p.stdout)
        return j.get("fingerprint")  # string or None
    except Exception as e:
        print("fpcalc error:", e)
        return None

def compute_audio_embedding(filepath: str, sr=22050, n_mels=128):
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mels)
    feat = np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])
    return feat.astype("float32")

# ----- FAISS helpers -----
def build_faiss_index(embeddings: np.ndarray, use_inner_prod=True):
    dim = embeddings.shape[1]
    if use_inner_prod:
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_vec: np.ndarray, k=10, inner_prod=True):
    q = query_vec.copy().reshape(1, -1)
    if inner_prod:
        faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

# ----- TEXT ALIGNMENT -----
def sentence_align(a_text: str, b_text: str, threshold=0.8):
    # sentence-level alignment using SequenceMatcher
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk import sent_tokenize
    sa = sent_tokenize(a_text)
    sb = sent_tokenize(b_text)
    matches = []
    for i, s_a in enumerate(sa):
        for j, s_b in enumerate(sb):
            ratio = SequenceMatcher(None, s_a, s_b).ratio()
            if ratio >= threshold:
                matches.append({
                    "a_index": i, "b_index": j,
                    "ratio": float(ratio),
                    "a_sentence": s_a,
                    "b_sentence": s_b
                })
    return matches

# ----- AUDIO ALIGNMENT (DTW) -----
def audio_dtw_segments(file_a, file_b, sr=22050, hop_length=512, top_matches=5):
    ya, _ = librosa.load(file_a, sr=sr, mono=True)
    yb, _ = librosa.load(file_b, sr=sr, mono=True)
    Ca = librosa.feature.chroma_cqt(y=ya, sr=sr)
    Cb = librosa.feature.chroma_cqt(y=yb, sr=sr)
    D, wp = librosa.sequence.dtw(X=Ca, Y=Cb, metric="cosine")
    # wp: list of (ia, ib) aligned frames in reversed order
    wp = np.array(wp)[::-1]  # make chronological
    # group contiguous matches into segments
    segments = []
    if wp.size == 0:
        return segments
    cur = [wp[0,0], wp[0,1]]
    for ia, ib in wp[1:]:
        if ia == cur[1]+1 or ib == cur[1]+1 or abs(ia - cur[0]) <= 3:
            cur[1] = ia
        else:
            segments.append(tuple(cur))
            cur = [ia, ib]
    segments.append(tuple(cur))
    # convert to seconds approx
    secs = []
    for a0, a1 in segments[:top_matches]:
        a_start = (a0 * hop_length) / sr
        a_end = (a1 * hop_length) / sr
        secs.append({"a_start": float(a_start), "a_end": float(a_end)})
    return secs
