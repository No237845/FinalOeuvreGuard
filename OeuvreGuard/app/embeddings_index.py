# app/embeddings_index.py
import faiss, os, json
import numpy as np

BASE_DIR = "storage"
TEXT_INDEX_PATH = os.path.join(BASE_DIR, "faiss_text.index")
AUDIO_INDEX_PATH = os.path.join(BASE_DIR, "faiss_audio.index")
TEXT_METAS_PATH = os.path.join(BASE_DIR, "faiss_text_metas.json")
AUDIO_METAS_PATH = os.path.join(BASE_DIR, "faiss_audio_metas.json")


TEXT_INDEX = None
AUDIO_INDEX = None
TEXT_METAS = []
AUDIO_METAS = []

def _load_index(path):
    return faiss.read_index(path) if os.path.exists(path) else None

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_index():
    global TEXT_INDEX, AUDIO_INDEX, TEXT_METAS, AUDIO_METAS
    TEXT_INDEX = _load_index(TEXT_INDEX_PATH)
    AUDIO_INDEX = _load_index(AUDIO_INDEX_PATH)
    TEXT_METAS = _load_json(TEXT_METAS_PATH)
    AUDIO_METAS = _load_json(AUDIO_METAS_PATH)
    return (TEXT_INDEX, TEXT_METAS), (AUDIO_INDEX, AUDIO_METAS)

def save_all():
    if TEXT_INDEX is not None: _save_index(TEXT_INDEX, TEXT_INDEX_PATH)
    if AUDIO_INDEX is not None: _save_index(AUDIO_INDEX, AUDIO_INDEX_PATH)
    _save_json(TEXT_METAS, TEXT_METAS_PATH)
    _save_json(AUDIO_METAS, AUDIO_METAS_PATH)

def add_text(embedding: np.ndarray, meta: dict):
    global TEXT_INDEX, TEXT_METAS
    emb = embedding.reshape(1, -1).astype("float32")
    if TEXT_INDEX is None:
        dim = emb.shape[1]
        TEXT_INDEX = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    TEXT_INDEX.add(emb)
    TEXT_METAS.append(meta)
    save_all()

def add_audio(embedding: np.ndarray, meta: dict):
    global AUDIO_INDEX, AUDIO_METAS
    emb = embedding.reshape(1, -1).astype("float32")
    if AUDIO_INDEX is None:
        dim = emb.shape[1]
        AUDIO_INDEX = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    AUDIO_INDEX.add(emb)
    AUDIO_METAS.append(meta)
    save_all()

def search_text(query_emb: np.ndarray, k=10):
    if TEXT_INDEX is None:
        return [], []
    q = query_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = TEXT_INDEX.search(q, k)
    return D[0], I[0]

def search_audio(query_emb: np.ndarray, k=10):
    if AUDIO_INDEX is None:
        return [], []
    q = query_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = AUDIO_INDEX.search(q, k)
    return D[0], I[0]

def save_index(index, metas, is_audio=False):
    """Sauvegarde FAISS + métadonnées."""
    os.makedirs("storage/faiss", exist_ok=True)

    if is_audio:
        faiss.write_index(index, INDEX_AUDIO_PATH)
        with open(METAS_AUDIO_PATH, "wb") as f:
            pickle.dump(metas, f)
    else:
        faiss.write_index(index, INDEX_TEXT_PATH)
        with open(METAS_TEXT_PATH, "wb") as f:
            pickle.dump(metas, f)