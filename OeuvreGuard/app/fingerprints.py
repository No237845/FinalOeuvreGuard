# app/fingerprints.py
import hashlib
import base64
import json
import os
import tempfile
from typing import Union, Tuple, Dict, Any

import numpy as np

# Si tu as déjà processing.py dans ton projet, on réutilise certaines fonctions (embeddings, simhash, audio features).
# Sinon tu peux remplacer les appels par tes propres implémentations.
try:
    from . import processing
except Exception:
    import processing

# Optional: use librosa if available (audio features)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# Optional: use chromaprint fpcalc if installed on system
import subprocess
def _call_fpcalc(path: str) -> Union[str, None]:
    """
    Appelle fpcalc (Chromaprint) si disponible. Retourne la fingerprint string ou None.
    """
    try:
        p = subprocess.run(["fpcalc", "-json", path], capture_output=True, text=True, check=True)
        import json
        j = json.loads(p.stdout)
        return j.get("fingerprint")
    except Exception:
        return None

# -------------------------
# Helpers de sérialisation
# -------------------------
def _hash_bytes(b: bytes) -> str:
    """Retourne SHA256 hexdigest d'un blob b."""
    return hashlib.sha256(b).hexdigest()

def _normalize_array(arr: np.ndarray) -> bytes:
    """Normalise et convertit un array numpy en bytes (deterministic)."""
    a = np.asarray(arr, dtype=np.float32)
    # normaliser pour réduire variance non déterministe:
    # on quantize en int16 sur plage min/max
    amin, amax = a.min(), a.max()
    if amax - amin < 1e-6:
        scaled = np.zeros_like(a, dtype=np.int16)
    else:
        norm = (a - amin) / (amax - amin)
        scaled = (norm * 32767).astype(np.int16)
    return scaled.tobytes()

# -------------------------
# Empreinte Texte
# -------------------------
def fingerprint_text(text: str, use_embedding: bool = True) -> Dict[str, Any]:
    """
    Calcule un ensemble d'empreintes pour un texte :
      - simhash (int)
      - embedding (float32 np.array) (optionnel)
      - stylometry features (dict)
    Retour: dict contenant les éléments bruts.
    """
    out = {}
    # SimHash (string int)
    try:
        sim = processing.compute_simhash(text)
        out["simhash"] = str(sim)
    except Exception:
        out["simhash"] = None

    # Embedding
    if use_embedding:
        try:
            emb = processing.compute_text_embedding(text)
            out["embedding"] = emb.astype("float32")
        except Exception:
            out["embedding"] = None
    else:
        out["embedding"] = None

    # Quelques features stylométriques simples
    try:
        tokens = text.split()
        nb_tokens = len(tokens)
        avg_len = float(np.mean([len(t) for t in tokens])) if nb_tokens > 0 else 0.0
        uniq = len(set(tokens))
        type_token_ratio = float(uniq) / nb_tokens if nb_tokens>0 else 0.0
        nb_sentences = text.count('.') + text.count('!') + text.count('?')
        out["stylometry"] = {
            "nb_tokens": int(nb_tokens),
            "avg_token_len": float(avg_len),
            "type_token_ratio": float(type_token_ratio),
            "nb_sentences": int(nb_sentences)
        }
    except Exception:
        out["stylometry"] = {}

    return out

def compute_text_256bit_hash(text: str, reduce_emb_dim: int = 64) -> str:
    """
    Produit une empreinte 256 bits (hex SHA256) à partir du texte :
    concatène : simhash + truncated embedding bytes + stylometry json -> SHA256
    """
    fp = fingerprint_text(text, use_embedding=True)
    pieces = []
    # simhash
    if fp.get("simhash") is not None:
        pieces.append(fp["simhash"].encode())
    # embedding (truncate to reduce size)
    emb = fp.get("embedding")
    if emb is not None:
        # reduce dimension deterministically by taking first N values
        emb_trunc = np.asarray(emb, dtype=np.float32)[:reduce_emb_dim]
        pieces.append(_normalize_array(emb_trunc))
    # stylometry
    styl = fp.get("stylometry", {})
    pieces.append(json.dumps(styl, sort_keys=True).encode())

    blob = b"||".join(pieces)
    return _hash_bytes(blob)

# -------------------------
# Empreinte Audio
# -------------------------
def fingerprint_audio_from_path(path: str, compute_additional: bool = True) -> Dict[str, Any]:
    """
    Calcule fingerprints audio. Retourne dict:
      - chromaprint_fp (str) si fpcalc présent
      - mel_mean_std (np.array)
      - chroma_mean_std (np.array)
    """
    out = {}
    # Chromaprint
    chroma_fp = _call_fpcalc(path)
    out["chromaprint"] = chroma_fp

    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
            mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel = librosa.power_to_db(mels)
            feat = np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)]).astype("float32")
            out["mel_feat"] = feat
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_feat = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype("float32")
            out["chroma_feat"] = chroma_feat
            # tempo / rhythm
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
            out["tempo"] = tempo
        except Exception:
            out.setdefault("mel_feat", None)
            out.setdefault("chroma_feat", None)
            out.setdefault("tempo", None)
    else:
        out.setdefault("mel_feat", None)
        out.setdefault("chroma_feat", None)
        out.setdefault("tempo", None)

    return out

def fingerprint_audio_from_bytes(b: bytes, compute_additional: bool = True) -> Dict[str, Any]:
    """
    Si on reçoit bytes, on crée un tmp file et appelle fingerprint_audio_from_path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(b)
        tmp_path = tmp.name
    try:
        res = fingerprint_audio_from_path(tmp_path, compute_additional=compute_additional)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return res

def compute_audio_256bit_hash_from_bytes(b: bytes) -> str:
    """
    Construit une empreinte 256 bits audio :
    combine chromaprint (si dispo) + mel/chroma features + tempo -> SHA256
    """
    fp = fingerprint_audio_from_bytes(b)
    pieces = []
    if fp.get("chromaprint"):
        pieces.append(fp["chromaprint"].encode())
    if fp.get("mel_feat") is not None:
        pieces.append(_normalize_array(fp["mel_feat"]))
    if fp.get("chroma_feat") is not None:
        pieces.append(_normalize_array(fp["chroma_feat"]))
    if fp.get("tempo") is not None:
        pieces.append(str(fp["tempo"]).encode())

    blob = b"||".join(pieces) if pieces else b""
    if not blob:
        # fallback: hash raw bytes
        return _hash_bytes(b)
    return _hash_bytes(blob)

# -------------------------
# Empreinte finale commune (texte ou audio)
# -------------------------
def compute_emp_hash_for_upload(file_bytes: bytes, filename: str, file_type: str):
    """
    Version CORRIGÉE : sauvegarde temporairement le fichier pour garantir
    une extraction de texte correcte pour PDF / DOCX / etc.
    """

    import tempfile
    import os
    import hashlib
    import json

    details = {"type": file_type, "filename": filename}

    # --------------------------------------
    # 1) Si c'est un texte → extraire VRAIMENT le texte
    # --------------------------------------
    if file_type == "texte":

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Extraction texte depuis fichier (correct)
            text = processing.extract_text(tmp_path)

        except Exception:
            text = ""

        finally:
            # Nettoyage
            try:
                os.remove(tmp_path)
            except:
                pass

        # fingerprint complet
        details["components"] = fingerprint_text(text)
        hex_hash = compute_text_256bit_hash(text)
        details["emp_hash"] = hex_hash

        return hex_hash, details

    # --------------------------------------
    # 2) Audio
    # --------------------------------------
    elif file_type == "musique":
        audio_fp = fingerprint_audio_from_bytes(file_bytes)
        details["components"] = audio_fp
        hex_hash = compute_audio_256bit_hash_from_bytes(file_bytes)
        details["emp_hash"] = hex_hash
        return hex_hash, details

    # --------------------------------------
    # 3) fallback
    # --------------------------------------
    else:
        h = hashlib.sha256(file_bytes).hexdigest()
        details["components"] = {}
        details["emp_hash"] = h
        return h, details

