import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

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

BERT_MODEL_DIR = "./bert_model_directory"

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
