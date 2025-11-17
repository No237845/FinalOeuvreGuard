import os
import torch
""" from fingerprints import compute_emp_hash_for_upload

# test texte
h, d = compute_emp_hash_for_upload("Ceci est un texte de test.".encode("utf-8"), "test.txt", "texte")
print("hash texte:", h)
print(d["components"]["stylometry"]) """

# test audio (si tu as un wav)
#with open("tests/sample.wav", "rb") as f:
#    b = f.read()
#h2, d2 = compute_emp_hash_for_upload(b, "sample.wav", "musique")
#print("hash audio:", h2)

""" from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_DIR = os.path.join(BASE_DIR, "bert_model_directory")

print("[BERT] Chargement depuis :", BERT_MODEL_DIR) """
""" try:
    tok = BertTokenizer.from_pretrained(DIR)
    model = BertForSequenceClassification.from_pretrained(DIR)
    print("OK : Chargement réussi")
except Exception as e:
    print("ERREUR :", e) """
# try:
#     tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
#     bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
#     bert_model.eval()
#     print("[BERT] Modèle chargé avec succès 👍")
# except Exception as e:
#     print(f"[BERT] ERREUR : {e}")
#     tokenizer = None
#     bert_model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier /app
MODEL_PATH = os.path.join(BASE_DIR, "audio_model_final.pth")

print("Chargement modèle depuis :", MODEL_PATH)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
print(state_dict.keys())
print(state_dict["fc1.weight"].shape)
print(state_dict["fc2.weight"].shape)
print(state_dict["fc3.weight"].shape)
