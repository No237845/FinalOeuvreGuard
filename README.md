🛡️ OeuvreGuard
Système intelligent d’enregistrement, de détection IA, d’analyse de plagiat et de certification numérique pour œuvres textuelles & audio
📌 Description du projet

OeuvreGuard est une solution complète permettant :

✔ L’enregistrement sécurisé d’œuvres (texte & musique)
✔ La génération d’une empreinte numérique unique (SimHash + fingerprint audio)
✔ La détection IA

Textes → Modèle BERT finetuné

Audio → Modèle neuronal basé MFCC
✔ L’analyse de similarité & plagiat (FAISS + SentenceTransformer)
✔ La génération automatique de rapports légaux PDF
✔ La création d’un certificat d’enregistrement BBDA
✔ L’upload des fichiers sur IPFS pour traçabilité immuable

Le projet inclut :

Une API FastAPI

Une base PostgreSQL avec SQLAlchemy

Des modules IA (BERT, CNN audio)

Un pipeline stable texte/audio

Un générateur PDF légal

Un certificat individuel signé

📂 Architecture du projet
OeuvreGuard/
│── app/
│   ├── main.py                  # API principale FastAPI
│   ├── models.py                # Modèles SQLAlchemy
│   ├── db.py                    # Connexion DB
│   ├── processing.py            # Embedding + SimHash + audio pipeline
│   ├── fingerprints.py          # Empreinte numérique
│   ├── embeddings_index.py      # FAISS : index texte/audio
│   ├── report_utils.py          # Génération PDF légal
│   ├── certificat_utils.py      # Certificat PDF
│   ├── bert_model_directory/    # Modèle BERT finetuné
│   ├── audio_model_final.pth    # Modèle audio IA
│   └── storage/
│       ├── files/               # Œuvres originales
│       ├── certificats/         # Certificats PDF
│       ├── reports/             # Rapports légaux
│       └── empreintes/          # Détails fingerprints
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│
├── ingest_index.py              # Construction FAISS
├── requirements.txt
├── README.md
└── run.bat                      # Lancement Windows

⚙️ Pré-requis
✔ Installer Python 3.10+

✔ Installer PostgreSQL
✔ Installer FFmpeg (obligatoire pour audio)
✔ Installer IPFS local
✔ Installer les dépendances

📦 Installation
1️⃣ Cloner le projet
git clone https://github.com/ton-compte/OeuvreGuard.git
cd OeuvreGuard

2️⃣ Installer les modules Python
pip install -r requirements.txt

3️⃣ Démarrer IPFS
ipfs daemon

4️⃣ Modifier les paramètres DB dans db.py
dbname="oeuvre_guard_db"
user="postgres"
password="VOTRE_MDP"
port="5434"

5️⃣ Générer les index FAISS
python ingest_index.py

6️⃣ Lancer l’API
uvicorn app.main:app --reload --port 8000

🚀 Fonctionnalités
🔹 1. Upload d’œuvre /upload

Identification du type (texte/audio)

Upload IPFS

Extraction & embedding

Détection IA (texte/audio)

Empreinte numérique unique

Génération certificat PDF

Génération rapport légal

Stockage DB + IPFS

🔹 2. Analyse de plagiat /plagiat

Extraction texte

Embedding + FAISS

Alignement avancé NLTK

Rapport PDF détaillé

🔹 3. Téléchargement de rapports
/report/{uid}
/certificat/{uid}
/plagiat/report/{analyse_id}

🧠 Modèles IA utilisés
🔸 Texte (BERT)

BertForSequenceClassification

Labels : IA / Humain / Incertain

Seuil d’incertitude réglable

🔸 Audio

Réseau neuronal simple :

20 MFCC → FC(20→64) → FC(64→32) → FC(32→2)

🧪 Tests
Tester upload texte
curl -X POST -F "fichier=@test.pdf" -F "titre=Test" -F "auteur=Moi" -F "genre=Roman" -F "langue=Fr" http://127.0.0.1:8000/upload

Tester plagiat
curl -X POST -F "fichier=@doc.pdf" http://127.0.0.1:8000/plagiat

🖨️ Génération automatique de certificats

Chaque upload génère :

✔ Identifiant unique
✔ Hash numérique
✔ Date & heure
✔ Signature BBDA
✔ PDF exportable

🛡️ Sécurité & Authenticité

Traces IPFS immuables

SimHash / fingerprint audio

Rapports PDF signables
