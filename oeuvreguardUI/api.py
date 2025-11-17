from fastapi import FastAPI, Form, UploadFile, File
import psycopg2
import ipfshttpclient
import tempfile
import mimetypes
import datetime

app = FastAPI()

def get_conn():
    return psycopg2.connect(
        dbname="oeuvre_guard_db",
        user="postgres",
        password="qwerty123456",
        host="localhost",  
        port="5434"        
    )

# Upload vers IPFS
def upload_to_ipfs(file_bytes: bytes) -> str:
    client = ipfshttpclient.connect()  # IPFS daemon doit tourner
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        res = client.add(tmp.name)
    return res["Hash"]

# Détection du type en fonction du format
def detect_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if mime.startswith("audio/"):
            return "musique"
        elif mime in ("application/pdf",
                      "application/msword",
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                      "text/plain"):
            return "texte"
    return "autre"

# Génération d’un ID unique
def generate_id(file_type: str) -> str:
    year = datetime.datetime.now().year
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    if file_type == "musique":
        prefix = "MUS"
    elif file_type == "texte":
        prefix = "BOOK"
    else:
        prefix = "GEN"
    return f"{prefix}-{year}-{timestamp}"

@app.post("/upload")
async def upload(
    titre: str = Form(...),
    auteur: str = Form(...),
    genre: str = Form(...),
    langue: str = Form(...),
    fichier: UploadFile = File(...)
):
    # Lire le fichier
    file_bytes = await fichier.read()

    # Upload vers IPFS
    cid = upload_to_ipfs(file_bytes)
    fichier_url = f"ipfs://{cid}"

    # Détecter type automatiquement
    type_oeuvre = detect_type(fichier.filename)

    # Générer ID
    oeuvre_id = generate_id(type_oeuvre)

    # Enregistrer en base
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO oeuvre (id, type, titre, auteur, genre, langue, fichier_nom, ipfs_cid, fichier_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (oeuvre_id, type_oeuvre, titre, auteur, genre, langue, fichier.filename, cid, fichier_url)
    )
    conn.commit()
    cur.close()
    conn.close()

    # Retour JSON
    return {
        "oeuvre": {
            "id": oeuvre_id,
            "type": type_oeuvre,
            "titre": titre,
            "auteur": auteur,
            "genre": genre,
            "langue": langue,
            "fichier_url": fichier_url
        }
    }
