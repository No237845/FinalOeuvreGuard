# app/utils.py
import uuid
import os

def make_uid(prefix="BF"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def save_upload_file(upload_file, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(upload_file.file.read())
    return dest_path
