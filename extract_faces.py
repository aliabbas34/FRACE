import os
import sqlite3
import json
import face_recognition
from tqdm import tqdm

def extract_faces(db_name, image_folder):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    all_files = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(root, file))

    print(f"Found {len(all_files)} images. Starting extraction...")

    batch_size = 50  # Commit every 50 images
    for i, path in tqdm(enumerate(all_files), total=len(all_files), unit="img"):
        try:
            c.execute("SELECT 1 FROM faces WHERE image_path = ?", (path,))
            if c.fetchone():
                continue

            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, locations)

            for (top, right, bottom, left), embedding in zip(locations, encodings):
                c.execute("""
                    INSERT OR IGNORE INTO faces (image_path, x, y, w, h, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (path, left, top, right - left, bottom - top, json.dumps(embedding.tolist())))
            
            if (i + 1) % batch_size == 0:
                conn.commit()

        except Exception as e:
            print(f"\nSkipping {path} due to error: {e}")
            continue

    conn.commit()
    conn.close()
    print("Face extraction complete.")