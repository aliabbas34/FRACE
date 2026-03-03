import os
import sqlite3
from reduce_image_size import compress_images
from correct_image_orientation import fix_orientation
from extract_faces import extract_faces
from cluster_faces import cluster_faces

# CONFIG
ORIGINAL_PHOTOS = "original-photos"
COMPRESSED_PHOTOS = "compressed-photos"
DB_NAME = "faces.db"

os.makedirs(ORIGINAL_PHOTOS, exist_ok=True)
os.makedirs(COMPRESSED_PHOTOS, exist_ok=True)

# DATABASE SETUP
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT UNIQUE,
        x INTEGER,
        y INTEGER,
        w INTEGER,
        h INTEGER,
        embedding TEXT
    )
    """)

    conn.commit()
    conn.close()

# PIPELINE
if __name__ == "__main__":
    compress_images(ORIGINAL_PHOTOS, COMPRESSED_PHOTOS)
    fix_orientation(COMPRESSED_PHOTOS)
    init_db()
    extract_faces(DB_NAME, COMPRESSED_PHOTOS)
    cluster_faces(0.4, 4, DB_NAME)
    print("All done.")