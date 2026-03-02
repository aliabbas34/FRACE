import os
import sqlite3
import json
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def sort_faces_to_folders(clusters, face_data, min_samples):
    
    base_dir = "clustered_faces"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for label, face_ids in tqdm(clusters.items(), unit= "cluster"):
        if len(face_ids) < min_samples:
            print(f"Skipping cluster {label} with only {len(face_ids)} faces (less than min_samples={min_samples}).")
            continue
        # Create the specific folder for this label (folder_0, folder_1, etc.)
        folder_name = os.path.join(base_dir, f"folder_{label}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        actual_photos_path = os.path.join(folder_name, "photos")
        if not os.path.exists(actual_photos_path):
            os.makedirs(actual_photos_path)
        
        print(f"Processing {folder_name}...")
        first_crop = True
        for face_id in face_ids:
            data = face_data[face_id]
            image_path = data['path']
            
            try:
                with Image.open(image_path) as img:
                    if(first_crop):
                        x,y,w,h = data['box']
                        face_crop = img.crop((x, y, x + w, y + h))
                        save_path = os.path.join(folder_name, f"face_{face_id}.jpg")
                        face_crop.save(save_path)
                        first_crop = False
                    img.save(os.path.join(actual_photos_path, os.path.basename(image_path)))
            except Exception as e:
                print(f"Error processing face {face_id} from {image_path}: {e}")


def cluster_faces(eps, min_samples, db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute("SELECT id, image_path, x,y,w,h, embedding FROM faces")
    rows = c.fetchall()
    conn.commit()
    conn.close()

    if not rows:
        print("No data found")
        return
    
    face_data={}
    ids = []
    embeddings = []

    for row in rows:
        face_id, image_path, x,y,w,h, embedding_json = row
        face_data[face_id] = {'path': image_path, 'box': (x,y,w,h)}
        ids.append(face_id)
        embeddings.append(json.loads(embedding_json))

    embeddings = np.array(embeddings)
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=eps, 
        metric='euclidean', 
        linkage='average'
    )
    labels = clusterer.fit_predict(embeddings)

    face_groups = {}

    for i, label in enumerate(labels):
        if label not in face_groups:
            face_groups[label] = []
        face_groups[label].append(ids[i])

    print(f"Found {len(face_groups)} unique people.")
    sort_faces_to_folders(face_groups, face_data, min_samples)

    print("Clustering complete.")
