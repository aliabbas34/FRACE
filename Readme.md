# FRACE  
**F**acial **R**ecognition **A**nd **C**lustering **E**ngine

FRACE is a Python-based pipeline that processes raw images, corrects orientation, extracts faces, generates embeddings, stores metadata in SQLite, and performs hierarchical clustering to automatically group images by person.

The engine transforms an unstructured photo collection into organized, person-wise folders.

---

## 🚀 Pipeline Overview

FRACE processes images through the following stages:

1. **Image Compression**
2. **Orientation Correction**
3. **Face Detection & Embedding Extraction**
4. **Database Storage**
5. **Agglomerative Clustering**
6. **Cluster-based Folder Generation**

---

## 🧠 Architecture Flow

```
original-photos/
        ↓
compressed-photos/
        ↓
Orientation Correction (in-place)
        ↓
Face Extraction
        ↓
SQLite Database (faces.db)
        ↓
Agglomerative Clustering
        ↓
Clustered Output Folders (clustered_faces)
```

---

## 📁 Project Structure

```
FRACE/
│
├── original-photos/        # Input images (create this folder with same name and paste the input images)
├── compressed-photos/      # Auto-generated compressed images
├── clustered_faces/        # Generated cluster folders (output)
│
├── main.py
├── extract-faces.py
├── cluster-faces.py
├── correct-image-orientation.py
├── reduce-image-size.py
│
├── requirements.txt
├── README.md
│
└── faces.db (ignored from Git)
```

> Virtual environments and database files are excluded from version control.

---

## 🔍 Stage-by-Stage Breakdown

### 1️⃣ Image Compression

- Reads images from `original-photos/`
- Resizes and compresses images
- Saves them into `compressed-photos/`

Purpose:
- Reduce processing time
- Improve performance efficiency

---

### 2️⃣ Orientation Correction

- Uses `check_orientation` library
- Internally uses **swsl_resnext50_32x4d** model with pretrained weights
- Fixes orientation **in-place** inside `compressed-photos/`

This ensures consistent face detection performance.

---

### 3️⃣ Face Extraction

Uses:

- `face_recognition` library
- HOG based detection

For each image:

- Detects all faces
- Generates:
  - Bounding box coordinates
  - 128-dimensional face embeddings
- Stores:
  - Image path
  - Face coordinates
  - Embeddings

All stored in **SQLite database (`faces.db`)**

---

### 4️⃣ Database Layer

- SQLite used for persistent storage
- Embeddings stored as serialized vectors
- Enables:
  - Efficient retrieval
  - Re-clustering without reprocessing images

---

### 5️⃣ Clustering

After face extraction:

- Fetches all embeddings from SQLite
- Converts them into NumPy arrays
- Uses:

```
AgglomerativeClustering(metric="euclidean")
```

From:

- `sklearn.cluster`

Why Agglomerative?
- Does not require predefined cluster count
- Performs well for hierarchical face grouping

---

### 6️⃣ Cluster Output Generation

For each cluster:

- A folder is created
- Inside each cluster folder:
  - An **avatar image** (representative face)
  - A subfolder containing all images belonging to that person

Final result:
A structured directory of grouped faces.

---

## 🛠 Tech Stack

- Python3
- NumPy
- Pillow
- face_recognition
- scikit-learn(AgglomerativeClustering)
- SQLite
- check_orientation(swsl_resnext50_32x4d pretrained model)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<username>/FRACE.git
cd FRACE
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Images

Place all input images inside:

```
original-photos/
```

---

### 5️⃣ Run the Engine

```bash
python main.py
```

---

## 🗄 Database Notes

- `faces.db` is auto-generated
- Not committed to Git
- Safe to delete if you want to re-run full pipeline

---

## 📊 Performance Considerations

- Compression significantly reduces compute time
- Orientation correction improves detection accuracy
- Database layer avoids redundant embedding generation
- Clustering complexity grows with number of faces

---

## ⚠️ Limitations

- Performance depends on image quality
- Extremely large datasets may require optimization
- Agglomerative clustering may become expensive for very large embedding sets

---

## 👤 Author

Ali Abbas

Email: aliabbas7317@gmail.com