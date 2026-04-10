
# 🎧 Deezer Audio-Only Embedding Centroid Pipeline

A Python pipeline that converts **Deezer playlist tracks into audio embeddings** using **MusicFM**, builds **24-hour prototypes and 360 micro-centroids**, and classifies new playlist tracks using **FAISS similarity search**.

The system uses **Deezer 30-second preview clips only** and **never stores audio files**, only embeddings.

---





# 📌 Overview

The pipeline performs the following steps:

1. **Fetch reference playlists**
2. **Download preview clips**
3. **Generate embeddings using MusicFM**
4. **Store vectors in cache**
5. **Build 24 hour prototypes**
6. **Generate 15 micro-centroids per hour**
7. **Classify new tracks**
8. **Export results**

Architecture:

```
STEP 1  MusicFM Embedding Engine
STEP 2  Deezer ingestion + SQLite cache
STEP 3  Prototype + centroid generation
STEP 4  FAISS hierarchical classifier
STEP 5  Batched orchestration pipeline
```

---

# 📦 Installation

Install dependencies:

```bash
pip install librosa==0.11.0 torch==2.11.0 faiss-cpu==1.13.2  huggingface_hub==1.10.0  scikit-learn==1.8.0  transformers==5.5.1  spotipy==2.26.0
```


```bash
pip install torch librosa numpy pandas scikit-learn faiss-cpu huggingface_hub requests
```


Clone MusicFM repository automatically inside Step 1.

---

# 📂 Project Structure

```
project/
│
├── vectors/                    # cached embeddings
├── outputs/
│   └── assignment_results.csv
│
├── music_cache.db              # SQLite vector cache
│
├── step1_embedding_engine.py
├── step2_ingestion.py
├── step3_reference_builder.py
├── step4_classifier.py
├── step5_pipeline.py
│
└── README.md
```

---

# ⚙️ Pipeline Configuration

Three parameters control runtime behaviour.

| Parameter    | Description                |
| ------------ | -------------------------- |
| `LIMIT`      | Max tracks per playlist    |
| `BATCH_SIZE` | Tracks processed per batch |
| `WORKERS`    | Parallel embedding threads |

STEP 1
Embedding Engine (MusicFM)

STEP 2
Deezer ingestion

STEP 3
Embedding + NumPy cache

STEP 4
Prototype + centroid build

STEP 5
Classification pipeline
    ├─ batch embedding
    ├─ FAISS classification
    ├─ CSV export
    ├─ Spotify track matching
    └─ Spotify playlist creation


Deezer Ingestion
      ↓
Embedding + Cache
      ↓
Reference Model Build
      ↓
FAISS Classification
      ↓
CSV Export
      ↓
Spotify Track Matching
      ↓
Spotify Playlist Creation


Example:

```python
LIMIT = 20
BATCH_SIZE = 10
WORKERS = 4
```

---

# 📥 Input Format

Reference playlists define the **24-hour structure**.

Example:

```python
real_reference_data = { 
    'deezer_id': [
        '2252220162','2296059482','11081408402',
        '146819501','11663792464','13098300343',
        '10746894082','1902101402','1947984342',
        '1306978785','1746835762','13668653381',
        '8970791602','10783752422','10872142682',
        '6006188884','8666347202','3035482146',
        '7214404404','1393810385','3881674142',
        '11556358124','12993355223','1295485847'
    ],

    'hour_id': list(range(1,25))
}
```

Each playlist corresponds to **one hour of the day**.

---

# ▶️ Running the Pipeline

Example execution:

```python
run_full_pipeline(
    real_reference_data,
    input_playlists=["1363560485"],
    LIMIT=10,
    BATCH_SIZE=5,
    WORKERS=4
)
```

---

# 🔄 Pipeline Flow

### 1️⃣ Build Reference Dataset

Reference playlists are expanded into track-level references.

```
playlist → tracks → embeddings
```

---

### 2️⃣ Generate Embeddings

Each preview clip is processed with MusicFM:

```
30s preview → latent → pooled embedding
```

Output vector:

```
1024-dimension normalized embedding
```

Vectors are stored as:

```
vectors/<deezer_track_id>.npy
```

---

### 3️⃣ Build Hour Prototypes

For each hour:

```
prototype = mean(embeddings)
```

Output:

```
24 prototypes
```

---

### 4️⃣ Build Micro-Centroids

Tracks are clustered with MiniBatchKMeans.

```
15 clusters per hour
```

Output:

```
360 centroids
```

Centroid IDs:

```
hour 01 → 001-015
hour 02 → 016-030
...
hour 24 → 346-360
```

---

### 5️⃣ Classification

Tracks are classified in two stages:

```
1️⃣ Compare to 24 prototypes
2️⃣ Compare to 15 centroids in selected hour
```

Similarity metric:

```
cosine similarity
```

Implemented using **FAISS vector search**.

---

# 📊 Output Format

Results are stored in:

```
outputs/assignment_results.csv
```

Example:

| deezer_id  | assigned_hour | assigned_centroid | best_score | margin |
| ---------- | ------------- | ----------------- | ---------- | ------ |
| 3763842212 | 11            | 151               | 0.832      | 0.005  |
| 3589389951 | 12            | 167               | 0.846      | 0.085  |

Columns:

| Column              | Description               |
| ------------------- | ------------------------- |
| `assigned_hour`     | predicted hour            |
| `assigned_centroid` | cluster ID                |
| `best_score`        | similarity score          |
| `second_best_score` | runner-up score           |
| `margin`            | classification confidence |

---

# 📁 Output Files

outputs/
    assignment_results.csv
    centroids_360.csv
    hour_prototypes_24.csv
    reference_embeddings_summary.csv
    missing_preview.csv
    unmatched_spotify.csv

## CSV File Infoz


---

### **reference_embeddings_summary.csv**

Summarizes how many reference tracks were successfully embedded for each hour bucket (1–24).
Useful for verifying that the reference dataset has sufficient coverage per hour.

---

### **hour_prototypes_24.csv**

Contains the **24 prototype vectors**, one representing the average embedding of reference tracks for each hour.
These prototypes act as the **first-stage classifier** to determine the most likely hour.

---

### **centroids_360.csv**

Stores the **360 micro-centroids (15 per hour)** generated using K-Means clustering on the reference embeddings.
These centroids refine classification by identifying the closest cluster within the predicted hour.

---

### **assignment_results.csv**

Contains the final classification results for each processed track.
Includes the predicted hour, assigned centroid, similarity scores, and confidence margin.

---

### **missing_preview.csv**

Lists tracks that were skipped because Deezer did not provide a 30-second preview clip.
Helps identify gaps in the dataset where audio embeddings could not be generated.

---

### **run_summary.csv**

Provides a high-level summary of the pipeline execution.
Includes counts of reference tracks, prototypes created, centroids generated, classified tracks, and missing previews.




# 🚀 Performance

Typical speeds:

| Dataset    | Runtime    |
| ---------- | ---------- |
| 100 tracks | ~1 minute  |
| 1k tracks  | ~6 minutes |
| 25k tracks | ~2 hours   |

Parallel embedding greatly reduces runtime.

---

# ⚠️ Deezer Preview Limitation

The pipeline only processes tracks that have a **30-second preview clip**.

Tracks without preview are skipped.

You can check preview availability:

```python
track["preview"]
```

---

# 🔧 Debugging

Useful checks:

### Check vectors

```python
len(os.listdir("vectors"))
```

### Check prototype dataframe

```python
print(df_prototypes.head())
```

### Check embedding failures

```
Skipped: track_id {'status':'no_preview'}
```

---

# 🧠 Scaling Recommendations

For production reference models:

```
30–100 tracks per hour
```

Example:

```
24 hours × 40 tracks ≈ 960 reference tracks
```

---

# 📈 Future Improvements

Possible upgrades:

• async download pipeline
• distributed embedding workers
• vector database (FAISS IVF / HNSW)
• centroid visualization
• automated playlist ingestion

---

# 📜 License

This project uses the **MusicFM model** and Deezer preview API.

Ensure compliance with respective licenses.

---

# 🙌 Acknowledgements

MusicFM model by:

**Minz Won**

Repository:

```
https://github.com/minzwon/musicfm
```

---

If you'd like, I can also generate a **much more advanced README** including:

* architecture diagrams
* pipeline flow charts
* vector space visualizations
* system scaling design (100k+ tracks)

which makes the project look **very professional / publication-ready**.
