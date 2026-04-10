# ==========================================
# STEP 3: Reference Model Builder
# ==========================================

import pandas as pd
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans


def load_vector(path):
    """Load, validate, and L2-normalize a stored embedding vector from disk."""

    if not os.path.exists(path):
        return None

    v=np.load(path)

    if not np.isfinite(v).all():
        return None

    norm=np.linalg.norm(v)

    if norm>0:
        v=v/norm

    return v


def generate_hour_prototypes(df_reference, vector_dir="vectors"):
    """Build normalized prototype vectors for each hour bucket from reference tracks."""

    rows = []
    missing_hours = []

    # ensure required columns exist
    if "deezer_id" not in df_reference.columns or "hour_id" not in df_reference.columns:
        raise ValueError("df_reference must contain 'deezer_id' and 'hour_id' columns")

    for hour in range(1, 25):

        subset = df_reference[df_reference["hour_id"] == hour]

        if subset.empty:
            missing_hours.append(hour)
            continue

        vectors = []

        for tid in subset["deezer_id"]:

            path = os.path.join(vector_dir, f"{tid}.npy")

            vec = load_vector(path)

            if vec is not None:
                vectors.append(vec)

        # skip if no valid vectors
        if len(vectors) == 0:
            print(f"[WARN] Hour {hour}: no valid vectors found")
            missing_hours.append(hour)
            continue

        try:
            X = np.stack(vectors)
        except Exception as e:
            print(f"[WARN] Hour {hour}: vector stacking failed ({e})")
            missing_hours.append(hour)
            continue

        proto = np.mean(X, axis=0)

        norm = np.linalg.norm(proto)

        if norm > 0:
            proto = proto / norm
        else:
            print(f"[WARN] Hour {hour}: prototype norm is zero")
            missing_hours.append(hour)
            continue

        rows.append({
            "hour_id": hour,
            "prototype_vector": proto,
            "track_count": len(vectors)
        })

    # Always return a valid dataframe schema
    df = pd.DataFrame(rows, columns=["hour_id", "prototype_vector", "track_count"])

    if df.empty:
        raise RuntimeError(
            "No prototypes could be generated. "
            "Ensure embeddings exist in the vectors/ directory."
        )

    if missing_hours:
        print(f"[INFO] Skipped hours with no vectors: {missing_hours}")

    return df


def generate_micro_centroids(hour,X):
    """Cluster one hour's embeddings into adaptive MiniBatchKMeans micro-centroids."""
    n_tracks = len(X)
    if n_tracks == 0:
        return []

    # Adaptive granularity: denser centroid mesh for hours with more tracks.
    min_k = 8
    max_k = 64
    alpha = 2.0
    proposed_k = int(np.sqrt(n_tracks) * alpha)
    k = min(max_k, n_tracks, max(min_k, proposed_k))

    batch_size = min(1024, max(64, 4 * k))

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=42,
        n_init=10,
        max_iter=300,
        max_no_improvement=20,
        reassignment_ratio=0.01,
        init_size=min(max(3 * k, batch_size), n_tracks),
    )

    kmeans.fit(X)

    centers = kmeans.cluster_centers_
    cluster_sizes = np.bincount(kmeans.labels_, minlength=k)

    rows = []

    for i, c in enumerate(centers):
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm

        rows.append({
            "centroid_id": f"{hour:02d}_{i:03d}",
            "hour_id": hour,
            "vector": c,
            "cluster_size": int(cluster_sizes[i]),
            "hour_cluster_count": int(k),
        })

    return rows


def build_all_centroids(df_reference,vector_dir="vectors"):
    """Generate micro-centroids for all hours and return a centroid dataframe."""

    rows=[]

    for hour in range(1,25):

        subset=df_reference[df_reference.hour_id==hour]

        vecs=[]

        for tid in subset.deezer_id:

            p=os.path.join(vector_dir,f"{tid}.npy")

            v=load_vector(p)

            if v is not None:
                vecs.append(v)

        if vecs:

            X=np.stack(vecs)

            rows.extend(generate_micro_centroids(hour,X))

    return pd.DataFrame(
        rows,
        columns=[
            "centroid_id",
            "hour_id",
            "vector",
            "cluster_size",
            "hour_cluster_count",
        ],
    )

