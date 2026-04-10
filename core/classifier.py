# ==========================================
# STEP 4: FAISS Hierarchical Classifier
# ==========================================

import faiss
import numpy as np
import pandas as pd


def build_faiss_indexes(df_prototypes, df_centroids):
    """Build FAISS prototype and per-hour centroid indexes plus centroid metadata."""

    # ---------- Validation ----------
    if "prototype_vector" not in df_prototypes.columns:
        raise ValueError(
            "prototype_vector column missing. "
            "Check STEP 3 generate_hour_prototypes()."
        )

    if "vector" not in df_centroids.columns:
        raise ValueError(
            "vector column missing in centroids dataframe."
        )

    if df_prototypes.empty:
        raise ValueError("df_prototypes is empty. No reference embeddings found.")

    # ---------- Build prototype index ----------

    proto_vecs = np.stack(
        df_prototypes["prototype_vector"].values
    ).astype("float32")

    proto_hours = df_prototypes["hour_id"].values

    dim = proto_vecs.shape[1]

    proto_index = faiss.IndexFlatIP(dim)
    proto_index.add(proto_vecs)

    # ---------- Build centroid indexes ----------

    centroid_indexes = {}
    centroid_meta = {}

    for hour in sorted(df_centroids["hour_id"].unique()):

        subset = df_centroids[df_centroids["hour_id"] == hour]

        if subset.empty:
            continue

        vecs = np.stack(subset["vector"].values).astype("float32")

        ids = subset["centroid_id"].astype(str).values
        sizes = subset["cluster_size"].fillna(1).astype("float32").values

        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)

        prior = sizes / max(float(np.sum(sizes)), 1.0)

        centroid_indexes[hour] = idx
        centroid_meta[hour] = {
            "ids": ids,
            "sizes": sizes,
            "prior": prior,
        }

    return proto_index, proto_hours, centroid_indexes, centroid_meta


def classify_track(
    vec,
    proto_index,
    proto_hours,
    centroid_indexes,
    centroid_meta,
    top_hour_k=3,
    top_centroid_k=3,
    prior_weight=0.08,
):
    """Classify one embedding by scoring centroid candidates across top prototype hours."""

    vec = vec.astype("float32").reshape(1, -1)

    hour_k = max(1, min(int(top_hour_k), len(proto_hours)))
    proto_scores, proto_idx = proto_index.search(vec, hour_k)

    candidates = []

    for rank, proto_pos in enumerate(proto_idx[0]):
        if proto_pos < 0:
            continue

        hour = int(proto_hours[proto_pos])
        if hour not in centroid_indexes:
            continue

        cent_index = centroid_indexes[hour]
        hour_cent_k = max(1, min(int(top_centroid_k), int(cent_index.ntotal)))
        cent_scores, cent_idx = cent_index.search(vec, hour_cent_k)

        meta = centroid_meta[hour]
        hour_proto_score = float(proto_scores[0][rank])

        for local_rank, cent_pos in enumerate(cent_idx[0]):
            if cent_pos < 0:
                continue

            raw_score = float(cent_scores[0][local_rank])
            prior = float(meta["prior"][cent_pos])
            adjusted_score = raw_score + prior_weight * prior

            candidates.append({
                "hour": hour,
                "centroid": meta["ids"][cent_pos],
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "hour_score": hour_proto_score,
                "cluster_size": int(meta["sizes"][cent_pos]),
            })

    if not candidates:
        raise RuntimeError("No centroid candidates available for classification.")

    candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else best

    return {
        "assigned_hour": best["hour"],
        "assigned_centroid": best["centroid"],
        "best_score": best["raw_score"],
        "second_best_score": second["raw_score"],
        "margin": float(best["adjusted_score"] - second["adjusted_score"]),
        "best_hour_score": best["hour_score"],
        "cluster_size": best["cluster_size"],
    }
