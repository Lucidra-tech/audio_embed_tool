# ==========================================
# STEP 2: Deezer Ingestion + Cache + Embedding
# ==========================================

import os, threading
import sqlite3
import requests
import numpy as np
import time


class CacheManager:
    """SQLite-backed cache for persisting deezer track id to embedding file mappings."""

    def __init__(self, db_path="music_cache.db"):
        """Initialize cache database and ensure the track cache table exists."""
        self.db_path = db_path
        self.local = threading.local()

        # Create table once
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS track_cache(
            deezer_id TEXT PRIMARY KEY,
            vector_path TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self):
        """
        Each thread gets its own SQLite connection.
        """
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_path)
        return self.local.conn

    def get_vector(self, deezer_id):
        """Return cached embedding file path for a Deezer track id if available."""

        conn = self._get_conn()

        cur = conn.execute(
            "SELECT vector_path FROM track_cache WHERE deezer_id=?",
            (deezer_id,)
        )

        row = cur.fetchone()

        return row[0] if row else None

    def save(self, deezer_id, path):
        """Insert or update cached embedding path for a Deezer track id."""

        conn = self._get_conn()

        conn.execute(
            "INSERT OR REPLACE INTO track_cache(deezer_id,vector_path) VALUES (?,?)",
            (deezer_id, path)
        )

        conn.commit()

class DeezerIngestor:
    """Deezer API client for retrieving playlist tracks with optional limit."""

    BASE_URL="https://api.deezer.com"

    def get_playlist_tracks(self, playlist_id, LIMIT=None):
        """Fetch tracks from a Deezer playlist, following pagination until limit/end."""

        url=f"{self.BASE_URL}/playlist/{playlist_id}/tracks"

        tracks=[]

        while url:

            r=requests.get(url,timeout=10).json()

            if "error" in r:
                print("API error:",r["error"])
                break

            tracks.extend(r["data"])

            if LIMIT and len(tracks)>=LIMIT:
                break

            url=r.get("next")

            time.sleep(0.2)

        if LIMIT:
            tracks=tracks[:LIMIT]

        return tracks


# ---------------------------------------------------------
# 3. Safe Processing Workflow (Used by Step 5 Orchestrator)
# ---------------------------------------------------------

def process_single_track(track,engine,cache,ingestor,vector_dir="vectors"):
    """Download preview audio, compute embedding, save vector, and update cache."""

    os.makedirs(vector_dir,exist_ok=True)

    tid=str(track["id"])

    cached=cache.get_vector(tid)

    if cached and os.path.exists(cached):
        return {"status":"cached","vector_path":cached}

    preview=track.get("preview")

    if not preview:
        return {
            "status":"no_preview",
            "deezer_id": tid
        }

    temp_file=f"temp_{tid}.mp3"
    vec_path=os.path.join(vector_dir,f"{tid}.npy")

    try:

        r=requests.get(preview,timeout=10)

        if r.status_code!=200:
            return {"status":"download_failed"}

        with open(temp_file,"wb") as f:
            f.write(r.content)

        if os.path.getsize(temp_file)<10000:
            return {"status":"corrupt"}

        vector=engine.get_track_embedding(temp_file)

        if vector is None:
            return {"status":"embedding_failed"}

        np.save(vec_path,vector)

        cache.save(tid,vec_path)

        return {"status":"success","vector_path":vec_path}

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)


def ensure_reference_embeddings(reference_df, engine, cache, ingestor):
    """Ensure vector files exist for all reference deezer ids in the dataframe."""

    print("Ensuring reference embeddings exist...")

    os.makedirs("vectors", exist_ok=True)

    for tid in reference_df["deezer_id"]:

        vec_path = os.path.join("vectors", f"{tid}.npy")

        if os.path.exists(vec_path):
            continue

        # fetch track metadata from Deezer
        track_url = f"https://api.deezer.com/track/{tid}"

        try:
            track = requests.get(track_url).json()
        except:
            print(f"[WARN] Failed to fetch track {tid}")
            continue

        if not track.get("preview"):
            print(f"[WARN] No preview for track {tid}")
            continue

        result = process_single_track(
            track,
            engine,
            cache,
            ingestor
        )

        if result.get("vector_path"):
            print(f"Embedded reference track {tid}")



# Usage Example (Assuming 'engine' is instantiated from Step 1):
# cache = CacheManager()
# ingestor = DeezerIngestor()
# process_playlist("YOUR_PLAYLIST_ID", engine, cache, ingestor)
