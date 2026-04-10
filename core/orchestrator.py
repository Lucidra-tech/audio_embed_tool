# ==========================================
# STEP 5: Full Pipeline (Clean Version)
# ==========================================

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import io
import zipfile
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyOAuth

try:
    from .caching_embedding import (
        CacheManager,
        DeezerIngestor,
        ensure_reference_embeddings,
        process_single_track,
    )
    from .classifier import build_faiss_indexes, classify_track
    from .embedding_engine import EmbeddingEngine
    from .reference_model import build_all_centroids, generate_hour_prototypes
except ImportError:
    from caching_embedding import (
        CacheManager,
        DeezerIngestor,
        ensure_reference_embeddings,
        process_single_track,
    )
    from classifier import build_faiss_indexes, classify_track
    from embedding_engine import EmbeddingEngine
    from reference_model import build_all_centroids, generate_hour_prototypes




PIPELINE_DF_CACHE = {
    "files": {},
    "reference": {},
    "playlists": {},
}


def reset_pipeline_df_cache():
    """Clear all in-memory dataframe caches for a fresh pipeline run."""
    PIPELINE_DF_CACHE["files"] = {}
    PIPELINE_DF_CACHE["reference"] = {}
    PIPELINE_DF_CACHE["playlists"] = {}


def get_pipeline_df_cache():
    """Return the global in-memory cache containing run dataframe artifacts."""
    return PIPELINE_DF_CACHE


def cache_export_dataframe(file_name, df):
    """Store a dataframe in the export cache under its target output filename."""
    if df is None:
        return
    PIPELINE_DF_CACHE["files"][file_name] = df.copy()


def export_cached_dataframes_to_csv(output_dir="outputs"):
    """Persist all cached dataframes to CSV files in the requested output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for file_name, df in PIPELINE_DF_CACHE.get("files", {}).items():
        df.to_csv(os.path.join(output_dir, file_name), index=False)


def download_all_csv_files(output_dir="outputs"):
    """Write cached CSVs to disk and return a ZIP archive as bytes."""
    export_cached_dataframes_to_csv(output_dir)

    output_path = Path(output_dir)
    csv_files = sorted(output_path.glob("*.csv"))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_file in csv_files:
            zf.write(csv_file, arcname=csv_file.name)

    buffer.seek(0)
    return buffer.getvalue()


class ExportManager:
    """Cache-first export manager that stages pipeline outputs as dataframes."""

    def __init__(self, output_dir="outputs"):
        """Initialize output directory metadata used by export operations."""

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_reference_summary(self, reference_df):
        """Cache per-hour reference embedding counts."""
        summary = (
            reference_df.groupby("hour_id")
            .size()
            .reset_index(name="track_count")
        )
        cache_export_dataframe("reference_embeddings_summary.csv", summary)

    def export_prototypes(self, df_prototypes):
        """Cache prototype vectors after converting arrays to CSV-friendly strings."""
        df = df_prototypes.copy()

        df["prototype_vector"] = df["prototype_vector"].apply(
            lambda x: ",".join(map(str, x))
        )
        cache_export_dataframe("hour_prototypes_24.csv", df)

    def export_centroids(self, df_centroids):
        """Cache centroid vectors after converting arrays to CSV-friendly strings."""
        df = df_centroids.copy()

        df["vector"] = df["vector"].apply(
            lambda x: ",".join(map(str, x))
        )
        cache_export_dataframe("centroids_360.csv", df)

    def export_missing_previews(self, missing_list, playlist_id):
        """Cache missing preview track rows for one playlist."""
        df = pd.DataFrame(missing_list)
        cache_export_dataframe(f"missing_preview_{playlist_id}.csv", df)

    def export_run_summary(self, summary_dict, playlist_id):
        """Cache one-row summary metrics for a processed playlist."""
        df = pd.DataFrame([summary_dict])
        cache_export_dataframe(f"run_summary_{playlist_id}.csv", df)


class SpotifyClient:
    """Thin wrapper around Spotipy for lookup and playlist creation workflows."""

    def __init__(self):
        """Create an authenticated Spotify client and capture the active user id."""

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                scope="playlist-modify-public playlist-modify-private"
            )
        )

        self.user_id = self.sp.current_user()["id"]


    def search_by_isrc(self, isrc):
        """Search Spotify for a track URI using ISRC and return the first match."""

        if not isrc:
            return None

        result = self.sp.search(
            q=f"isrc:{isrc}",
            type="track",
            limit=1
        )

        items = result["tracks"]["items"]

        if items:
            return items[0]["uri"]

        return None


    def search_by_text(self, title, artist):
        """Search Spotify by title/artist text and return the first matched URI."""

        query = f"track:{title} artist:{artist}"

        result = self.sp.search(
            q=query,
            type="track",
            limit=1
        )

        items = result["tracks"]["items"]

        if items:
            return items[0]["uri"]

        return None


    

    def create_playlist_chunks(self, name, uris):
        """Create one or more playlists and add URIs in chunked batches."""

        CHUNK = 10000
        links = []

        for i in range(0, len(uris), CHUNK):

            subset = uris[i:i+CHUNK]

            playlist = self.sp.user_playlist_create(
                self.user_id,
                f"{name}_{i//CHUNK+1}"
            )

            self.sp.playlist_add_items(
                playlist["id"],
                subset
            )

            links.append(playlist["external_urls"]["spotify"])

        return links



def create_playlist_into_spotify_save_unmatched(spotify_uris, unmatched, playlist_id):
      """Create Spotify playlists for matched URIs and cache unmatched rows."""

      playlist_links = []

      if spotify_uris:

        sp = SpotifyClient()

        playlist_links = sp.create_playlist_chunks(
            f"classified_playlist_{playlist_id}",
            spotify_uris
        )

        print("\nSpotify playlists created:")

      for link in playlist_links:
          print(link)

      cache_export_dataframe(
          f"unmatched_spotify_{playlist_id}.csv",
          pd.DataFrame(unmatched)
      )



def match_tracks_to_spotify(tracks):
    """Resolve input tracks to Spotify URIs and return matched/unmatched lists."""

    spotify = SpotifyClient()

    uris = []
    unmatched = []

    for t in tracks:

        uri = None

        isrc = t.get("isrc")

        if isrc:
            uri = spotify.search_by_isrc(isrc)

        if not uri:
            uri = spotify.search_by_text(
                t.get("title"),
                t.get("artist")
            )

        if uri:
            uris.append(uri)
        else:
            unmatched.append(t)

    return uris, unmatched




# ---------------------------------------------------------
# Parallel embedding
# ---------------------------------------------------------

def embed_tracks_parallel(tracks, engine, cache, ingestor, workers):
    """Embed tracks in parallel using cached vector files produced per track."""

    def process(track):
        """Embed one track and return deezer id plus vector when available."""

        result = process_single_track(
            track,
            engine,
            cache,
            ingestor
        )

        if result.get("vector_path"):
            vec = np.load(result["vector_path"])

            return {
                "deezer_id": str(track["id"]),
                "embedding": vec
            }

        return None

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = executor.map(process, tracks)

        for o in outputs:
            if o:
                results.append(o)

    return results


# ---------------------------------------------------------
# Build reference dataset
# ---------------------------------------------------------

def build_reference_from_playlists(real_reference_data, ingestor, LIMIT):
    """Fetch reference playlists and build a de-duplicated deezer/hour dataframe."""

    rows = []

    for playlist_id, hour in zip(
        real_reference_data["deezer_id"],
        real_reference_data["hour_id"]
    ):

        tracks = ingestor.get_playlist_tracks(
            playlist_id,
            LIMIT=LIMIT
        )

        for t in tracks:

            if not t.get("preview"):
                continue

            rows.append({
                "deezer_id": str(t["id"]),
                "hour_id": hour
            })

    df = pd.DataFrame(rows)

    return df.drop_duplicates("deezer_id")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run_full_pipeline(
    real_reference_data,
    input_playlists,
    LIMIT=False,
    BATCH_SIZE=20,
    WORKERS=4
):
    """Run end-to-end reference build, classification, and cache-backed exports."""
    reset_pipeline_df_cache()

    engine = EmbeddingEngine()
    cache = CacheManager()
    ingestor = DeezerIngestor()
    exporter = ExportManager()

    print("Building reference dataset...")

    reference_df = build_reference_from_playlists(
        real_reference_data,
        ingestor,
        LIMIT
    )

    ensure_reference_embeddings(
        reference_df,
        engine,
        cache,
        ingestor
    )

    print("Reference tracks:", len(reference_df))


    # ---------------------------------------------------------
    # Build reference model
    # ---------------------------------------------------------

    df_prototypes = generate_hour_prototypes(reference_df)
    df_centroids = build_all_centroids(reference_df)

    proto_index, proto_hours, centroid_indexes, centroid_meta = build_faiss_indexes(
        df_prototypes,
        df_centroids
    )

    exporter.export_reference_summary(reference_df)
    exporter.export_prototypes(df_prototypes)
    exporter.export_centroids(df_centroids)

    reference_summary_df = (
        reference_df.groupby("hour_id")
        .size()
        .reset_index(name="track_count")
    )
    PIPELINE_DF_CACHE["reference"]["reference_embeddings_summary"] = reference_summary_df.copy()
    PIPELINE_DF_CACHE["reference"]["hour_prototypes_24"] = df_prototypes.copy()
    PIPELINE_DF_CACHE["reference"]["centroids_360"] = df_centroids.copy()


    # ---------------------------------------------------------
    # Process input playlists
    # ---------------------------------------------------------

    for playlist_id in input_playlists:
        print("\nProcessing playlist:", playlist_id)

        tracks = ingestor.get_playlist_tracks(
            playlist_id,
            LIMIT=LIMIT
        )

        print("Tracks fetched:", len(tracks))


        playlist_results = []
        missing_previews = []


        # ---------------------------------------------------------
        # Batch loop
        # ---------------------------------------------------------

        for start in range(0, len(tracks), BATCH_SIZE):

            batch_tracks = tracks[start:start+BATCH_SIZE]

            print(
                f"Batch {start//BATCH_SIZE + 1} | {len(batch_tracks)} tracks"
            )


            # -------------------------------
            # Filter preview tracks
            # -------------------------------

            valid_tracks = []

            for track in batch_tracks:

                if not track.get("preview"):

                    missing_previews.append({
                        "deezer_id": track["id"],
                        "title": track.get("title"),
                        "artist": track.get("artist", {}).get("name")
                    })

                else:
                    valid_tracks.append(track)


            if not valid_tracks:
                continue


            # -------------------------------
            # Embed tracks
            # -------------------------------

            embedded = embed_tracks_parallel(
                valid_tracks,
                engine,
                cache,
                ingestor,
                WORKERS
            )

            if not embedded:
                continue


            # -------------------------------
            # Classification
            # -------------------------------

            for item in embedded:

                result = classify_track(
                    item["embedding"],
                    proto_index,
                    proto_hours,
                    centroid_indexes,
                    centroid_meta
                )

                result["deezer_id"] = item["deezer_id"]

                track_meta = next(
                    (t for t in valid_tracks if str(t["id"]) == item["deezer_id"]),
                    None
                )

                if track_meta:
                    result["title"] = track_meta.get("title")
                    result["artist"] = track_meta.get("artist", {}).get("name")
                    result["isrc"] = track_meta.get("isrc")

                playlist_results.append(result)


        # ---------------------------------------------------------
        # Export results
        # ---------------------------------------------------------

        df_results = pd.DataFrame(playlist_results)

        spotify_input = df_results.to_dict("records")

        spotify_uris, unmatched = match_tracks_to_spotify(spotify_input)
        
        create_playlist_into_spotify_save_unmatched(spotify_uris, unmatched, playlist_id)


        print(df_results.head())

        cache_export_dataframe(
            f"assignment_results_{playlist_id}.csv",
            df_results
        )

        exporter.export_missing_previews(
            missing_previews,
            playlist_id
        )


        summary = {
            "reference_tracks": len(reference_df),
            "prototypes_created": len(df_prototypes),
            "centroids_created": len(df_centroids),
            "classified_tracks": len(df_results),
            "missing_previews": len(missing_previews)
        }

        exporter.export_run_summary(summary, playlist_id)

        trace_key = str(playlist_id)
        PIPELINE_DF_CACHE["playlists"][trace_key] = {
            "playlist_id": str(playlist_id),
            "assignment_results": df_results.copy(),
            "missing_previews": pd.DataFrame(missing_previews).copy(),
            "unmatched_spotify": pd.DataFrame(unmatched).copy(),
            "run_summary": pd.DataFrame([summary]).copy(),
        }

        print("\nPlaylist completed.")
        print("Classified:", len(df_results))
        print("Missing previews:", len(missing_previews))
