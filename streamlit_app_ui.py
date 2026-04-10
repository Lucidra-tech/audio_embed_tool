from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from helper import get_reference_playlist_options, read_reference_playlist_csvs


st.set_page_config(
    page_title="Deezer Embedding Pipeline",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REFERENCE_PLAYLIST_DIR = PROJECT_ROOT / "reference_playlist"

DEFAULT_REFERENCE_DATA = {
    "deezer_id": [
        "2252220162",
        "2296059482",
        "11081408402",
        "146819501",
        "11663792464",
        "13098300343",
        "10746894082",
        "1902101402",
        "1947984342",
        "1306978785",
        "1746835762",
        "13668653381",
        "8970791602",
        "10783752422",
        "10872142682",
        "6006188884",
        "8666347202",
        "3035482146",
        "7214404404",
        "1393810385",
        "3881674142",
        "11556358124",
        "12993355223",
        "1295485847",
    ],
    "hour_id": list(range(1, 25)),
}

DEFAULT_INPUT_PLAYLISTS = ["1363560485"]


class ExportManager:
    """Small exporter shim for the current core orchestrator."""

    def __init__(self, output_dir: Path | str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_reference_summary(self, reference_df: pd.DataFrame) -> None:
        summary = (
            reference_df.groupby("hour_id")
            .size()
            .reset_index(name="embedded_track_count")
        )
        from core import orchestrator

        orchestrator.cache_export_dataframe("reference_embeddings_summary.csv", summary)

    def export_prototypes(self, df_prototypes: pd.DataFrame) -> None:
        self._export_vectors(df_prototypes, "hour_prototypes_24.csv")

    def export_centroids(self, df_centroids: pd.DataFrame) -> None:
        self._export_vectors(df_centroids, "centroids_360.csv")

    def export_missing_previews(self, rows: list[dict], playlist_id: str) -> None:
        from core import orchestrator

        orchestrator.cache_export_dataframe(
            f"missing_preview_{playlist_id}.csv",
            pd.DataFrame(rows),
        )

    def export_run_summary(self, summary: dict, playlist_id: str) -> None:
        from core import orchestrator

        orchestrator.cache_export_dataframe(
            f"run_summary_{playlist_id}.csv",
            pd.DataFrame([summary]),
        )

    def _export_vectors(self, df: pd.DataFrame, filename: str) -> None:
        serializable = df.copy()
        for column in serializable.columns:
            serializable[column] = serializable[column].apply(
                lambda value: value.tolist() if hasattr(value, "tolist") else value
            )
        from core import orchestrator

        orchestrator.cache_export_dataframe(filename, serializable)


def page_setup() -> None:
    st.title("Deezer Audio Embedding Pipeline")
    st.caption(
        "Build 24-hour MusicFM prototypes, classify Deezer playlist tracks, "
        "and review the exported CSV outputs."
    )


def default_reference_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_REFERENCE_DATA)


@st.cache_data(show_spinner=False)
def reference_playlist_csvs(modified_token: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    _ = modified_token
    return read_reference_playlist_csvs(REFERENCE_PLAYLIST_DIR)


@st.cache_data(show_spinner=False)
def reference_playlist_options(modified_token: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    _ = modified_token
    return get_reference_playlist_options(REFERENCE_PLAYLIST_DIR)


def reference_playlist_modified_token() -> tuple[tuple[str, float], ...]:
    if not REFERENCE_PLAYLIST_DIR.exists():
        return ()

    return tuple(
        (path.name, path.stat().st_mtime)
        for path in sorted(REFERENCE_PLAYLIST_DIR.glob("*.csv"))
    )


def parse_playlist_ids(raw_value: str) -> list[str]:
    ids = []

    for line in raw_value.replace(",", "\n").splitlines():
        cleaned = line.strip()
        if cleaned:
            ids.append(cleaned)

    return ids


def validate_reference_df(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []

    required = {"deezer_id", "hour_id"}
    missing = required.difference(df.columns)
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
        return False, errors

    if df.empty:
        errors.append("Reference table is empty.")

    if df["deezer_id"].astype(str).str.strip().eq("").any():
        errors.append("Every reference row needs a Deezer playlist ID.")

    hours = pd.to_numeric(df["hour_id"], errors="coerce")
    if hours.isna().any():
        errors.append("Hour IDs must be numbers from 1 to 24.")
    elif not hours.between(1, 24).all():
        errors.append("Hour IDs must stay between 1 and 24.")

    if len(set(hours.dropna().astype(int))) < 24:
        errors.append("The reference table should contain all 24 hour buckets.")

    return not errors, errors


def prepare_reference_data(df: pd.DataFrame) -> dict[str, list]:
    clean = df.copy()
    clean["deezer_id"] = clean["deezer_id"].astype(str).str.strip()
    clean["hour_id"] = pd.to_numeric(clean["hour_id"], errors="raise").astype(int)
    clean = clean.sort_values("hour_id")

    return {
        "deezer_id": clean["deezer_id"].tolist(),
        "hour_id": clean["hour_id"].tolist(),
    }


def build_sidebar() -> tuple[pd.DataFrame, list[str], int | bool, int, int, str]:
    with st.sidebar:
        st.header("Run Settings")

        st.subheader("API Keys")
        st.caption(
            "Optional runtime credentials. Values are kept in this session "
            "and exported to environment variables for the pipeline."
        )

        hf_token = st.text_input(
            "HF_TOKEN",
            value=st.session_state.get("HF_TOKEN", ""),
            type="password",
            help="Hugging Face token used for gated/private model downloads.",
        )
        client_id = st.text_input(
            "CLIENT_ID",
            value=st.session_state.get("CLIENT_ID", ""),
            type="password",
            help="Spotify OAuth client ID.",
        )
        client_secret = st.text_input(
            "CLIENT_SECRET",
            value=st.session_state.get("CLIENT_SECRET", ""),
            type="password",
            help="Spotify OAuth client secret.",
        )

        st.session_state["HF_TOKEN"] = hf_token
        st.session_state["CLIENT_ID"] = client_id
        st.session_state["CLIENT_SECRET"] = client_secret

        if hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token.strip()
        if client_id.strip():
            os.environ["CLIENT_ID"] = client_id.strip()
            os.environ["SPOTIPY_CLIENT_ID"] = client_id.strip()
        if client_secret.strip():
            os.environ["CLIENT_SECRET"] = client_secret.strip()
            os.environ["SPOTIPY_CLIENT_SECRET"] = client_secret.strip()

        st.divider()

        limit_enabled = st.checkbox("Limit tracks per playlist", value=True)
        limit_value = st.number_input(
            "Track limit",
            min_value=1,
            max_value=100_000,
            value=15,
            step=1,
            disabled=not limit_enabled,
        )

        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=1_000,
            value=2,
            step=1,
        )
        workers = st.number_input(
            "Embedding workers",
            min_value=1,
            max_value=32,
            value=2,
            step=1,
        )

        st.divider()
        st.subheader("Input Playlists")
        raw_playlists = st.text_area(
            "Deezer playlist IDs",
            value="\n".join(DEFAULT_INPUT_PLAYLISTS),
            help="Use one playlist ID per line, or separate IDs with commas.",
        )

        st.divider()
        st.subheader("Reference Source")
        reference_source = st.radio(
            "Choose reference playlists",
            ["Manual editor", "reference_playlist CSVs"],
            help=(
                "CSV mode reads every CSV in reference_playlist and uses "
                "deezer_playlist_id when that column exists."
            ),
        )

        if st.button("Reload reference CSVs"):
            reference_playlist_csvs.clear()
            reference_playlist_options.clear()
            st.session_state.pop("reference_df", None)
            st.rerun()

    reference_df = load_reference_source(reference_source)

    input_playlists = parse_playlist_ids(raw_playlists)
    limit = int(limit_value) if limit_enabled else False

    return reference_df, input_playlists, limit, int(batch_size), int(workers), reference_source


def load_reference_source(reference_source: str) -> pd.DataFrame:
    if reference_source == "reference_playlist CSVs":
        token = reference_playlist_modified_token()
        csv_reference_df = reference_playlist_options(token)

        if not csv_reference_df.empty:
            return csv_reference_df

        return st.session_state.get("reference_df", default_reference_df())

    return st.session_state.get("reference_df", default_reference_df())


def reference_editor(reference_df: pd.DataFrame, reference_source: str) -> pd.DataFrame:
    st.subheader("24-Hour Reference Playlists")
    st.write(
        "Each reference playlist represents one hour bucket used to build the "
        "prototype and micro-centroid model."
    )

    if reference_source == "reference_playlist CSVs":
        token = reference_playlist_modified_token()
        raw_reference_df = reference_playlist_csvs(token)

        if raw_reference_df.empty:
            st.warning(
                "No usable CSV files were found in reference_playlist. "
                "The editor is using the default reference data."
            )
        elif "deezer_playlist_id" not in raw_reference_df.columns:
            st.info(
                "Reference CSVs were found, but no deezer_playlist_id column "
                "was present. The UI will use deezer_id if available."
            )
        else:
            st.success(
                "Loaded reference playlists from deezer_playlist_id in "
                "reference_playlist CSVs."
            )

    editor_key = f"reference_editor_{reference_source.lower().replace(' ', '_')}"

    edited = st.data_editor(
        reference_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "deezer_id": st.column_config.TextColumn(
                "Deezer playlist ID",
                required=True,
            ),
            "hour_id": st.column_config.NumberColumn(
                "Hour",
                min_value=1,
                max_value=24,
                step=1,
                required=True,
            ),
        },
        key=editor_key,
    )

    col_a, _ = st.columns([1, 4])
    with col_a:
        if st.button("Reset references"):
            st.session_state["reference_df"] = default_reference_df()
            st.rerun()

    st.session_state["reference_df"] = edited
    return edited


def load_pipeline():
    """Load the core pipeline lazily so the page can render before heavy imports."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from core import orchestrator
    from core.caching_embedding import (
        CacheManager,
        DeezerIngestor,
        ensure_reference_embeddings,
        process_single_track,
    )
    from core.classifier import build_faiss_indexes, classify_track
    from core.reference_model import build_all_centroids, generate_hour_prototypes

    orchestrator.CacheManager = CacheManager
    orchestrator.DeezerIngestor = DeezerIngestor
    orchestrator.ensure_reference_embeddings = ensure_reference_embeddings
    orchestrator.process_single_track = process_single_track
    orchestrator.build_faiss_indexes = build_faiss_indexes
    orchestrator.classify_track = classify_track
    orchestrator.build_all_centroids = build_all_centroids
    orchestrator.generate_hour_prototypes = generate_hour_prototypes

    if not hasattr(orchestrator, "EmbeddingEngine"):
        try:
            from core import musicfm25hz

            sys.modules.setdefault("musicfm_25hz", musicfm25hz)
            from core.embedding_engine import EmbeddingEngine

            orchestrator.EmbeddingEngine = EmbeddingEngine
        except Exception as exc:
            raise RuntimeError(
                "The Streamlit UI loaded, but the embedding engine could not be "
                "imported. The backend currently expects the MusicFM model files "
                "and may need its imports/paths fixed before a full run can start."
            ) from exc

    return orchestrator.run_full_pipeline


@st.cache_resource(show_spinner=False)
def ensure_musicfm_downloaded(hf_token: str = "") -> dict[str, str]:
    """Download MusicFM assets once per Streamlit runtime."""
    from core.embedding_engine import download_musicfm_assets

    return download_musicfm_assets(hf_token=hf_token or None)


def run_pipeline(
    reference_data: dict[str, list],
    input_playlists: list[str],
    limit: int | bool,
    batch_size: int,
    workers: int,
) -> str:
    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        run_full_pipeline = load_pipeline()
        run_full_pipeline(
            reference_data,
            input_playlists,
            LIMIT=limit,
            BATCH_SIZE=batch_size,
            WORKERS=workers,
        )

    return output_buffer.getvalue()


def output_files() -> Iterable[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        OUTPUT_DIR.glob("*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(path: str, modified_time: float) -> pd.DataFrame:
    _ = modified_time
    return pd.read_csv(path)


def output_dashboard() -> None:
    st.subheader("Output Files")

    files = list(output_files())
    if files:
        selected = st.selectbox(
            "Choose a CSV to preview",
            files,
            format_func=lambda path: path.name,
        )

        df = read_csv(str(selected), selected.stat().st_mtime)

        left, right, third = st.columns(3)
        left.metric("Rows", len(df))
        right.metric("Columns", len(df.columns))
        third.metric("File", selected.name)

        st.dataframe(df, use_container_width=True, height=360)
        st.download_button(
            "Download CSV",
            data=selected.read_bytes(),
            file_name=selected.name,
            mime="text/csv",
        )
    else:
        st.info(
            "No CSV files on disk yet. You can still use the ZIP export button "
            "to materialize cached pipeline outputs."
        )

    st.divider()
    st.caption("Export and download all available CSV outputs as one ZIP file.")

    try:
        from core import orchestrator

        all_csv_zip = orchestrator.download_all_csv_files(str(OUTPUT_DIR))
        st.download_button(
            "Download All CSVs (ZIP)",
            data=all_csv_zip,
            file_name="all_csv_outputs.zip",
            mime="application/zip",
        )
    except Exception as exc:
        st.warning(f"Could not prepare all CSV downloads: {exc}")


def main() -> None:
    page_setup()

    (
        reference_df,
        input_playlists,
        limit,
        batch_size,
        workers,
        reference_source,
    ) = build_sidebar()

    tab_run, tab_outputs, tab_notes = st.tabs(["Run Pipeline", "Outputs", "Notes"])

    with tab_run:
        edited_reference_df = reference_editor(reference_df, reference_source)
        is_valid, errors = validate_reference_df(edited_reference_df)

        if errors:
            for error in errors:
                st.warning(error)

        if not input_playlists:
            st.warning("Add at least one Deezer input playlist ID in the sidebar.")
            is_valid = False

        summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
        summary_col_1.metric("Reference rows", len(edited_reference_df))
        summary_col_2.metric("Input playlists", len(input_playlists))
        summary_col_3.metric("Track limit", "No limit" if limit is False else limit)

        run_clicked = st.button(
            "Run full pipeline",
            type="primary",
            disabled=not is_valid,
        )

        if run_clicked:
            reference_data = prepare_reference_data(edited_reference_df)

            with st.status("Running pipeline...", expanded=True) as status:
                try:
                    hf_token = (
                        os.environ.get("HF_TOKEN")
                        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                        or ""
                    )
                    ensure_musicfm_downloaded(hf_token)

                    logs = run_pipeline(
                        reference_data,
                        input_playlists,
                        limit,
                        batch_size,
                        workers,
                    )
                    status.update(
                        label="Pipeline completed",
                        state="complete",
                        expanded=False,
                    )
                    st.success("Pipeline completed.")
                    if logs.strip():
                        st.subheader("Run Log")
                        st.code(logs, language="text")
                    st.cache_data.clear()
                except Exception as exc:
                    status.update(
                        label="Pipeline stopped",
                        state="error",
                        expanded=True,
                    )
                    st.error(str(exc))
                    st.exception(exc)

    with tab_outputs:
        output_dashboard()

    with tab_notes:
        st.subheader("Pipeline Notes")
        st.write(
            "The backend uses Deezer 30-second preview clips, MusicFM embeddings, "
            "24 hourly prototypes, 360 micro-centroids, and FAISS similarity search."
        )
        st.write(
            "Expected CSV exports include assignment results, reference summaries, "
            "hour prototypes, centroids, missing previews, unmatched Spotify tracks, "
            "and run summaries."
        )
        st.write(
            "A full run may need Spotify credentials and MusicFM model assets "
            "available in the environment before the backend can complete."
        )


if __name__ == "__main__":
    main()
