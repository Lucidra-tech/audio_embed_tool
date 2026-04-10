from __future__ import annotations

from pathlib import Path

import pandas as pd


REFERENCE_PLAYLIST_DIR = "reference_playlist"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def read_reference_playlist_csvs(
    directory: str | Path = REFERENCE_PLAYLIST_DIR,
) -> pd.DataFrame:
    """Read and combine all CSV files from the reference playlist directory."""
    directory_path = Path(directory)

    if not directory_path.exists() or not directory_path.is_dir():
        return pd.DataFrame()

    frames = []

    for csv_path in sorted(directory_path.glob("*.csv")):
        df = _read_csv_with_fallback(csv_path)
        if df.empty:
            continue

        df.columns = [str(column).strip() for column in df.columns]
        df["source_file"] = csv_path.name

        if "deezer_playlist_id" in df.columns:
            df["deezer_id"] = df["deezer_playlist_id"]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_reference_playlist_options(
    directory: str | Path = REFERENCE_PLAYLIST_DIR,
) -> pd.DataFrame:
    """Return normalized reference rows with deezer_id and hour_id when possible."""
    df = read_reference_playlist_csvs(directory)

    if df.empty or "deezer_id" not in df.columns:
        return pd.DataFrame(columns=["deezer_id", "hour_id"])

    result = df.copy()
    result["deezer_id"] = result["deezer_id"].astype(str).str.strip()
    result = result[result["deezer_id"].ne("")]

    if "hour_id" in result.columns:
        result["hour_id"] = pd.to_numeric(result["hour_id"], errors="coerce")
    elif "hour" in result.columns:
        result["hour_id"] = pd.to_numeric(result["hour"], errors="coerce")
    else:
        result["hour_id"] = range(1, len(result) + 1)

    result = result.dropna(subset=["hour_id"])
    result["hour_id"] = result["hour_id"].astype(int)
    result = result[result["hour_id"].between(1, 24)]

    return result[["deezer_id", "hour_id"]].drop_duplicates().reset_index(drop=True)


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, dtype=str, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue

    return pd.read_csv(csv_path, dtype=str)
