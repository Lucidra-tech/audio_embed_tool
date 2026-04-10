import os # NO issue
from pathlib import Path
import sys
import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download


# Construct the absolute path to the 'core' directory using the current working directory
core_path = os.path.join(os.getcwd(), 'core')

# Insert the 'core' path at the beginning of sys.path to prioritize it
if core_path not in sys.path:
    sys.path.insert(0, core_path)

# Now Python can successfully locate and import the module
from musicfm25hz import MusicFM25Hz

REPO_ID = "minzwon/MusicFM"
DATA_DIR = Path(os.getcwd()) / "data"
MODEL_FILENAME = "pretrained_fma.pt"
STATS_FILENAME = "msd_stats.json"


def download_musicfm_assets(
    data_dir: str | Path = DATA_DIR,
    hf_token: str | None = None,
) -> dict[str, str]:
    """Download model artifacts if needed and return absolute paths."""
    target_dir = Path(data_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    stats_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=STATS_FILENAME,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    return {
        "model_path": str(Path(model_path).resolve()),
        "stats_path": str(Path(stats_path).resolve()),
    }


class EmbeddingEngine:
    """MusicFM embedding runner that loads model assets and encodes audio clips."""

    def __init__(self, model_path: str | None = None, stats_path: str | None = None):
        """Initialize MusicFM model on available device using resolved artifact paths."""
        if not model_path or not stats_path:
            artifacts = download_musicfm_assets()
            model_path = model_path or artifacts["model_path"]
            stats_path = stats_path or artifacts["stats_path"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing MusicFM on {self.device}...")

        self.model = MusicFM25Hz(
            is_flash=False,
            stat_path=stats_path,
            model_path=model_path
        ).to(self.device).eval()

    def get_track_embedding(self, audio_path):
        """Loads audio, extracts latent representation, pools, and normalizes."""
        try:
            wav, _ = librosa.load(audio_path, sr=24000, duration=30.0)
            wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)

            with torch.no_grad():
                latent = self.model.get_latent(wav_tensor, layer_ix=7)
                embedding = torch.mean(latent, dim=1).cpu().numpy().flatten()

            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            return None
