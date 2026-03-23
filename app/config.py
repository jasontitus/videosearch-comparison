import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://videosearch:videosearch@localhost:5432/videosearch",
)

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "pytorch")  # "pytorch" or "gguf"
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8090")
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "/models/jina-embeddings-v4-text-retrieval-Q8_0.gguf")

VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", str(BASE_DIR / "videos")))
THUMBNAILS_DIR = Path(os.getenv("THUMBNAILS_DIR", str(BASE_DIR / "thumbnails")))

if not GCS_BUCKET:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    override = os.getenv("DEVICE")
    if override:
        return override

    import torch

    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
