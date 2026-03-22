import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://videosearch:videosearch@localhost:5432/videosearch",
)

VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", str(BASE_DIR / "videos")))
THUMBNAILS_DIR = Path(os.getenv("THUMBNAILS_DIR", str(BASE_DIR / "thumbnails")))

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
