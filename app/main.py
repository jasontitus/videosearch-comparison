from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import THUMBNAILS_DIR, VIDEOS_DIR
from app.database import init_db
from app.routes.search import router as search_router
from app.utils.mps_compat import apply_mps_patches

# Apply MPS compatibility patches before any model loading
apply_mps_patches()

# Force pipeline registration on import
import app.pipelines  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Video Search Eval Workbench", lifespan=lifespan)

# --- API routes (must be registered before catch-all mounts) ---
app.include_router(search_router)

# --- static file mounts ---
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
