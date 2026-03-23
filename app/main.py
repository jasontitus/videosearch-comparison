from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import GCS_BUCKET, THUMBNAILS_DIR, VIDEOS_DIR
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

# --- Upload route ---
from app.routes.upload import router as upload_router  # noqa: E402

app.include_router(upload_router)

if GCS_BUCKET:
    # Cloud mode: redirect media requests to GCS public URLs
    GCS_BASE = f"https://storage.googleapis.com/{GCS_BUCKET}"

    @app.get("/videos/{filepath:path}")
    def serve_video(filepath: str):
        from urllib.parse import quote

        return RedirectResponse(f"{GCS_BASE}/videos/{quote(filepath)}")

    @app.get("/thumbnails/{filepath:path}")
    def serve_thumbnail(filepath: str):
        from urllib.parse import quote

        return RedirectResponse(f"{GCS_BASE}/thumbnails/{quote(filepath)}")
else:
    # Local mode: serve from filesystem
    app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
    app.mount(
        "/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails"
    )

# --- Static assets (always from container/local) ---
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
