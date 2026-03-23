"""Video upload and management endpoints."""

import os

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import GCS_BUCKET, VIDEOS_DIR
from app.database import Video, get_db

# Cloud Run job to trigger after upload (set via env var so it's optional)
INGEST_JOB_NAME = os.getenv("INGEST_JOB_NAME", "ingest-new")
INGEST_JOB_REGION = os.getenv("INGEST_JOB_REGION", "europe-west1")

router = APIRouter(prefix="/api")


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """List all videos with per-pipeline embedding counts."""
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    result = []
    for v in videos:
        counts = db.execute(
            text(
                "SELECT pipeline_name, COUNT(*) as cnt "
                "FROM embeddings WHERE video_id = :vid "
                "GROUP BY pipeline_name"
            ),
            {"vid": v.id},
        ).fetchall()
        result.append(
            {
                "id": v.id,
                "filename": v.filename,
                "duration": v.duration,
                "pipelines": {r.pipeline_name: r.cnt for r in counts},
            }
        )
    return result


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a video file. Stores to GCS in cloud mode, local filesystem otherwise."""
    if not file.filename or not file.filename.endswith(".mp4"):
        return {"error": "Only .mp4 files are accepted"}

    filename = file.filename

    # Check for duplicate
    existing = db.query(Video).filter_by(filename=filename).first()
    if existing:
        return {"error": f"Video '{filename}' already exists", "id": existing.id}

    if GCS_BUCKET:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"videos/{filename}")
        blob.upload_from_file(file.file, content_type="video/mp4")
    else:
        dest = VIDEOS_DIR / filename
        with open(dest, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)

    video = Video(filename=filename)
    db.add(video)
    db.commit()

    # Trigger ingest job in the background
    ingest_triggered = False
    if GCS_BUCKET:
        try:
            ingest_triggered = _trigger_ingest_job()
        except Exception as exc:
            print(f"[upload] Failed to trigger ingest job: {exc}")

    return {
        "id": video.id,
        "filename": filename,
        "status": "uploaded",
        "ingest_triggered": ingest_triggered,
    }


def _trigger_ingest_job() -> bool:
    """Trigger the ingest-new Cloud Run job via the Cloud Run Admin API."""
    try:
        from google.cloud import run_v2

        client = run_v2.JobsClient()
        job_name = f"projects/{os.environ.get('GOOGLE_CLOUD_PROJECT', 'videosearch-comparison')}/locations/{INGEST_JOB_REGION}/jobs/{INGEST_JOB_NAME}"
        client.run_job(name=job_name)
        print(f"[upload] Triggered ingest job: {INGEST_JOB_NAME}")
        return True
    except Exception as exc:
        print(f"[upload] Could not trigger ingest job: {exc}")
        return False
