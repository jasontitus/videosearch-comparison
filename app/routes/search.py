from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_device
from app.database import get_db
from app.pipelines.registry import (
    get_pipeline,
    list_pipeline_info,
    list_pipeline_names,
)

router = APIRouter(prefix="/api")


@router.get("/status")
def status(db: Session = Depends(get_db)):
    """Overview of ingested data per pipeline."""
    pipelines = list_pipeline_info()
    for p in pipelines:
        p["count"] = db.execute(
            text("SELECT COUNT(*) FROM embeddings WHERE pipeline_name = :name"),
            {"name": p["name"]},
        ).scalar()

    total_videos = db.execute(text("SELECT COUNT(*) FROM videos")).scalar()
    return {"total_videos": total_videos, "pipelines": pipelines}


@router.get("/search")
def search(q: str, db: Session = Depends(get_db)):
    """Search all pipelines and return top-5 results per pipeline."""
    device = get_device()
    results = {}

    for pipeline_name in list_pipeline_names():
        try:
            pipeline = get_pipeline(pipeline_name, device=device)
            pipeline.ensure_loaded()

            query_vec = pipeline.embed_text(q)

            rows = db.execute(
                text(
                    """
                    SELECT e.id,
                           v.filename,
                           e.timestamp_start,
                           e.timestamp_end,
                           e.thumbnail_path,
                           1 - (e.embedding <=> :qvec ::vector) AS similarity
                    FROM embeddings e
                    JOIN videos v ON v.id = e.video_id
                    WHERE e.pipeline_name = :pname
                    ORDER BY e.embedding <=> :qvec ::vector
                    LIMIT 5
                    """
                ),
                {"qvec": str(query_vec.tolist()), "pname": pipeline_name},
            ).fetchall()

            results[pipeline_name] = {
                "display_name": pipeline.display_name,
                "results": [
                    {
                        "id": r.id,
                        "filename": r.filename,
                        "timestamp_start": r.timestamp_start,
                        "timestamp_end": r.timestamp_end,
                        "thumbnail_path": r.thumbnail_path,
                        "similarity": round(float(r.similarity), 4),
                    }
                    for r in rows
                ],
            }
        except Exception as exc:
            db.rollback()
            results[pipeline_name] = {
                "display_name": pipeline_name,
                "error": str(exc),
                "results": [],
            }

    return results
