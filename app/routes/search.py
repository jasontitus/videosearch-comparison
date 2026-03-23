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

    from app.config import GCS_BUCKET

    result = {"total_videos": total_videos, "pipelines": pipelines}
    if GCS_BUCKET:
        result["media_base"] = f"https://storage.googleapis.com/{GCS_BUCKET}"
    return result


@router.get("/search")
def search(q: str, db: Session = Depends(get_db)):
    """Search all pipelines and return top-5 results per pipeline.

    Pipelines that share the same MODEL_ID (e.g. the three Jina v4 variants)
    are grouped so the model loads once, the text query is encoded once, and
    the same query vector is reused for all DB searches in that group.

    Models stay resident in GPU memory between requests (~15.6GB total for
    Jina v4 float16 + PE-Core, well within the L4's 24GB).
    """
    device = get_device()
    results = {}

    # Group pipelines by MODEL_ID so we load each model only once
    groups: dict[str, list[str]] = {}
    for pname in list_pipeline_names():
        pipeline = get_pipeline(pname, device=device)
        model_id = getattr(pipeline, "MODEL_ID", pname)
        groups.setdefault(model_id, []).append(pname)

    for model_id, pipeline_names in groups.items():
        # Pick the first pipeline in the group as the text encoder
        encoder_pipeline = get_pipeline(pipeline_names[0], device=device)

        try:
            encoder_pipeline.ensure_loaded()
            query_vec = encoder_pipeline.embed_text(q)
        except Exception as exc:
            for pname in pipeline_names:
                p = get_pipeline(pname, device=device)
                results[pname] = {
                    "display_name": p.display_name,
                    "error": str(exc),
                    "results": [],
                }
            continue

        # Search each pipeline in this group with the shared query vector
        for pname in pipeline_names:
            pipeline = get_pipeline(pname, device=device)
            try:
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
                    {"qvec": str(query_vec.tolist()), "pname": pname},
                ).fetchall()

                results[pname] = {
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
                results[pname] = {
                    "display_name": pipeline.display_name,
                    "error": str(exc),
                    "results": [],
                }

    return results


@router.get("/benchmark")
def benchmark(q: str = "a dog running in a field"):
    """Compare text encoding across dtypes (float32, bfloat16, float16).

    Loads jina_native_3d at each dtype, encodes the query, reports timing
    and cosine similarity vs float32 baseline. Unloads between each run.
    """
    import os
    import time

    import numpy as np

    from app.pipelines.registry import clear_shared_models

    device = get_device()
    dtypes = ["float32", "bfloat16", "float16"]
    embeddings = {}
    report = {"query": q, "device": device, "results": []}

    for dtype_name in dtypes:
        os.environ["JINA_DTYPE"] = dtype_name

        # Force a fresh pipeline instance
        from app.pipelines.registry import _instances
        _instances.pop("jina_native_3d", None)

        pipeline = get_pipeline("jina_native_3d", device=device)

        try:
            t0 = time.perf_counter()
            pipeline.ensure_loaded()
            load_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            emb = pipeline.embed_text(q)
            encode_time = time.perf_counter() - t0

            embeddings[dtype_name] = emb
            entry = {
                "dtype": dtype_name,
                "load_time_s": round(load_time, 2),
                "encode_time_s": round(encode_time, 4),
                "emb_norm": round(float(np.linalg.norm(emb)), 6),
                "emb_first5": [round(float(x), 6) for x in emb[:5]],
            }

            # Cosine similarity vs float32 baseline
            if "float32" in embeddings and dtype_name != "float32":
                baseline = embeddings["float32"]
                cosine = float(np.dot(emb, baseline) / (
                    np.linalg.norm(emb) * np.linalg.norm(baseline)
                ))
                entry["cosine_vs_fp32"] = round(cosine, 8)
                entry["max_abs_diff"] = round(float(np.max(np.abs(emb - baseline))), 8)

            report["results"].append(entry)
        except Exception as exc:
            report["results"].append({"dtype": dtype_name, "error": str(exc)})
        finally:
            pipeline.unload()
            _instances.pop("jina_native_3d", None)
            clear_shared_models()
            _free_accelerator_memory(device)

    # Restore default
    os.environ["JINA_DTYPE"] = "bfloat16"

    return report


def _free_accelerator_memory(device: str) -> None:
    """GC + clear CUDA/MPS caches to reclaim GPU memory between pipelines."""
    import gc

    from app.pipelines.registry import clear_shared_models

    clear_shared_models()
    gc.collect()
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except ImportError:
        pass
