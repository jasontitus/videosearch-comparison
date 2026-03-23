#!/usr/bin/env python3
"""Ingest new videos — processes videos that are missing embeddings for any pipeline.

Downloads videos from GCS, extracts frames, runs all pipelines, and saves
embeddings to Cloud SQL. Pipelines are run in memory-safe groups (same as
ingest.py) so only one model family is resident at a time.

Designed to run as a Cloud Run job.

Usage:
    python ingest_new.py
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from pathlib import Path

from app.utils.mps_compat import apply_mps_patches

apply_mps_patches()

import app.pipelines  # noqa: F401, E402
from app.config import BASE_DIR, GCS_BUCKET, get_device  # noqa: E402
from app.database import Embedding, SessionLocal, Video, init_db  # noqa: E402
from app.pipelines.registry import get_pipeline  # noqa: E402
from app.utils.video import cleanup_frame_cache, extract_frames_to_disk  # noqa: E402

# Pipeline processing order — shared-model pipelines grouped together.
# Memory is freed between groups.
PIPELINE_GROUPS = [
    ["pe_core"],
    ["jina_single", "jina_grid"],
    ["jina_native_3d"],
]


def download_video_from_gcs(filename: str, dest_dir: Path) -> Path:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"videos/{filename}")
    dest = dest_dir / filename
    blob.download_to_filename(str(dest))
    return dest


def upload_thumbnails_to_gcs(thumb_dir: Path, video_filename: str) -> None:
    """Upload generated thumbnails to GCS so they're accessible via the CDN."""
    from google.cloud import storage

    stem = Path(video_filename).stem
    local_thumb_dir = thumb_dir / stem
    if not local_thumb_dir.exists():
        return

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    for thumb_file in sorted(local_thumb_dir.iterdir()):
        blob = bucket.blob(f"thumbnails/{stem}/{thumb_file.name}")
        blob.upload_from_filename(str(thumb_file), content_type="image/jpeg")
    print(f"  Uploaded {len(list(local_thumb_dir.iterdir()))} thumbnails to GCS")


def _memory_cleanup(device: str) -> None:
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


def main() -> None:
    init_db()
    device = get_device()
    print(f"Device: {device}")
    print(f"GCS bucket: {GCS_BUCKET or '(local)'}")

    db = SessionLocal()
    try:
        videos = db.query(Video).order_by(Video.id).all()
        print(f"Found {len(videos)} video(s) in database")

        all_pipelines = [p for group in PIPELINE_GROUPS for p in group]

        # Find videos missing embeddings for any pipeline
        to_process = []
        for v in videos:
            missing = []
            for pname in all_pipelines:
                count = db.query(Embedding).filter_by(
                    video_id=v.id, pipeline_name=pname
                ).count()
                if count == 0:
                    missing.append(pname)
            if missing:
                to_process.append((v, missing))

        if not to_process:
            print("All videos fully ingested. Nothing to do!")
            return

        print(f"{len(to_process)} video(s) need processing:")
        for v, missing in to_process:
            print(f"  {v.filename}: missing {', '.join(missing)}")

        frame_cache = Path(tempfile.mkdtemp(prefix="ingest_frames_"))
        thumb_cache = Path(tempfile.mkdtemp(prefix="ingest_thumbs_"))

        for vi, (video, missing_pipelines) in enumerate(to_process):
            print(f"\n[{vi+1}/{len(to_process)}] {video.filename}")

            # Download video
            if GCS_BUCKET:
                video_dir = Path(tempfile.mkdtemp(prefix="ingest_vid_"))
                print(f"  Downloading from GCS...")
                t0 = time.perf_counter()
                video_path = download_video_from_gcs(video.filename, video_dir)
                print(f"  Downloaded in {time.perf_counter()-t0:.1f}s")
            else:
                video_path = BASE_DIR / "videos" / video.filename
                video_dir = None

            # Extract frames
            t0 = time.perf_counter()
            frame_paths, thumb_paths, timestamps, duration = extract_frames_to_disk(
                str(video_path), frame_cache, thumb_cache, video.filename
            )
            print(f"  {len(frame_paths)} frames extracted in {time.perf_counter()-t0:.1f}s")

            # Upload thumbnails to GCS if needed
            if GCS_BUCKET:
                upload_thumbnails_to_gcs(thumb_cache, video.filename)

            # Update video duration if not set
            if not video.duration:
                video.duration = duration
                db.commit()

            # Process each pipeline group
            for group in PIPELINE_GROUPS:
                group_pipelines = [p for p in group if p in missing_pipelines]
                if not group_pipelines:
                    continue

                for pname in group_pipelines:
                    pipeline = get_pipeline(pname, device=device)
                    print(f"  [{pipeline.display_name}] embedding...")

                    try:
                        pipeline.ensure_loaded()
                        t0 = time.perf_counter()
                        emb_results = pipeline.embed_frames(frame_paths, timestamps)
                        embed_elapsed = time.perf_counter() - t0
                    except Exception as exc:
                        import traceback
                        print(f"    ERROR: {repr(exc)}")
                        traceback.print_exc()
                        continue

                    embed_fps = len(frame_paths) / embed_elapsed if embed_elapsed > 0 else 0
                    print(f"    {len(emb_results)} embeddings in {embed_elapsed:.1f}s ({embed_fps:.1f} fps)")

                    for er in emb_results:
                        thumb = thumb_paths[0]
                        for idx, ts in enumerate(timestamps):
                            if abs(ts - er.timestamp_start) < 0.1:
                                thumb = thumb_paths[idx]
                                break

                        db.add(Embedding(
                            video_id=video.id,
                            pipeline_name=pname,
                            timestamp_start=er.timestamp_start,
                            timestamp_end=er.timestamp_end,
                            embedding=er.embedding.tolist(),
                            thumbnail_path=thumb,
                        ))
                    db.commit()
                    pipeline.unload()

                from app.pipelines.registry import clear_shared_models
                clear_shared_models()
                _memory_cleanup(device)

            # Cleanup
            cleanup_frame_cache(frame_cache, video.filename)
            if video_dir:
                shutil.rmtree(video_dir, ignore_errors=True)
            print(f"  Done: {video.filename}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Summary
    db2 = SessionLocal()
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    for pname in all_pipelines:
        count = db2.query(Embedding).filter_by(pipeline_name=pname).count()
        print(f"  {pname}: {count} embeddings")
    db2.close()


if __name__ == "__main__":
    main()
