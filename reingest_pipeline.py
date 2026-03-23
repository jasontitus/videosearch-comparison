#!/usr/bin/env python3
"""Re-ingest videos for a single pipeline (e.g. jina_native_3d).

Downloads videos from GCS, extracts frames, embeds, and saves to Cloud SQL.
Designed to run as a Cloud Run job or locally via cloud-sql-proxy.

Usage:
    python reingest_pipeline.py --pipeline jina_native_3d [--force]

    --force   Delete existing embeddings for this pipeline and re-process all videos.
              Without --force, only processes videos that have no embeddings for this pipeline.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
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


def download_video_from_gcs(filename: str, dest_dir: Path) -> Path:
    """Download a video from GCS to a local temp directory."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"videos/{filename}")
    dest = dest_dir / filename
    blob.download_to_filename(str(dest))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ingest a single pipeline")
    parser.add_argument("--pipeline", required=True, help="Pipeline name to re-ingest")
    parser.add_argument("--force", action="store_true", help="Delete existing embeddings and redo all")
    args = parser.parse_args()

    init_db()
    device = get_device()
    print(f"Device: {device}")
    print(f"Pipeline: {args.pipeline}")
    print(f"GCS bucket: {GCS_BUCKET or '(local)'}")
    print(f"Force: {args.force}")

    pipeline = get_pipeline(args.pipeline, device=device)

    db = SessionLocal()
    try:
        videos = db.query(Video).order_by(Video.id).all()
        print(f"Found {len(videos)} video(s) in database")

        if args.force:
            deleted = db.query(Embedding).filter_by(pipeline_name=args.pipeline).delete()
            db.commit()
            print(f"Deleted {deleted} existing {args.pipeline} embeddings")

        # Find videos that need processing
        to_process = []
        for v in videos:
            count = db.query(Embedding).filter_by(
                video_id=v.id, pipeline_name=args.pipeline
            ).count()
            if count == 0 or args.force:
                to_process.append(v)

        print(f"{len(to_process)} video(s) need processing")
        if not to_process:
            print("Nothing to do!")
            return

        pipeline.ensure_loaded()

        frame_cache = Path(tempfile.mkdtemp(prefix="reingest_frames_"))
        thumb_cache = Path(tempfile.mkdtemp(prefix="reingest_thumbs_"))

        for i, video in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}] {video.filename}")

            # Get the video file
            if GCS_BUCKET:
                video_dir = Path(tempfile.mkdtemp(prefix="reingest_vid_"))
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

            # Embed
            t0 = time.perf_counter()
            try:
                emb_results = pipeline.embed_frames(frame_paths, timestamps)
            except Exception as exc:
                import traceback
                print(f"  ERROR: {repr(exc)}")
                traceback.print_exc()
                cleanup_frame_cache(frame_cache, video.filename)
                if video_dir:
                    shutil.rmtree(video_dir, ignore_errors=True)
                continue

            embed_elapsed = time.perf_counter() - t0
            embed_fps = len(frame_paths) / embed_elapsed if embed_elapsed > 0 else 0
            print(f"  {len(emb_results)} embeddings in {embed_elapsed:.1f}s ({embed_fps:.1f} fps)")

            # Delete old embeddings for this video+pipeline if force
            if args.force:
                db.query(Embedding).filter_by(
                    video_id=video.id, pipeline_name=args.pipeline
                ).delete()

            # Save embeddings
            for er in emb_results:
                thumb = thumb_paths[0]
                for idx, ts in enumerate(timestamps):
                    if abs(ts - er.timestamp_start) < 0.1:
                        thumb = thumb_paths[idx]
                        break

                db.add(Embedding(
                    video_id=video.id,
                    pipeline_name=args.pipeline,
                    timestamp_start=er.timestamp_start,
                    timestamp_end=er.timestamp_end,
                    embedding=er.embedding.tolist(),
                    thumbnail_path=thumb,
                ))
            db.commit()

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
        pipeline.unload()
        gc.collect()

    # Count final embeddings
    db2 = SessionLocal()
    final_count = db2.query(Embedding).filter_by(pipeline_name=args.pipeline).count()
    db2.close()
    print(f"\nComplete! {args.pipeline} now has {final_count} embeddings.")


if __name__ == "__main__":
    main()
