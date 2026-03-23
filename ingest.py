#!/usr/bin/env python3
"""Ingest videos from ./videos through all registered embedding pipelines.

Memory-safe: frames are saved to disk during extraction (with thumbnails),
then each pipeline loads small batches from disk.  Models are unloaded and
garbage-collected between pipeline groups so only one model family is
resident at a time.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Apply MPS compatibility patches before any model loading
from app.utils.mps_compat import apply_mps_patches

apply_mps_patches()

# Ensure pipeline modules are imported (triggers @register decorators)
import app.pipelines  # noqa: F401
from app.config import BASE_DIR, THUMBNAILS_DIR, VIDEOS_DIR, get_device
from app.database import Embedding, SessionLocal, Video, init_db
from app.pipelines.registry import clear_shared_models, get_all_pipelines
from app.utils.perf_logger import PerfLogger
from app.utils.video import cleanup_frame_cache, extract_frames_to_disk

FRAME_CACHE_DIR = BASE_DIR / ".frame_cache"

# Pipeline processing order — shared-model pipelines (jina_single, jina_grid)
# are grouped so the model loads once.  Memory is freed between groups.
PIPELINE_GROUPS = [
    ["pe_core"],
    ["jina_single", "jina_grid"],
    ["jina_native_3d"],
]


def _memory_cleanup() -> None:
    """Force garbage collection and clear accelerator caches."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.synchronize()
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    init_db()
    device = get_device()
    print(f"Device: {device}")

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {VIDEOS_DIR}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s)")

    pipelines = get_all_pipelines(device=device)
    pipeline_order = [p for group in PIPELINE_GROUPS for p in group]
    print(f"Pipelines: {', '.join(pipeline_order)}")

    db = SessionLocal()
    perf = PerfLogger()
    for group in PIPELINE_GROUPS:
        for pname in group:
            p = pipelines[pname]
            perf.add_pipeline(pname, p.display_name)

    try:
        for video_path in tqdm(video_files, desc="Videos", unit="vid"):
            filename = video_path.name

            existing = db.query(Video).filter_by(filename=filename).first()
            if existing:
                print(f"  Skipping {filename} (already ingested)")
                continue

            print(f"\n  Processing {filename} …")

            # --- Phase 1: extract frames + thumbnails to disk ---
            t0 = time.perf_counter()
            frame_paths, thumb_paths, timestamps, duration = extract_frames_to_disk(
                str(video_path), FRAME_CACHE_DIR, THUMBNAILS_DIR, filename
            )
            extract_elapsed = time.perf_counter() - t0
            extract_fps = len(frame_paths) / extract_elapsed if extract_elapsed > 0 else 0
            print(
                f"    {len(frame_paths)} frames extracted ({duration:.1f}s video) "
                f"in {extract_elapsed:.1f}s ({extract_fps:.1f} frames/s)"
            )

            vt = perf.log_extraction(filename, duration, len(frame_paths), extract_elapsed)

            # --- Phase 2: DB record (commit immediately so searches can find the video) ---
            video_record = Video(filename=filename, duration=duration)
            db.add(video_record)
            db.commit()

            # --- Phase 3: embed with each pipeline group sequentially ---
            for group in PIPELINE_GROUPS:
                for pname in group:
                    pipeline = pipelines[pname]
                    print(f"    [{pipeline.display_name}] embedding …")
                    try:
                        pipeline.ensure_loaded()
                        t0 = time.perf_counter()
                        emb_results = pipeline.embed_frames(frame_paths, timestamps)
                        embed_elapsed = time.perf_counter() - t0
                    except Exception as exc:
                        import traceback

                        print(f"      ERROR: {repr(exc)}")
                        traceback.print_exc()
                        continue

                    n_frames = len(frame_paths)
                    embed_fps = n_frames / embed_elapsed if embed_elapsed > 0 else 0
                    perf.log_pipeline(vt, pname, n_frames, len(emb_results), embed_elapsed)

                    for er in emb_results:
                        # Find the thumbnail whose timestamp matches ts_start
                        thumb = thumb_paths[0]
                        for idx, ts in enumerate(timestamps):
                            if abs(ts - er.timestamp_start) < 0.1:
                                thumb = thumb_paths[idx]
                                break

                        db.add(
                            Embedding(
                                video_id=video_record.id,
                                pipeline_name=pname,
                                timestamp_start=er.timestamp_start,
                                timestamp_end=er.timestamp_end,
                                embedding=er.embedding.tolist(),
                                thumbnail_path=thumb,
                            )
                        )

                    # Commit after each pipeline so results are searchable immediately
                    db.commit()

                    print(
                        f"      {len(emb_results)} embeddings saved "
                        f"in {embed_elapsed:.1f}s ({embed_fps:.2f} frames/s)"
                    )

                    # Release this pipeline's model references
                    pipeline.unload()

                # Free shared models between groups and reclaim memory
                clear_shared_models()
                _memory_cleanup()

            # --- Phase 4: clean up cached frames ---
            cleanup_frame_cache(FRAME_CACHE_DIR, filename)
            print(f"  Done: {filename}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # --- Performance report ---
    print(perf.summary(device=device))
    perf.save_json(str(BASE_DIR / "perf_report.json"))
    print(f"\nDetailed report saved to perf_report.json")


if __name__ == "__main__":
    main()
