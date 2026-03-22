"""Utilities for frame extraction and thumbnail generation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image


def extract_frames_to_disk(
    video_path: str,
    frames_dir: Path,
    thumbnails_dir: Path,
    video_filename: str,
    fps: int = 1,
    thumb_size: Tuple[int, int] = (320, 180),
) -> Tuple[List[Path], List[str], List[float], float]:
    """Extract frames from *video_path*, saving full-res frames and thumbnails to disk.

    Returns (frame_paths, thumb_rel_paths, timestamps, duration).
    Only one frame is held in memory at a time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0.0
    frame_interval = max(1, int(round(video_fps / fps)))

    stem = Path(video_filename).stem
    frame_out = frames_dir / stem
    thumb_out = thumbnails_dir / stem
    frame_out.mkdir(parents=True, exist_ok=True)
    thumb_out.mkdir(parents=True, exist_ok=True)

    frame_paths: List[Path] = []
    thumb_rel_paths: List[str] = []
    timestamps: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = round(frame_idx / video_fps, 2)
            fname = f"{ts:.2f}.jpg"

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Save full-resolution frame
            frame_path = frame_out / fname
            img.save(frame_path, "JPEG", quality=95)
            frame_paths.append(frame_path)

            # Save thumbnail (thumbnail modifies in-place, no copy needed)
            img.thumbnail(thumb_size)
            img.save(thumb_out / fname, "JPEG", quality=85)
            thumb_rel_paths.append(f"{stem}/{fname}")

            timestamps.append(ts)
            del img

        frame_idx += 1

    cap.release()
    return frame_paths, thumb_rel_paths, timestamps, duration


def cleanup_frame_cache(frames_dir: Path, video_filename: str) -> None:
    """Remove cached full-resolution frames for a video."""
    stem = Path(video_filename).stem
    target = frames_dir / stem
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
