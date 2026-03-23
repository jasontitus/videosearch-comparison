"""Performance logging and cost estimation for ingestion runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


# Cloud Run pricing (us-central1, as of 2026-03)
# https://cloud.google.com/run/pricing
CLOUD_RUN_GPU_L4_PER_SEC = 0.000233  # $0.8388/hr for L4 GPU
CLOUD_RUN_VCPU_PER_SEC = 0.0000240   # $0.0864/hr per vCPU
CLOUD_RUN_MEM_GIB_PER_SEC = 0.0000025  # $0.009/hr per GiB
DEFAULT_VCPUS = 4
DEFAULT_MEM_GIB = 16


@dataclass
class PipelineTiming:
    name: str
    display_name: str
    total_frames: int = 0
    total_embeddings: int = 0
    total_seconds: float = 0.0
    video_timings: list = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.total_frames / self.total_seconds if self.total_seconds > 0 else 0

    @property
    def sec_per_frame(self) -> float:
        return self.total_seconds / self.total_frames if self.total_frames > 0 else 0


@dataclass
class VideoTiming:
    filename: str
    duration_sec: float
    num_frames: int
    extraction_sec: float
    pipeline_timings: dict = field(default_factory=dict)  # pipeline_name -> seconds

    @property
    def total_sec(self) -> float:
        return self.extraction_sec + sum(self.pipeline_timings.values())


class PerfLogger:
    def __init__(self):
        self.pipelines: dict[str, PipelineTiming] = {}
        self.videos: list[VideoTiming] = []
        self.run_start = time.time()

    def add_pipeline(self, name: str, display_name: str):
        if name not in self.pipelines:
            self.pipelines[name] = PipelineTiming(name=name, display_name=display_name)

    def log_extraction(self, filename: str, duration: float, num_frames: int, elapsed: float):
        vt = VideoTiming(
            filename=filename, duration_sec=duration,
            num_frames=num_frames, extraction_sec=elapsed,
        )
        self.videos.append(vt)
        return vt

    def log_pipeline(self, video_timing: VideoTiming, pipeline_name: str,
                     num_frames: int, num_embeddings: int, elapsed: float):
        video_timing.pipeline_timings[pipeline_name] = elapsed
        pt = self.pipelines[pipeline_name]
        pt.total_frames += num_frames
        pt.total_embeddings += num_embeddings
        pt.total_seconds += elapsed
        pt.video_timings.append({
            "filename": video_timing.filename,
            "frames": num_frames,
            "embeddings": num_embeddings,
            "seconds": round(elapsed, 1),
            "fps": round(num_frames / elapsed, 3) if elapsed > 0 else 0,
        })

    def summary(self, device: str = "unknown") -> str:
        total_elapsed = time.time() - self.run_start
        total_video_duration = sum(v.duration_sec for v in self.videos)
        total_frames = sum(v.num_frames for v in self.videos)

        lines = [
            "",
            "=" * 70,
            f"INGESTION PERFORMANCE REPORT",
            f"Device: {device}  |  Videos: {len(self.videos)}  |  "
            f"Frames: {total_frames}  |  Wall time: {total_elapsed:.0f}s",
            f"Total video duration: {total_video_duration:.0f}s "
            f"({total_video_duration / 60:.1f} min)",
            "=" * 70,
            "",
            f"{'Pipeline':<30} {'Frames':>7} {'Embeds':>7} {'Time':>8} "
            f"{'FPS':>7} {'s/frame':>8} {'s/min-video':>12}",
            "-" * 70,
        ]

        for pt in self.pipelines.values():
            if pt.total_seconds == 0:
                lines.append(f"{pt.display_name:<30} {'(no data)':>7}")
                continue
            # How many seconds of processing per minute of source video
            sec_per_min_video = (pt.total_seconds / total_video_duration * 60
                                 if total_video_duration > 0 else 0)
            lines.append(
                f"{pt.display_name:<30} {pt.total_frames:>7} "
                f"{pt.total_embeddings:>7} {pt.total_seconds:>7.0f}s "
                f"{pt.fps:>7.2f} {pt.sec_per_frame:>7.2f}s "
                f"{sec_per_min_video:>11.1f}s"
            )

        # Cost estimation
        lines.extend([
            "",
            "COST ESTIMATE (Cloud Run with L4 GPU, "
            f"{DEFAULT_VCPUS} vCPUs, {DEFAULT_MEM_GIB} GiB)",
            "-" * 70,
        ])

        cost_per_sec = (
            CLOUD_RUN_GPU_L4_PER_SEC
            + DEFAULT_VCPUS * CLOUD_RUN_VCPU_PER_SEC
            + DEFAULT_MEM_GIB * CLOUD_RUN_MEM_GIB_PER_SEC
        )

        for pt in self.pipelines.values():
            if pt.total_seconds == 0:
                continue
            cost = pt.total_seconds * cost_per_sec
            cost_per_min_video = (cost / total_video_duration * 60
                                  if total_video_duration > 0 else 0)
            lines.append(
                f"{pt.display_name:<30} "
                f"${cost:>7.4f} total  |  "
                f"${cost_per_min_video:.4f}/min of video"
            )

        total_processing = sum(pt.total_seconds for pt in self.pipelines.values())
        total_cost = total_processing * cost_per_sec
        total_cost_per_min = (total_cost / total_video_duration * 60
                              if total_video_duration > 0 else 0)
        lines.extend([
            "-" * 70,
            f"{'TOTAL':<30} ${total_cost:>7.4f} total  |  "
            f"${total_cost_per_min:.4f}/min of video",
            f"{'(Cloud Run rate)':<30} ${cost_per_sec * 3600:.2f}/hr",
            "=" * 70,
        ])

        return "\n".join(lines)

    def save_json(self, path: str):
        data = {
            "run_start": self.run_start,
            "total_elapsed": time.time() - self.run_start,
            "videos": [
                {
                    "filename": v.filename,
                    "duration_sec": v.duration_sec,
                    "num_frames": v.num_frames,
                    "extraction_sec": v.extraction_sec,
                    "pipeline_timings": v.pipeline_timings,
                    "total_sec": v.total_sec,
                }
                for v in self.videos
            ],
            "pipelines": {
                name: {
                    "display_name": pt.display_name,
                    "total_frames": pt.total_frames,
                    "total_embeddings": pt.total_embeddings,
                    "total_seconds": pt.total_seconds,
                    "fps": pt.fps,
                    "sec_per_frame": pt.sec_per_frame,
                    "per_video": pt.video_timings,
                }
                for name, pt in self.pipelines.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))
