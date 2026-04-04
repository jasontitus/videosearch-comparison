#!/usr/bin/env python3
"""Ingest videos with PE-Core + Gemma 4 (AV and V-only), then compare search results.

Self-contained script that:
1. Sets up the database tables
2. Extracts frames from all videos
3. Ingests with PE-Core (fast baseline)
4. Ingests with Gemma 4 in two modes: video-only and audio+video
5. Runs search queries and compares results across all 3 pipelines

Usage: .venv-gemma4/bin/python run_comparison.py
"""

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_URL = "postgresql://videosearch:videosearch@localhost:5433/videosearch"
os.environ["DATABASE_URL"] = DB_URL

PROJECT_DIR = Path(__file__).parent
VIDEOS_DIR = PROJECT_DIR / "videos"
FRAME_CACHE = PROJECT_DIR / ".frame_cache"
THUMBNAILS_DIR = PROJECT_DIR / "thumbnails"

GEMMA_MODEL_ID = "mlx-community/gemma-4-e4b-it-bf16"
PRE_CONTEXT = 2   # seconds
EMBED_WINDOW = 3  # seconds
SEARCH_QUERIES = ["dog", "running", "rifle", "police", "car", "person speaking"]


def banner(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}\n")


def force_eval(tensor):
    """Force MLX lazy graph evaluation."""
    mx.eval(tensor)
    return tensor


def mx_to_numpy(tensor):
    """Convert MLX array to numpy, handling bfloat16."""
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float32)
        force_eval(tensor)
    return np.array(tensor)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
def setup_db():
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
    from sqlalchemy.orm import declarative_base, sessionmaker
    from pgvector.sqlalchemy import Vector
    from datetime import datetime

    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()

    class Video(Base):
        __tablename__ = "videos"
        id = Column(Integer, primary_key=True)
        filename = Column(String, unique=True, nullable=False)
        duration = Column(Float)
        created_at = Column(DateTime, default=datetime.utcnow)

    class Embedding(Base):
        __tablename__ = "embeddings"
        id = Column(Integer, primary_key=True)
        video_id = Column(Integer, nullable=False)
        pipeline_name = Column(String, nullable=False)
        timestamp_start = Column(Float, nullable=False)
        timestamp_end = Column(Float)
        embedding = Column(Vector())
        thumbnail_path = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)

    return engine, Session, Video, Embedding


# ---------------------------------------------------------------------------
# Frame extraction (from app/utils/video.py logic)
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, fps: int = 1):
    """Extract frames at 1 FPS, return (frame_paths, timestamps, duration)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0.0
    frame_interval = max(1, int(round(video_fps / fps)))

    stem = Path(video_path).stem
    frame_out = FRAME_CACHE / stem
    frame_out.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    timestamps = []

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
            frame_path = frame_out / fname
            img.save(frame_path, "JPEG", quality=95)
            frame_paths.append(frame_path)
            timestamps.append(ts)
            del img
        frame_idx += 1

    cap.release()
    return frame_paths, timestamps, duration


# ---------------------------------------------------------------------------
# PE-Core pipeline (Meta Perception Encoder CLIP)
# ---------------------------------------------------------------------------
class PECorePipeline:
    name = "pe_core"
    display_name = "PE-Core L14"
    MODEL_NAME = "PE-Core-L14-336"
    BATCH_SIZE = 16

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        import torch
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        print(f"  Loading {self.MODEL_NAME}...")
        model = pe.CLIP.from_config(self.MODEL_NAME, pretrained=True)
        try:
            model = model.half().to("mps").eval()
        except NotImplementedError:
            model = model.to_empty(device="mps", dtype=torch.float16)
            checkpoint = pe.CLIP.from_config(self.MODEL_NAME, pretrained=True)
            model.load_state_dict(checkpoint.state_dict(), assign=True)
            model = model.half().to("mps").eval()
            del checkpoint
        self.model = model
        self.preprocess = pe_transforms.get_image_transform(model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
        print(f"  PE-Core loaded on MPS")

    def embed_frames(self, frame_paths, timestamps):
        import torch

        self.load()
        results = []
        for i in range(0, len(frame_paths), self.BATCH_SIZE):
            batch_paths = frame_paths[i:i + self.BATCH_SIZE]
            batch_ts = timestamps[i:i + self.BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            tensors = torch.stack([self.preprocess(img) for img in images]).half().to("mps")
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings = feats.cpu().float().numpy()
            for j, emb in enumerate(embeddings):
                ts = batch_ts[j]
                results.append((ts, ts + 1.0, emb))
            del images, tensors, feats
        return results

    def embed_text(self, text):
        import torch

        self.load()
        tokens = self.tokenizer([text]).to("mps")
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.squeeze(0).cpu().float().numpy()

    def unload(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        gc.collect()
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Gemma 4 pipeline
# ---------------------------------------------------------------------------
class Gemma4Pipeline:
    def __init__(self, name, display_name, include_audio=True):
        self.name = name
        self.display_name = display_name
        self.include_audio = include_audio
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        from mlx_vlm import load
        print(f"  Loading {GEMMA_MODEL_ID}...")
        self.model, self.processor = load(GEMMA_MODEL_ID)
        print(f"  Gemma 4 E4B loaded (audio={'ON' if self.include_audio else 'OFF'})")

    def _extract_audio_segment(self, video_path, start_sec, duration_sec):
        """Extract a WAV audio segment via ffmpeg. Returns path or None."""
        import tempfile
        audio_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-ss", str(start_sec), "-t", str(duration_sec),
                    "-ar", "16000", "-ac", "1",
                    "-f", "wav", "-y", "-loglevel", "error",
                    str(audio_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or audio_path.stat().st_size < 100:
                audio_path.unlink(missing_ok=True)
                return None
            return audio_path
        except Exception:
            audio_path.unlink(missing_ok=True)
            return None

    def _embed_window(self, frame_paths, video_path=None, window_start_sec=0.0):
        """Embed a window of frames (+ optional audio) via hidden state extraction."""
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # Build message content
        content = []

        # Audio (if enabled and available)
        audio_path = None
        if self.include_audio and video_path:
            total_window = len(frame_paths)  # 1 FPS, so num frames ~ seconds
            audio_path = self._extract_audio_segment(
                video_path, window_start_sec, total_window
            )
            if audio_path:
                content.append({"type": "audio", "audio": str(audio_path)})

        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": "Describe what is happening in these frames."})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        proc_kwargs = {"text": prompt, "images": images, "return_tensors": "np"}
        if audio_path:
            proc_kwargs["audios"] = [str(audio_path)]

        try:
            inputs = self.processor(**proc_kwargs)
        except Exception:
            # Fallback: try without audio if audio processing fails
            if audio_path:
                content_no_audio = [c for c in content if c.get("type") != "audio"]
                messages_no_audio = [{"role": "user", "content": content_no_audio}]
                prompt_no_audio = self.processor.tokenizer.apply_chat_template(
                    messages_no_audio, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(text=prompt_no_audio, images=images, return_tensors="np")
            else:
                raise

        input_ids = mx.array(inputs["input_ids"])

        pixel_values = None
        for key in ["pixel_values", "pixel_values_images"]:
            if key in inputs:
                pv = inputs[key]
                pixel_values = [mx.array(p) for p in pv] if isinstance(pv, list) else mx.array(pv)
                break

        # Pass through any extra inputs (audio features, grid_thw, etc.)
        extra_kwargs = {}
        for key in inputs:
            if key not in ["input_ids", "attention_mask", "pixel_values", "pixel_values_images"]:
                val = inputs[key]
                if isinstance(val, np.ndarray):
                    extra_kwargs[key] = mx.array(val)
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
                    extra_kwargs[key] = [mx.array(v) for v in val]

        emb_features = self.model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **extra_kwargs,
        )

        embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
        kwargs_fwd = {}
        if hasattr(emb_features, "per_layer_inputs"):
            kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

        hidden_states = self.model.language_model.model(
            inputs=None,
            inputs_embeds=embeds,
            **kwargs_fwd,
        )
        force_eval(hidden_states)

        # Mean pool over all tokens, then normalize
        pooled = mx.mean(hidden_states, axis=1)
        norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normalized = pooled / norm
        force_eval(normalized)

        embedding = mx_to_numpy(normalized).flatten()

        # Cleanup
        if audio_path:
            audio_path.unlink(missing_ok=True)
        del images

        return embedding

    def embed_frames(self, frame_paths, timestamps, video_path=None):
        """Embed frames using rolling window: 2s pre-context + 3s embed window."""
        self.load()
        results = []
        n = len(frame_paths)

        stride = EMBED_WINDOW
        for win_start in range(0, n, stride):
            # Pre-context: up to PRE_CONTEXT frames before the embed window
            pre_start = max(0, win_start - PRE_CONTEXT)
            # Embed window: EMBED_WINDOW frames starting at win_start
            win_end = min(win_start + EMBED_WINDOW, n)

            if win_start >= n:
                break

            window_paths = frame_paths[pre_start:win_end]
            if len(window_paths) == 0:
                continue

            ts_start = timestamps[win_start]
            ts_end = timestamps[win_end - 1] + 1.0
            window_start_sec = timestamps[pre_start]

            try:
                emb = self._embed_window(window_paths, video_path, window_start_sec)
                if not np.isnan(emb).any():
                    results.append((ts_start, ts_end, emb))
                else:
                    print(f"        NaN at t={ts_start:.0f}s, skipping")
            except Exception as e:
                print(f"        Error at t={ts_start:.0f}s: {e}")

        return results

    def embed_text(self, text):
        """Encode a text query via hidden state extraction."""
        self.load()
        messages = [{"role": "user", "content": text}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        emb_features = self.model.get_input_embeddings(input_ids=input_ids)
        embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
        kwargs_fwd = {}
        if hasattr(emb_features, "per_layer_inputs"):
            kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

        hidden_states = self.model.language_model.model(
            inputs=None, inputs_embeds=embeds, **kwargs_fwd,
        )
        force_eval(hidden_states)

        # Last-token pooling for text
        last_token = hidden_states[:, -1, :]
        norm = mx.linalg.norm(last_token, axis=-1, keepdims=True)
        normalized = last_token / norm
        force_eval(normalized)

        return mx_to_numpy(normalized).flatten()

    def unload(self):
        self.model = None
        self.processor = None
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    banner("VIDEO SEARCH COMPARISON: PE-Core vs Gemma 4 (V) vs Gemma 4 (AV)")

    # Setup DB
    print("Setting up database...")
    engine, Session, Video, Embedding = setup_db()
    db = Session()

    # Find videos
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files in {VIDEOS_DIR}")
        sys.exit(1)
    print(f"Found {len(video_files)} videos")

    # Pipelines
    pe_core = PECorePipeline()
    gemma4_v = Gemma4Pipeline("gemma4_v", "Gemma 4 E4B (Video Only)", include_audio=False)
    gemma4_av = Gemma4Pipeline("gemma4_av", "Gemma 4 E4B (Audio+Video)", include_audio=True)

    all_pipelines = [pe_core, gemma4_v, gemma4_av]

    # -----------------------------------------------------------------------
    # Phase 1: Extract frames
    # -----------------------------------------------------------------------
    banner("PHASE 1: Frame Extraction")
    video_data = {}
    for vf in video_files:
        stem = vf.stem
        cached = FRAME_CACHE / stem
        if cached.exists() and len(list(cached.glob("*.jpg"))) > 0:
            frame_paths = sorted(cached.glob("*.jpg"))
            timestamps = [float(p.stem) for p in frame_paths]
            import cv2
            cap = cv2.VideoCapture(str(vf))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"  {vf.name[:60]}: {len(frame_paths)} frames (cached)")
        else:
            t0 = time.perf_counter()
            frame_paths, timestamps, duration = extract_frames(str(vf))
            elapsed = time.perf_counter() - t0
            print(f"  {vf.name[:60]}: {len(frame_paths)} frames in {elapsed:.1f}s ({duration:.0f}s video)")

        video_data[vf.name] = {
            "path": vf,
            "frame_paths": frame_paths,
            "timestamps": timestamps,
            "duration": duration,
        }

        # Create video record if not exists
        existing = db.query(Video).filter_by(filename=vf.name).first()
        if not existing:
            db.add(Video(filename=vf.name, duration=duration))
            db.commit()

    total_frames = sum(len(v["frame_paths"]) for v in video_data.values())
    print(f"\n  Total: {total_frames} frames across {len(video_files)} videos")

    # -----------------------------------------------------------------------
    # Phase 2: Ingest with each pipeline
    # -----------------------------------------------------------------------
    for pipeline in all_pipelines:
        banner(f"PHASE 2: Ingesting with {pipeline.display_name}")

        # Check if already ingested
        from sqlalchemy import text as sql_text
        count = db.execute(
            sql_text("SELECT COUNT(*) FROM embeddings WHERE pipeline_name = :name"),
            {"name": pipeline.name},
        ).scalar()
        if count > 0:
            print(f"  Already have {count} embeddings, skipping")
            continue

        t0_total = time.perf_counter()
        total_embeddings = 0

        for vname, vdata in video_data.items():
            video_record = db.query(Video).filter_by(filename=vname).first()
            t0 = time.perf_counter()

            if isinstance(pipeline, Gemma4Pipeline):
                emb_results = pipeline.embed_frames(
                    vdata["frame_paths"], vdata["timestamps"],
                    video_path=vdata["path"],
                )
            else:
                emb_results = pipeline.embed_frames(
                    vdata["frame_paths"], vdata["timestamps"],
                )

            elapsed = time.perf_counter() - t0
            n_frames = len(vdata["frame_paths"])
            fps = n_frames / elapsed if elapsed > 0 else 0

            for ts_start, ts_end, emb in emb_results:
                db.add(Embedding(
                    video_id=video_record.id,
                    pipeline_name=pipeline.name,
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    embedding=emb.tolist(),
                ))

            db.commit()
            total_embeddings += len(emb_results)
            print(f"  {vname[:50]}: {len(emb_results)} embeddings in {elapsed:.1f}s ({fps:.2f} fps)")

        elapsed_total = time.perf_counter() - t0_total
        print(f"\n  Total: {total_embeddings} embeddings in {elapsed_total:.1f}s")

        # Unload to free memory for next pipeline
        pipeline.unload()

    # -----------------------------------------------------------------------
    # Phase 3: Search comparison
    # -----------------------------------------------------------------------
    banner("PHASE 3: Search Comparison")

    from sqlalchemy import text as sql_text

    # Load pipelines for text encoding
    pe_core_search = PECorePipeline()
    gemma4_search = Gemma4Pipeline("gemma4_search", "search", include_audio=False)

    for query in SEARCH_QUERIES:
        print(f"\n  Query: '{query}'")
        print(f"  {'-'*66}")

        # Encode query with each pipeline's text encoder
        query_vectors = {}

        # PE-Core text encoding
        pe_core_search.load()
        query_vectors["pe_core"] = pe_core_search.embed_text(query)

        # Gemma 4 text encoding (shared between AV and V-only)
        gemma4_search.load()
        query_vectors["gemma4_av"] = gemma4_search.embed_text(query)
        query_vectors["gemma4_v"] = query_vectors["gemma4_av"]  # same encoder

        for pname, display in [
            ("pe_core", "PE-Core L14"),
            ("gemma4_v", "Gemma 4 (V)"),
            ("gemma4_av", "Gemma 4 (AV)"),
        ]:
            qvec = query_vectors[pname]
            try:
                rows = db.execute(
                    sql_text("""
                        SELECT v.filename,
                               e.timestamp_start,
                               e.timestamp_end,
                               1 - (e.embedding <=> :qvec ::vector) AS similarity
                        FROM embeddings e
                        JOIN videos v ON v.id = e.video_id
                        WHERE e.pipeline_name = :pname
                        ORDER BY e.embedding <=> :qvec ::vector
                        LIMIT 5
                    """),
                    {"qvec": str(qvec.tolist()), "pname": pname},
                ).fetchall()

                print(f"\n  [{display}]")
                for i, r in enumerate(rows):
                    fname = r.filename[:45]
                    print(f"    {i+1}. {r.similarity:.4f}  t={r.timestamp_start:.0f}-{r.timestamp_end:.0f}s  {fname}")

            except Exception as e:
                db.rollback()
                print(f"\n  [{display}] ERROR: {e}")

    # Summary stats
    banner("EMBEDDING STATS")
    for pname in ["pe_core", "gemma4_v", "gemma4_av"]:
        count = db.execute(
            sql_text("SELECT COUNT(*) FROM embeddings WHERE pipeline_name = :name"),
            {"name": pname},
        ).scalar()
        print(f"  {pname}: {count} embeddings")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
