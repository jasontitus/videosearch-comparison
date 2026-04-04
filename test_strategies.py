#!/usr/bin/env python3
"""Test embedding strategies for Gemma 4 to improve retrieval discrimination.

Strategies tested:
  Video pooling:
    A. mean_all     -- mean pool over all tokens (baseline, current)
    B. vision_only  -- mean pool only over vision tokens (id=258880)
    C. last_token   -- last token only (decoder convention)

  Text encoding:
    1. chat_last    -- chat template + last-token pool (baseline, current)
    2. chat_mean    -- chat template + mean pool
    3. raw_last     -- raw text, no template, last-token pool
    4. raw_mean     -- raw text, no template, mean pool

Re-ingests Gemma 4 V-only with all 3 video strategies, then tests all
text strategies against each. Prints a comparison matrix.

Usage: .venv-gemma4/bin/python test_strategies.py
"""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["DATABASE_URL"] = "postgresql://videosearch:videosearch@localhost:5433/videosearch"

import mlx.core as mx
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).parent
VIDEOS_DIR = PROJECT_DIR / "videos"
FRAME_CACHE = PROJECT_DIR / ".frame_cache"

GEMMA_MODEL_ID = "mlx-community/gemma-4-e4b-it-bf16"
IMAGE_TOKEN_ID = 258880
PRE_CONTEXT = 2
EMBED_WINDOW = 3
QUERIES = ["dog", "running", "rifle", "police", "car", "person speaking"]


def mlx_sync(t):
    """Force MLX lazy graph evaluation."""
    mx.eval(t)
    return t


def mx_to_numpy(t):
    if t.dtype == mx.bfloat16:
        t = t.astype(mx.float32)
        mlx_sync(t)
    return np.array(t)


def setup_db():
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
    from sqlalchemy.orm import declarative_base, sessionmaker
    from pgvector.sqlalchemy import Vector
    from datetime import datetime

    engine = create_engine(os.environ["DATABASE_URL"])
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


def extract_hidden_states(model, processor, prompt, images=None):
    """Run forward pass and return raw hidden states + input_ids."""
    proc_kwargs = {"text": prompt, "return_tensors": "np"}
    if images:
        proc_kwargs["images"] = images

    inputs = processor(**proc_kwargs)
    input_ids = mx.array(inputs["input_ids"])
    input_ids_np = inputs["input_ids"][0]

    pixel_values = None
    for key in ["pixel_values", "pixel_values_images"]:
        if key in inputs:
            pv = inputs[key]
            pixel_values = [mx.array(p) for p in pv] if isinstance(pv, list) else mx.array(pv)
            break

    extra_kwargs = {}
    for key in inputs:
        if key not in ["input_ids", "attention_mask", "pixel_values", "pixel_values_images"]:
            val = inputs[key]
            if isinstance(val, np.ndarray):
                extra_kwargs[key] = mx.array(val)

    emb_features = model.get_input_embeddings(
        input_ids=input_ids, pixel_values=pixel_values, **extra_kwargs
    )
    embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
    kwargs_fwd = {}
    if hasattr(emb_features, "per_layer_inputs"):
        kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

    hidden_states = model.language_model.model(
        inputs=None, inputs_embeds=embeds, **kwargs_fwd
    )
    mlx_sync(hidden_states)
    return hidden_states, input_ids_np


def pool_and_normalize(hidden_states, strategy, input_ids_np=None):
    """Apply pooling strategy to hidden states."""
    if strategy == "mean_all":
        pooled = mx.mean(hidden_states, axis=1)
    elif strategy == "vision_only":
        mask = (input_ids_np == IMAGE_TOKEN_ID)
        if not mask.any():
            pooled = mx.mean(hidden_states, axis=1)
        else:
            # Convert hidden states to numpy for masked indexing, then back
            hs_np = mx_to_numpy(hidden_states)  # (1, seq, dim)
            vision_np = hs_np[0, mask, :]  # (n_vision, dim)
            pooled_np = vision_np.mean(axis=0, keepdims=True)  # (1, dim)
            pooled = mx.array(pooled_np)
    elif strategy == "last_token":
        pooled = hidden_states[:, -1, :]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    mlx_sync(pooled)
    norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
    normalized = pooled / norm
    mlx_sync(normalized)
    return mx_to_numpy(normalized).flatten()


def main():
    print("=" * 70)
    print("  EMBEDDING STRATEGY COMPARISON")
    print("=" * 70)

    engine, Session, Video, Embedding = setup_db()
    db = Session()

    print("\nLoading Gemma 4 E4B...")
    from mlx_vlm import load
    model, processor = load(GEMMA_MODEL_ID)
    print("Model loaded.")

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} videos\n")

    VIDEO_STRATEGIES = ["mean_all", "vision_only", "last_token"]
    TEXT_STRATEGIES = ["chat_last", "chat_mean", "raw_last", "raw_mean"]

    # Phase 1: Ingest with all 3 video pooling strategies
    for vstrat in VIDEO_STRATEGIES:
        pname = f"g4_{vstrat}"
        from sqlalchemy import text as sql_text
        count = db.execute(
            sql_text("SELECT COUNT(*) FROM embeddings WHERE pipeline_name = :n"),
            {"n": pname},
        ).scalar()
        if count > 0:
            print(f"  [{pname}] Already has {count} embeddings, skipping")
            continue

        print(f"\n  Ingesting with video strategy: {vstrat}")
        t0_total = time.perf_counter()
        total = 0

        for vf in video_files:
            video_record = db.query(Video).filter_by(filename=vf.name).first()
            if not video_record:
                continue

            cached = FRAME_CACHE / vf.stem
            if not cached.exists():
                continue
            frame_paths = sorted(cached.glob("*.jpg"))
            timestamps = [float(p.stem) for p in frame_paths]
            n = len(frame_paths)

            for win_start in range(0, n, EMBED_WINDOW):
                pre_start = max(0, win_start - PRE_CONTEXT)
                win_end = min(win_start + EMBED_WINDOW, n)
                if win_start >= n:
                    break

                window_paths = frame_paths[pre_start:win_end]
                if not window_paths:
                    continue

                ts_start = timestamps[win_start]
                ts_end = timestamps[win_end - 1] + 1.0

                images = [Image.open(p).convert("RGB") for p in window_paths]
                content = [{"type": "image"} for _ in images]
                content.append({"type": "text", "text": "Describe what is happening."})
                messages = [{"role": "user", "content": content}]
                prompt = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                try:
                    hidden_states, input_ids_np = extract_hidden_states(
                        model, processor, prompt, images
                    )
                    emb = pool_and_normalize(hidden_states, vstrat, input_ids_np)

                    if not np.isnan(emb).any():
                        db.add(Embedding(
                            video_id=video_record.id,
                            pipeline_name=pname,
                            timestamp_start=ts_start,
                            timestamp_end=ts_end,
                            embedding=emb.tolist(),
                        ))
                        total += 1
                except Exception as e:
                    print(f"      Error at t={ts_start:.0f}s: {e}")

                del images

            db.commit()

        elapsed = time.perf_counter() - t0_total
        print(f"    {total} embeddings in {elapsed:.1f}s")

    # Phase 2: Search comparison
    print("\n" + "=" * 70)
    print("  SEARCH RESULTS (video_strategy / text_strategy)")
    print("=" * 70)

    from sqlalchemy import text as sql_text

    # Load PE-Core for reference
    pe_model = None
    pe_tokenizer = None
    pe_count = db.execute(
        sql_text("SELECT COUNT(*) FROM embeddings WHERE pipeline_name = 'pe_core'"),
    ).scalar()
    if pe_count > 0:
        try:
            import torch
            import core.vision_encoder.pe as pe_mod
            import core.vision_encoder.transforms as pe_transforms
            pe_model = pe_mod.CLIP.from_config("PE-Core-L14-336", pretrained=True)
            pe_model = pe_model.half().to("mps").eval()
            pe_tokenizer = pe_transforms.get_text_tokenizer(pe_model.context_length)
            print("  PE-Core loaded for reference\n")
        except Exception as e:
            print(f"  PE-Core not available: {e}\n")

    for query in QUERIES:
        print(f"\n  Query: '{query}'")
        print(f"  {'Strategy':<28s}  {'Top-1':>7s}  {'Top-3':>7s}  {'Spread':>7s}  {'Top result'}")
        print(f"  {'-'*90}")

        # Compute all text embeddings
        text_embeddings = {}
        msgs = [{"role": "user", "content": query}]
        chat_prompt = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        hs_chat, ids_chat = extract_hidden_states(model, processor, chat_prompt)
        text_embeddings["chat_last"] = pool_and_normalize(hs_chat, "last_token", ids_chat)
        text_embeddings["chat_mean"] = pool_and_normalize(hs_chat, "mean_all", ids_chat)

        hs_raw, ids_raw = extract_hidden_states(model, processor, query)
        text_embeddings["raw_last"] = pool_and_normalize(hs_raw, "last_token", ids_raw)
        text_embeddings["raw_mean"] = pool_and_normalize(hs_raw, "mean_all", ids_raw)

        for vstrat in VIDEO_STRATEGIES:
            pname = f"g4_{vstrat}"
            for tstrat in TEXT_STRATEGIES:
                qvec = text_embeddings[tstrat]
                label = f"{vstrat}/{tstrat}"

                try:
                    rows = db.execute(
                        sql_text("""
                            SELECT v.filename, e.timestamp_start, e.timestamp_end,
                                   1 - (e.embedding <=> :qvec ::vector) AS similarity
                            FROM embeddings e JOIN videos v ON v.id = e.video_id
                            WHERE e.pipeline_name = :pname
                            ORDER BY e.embedding <=> :qvec ::vector LIMIT 5
                        """),
                        {"qvec": str(qvec.tolist()), "pname": pname},
                    ).fetchall()

                    if rows:
                        top1 = rows[0].similarity
                        top3 = rows[2].similarity if len(rows) > 2 else rows[-1].similarity
                        spread = top1 - rows[4].similarity if len(rows) > 4 else top1 - rows[-1].similarity
                        top_file = rows[0].filename[:30]
                        top_t = f"t={rows[0].timestamp_start:.0f}-{rows[0].timestamp_end:.0f}s"
                        print(f"  {label:<28s}  {top1:>7.4f}  {top3:>7.4f}  {spread:>7.4f}  {top_t} {top_file}")
                except Exception as e:
                    db.rollback()
                    print(f"  {label:<28s}  ERROR: {e}")

        # PE-Core reference
        if pe_model is not None:
            import torch
            tokens = pe_tokenizer([query]).to("mps")
            with torch.no_grad():
                feats = pe_model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            pe_qvec = feats.squeeze(0).cpu().float().numpy()

            rows = db.execute(
                sql_text("""
                    SELECT v.filename, e.timestamp_start, e.timestamp_end,
                           1 - (e.embedding <=> :qvec ::vector) AS similarity
                    FROM embeddings e JOIN videos v ON v.id = e.video_id
                    WHERE e.pipeline_name = 'pe_core'
                    ORDER BY e.embedding <=> :qvec ::vector LIMIT 5
                """),
                {"qvec": str(pe_qvec.tolist()), "pname": "pe_core"},
            ).fetchall()
            if rows:
                top1 = rows[0].similarity
                top3 = rows[2].similarity if len(rows) > 2 else rows[-1].similarity
                spread = top1 - rows[4].similarity if len(rows) > 4 else top1 - rows[-1].similarity
                top_file = rows[0].filename[:30]
                top_t = f"t={rows[0].timestamp_start:.0f}-{rows[0].timestamp_end:.0f}s"
                print(f"  {'PE-Core (reference)':<28s}  {top1:>7.4f}  {top3:>7.4f}  {spread:>7.4f}  {top_t} {top_file}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
