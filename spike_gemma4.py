#!/usr/bin/env python3
"""Spike: Gemma 4 E4B hidden-state extraction for video search embeddings.

Tests:
1. Load model via mlx-vlm
2. Extract hidden states from a single image
3. Extract hidden states from multiple video frames
4. Extract hidden states from audio+video
5. Extract text embeddings and compute cosine similarity
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "mlx-community/gemma-4-e4b-it-bf16"
PROJECT_DIR = Path(__file__).parent
VIDEOS_DIR = PROJECT_DIR / "videos"
FRAME_CACHE = PROJECT_DIR / ".frame_cache"


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def mx_evaluate(tensor):
    """Force MLX lazy evaluation."""
    mx.eval(tensor)
    return tensor


def mx_to_numpy(tensor):
    """Convert MLX array to numpy, handling bfloat16 by casting to float32 first."""
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float32)
        mx.eval(tensor)
    return np.array(tensor)


# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------
def load_model():
    banner("Step 1: Loading Gemma 4 E4B bf16 via mlx-vlm")
    from mlx_vlm import load

    t0 = time.perf_counter()
    model, processor = load(MODEL_ID)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s")

    # Inspect model structure
    print(f"\n  Model type: {type(model).__name__}")
    if hasattr(model, "language_model"):
        lm = model.language_model
        print(f"  Language model type: {type(lm).__name__}")
        if hasattr(lm, "model"):
            tm = lm.model
            print(f"  Text model type: {type(tm).__name__}")
            if hasattr(tm, "layers"):
                print(f"  Num layers: {len(tm.layers)}")
            if hasattr(tm, "norm"):
                print(f"  Final norm type: {type(tm.norm).__name__}")
    if hasattr(model, "vision_tower"):
        print(f"  Vision tower type: {type(model.vision_tower).__name__}")
    if hasattr(model, "audio_tower"):
        print(f"  Audio tower: {type(model.audio_tower).__name__}")
    else:
        print("  Audio tower: NOT PRESENT")

    # Try to find hidden_size from config
    if hasattr(model, "config"):
        cfg = model.config
        if hasattr(cfg, "hidden_size"):
            print(f"  hidden_size: {cfg.hidden_size}")
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            print(f"  hidden_size: {cfg.text_config.hidden_size}")

    return model, processor


# ---------------------------------------------------------------------------
# Step 2: Extract hidden states from a single test image
# ---------------------------------------------------------------------------
def test_single_image(model, processor):
    banner("Step 2: Hidden states from a single test image")

    # Create a simple test image (red gradient)
    img = Image.new("RGB", (384, 384), color=(200, 50, 50))

    # Build input using the processor's chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Apply chat template to get the text prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"  Prompt (first 200 chars): {prompt[:200]}...")

    # Process through the processor
    inputs = processor(text=prompt, images=[img], return_tensors="np")
    print(f"  Input keys: {list(inputs.keys())}")

    input_ids = mx.array(inputs["input_ids"])
    print(f"  input_ids shape: {input_ids.shape}")

    pixel_values = None
    for key in ["pixel_values", "pixel_values_images"]:
        if key in inputs:
            pv = inputs[key]
            if isinstance(pv, list):
                pixel_values = [mx.array(p) for p in pv]
            else:
                pixel_values = mx.array(pv)
            print(f"  {key} shape: {pixel_values.shape if hasattr(pixel_values, 'shape') else f'list of {len(pixel_values)} items'}")
            break

    if pixel_values is None:
        print("  WARNING: No pixel_values found in inputs!")
        return None

    # Try to extract hidden states
    print("\n  Attempting hidden state extraction...")

    # Method 1: Try calling get_input_embeddings if it exists
    if hasattr(model, "get_input_embeddings"):
        try:
            print("  Trying model.get_input_embeddings()...")
            kwargs = {}
            if "image_grid_thw" in inputs:
                kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

            emb_features = model.get_input_embeddings(
                input_ids=input_ids,
                pixel_values=pixel_values,
                **kwargs,
            )
            print(f"  emb_features type: {type(emb_features)}")

            if hasattr(emb_features, "inputs_embeds"):
                embeds = emb_features.inputs_embeds
                print(f"  inputs_embeds shape: {embeds.shape}")
            elif isinstance(emb_features, mx.array):
                embeds = emb_features
                print(f"  embeddings shape: {embeds.shape}")
            else:
                embeds = emb_features
                print(f"  embeddings: {type(emb_features)}")

            # Now pass through the transformer (skip lm_head)
            lm_model = model.language_model.model
            kwargs_fwd = {}
            if hasattr(emb_features, "per_layer_inputs"):
                kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

            hidden_states = lm_model(
                inputs=None,
                inputs_embeds=embeds if isinstance(embeds, mx.array) else embeds,
                **kwargs_fwd,
            )
            mx_evaluate(hidden_states)
            print(f"  Hidden states shape: {hidden_states.shape}")
            print(f"  Hidden states dtype: {hidden_states.dtype}")

            # Check for NaN/degenerate
            hs_np = mx_to_numpy(hidden_states)
            print(f"  Contains NaN: {np.isnan(hs_np).any()}")
            print(f"  Min: {hs_np.min():.6f}, Max: {hs_np.max():.6f}, Mean: {hs_np.mean():.6f}")

            # Mean pool -> normalize
            pooled = mx.mean(hidden_states, axis=1)  # [1, hidden_dim]
            norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
            normalized = pooled / norm
            mx_evaluate(normalized)
            embedding = mx_to_numpy(normalized).flatten()
            print(f"\n  Final embedding dim: {embedding.shape[0]}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
            print(f"  First 10 values: {embedding[:10]}")

            return embedding

        except Exception as e:
            print(f"  get_input_embeddings failed: {e}")
            import traceback
            traceback.print_exc()

    # Method 2: Try direct forward with output_hidden_states
    print("\n  Trying direct model forward...")
    try:
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        print(f"  Output type: {type(outputs)}")
        if hasattr(outputs, "shape"):
            print(f"  Output shape: {outputs.shape}")
            print("  (These are logits, not hidden states -- need to go deeper)")
    except Exception as e:
        print(f"  Direct forward failed: {e}")
        import traceback
        traceback.print_exc()

    return None


# ---------------------------------------------------------------------------
# Step 3: Multiple video frames
# ---------------------------------------------------------------------------
def test_video_frames(model, processor):
    banner("Step 3: Hidden states from multiple video frames (5 frames)")

    # Check if we have actual video frames in .frame_cache
    actual_frames = []
    if FRAME_CACHE.exists():
        for video_dir in sorted(FRAME_CACHE.iterdir()):
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("*.jpg"))[:5]
                if len(frames) >= 5:
                    actual_frames = frames
                    print(f"  Using real frames from {video_dir.name}")
                    break

    if actual_frames:
        images = [Image.open(f).convert("RGB") for f in actual_frames]
        print(f"  Loaded {len(images)} frames, size: {images[0].size}")
    else:
        # Generate synthetic frames with different colors
        print("  No cached frames found, generating synthetic frames")
        colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (200, 50, 200)]
        images = [Image.new("RGB", (384, 384), color=c) for c in colors]

    # Build multi-image input
    content = []
    for i in range(len(images)):
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Describe these video frames."})

    messages = [{"role": "user", "content": content}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=prompt, images=images, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    print(f"  input_ids shape: {input_ids.shape}")

    pixel_values = None
    for key in ["pixel_values", "pixel_values_images"]:
        if key in inputs:
            pv = inputs[key]
            if isinstance(pv, list):
                pixel_values = [mx.array(p) for p in pv]
            else:
                pixel_values = mx.array(pv)
            print(f"  {key} shape/len: {pixel_values.shape if hasattr(pixel_values, 'shape') else len(pixel_values)}")
            break

    if pixel_values is None:
        print("  No pixel_values -- skipping")
        return None

    try:
        t0 = time.perf_counter()

        kwargs = {}
        if "image_grid_thw" in inputs:
            kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

        emb_features = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features

        kwargs_fwd = {}
        if hasattr(emb_features, "per_layer_inputs"):
            kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

        hidden_states = model.language_model.model(
            inputs=None,
            inputs_embeds=embeds,
            **kwargs_fwd,
        )
        mx_evaluate(hidden_states)
        elapsed = time.perf_counter() - t0

        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Forward pass time: {elapsed:.2f}s")

        hs_np = mx_to_numpy(hidden_states)
        print(f"  Contains NaN: {np.isnan(hs_np).any()}")

        # Mean pool -> normalize
        pooled = mx.mean(hidden_states, axis=1)
        norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normalized = pooled / norm
        mx_evaluate(normalized)
        embedding = mx_to_numpy(normalized).flatten()
        print(f"  Embedding dim: {embedding.shape[0]}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")

        return embedding

    except Exception as e:
        print(f"  Multi-frame extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Step 4: Audio + Video
# ---------------------------------------------------------------------------
def test_audio_video(model, processor):
    banner("Step 4: Audio + Video hidden states")

    # Check if model has audio tower
    if not hasattr(model, "audio_tower") or model.audio_tower is None:
        print("  Model does not have an audio tower -- skipping")
        print("  (This model checkpoint may not include audio weights)")
        return None

    # Find a video with an audio track
    video_files = sorted(VIDEOS_DIR.glob("*.mp4")) if VIDEOS_DIR.exists() else []
    if not video_files:
        print("  No video files found -- skipping audio test")
        return None

    video_path = video_files[0]
    print(f"  Using video: {video_path.name}")

    # Extract 5s audio segment via ffmpeg
    import subprocess
    import tempfile

    audio_path = Path(tempfile.mktemp(suffix=".wav"))
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-ss", "0", "-t", "5",
                "-ar", "16000", "-ac", "1",
                "-f", "wav", "-y",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"  ffmpeg failed: {result.stderr[:200]}")
            return None
        print(f"  Extracted 5s audio to {audio_path} ({audio_path.stat().st_size} bytes)")
    except FileNotFoundError:
        print("  ffmpeg not found -- skipping audio test")
        return None

    # Load first 5 frames from this video
    frames = []
    if FRAME_CACHE.exists():
        stem = video_path.stem
        frame_dir = FRAME_CACHE / stem
        if frame_dir.exists():
            frame_files = sorted(frame_dir.glob("*.jpg"))[:5]
            frames = [Image.open(f).convert("RGB") for f in frame_files]

    if not frames:
        print("  No cached frames for this video -- using synthetic frames")
        frames = [Image.new("RGB", (384, 384), color=(100, 100, 100)) for _ in range(5)]

    print(f"  Using {len(frames)} frames")

    # Build input with audio + images
    content = []
    content.append({"type": "audio", "audio": str(audio_path)})
    for _ in frames:
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Describe what is happening."})

    messages = [{"role": "user", "content": content}]

    try:
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=prompt,
            images=frames,
            audios=[str(audio_path)],
            return_tensors="np",
        )
        print(f"  Input keys: {list(inputs.keys())}")
        input_ids = mx.array(inputs["input_ids"])
        print(f"  input_ids shape: {input_ids.shape}")

        pixel_values = None
        for key in ["pixel_values", "pixel_values_images"]:
            if key in inputs:
                pv = inputs[key]
                pixel_values = [mx.array(p) for p in pv] if isinstance(pv, list) else mx.array(pv)
                break

        # Build kwargs for all special inputs
        extra_kwargs = {}
        for key in inputs:
            if key not in ["input_ids", "attention_mask", "pixel_values", "pixel_values_images"]:
                val = inputs[key]
                if isinstance(val, np.ndarray):
                    extra_kwargs[key] = mx.array(val)
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
                    extra_kwargs[key] = [mx.array(v) for v in val]

        t0 = time.perf_counter()
        emb_features = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **extra_kwargs,
        )

        embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
        kwargs_fwd = {}
        if hasattr(emb_features, "per_layer_inputs"):
            kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs

        hidden_states = model.language_model.model(
            inputs=None,
            inputs_embeds=embeds,
            **kwargs_fwd,
        )
        mx_evaluate(hidden_states)
        elapsed = time.perf_counter() - t0

        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Forward pass time: {elapsed:.2f}s")

        hs_np = mx_to_numpy(hidden_states)
        print(f"  Contains NaN: {np.isnan(hs_np).any()}")

        pooled = mx.mean(hidden_states, axis=1)
        norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normalized = pooled / norm
        mx_evaluate(normalized)
        embedding = mx_to_numpy(normalized).flatten()
        print(f"  Embedding dim: {embedding.shape[0]}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")

        return embedding

    except Exception as e:
        print(f"  Audio+video extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        audio_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 5: Text embedding + cosine similarity
# ---------------------------------------------------------------------------
def test_text_similarity(model, processor, image_embedding):
    banner("Step 5: Text embedding + cosine similarity")

    if image_embedding is None:
        print("  No image embedding available -- skipping")
        return

    queries = [
        "a red image",
        "a blue ocean with waves",
        "a person walking a dog",
    ]

    text_embeddings = []
    for query in queries:
        messages = [{"role": "user", "content": query}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        # Get text embeddings through the same path
        if hasattr(model, "get_input_embeddings"):
            emb_features = model.get_input_embeddings(input_ids=input_ids)
            embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
            kwargs_fwd = {}
            if hasattr(emb_features, "per_layer_inputs"):
                kwargs_fwd["per_layer_inputs"] = emb_features.per_layer_inputs
        else:
            # Fallback: just embed tokens
            embeds = model.language_model.model.embed_tokens(input_ids)
            kwargs_fwd = {}

        hidden_states = model.language_model.model(
            inputs=None,
            inputs_embeds=embeds,
            **kwargs_fwd,
        )
        mx_evaluate(hidden_states)

        # Last-token pooling for text (decoder convention)
        last_token = hidden_states[:, -1, :]
        norm = mx.linalg.norm(last_token, axis=-1, keepdims=True)
        normalized = last_token / norm
        mx_evaluate(normalized)
        text_emb = mx_to_numpy(normalized).flatten()
        text_embeddings.append(text_emb)

    # Compute cosine similarities
    print(f"\n  Image embedding (from Step 2, red image):")
    print(f"  Dim: {image_embedding.shape[0]}")
    print()

    for query, text_emb in zip(queries, text_embeddings):
        cosine_sim = float(np.dot(image_embedding, text_emb))
        print(f"  '{query}' -> cosine similarity: {cosine_sim:.4f}")

    # Also compute text-text similarities
    print(f"\n  Text-text similarities:")
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            sim = float(np.dot(text_embeddings[i], text_embeddings[j]))
            print(f"  '{queries[i]}' vs '{queries[j]}' -> {sim:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Python: {sys.version}")
    print(f"MLX device: {mx.default_device()}")

    model, processor = load_model()

    # Step 2: Single image
    img_emb = test_single_image(model, processor)

    # Step 3: Multiple frames
    multi_emb = test_video_frames(model, processor)

    # Step 4: Audio + Video
    av_emb = test_audio_video(model, processor)

    # Step 5: Text similarity
    test_text_similarity(model, processor, img_emb)

    # Summary
    banner("SPIKE SUMMARY")
    results = {
        "Single image hidden states": img_emb is not None,
        "Multi-frame hidden states": multi_emb is not None,
        "Audio+video hidden states": av_emb is not None,
    }
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}")

    if img_emb is not None:
        print(f"\n  Embedding dimension: {img_emb.shape[0]}")

    if img_emb is not None and multi_emb is not None:
        sim = float(np.dot(img_emb, multi_emb))
        print(f"  Single vs multi-frame cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
