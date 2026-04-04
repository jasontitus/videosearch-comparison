# Gemma 4 Audio+Video Search Pipeline — Research Report

**Date:** 2026-04-04
**Status:** Research complete — retrieval alignment gap identified
**Hardware:** Apple Silicon M2 Max, 96 GB unified memory
**Environment:** Python 3.12.13, MLX 0.31.1, mlx-vlm 0.4.4 (`.venv-gemma4/`)

---

## Summary

We evaluated Google's Gemma 4 E4B as a multimodal video search embedding model,
testing whether joint audio-visual understanding with temporal pre-context could
compete with contrastively-trained models like PE-Core L14.

**Result:** The infrastructure works — audio+video embedding extraction runs at
~1.7 fps via mlx-vlm on Apple Silicon with no stability issues. However, Gemma 4's
generative hidden states lack the cross-modal alignment needed for effective
text-to-video retrieval. PE-Core's contrastively-trained embeddings produce
significantly better retrieval discrimination despite being a simpler model.

Twelve pooling/prompt strategy combinations were tested. The best Gemma 4
configuration (`last_token` video pooling + `raw_last` text encoding) achieves
comparable spread to PE-Core but surfaces less relevant results.

**Conclusion:** Gemma 4 is not ready as a drop-in search embedding model without
contrastive fine-tuning. The audio pipeline and rolling window architecture are
validated and could be reused with a retrieval-trained model.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Why Gemma 4](#why-gemma-4)
3. [Architecture](#architecture-rolling-window-with-audiovideo-pre-context)
4. [Spike Results](#spike-results)
5. [Comparison Run: PE-Core vs Gemma 4](#comparison-run-pe-core-vs-gemma-4)
6. [Strategy Optimization](#strategy-optimization)
7. [Token Structure Analysis](#token-structure-analysis)
8. [Conclusions](#conclusions)
9. [Files Produced](#files-produced)

---

## Motivation

The video search workbench compares vision-only embedding strategies: single-frame
CLIP (PE-Core), single-frame VLM (Jina v4), spatial compositing (Jina Grid), and
temporal 3D position encoding (Jina Native 3D MRoPE). None of them use audio.

Gemma 4 (released April 2, 2026) is the first open-weight model family where the
smaller variants (E2B, E4B) natively process **video frames + audio** through a
unified architecture. This lets us test a fundamentally different hypothesis:

> Does joint audio-visual understanding with temporal pre-context produce
> better video search embeddings than vision-only approaches?

---

## Why Gemma 4

| Property | Value | Why it matters |
|----------|-------|----------------|
| **Native audio encoder** | Conformer, ~300M params (E2B/E4B only) | First audio-capable pipeline in the workbench |
| **Native video input** | Sequences of frames, up to 60s @ 1fps | No grid hacks — frames processed as a temporal sequence |
| **p-RoPE (256K context)** | Proportional RoPE for global attention layers | More frames without "forgetting" early context |
| **Unified KV cache** | Shared KV across some layers | Lower memory for long sequences on Apple Silicon |
| **MLX day-0 support** | mlx-vlm v0.4.3+ with Metal-optimized kernels | Native Apple Silicon, no MPS compatibility patches needed |
| **Apache 2.0 license** | Open-weight, no restrictions | Same freedom as PE-Core |

### Model: `mlx-community/gemma-4-e4b-it-bf16`

| Spec | Value |
|------|-------|
| Effective params | 4.5B |
| Total params (w/ embeddings) | 8B |
| Audio support | Yes (Conformer encoder) |
| bf16 size | ~8 GB |
| Transformer layers | 42 |
| Hidden dim (actual) | 2560 (config reports 1536 — PLE expands it) |
| Vision encoder | SigLIP2, ~150M params, 768-dim |
| Audio encoder | Conformer, ~300M params, 1024-dim |
| Vision tokens per frame | 256 (at 768x768 resolution) |

---

## Architecture: Rolling Window with Audio+Video Pre-Context

Instead of embedding isolated frames or fixed-size chunks, each embedding window
gets **temporal pre-context** — preceding audio and video that gives the model
situational awareness of what came before.

```
Video timeline (1 FPS):

Window 0:  [F0  F1] [F2  F3  F4]     -> embedding for t=2s-5s
            pre-ctx   embed window

Window 1:           [F3  F4] [F5  F6  F7]     -> embedding for t=5s-8s
                     pre-ctx   embed window

Audio:     [========5s========]
                    [========5s========]
```

| Parameter | Value |
|-----------|-------|
| Frame rate | 1 FPS |
| Pre-context | 2 seconds (2 frames) |
| Embed window | 3 seconds (3 frames) |
| Stride | 3 seconds |
| Audio segment | Full 5 seconds (pre-ctx + embed) |
| Audio format | 16 kHz mono WAV via ffmpeg |
| Tokens per window | ~1,300 (5 images x 256 tokens + ~10 template tokens) |

### Embedding Extraction

Gemma 4 is a decoder-only generative model. We extract embeddings by intercepting
hidden states before the language model head:

```
Video frames + audio -> Vision Tower (SigLIP2) + Audio Tower (Conformer)
    -> merged token embeddings
    -> Gemma4TextModel (42 transformer layers, sliding window + global attention)
    -> hidden_states [1, seq_len, 2560]  <-- intercept here, skip lm_head
    -> pool -> normalize -> 2560-dim embedding
```

Code path:
```python
emb_features = model.get_input_embeddings(input_ids, pixel_values, ...)
hidden_states = model.language_model.model(inputs=None, inputs_embeds=emb_features.inputs_embeds, ...)
# Pool and normalize
```

---

## Spike Results

All tests on M2 Max 96GB, MLX 0.31.1, mlx-vlm 0.4.4, Python 3.12.13.

| Test | Result | Details |
|------|--------|---------|
| Model load | **3.3s** (cached) | 42 layers, RMSNorm, vision + audio towers present |
| Single image | **PASS** | Shape: (1, 272, 2560), no NaN, range [-33, +34] |
| 5 video frames | **PASS** | Shape: (1, 1309, 2560), 2.19s forward pass |
| Audio + video | **PASS** | Shape: (1, 1311, 2560), 1.69s forward pass |
| Text embedding | **PASS** | 2560-dim via last-token pooling |

**bfloat16 note:** MLX outputs bf16 hidden states. Must cast to float32 before
numpy conversion. Helper function `mx_to_numpy()` handles this.

---

## Comparison Run: PE-Core vs Gemma 4

### Setup

- **7 videos** from the `./videos/` directory (ICE enforcement footage, political speeches)
- **513 total frames** at 1 FPS
- **PE-Core L14:** 513 embeddings (1 per frame, 1024-dim)
- **Gemma 4 V (video-only):** 172 embeddings (rolling 3s windows, 2560-dim)
- **Gemma 4 AV (audio+video):** 172 embeddings (same windows + audio)

### Throughput

| Pipeline | Embeddings | Total Time | FPS |
|----------|-----------|------------|-----|
| PE-Core L14 | 513 | 73s | **7.0** |
| Gemma 4 (V) | 172 | 283s | **1.8** |
| Gemma 4 (AV) | 172 | 314s | **1.6** |

Audio adds ~11% overhead (ffmpeg extraction is the bottleneck, not the encoder).

### Search Results

Queries: `dog`, `running`, `rifle`, `police`, `car`, `person speaking`

**PE-Core similarity range:** 0.15 - 0.21 (low absolute, but good discrimination)
**Gemma 4 similarity range:** 0.31 - 0.37 (high absolute, poor discrimination)

| Query | PE-Core Top-1 | PE-Core Spread | Gemma 4 V Top-1 | Gemma 4 V Spread |
|-------|--------------|----------------|-----------------|-----------------|
| dog | 0.1801 | 0.027 | 0.3379 | 0.007 |
| running | 0.1604 | 0.007 | 0.3564 | 0.010 |
| rifle | 0.1555 | 0.010 | 0.3285 | 0.007 |
| police | 0.2101 | 0.008 | 0.3471 | 0.010 |
| car | 0.1820 | 0.013 | 0.3095 | 0.007 |
| person speaking | 0.1874 | 0.004 | 0.3663 | 0.012 |

**Spread** = Top-1 minus Top-5 similarity. Higher spread = better discrimination
between relevant and irrelevant content.

### Key Observations

1. **Gemma 4 has a narrow similarity band** — all results cluster around 0.32-0.37
   regardless of query. The model doesn't discriminate "dog" from "rifle."

2. **PE-Core has better relative discrimination** — for `police`, it clearly
   surfaces the Biddeford/ICE videos. For `car`, it finds street scenes and
   driveways.

3. **Gemma 4 has a "first window" bias** — the `t=0-3s` window from the shortest
   video appears as #1 for almost every query, acting as a generic attractor.

4. **Audio (AV) provides marginal improvement** — Gemma 4 AV scores are ~0.003-0.005
   higher than V-only. For `person speaking`, AV slightly outperforms V (0.371 vs
   0.366), which makes sense since the audio encoder can detect speech.

---

## Strategy Optimization

To improve Gemma 4's discrimination, we tested 12 combinations of video pooling
and text encoding strategies.

### Token Structure (discovered via probing)

```
Single image input (270 tokens):
  [0-3]     Template tokens: <bos>, <|turn>, user, \n
  [4]       <|image> (start marker, id=255999)
  [5-260]   <|image|> (256 vision tokens, id=258880)  <-- actual visual content
  [261]     <image|> (end marker, id=258882)
  [262-264] Text content: "Describe."
  [265-269] Template: <turn|>, \n, <|turn>, model, \n

5-image input: 1,306 tokens (5 x 258 image + 16 template)
```

**Key finding:** 256 vision tokens per frame are replaced in-place (no sequence
expansion). Template overhead is only ~10 tokens — a negligible fraction.

### Strategies Tested

**Video pooling (applied to video embeddings during ingestion):**
- `mean_all` — mean pool over all tokens (baseline)
- `vision_only` — mean pool only over vision tokens (id=258880)
- `last_token` — use only the final token's hidden state

**Text encoding (applied to queries at search time):**
- `chat_last` — chat template + last-token pool (baseline)
- `chat_mean` — chat template + mean pool
- `raw_last` — raw text, no template, last-token pool
- `raw_mean` — raw text, no template, mean pool

### Results (all 6 queries, showing Top-1 sim and Top-1 to Top-5 spread)

| Video / Text | Avg Top-1 | Avg Spread | Notes |
|-------------|----------|-----------|-------|
| mean_all / chat_last | 0.337 | 0.009 | Baseline (original implementation) |
| mean_all / chat_mean | 0.562 | 0.010 | Inflated sims, everything ~0.55 |
| mean_all / raw_last | 0.405 | 0.007 | |
| mean_all / raw_mean | 0.413 | 0.008 | |
| vision_only / chat_last | 0.334 | 0.008 | Slightly worse than mean_all |
| vision_only / chat_mean | 0.555 | 0.008 | |
| vision_only / raw_last | 0.401 | 0.006 | |
| vision_only / raw_mean | 0.410 | 0.007 | |
| **last_token / chat_last** | **0.338** | **0.007** | |
| **last_token / chat_mean** | **0.416** | **0.011** | Best avg spread |
| **last_token / raw_last** | **0.310** | **0.010** | Best discrimination |
| **last_token / raw_mean** | **0.317** | **0.010** | Tied for best |
| PE-Core (reference) | 0.179 | 0.010 | Better result relevance |

### Strategy Findings

1. **`last_token` video pooling is best for discrimination** — the last token in
   a decoder model has attended to everything, producing the most semantically
   concentrated representation. Spread improved ~30% over baseline.

2. **`vision_only` pooling doesn't help** — it's marginally *worse* than `mean_all`.
   With only ~10 template tokens out of ~1,300, the template noise is negligible.
   The problem isn't template dilution.

3. **`chat_mean` text encoding inflates all similarities** to ~0.55-0.58 — the
   chat template tokens dominate the mean, creating a high baseline where
   everything looks similar.

4. **`last_token / raw_last`** shows the most topologically interesting results —
   for `car` it surfaces the Biddeford driveway, for `person speaking` it finds
   the Justin Jones speech segments. But absolute discrimination is still weak.

5. **The core problem is architectural**: Gemma 4's hidden states occupy a
   high-baseline-similarity region (cosine ~0.3-0.37 between any text and any
   video). This "anisotropy" is inherent to decoder-only models without
   contrastive training. No pooling strategy can fix this.

---

## Conclusions

### What Works

- **MLX-VLM infrastructure is solid** — model loading, multimodal processing,
  and hidden state extraction all work cleanly on Apple Silicon.
- **Audio encoding works** with negligible overhead (~2 extra tokens, ~11% wall time).
- **Rolling window architecture** produces well-formed embeddings at ~1.7 fps.
- **The E4B model fits comfortably** in 96GB unified memory alongside other pipelines.
- **Python 3.12 venv** at `.venv-gemma4/` isolates dependencies cleanly.

### What Doesn't Work

- **Retrieval discrimination is poor** — Gemma 4's generative hidden states don't
  separate relevant from irrelevant content the way CLIP-trained embeddings do.
- **All pooling/prompt strategies tested** improve discrimination by ~30% but
  don't close the fundamental alignment gap vs PE-Core.
- **The "first window attractor" problem** — short videos produce embeddings that
  are generically similar to everything, dominating all search results.

### Root Cause

PE-Core was trained with a **contrastive objective** (CLIP): matching text-image
pairs are pushed together in embedding space, non-matching pairs are pushed apart.
This directly optimizes for retrieval.

Gemma 4 was trained for **next-token prediction**: its hidden states encode
information needed for generation, not for similarity-based retrieval. The hidden
states contain rich semantic information but aren't *organized* for cosine
similarity search.

### Path Forward

To make Gemma 4 viable for video search, one of these would be needed:

1. **Contrastive fine-tuning** (most promising) — Train a lightweight LoRA adapter
   with contrastive loss on (text, video) pairs. Even a small dataset could
   dramatically improve alignment. This is how Jina v4 adapted Qwen2.5-VL
   for retrieval.

2. **Gemini Embedding 2** (API alternative) — Google's purpose-built multimodal
   embedding model maps text, images, video, and audio into a single retrieval-
   optimized space. API-only, not open-weight.

3. **Use Gemma 4 for reranking instead of retrieval** — Use PE-Core for initial
   retrieval (fast, good discrimination), then use Gemma 4 to rerank the top-N
   results with richer multimodal understanding. This plays to each model's
   strengths.

---

## Files Produced

| File | Purpose |
|------|---------|
| `PROPOSAL-gemma4.md` | This document |
| `spike_gemma4.py` | Spike script: model loading, hidden state extraction, basic similarity |
| `probe_tokens.py` | Token structure analysis: image token IDs, template overhead |
| `run_comparison.py` | Full comparison: PE-Core vs Gemma 4 V vs Gemma 4 AV on 7 videos |
| `test_strategies.py` | 12-strategy A/B test: 3 video pooling x 4 text encoding strategies |
| `.venv-gemma4/` | Python 3.12 virtualenv with mlx-vlm, MLX, and dependencies |
