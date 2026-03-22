# Video Search Pipeline Benchmarks

**Hardware:** Apple Silicon M2 Max (T6020), 96 GB unified memory
**Device:** MPS (Metal Performance Shaders)
**Date:** 2026-03-21

## Jina v4 Batch Size Scaling (MPS, float16)

Larger batches are **slower** — the Qwen2.5-VL vision encoder concatenates all
images into one sequence, so attention scales O(n²) with total tokens.

| Batch Size | Time (s) | Frames/s |
|:---:|---:|---:|
| 1 | 2.20 | **0.45** |
| 2 | 5.36 | 0.37 |
| 4 | 13.99 | 0.29 |
| 6 | 28.92 | 0.21 |
| 8 | 46.11 | 0.17 |

**Optimal batch size: 1** (0.45 fps)

## Pipeline Throughput Comparison (MPS, float16)

All measurements on MPS with float16 precision. Frame extraction at 1 FPS.

| Pipeline | Model Size | Embedding Dim | Batch Size | Avg FPS | Min | Max | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Frame extraction | — | — | — | ~28 | 24 | 57 | cv2 + JPEG save |
| **Meta PE-Core L14** | ~300M | 1024 | 16 | **7.00** | 6.28 | 7.23 | Fastest embedding pipeline |
| **Jina v4 Single Frame** | ~3.8B | 2048 | 1 | **0.19** | 0.19 | 0.20 | batch_size=1 optimal on MPS |
| **Jina v4 Grid Composite** | ~3.8B | 2048 | 1 grid/4 frames | **0.73** | 0.66 | 0.75 | 4 frames → 1 grid image |
| Jina v4 Native 3D | ~3.8B | varies | 4 | — | — | — | Chat template bug (WIP) |

*Measured across all 15 videos (5,764 total frames). Full run: 9h50m.*

## MPS Compatibility Notes

Jina v4 requires two monkey-patches for MPS (`app/utils/mps_compat.py`):

1. **Autocast disabled** — Jina hardcodes `torch.autocast(dtype=bfloat16)` which
   MPS does not fully support. We disable autocast entirely (runs in model's
   native dtype).

2. **SDPA contiguous tensors** — PyTorch 2.8 SDPA on MPS has a regression with
   non-contiguous tensors. Qwen2.5-VL's vision encoder produces these via
   `torch.split`. We make Q/K/V contiguous before `F.scaled_dot_product_attention`.

PE-Core and SigLIP 2 work natively on MPS without patches.

## Cost per Video (estimated)

For a typical 5-minute video (300 frames at 1 FPS):

| Pipeline | Time | Note |
|---|---:|---|
| Frame extraction | ~11s | |
| PE-Core L14 | ~43s | |
| Jina v4 Single Frame | ~26 min | |
| Jina v4 Grid Composite | ~6.8 min | 75 grid images |
| **Total (all 3 pipelines)** | **~34 min** | |

Full run across 15 videos (5,764 frames): **9h 50m**
