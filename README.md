# Multimodal Video Search Eval Workbench

A web application for evaluating and comparing different video embedding pipelines side-by-side. Drop `.mp4` files into a folder, run ingestion through multiple embedding pipelines, then search with natural language and compare results visually.

## Pipelines

All pipelines implement the `BaseEmbeddingPipeline` ABC so new models can be added by creating a single file with `@register`.

| Pipeline | Model | Strategy | Embedding Dim | MPS FPS |
|---|---|---|---:|---:|
| **Meta PE-Core L14** | `PE-Core-L14-336` (~300M) | Single frame, CLIP-style | 1024 | 5.92 |
| **Jina v4 Single Frame** | `jina-embeddings-v4` (~3.8B) | Single frame via sentence-transformers | 2048 | 0.45 |
| **Jina v4 Grid Composite** | `jina-embeddings-v4` (~3.8B) | 4 frames stitched into 2x2 grid | 2048 | 0.75 |
| **Jina v4 Native 3D** | `jina-embeddings-v4` (~3.8B) | 4-frame video chunks via Qwen2.5-VL MRoPE | varies | WIP |

**Vector space isolation:** Every pipeline uses its own text encoder for queries. PE-Core text vectors only query PE-Core image vectors. Jina text vectors only query Jina vectors.

## Architecture

```
Browser (Vanilla JS + Tailwind)
  └─ Search bar → N-column results grid (1 per pipeline)
       │
       ▼  HTTP
FastAPI (app/main.py)
  ├── GET /api/search?q=…  → top-5 per pipeline (cosine sim)
  ├── GET /api/status       → pipeline counts
  ├── /videos/*             → original .mp4s
  ├── /thumbnails/*         → pre-generated JPEGs
  └── /static/*             → JS/CSS
       │
       ▼  SQL
PostgreSQL 17 + pgvector (Docker)
  ├── videos (id, filename, duration)
  └── embeddings (video_id, pipeline_name, timestamps, vector, thumbnail)
```

## Quick Start

```bash
# 1. Start the database
docker compose up -d

# 2. Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/facebookresearch/perception_models.git"
pip install einops ftfy

# 3. Drop video files into the videos directory
cp /path/to/your/*.mp4 ./videos/

# 4. Run ingestion
python ingest.py

# 5. Start the web server
uvicorn app.main:app --port 8000

# 6. Open http://localhost:8000 and search!
```

## Memory-Safe Ingestion

The ingestion pipeline is designed to avoid OOM on machines with limited RAM:

- **Frames saved to disk** during extraction (one frame in memory at a time)
- **Thumbnails generated inline** during extraction (no separate copy pass)
- **Pipelines grouped by model** — shared-model pipelines run consecutively, then models are unloaded and `gc.collect()` + `torch.mps.empty_cache()` runs between groups
- **Each pipeline commits immediately** so results are searchable while ingestion continues
- **Temp frame cache** cleaned up per video after all pipelines finish

Peak memory: ~8 GB (one model at a time) vs ~27 GB before optimization.

## MPS Compatibility

Jina v4's Qwen2.5-VL vision encoder requires two monkey-patches for Apple Silicon MPS (`app/utils/mps_compat.py`):

1. **Autocast disabled** — Jina hardcodes `torch.autocast(dtype=bfloat16)` which MPS doesn't support
2. **SDPA contiguous tensors** — PyTorch SDPA on MPS has issues with non-contiguous tensors from `torch.split` in the vision encoder

PE-Core works natively on MPS without patches.

### Jina v4 Batch Size on MPS

Larger batches are **slower** due to quadratic attention scaling in the Qwen2.5-VL vision encoder:

| Batch Size | Frames/s |
|:---:|---:|
| 1 | **0.45** |
| 2 | 0.37 |
| 4 | 0.29 |
| 8 | 0.17 |

## Adding a New Pipeline

```python
# app/pipelines/my_pipeline.py
from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register

@register
class MyPipeline(BaseEmbeddingPipeline):
    name = "my_pipeline"
    display_name = "My New Pipeline"

    def _load_model(self): ...
    def embed_frames(self, frame_paths, timestamps): ...
    def embed_text(self, text): ...
    def unload(self): ...
```

Then add the import to `app/pipelines/__init__.py`.

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://videosearch:...` | Postgres connection string |
| `VIDEOS_DIR` | `./videos` | Directory containing .mp4 files |
| `THUMBNAILS_DIR` | `./thumbnails` | Output directory for thumbnails |
| `DEVICE` | auto-detect | Force `cpu`, `cuda`, or `mps` |

## Project Structure

```
videosearch-comparison/
├── docker-compose.yml              Postgres 17 + pgvector
├── requirements.txt                Python dependencies
├── ingest.py                       Memory-safe ingestion script
├── BENCHMARKS.md                   Performance measurements
├── app/
│   ├── main.py                     FastAPI application
│   ├── config.py                   Environment config + device detection
│   ├── database.py                 SQLAlchemy models (Video, Embedding)
│   ├── pipelines/
│   │   ├── base.py                 BaseEmbeddingPipeline ABC
│   │   ├── registry.py             @register decorator + shared model cache
│   │   ├── pe_core.py              Meta PE-Core L14
│   │   ├── jina_single.py          Jina v4 single-frame
│   │   ├── jina_grid.py            Jina v4 grid composite
│   │   └── jina_native3d.py        Jina v4 native 3D (WIP)
│   ├── routes/
│   │   └── search.py               /api/search, /api/status
│   └── utils/
│       ├── video.py                Frame extraction + thumbnails
│       └── mps_compat.py           MPS monkey-patches for Jina v4
├── static/
│   ├── index.html                  Frontend UI
│   └── app.js                      Search + video playback
├── videos/                         ← drop .mp4 files here
└── thumbnails/                     ← auto-generated
```
