# Deployment Guide — GCP Cloud Run

This project runs on GCP Cloud Run with a split architecture: a GPU service for video ingestion and a CPU service for search.

## Architecture

```
                    Firebase Hosting
                   (videosearch-comparison.web.app)
                            |
                     catch-all rewrite
                            |
                  Cloud Run (CPU, europe-west1)
                   videosearch-cpu
                   - GGUF text encoder (Jina v4, 3.3GB)
                   - PE-Core on CPU
                   - Scale-to-zero, ~$0.08/hr active
                            |
                    Cloud SQL (pgvector)
                   videosearch-db (us-central1)
                            |
                  Cloud Run Jobs (GPU, europe-west1)
                   - ingest-new: process new videos
                   - reingest-native3d: reprocess single pipeline
                   - L4 GPU, on-demand only
                            |
                    GCS Bucket
                   videosearch-comparison-media
                   - videos/, thumbnails/
                   - Public read (allUsers)
```

## GCP Resources

| Resource | Name | Region | Notes |
|----------|------|--------|-------|
| Cloud Run (GPU) | `videosearch` | europe-west1 | L4 GPU, 8 vCPU, 24GB RAM |
| Cloud Run (CPU) | `videosearch-cpu` | europe-west1 | 4 vCPU, 8GB RAM, scale-to-zero |
| Cloud Run Job | `ingest-new` | europe-west1 | Processes videos missing embeddings |
| Cloud Run Job | `reingest-native3d` | europe-west1 | Re-processes single pipeline |
| Cloud SQL | `videosearch-db` | us-central1 | PostgreSQL + pgvector |
| GCS Bucket | `videosearch-comparison-media` | us-central1 | Videos + thumbnails |
| Secret Manager | `database-url` | — | PostgreSQL connection string |
| Service Account | `videosearch-runner@...` | — | Used by Cloud Run services + jobs |
| Firebase Hosting | `videosearch-comparison` | — | CDN for static assets, API proxy |
| Artifact Registry | `videosearch` | us-central1 | Container images |

## Container Images

### `webapp-base:v1` — GPU base image (rarely rebuilt)
CUDA runtime, PyTorch, pip dependencies, pre-downloaded model weights. ~15GB.

```bash
gcloud builds submit --config=cloudbuild-base.yaml --timeout=2400
```

### `webapp:v7-gpu` — GPU app image (fast rebuild, ~30s)
App code on top of base image. Used by the GPU Cloud Run service and ingest jobs.

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp:v7-gpu --timeout=1800
```

### `webapp-search:v1` — CPU search image
Python 3.11 slim, PyTorch CPU, llama-cpp-python, PE-Core, GGUF Q8_0 model. ~5GB.

```bash
gcloud builds submit --config=cloudbuild-search.yaml --timeout=2400
```

## Deploying

### GPU Service

```bash
gcloud run deploy videosearch \
  --image=us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp:v7-gpu \
  --region=europe-west1 \
  --no-gpu-zonal-redundancy
```

### CPU Search Service

```bash
gcloud run deploy videosearch-cpu \
  --image=us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp-search:v1 \
  --region=europe-west1 \
  --cpu=4 --memory=8Gi \
  --cpu-boost \
  --min-instances=0 --max-instances=2 \
  --timeout=300 \
  --set-cloudsql-instances=videosearch-comparison:us-central1:videosearch-db \
  --service-account=videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --set-env-vars=GCS_BUCKET=videosearch-comparison-media,DEVICE=cpu,SEARCH_BACKEND=gguf \
  --set-secrets=DATABASE_URL=database-url:latest \
  --allow-unauthenticated
```

### Firebase Hosting

```bash
firebase deploy --only hosting --project=videosearch-comparison
```

The `firebase.json` rewrite sends all requests to the CPU search service. Update the `region` field if deploying to a different region.

## Environment Variables

| Variable | GPU Service | CPU Service | Description |
|----------|------------|-------------|-------------|
| `DEVICE` | `cuda` | `cpu` | PyTorch device |
| `SEARCH_BACKEND` | `pytorch` | `gguf` | Which text encoder to use |
| `GCS_BUCKET` | `videosearch-comparison-media` | same | GCS bucket for media |
| `DATABASE_URL` | (from secret) | (from secret) | PostgreSQL connection |
| `JINA_DTYPE` | `float16` | — | Jina v4 model precision (GPU only) |
| `GGUF_MODEL_PATH` | — | `/models/jina-embeddings-v4-text-retrieval-Q8_0.gguf` | Path to GGUF model |

## Ingestion

### Auto-ingest on upload
When a video is uploaded via the web UI, the upload endpoint triggers the `ingest-new` Cloud Run job automatically.

### Manual ingest for new videos
```bash
gcloud run jobs execute ingest-new --region=europe-west1
```
Finds all videos missing embeddings for any pipeline, downloads from GCS, extracts frames, runs all 4 pipelines, saves embeddings. Runs in memory-safe pipeline groups on the L4 GPU.

### Re-ingest a single pipeline
```bash
gcloud run jobs execute reingest-native3d --region=europe-west1
```
Deletes existing `jina_native_3d` embeddings and re-processes all videos. Useful after changing the pipeline code. To create a job for a different pipeline:

```bash
gcloud run jobs create reingest-PIPELINE \
  --image=us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp:v7-gpu \
  --region=europe-west1 \
  --cpu=8 --memory=24Gi --gpu=1 --gpu-type=nvidia-l4 \
  --max-retries=0 --task-timeout=3600 \
  --set-cloudsql-instances=videosearch-comparison:us-central1:videosearch-db \
  --service-account=videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --set-env-vars=GCS_BUCKET=videosearch-comparison-media,DEVICE=cuda,JINA_DTYPE=float16 \
  --set-secrets=DATABASE_URL=database-url:latest \
  --command=python --args=reingest_pipeline.py,--pipeline,PIPELINE_NAME,--force \
  --no-gpu-zonal-redundancy
```

## Pipelines

| Pipeline | Model | Params | Approach |
|----------|-------|--------|----------|
| `pe_core` | Meta PE-Core L14-336 | 300M | CLIP single-frame embedding |
| `jina_single` | Jina Embeddings v4 | 7.6B | Single frame, retrieval LoRA |
| `jina_grid` | Jina Embeddings v4 | 7.6B | 2x2 grid composite of 4 frames |
| `jina_native_3d` | Jina Embeddings v4 | 7.6B | Native video input with 3D MRoPE |

### Search text encoding
- **GPU mode (`SEARCH_BACKEND=pytorch`):** Loads Jina v4 via SentenceTransformer. All 3 Jina pipelines share one model instance (~15GB float16). PE-Core loads separately (~600MB).
- **CPU mode (`SEARCH_BACKEND=gguf`):** Loads `jina-embeddings-v4-text-retrieval-GGUF` Q8_0 (3.3GB) via llama-cpp-python. Retrieval LoRA pre-merged, vision removed. PE-Core loads on CPU.

## Dtype Benchmark (L4 GPU)

Tested text encoding for "a dog running in a field":

| Metric | float32 | bfloat16 | float16 |
|--------|---------|----------|---------|
| Load time | 41.4s | 11.7s | 14.6s |
| Encode time | 2.63s | 0.12s | 0.12s |
| Cosine vs fp32 | — | 0.99976 | 0.99996 |
| VRAM | ~30GB | ~15GB | ~15GB |

float16 is used in production: highest fidelity to float32, 21x faster, half the memory.

## Cost Comparison

| | GPU (L4) | CPU (GGUF) |
|--|----------|------------|
| Search latency | 0.1s | ~1-2s |
| Active cost/hr | ~$1.40 | ~$0.08-0.17 |
| Idle (scale-to-zero) | $0 | $0 |
| 100 searches/day | ~$5/month | < $1/month |
| Can run ingest | Yes | No |

## Initial Setup (one-time)

These steps were already performed for this project:

```bash
# Set project
gcloud config set project videosearch-comparison

# Create Artifact Registry repo
gcloud artifacts repositories create videosearch \
  --repository-format=docker --location=us-central1

# Create Cloud SQL instance with pgvector
gcloud sql instances create videosearch-db \
  --database-version=POSTGRES_15 --tier=db-f1-micro \
  --region=us-central1
# Then: create database, enable pgvector extension, create user

# Create GCS bucket
gcloud storage buckets create gs://videosearch-comparison-media \
  --location=us-central1
gcloud storage buckets add-iam-policy-binding gs://videosearch-comparison-media \
  --member=allUsers --role=roles/storage.objectViewer

# Create service account
gcloud iam service-accounts create videosearch-runner
gcloud projects add-iam-policy-binding videosearch-comparison \
  --member=serviceAccount:videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --role=roles/run.invoker

# Grant service account access to Cloud SQL, GCS, Secrets
gcloud projects add-iam-policy-binding videosearch-comparison \
  --member=serviceAccount:videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --role=roles/cloudsql.client
gcloud storage buckets add-iam-policy-binding gs://videosearch-comparison-media \
  --member=serviceAccount:videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

# Create database URL secret
echo -n "postgresql://USER:PASS@/videosearch?host=/cloudsql/videosearch-comparison:us-central1:videosearch-db" | \
  gcloud secrets create database-url --data-file=-
gcloud secrets add-iam-policy-binding database-url \
  --member=serviceAccount:videosearch-runner@videosearch-comparison.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

# Firebase Hosting
firebase init hosting --project=videosearch-comparison
firebase deploy --only hosting
```

## Cleaning Up Old Images

Container images with CUDA + models are ~15-20GB each. Clean up old tags:

```bash
# List images
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/videosearch-comparison/videosearch \
  --format="table(package,tags,createTime)"

# Delete old untagged images
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp \
  --include-tags --filter="NOT tags:*" --format="value(version)" | \
  xargs -I{} gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp@{} --quiet
```
