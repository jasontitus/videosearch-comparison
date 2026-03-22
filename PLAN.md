# Project: Multimodal Video Search Eval Workbench

I need you to build a standalone, containerized web application that allows me to evaluate and compare different video embedding and search strategies side-by-side. 

**The Goal:** I want to drop a pile of `.mp4` videos into a local folder, run an ingestion script that processes them through multiple distinct embedding pipelines, and then use a web UI to search a text query. The UI should display the top 5 results from *each* pipeline side-by-side so I can visually compare their accuracy.

## 1. Tech Stack & Architecture
* **Backend:** Python with FastAPI.
* **Database:** A local Postgres instance via Docker with the `pgvector` extension installed.
* **Frontend:** A clean, lightweight frontend (React or Vanilla JS + Tailwind) served by FastAPI.
* **Design Pattern:** Use a strict **Strategy Pattern** for the embedding pipelines. Create an abstract base class `BaseEmbeddingPipeline`. Every model/method should inherit from this so I can easily register new models in the future.

## 2. The Four Pipelines to Implement
Please implement the following four specific extraction strategies as separate pipeline classes. 
*(Note: All pipelines should extract frames at roughly 1 frame per second, or 1 chunk per second, depending on the strategy).*

* **Pipeline A (Baseline Vision): SigLIP 2**
  * Extract 1 frame per second.
  * Use the Hugging Face `timm` or standard `transformers` implementation of SigLIP 2.
  * Embed single frames independently.

* **Pipeline B (Advanced 2D): Jina Embeddings v4 (Single Frame)**
  * Extract 1 frame per second.
  * Use the `sentence-transformers` library to load `jinaai/jina-embeddings-v4` (`trust_remote_code=True`).
  * Embed single frames independently using `task="retrieval"`.

* **Pipeline C (Temporal Fallback): Jina v4 with Frame Grid Compositing**
  * Group frames into 4-second chunks.
  * Use PIL/OpenCV to stitch the 4 frames into a single 2x2 composite image.
  * Pass that single composite image to the Jina v4 `sentence-transformers` wrapper just like Pipeline B.
  * The resulting embedding represents that 4-second window.

* **Pipeline D (Native 3D Video MRoPE): Jina v4 via Qwen2.5-VL Processor**
  * *Crucial Instruction:* Do not use standard high-level image APIs for this. 
  * Group frames into 3-to-5 second chunks.
  * Use `qwen_vl_utils` and the `Qwen2_5_VLProcessor` to process the video chunk into `pixel_values_videos` and `video_grid_thw` tensors.
  * Load `jinaai/jina-embeddings-v4`.
  * Inject the processed 3D video tensors directly into the model's low-level forward pass to extract the hidden states, and apply mean pooling to generate the final 1D embedding vector.

## 3. Data Ingestion Flow
Write a script (`ingest.py`) that:
1. Scans a `./videos` directory for `.mp4` files.
2. For each video, routes the file through all active pipelines.
3. Saves the resulting embeddings to Postgres. 
4. *Schema thought:* You will likely need a table structure that links a `video_id`, `timestamp_start`, `pipeline_name`, and the `embedding` (vector) so they can be queried independently.

## 4. Frontend UI Requirements
* A prominent Search Bar at the top.
* A grid or column layout showing the results split by pipeline (e.g., Column 1: SigLIP, Column 2: Jina 2D, Column 3: Jina Grid, Column 4: Jina Native 3D).
* Show the Top 5 results for each column.
* **Result Card:** Each result must show:
  * The timestamp match (e.g., "00:01:23").
  * A static thumbnail of the video at that exact offset.
  * When clicked, the thumbnail should swap to an HTML5 `<video>` player that automatically starts playing the original video file at that exact timestamp using the `#t={seconds}` URL fragment.

Please write the complete project structure, the `docker-compose.yml` for Postgres, the backend API code, the pipeline classes, and the frontend code. Start with the backend architecture and pipeline classes.
