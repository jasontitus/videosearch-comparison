"""Pipeline B — Jina Embeddings v4, single-frame 2D."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import get_shared_model, register


@register
class JinaSingleFramePipeline(BaseEmbeddingPipeline):
    name = "jina_single"
    display_name = "Jina v4 (Single Frame)"
    MODEL_ID = "jinaai/jina-embeddings-v4"
    BATCH_SIZE = 1

    def _load_model(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        cache_key = f"st:{self.MODEL_ID}:{self.device}"

        def _loader() -> SentenceTransformer:
            print(f"[jina-st] Loading {self.MODEL_ID} via sentence-transformers …")
            return SentenceTransformer(
                self.MODEL_ID,
                trust_remote_code=True,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16},
            )

        self._model = get_shared_model(cache_key, _loader)
        print(f"[{self.name}] Ready on {self.device}")

    def embed_frames(
        self, frame_paths: List[Path], timestamps: List[float]
    ) -> List[EmbeddingResult]:
        self.ensure_loaded()
        results: List[EmbeddingResult] = []

        for i in range(0, len(frame_paths), self.BATCH_SIZE):
            batch_paths = frame_paths[i : i + self.BATCH_SIZE]
            batch_ts = timestamps[i : i + self.BATCH_SIZE]

            batch_frames = [Image.open(p).convert("RGB") for p in batch_paths]

            embeddings = self._model.encode(
                batch_frames, task="retrieval", batch_size=self.BATCH_SIZE
            )

            for ts, emb in zip(batch_ts, embeddings):
                results.append(EmbeddingResult(ts, ts + 1.0, np.asarray(emb)))

            del batch_frames

        return results

    def embed_text(self, text: str) -> np.ndarray:
        self.ensure_loaded()
        emb = self._model.encode(
            [text], task="retrieval", prompt_name="query"
        )
        return np.asarray(emb[0])

    def unload(self) -> None:
        self._model = None
        self._loaded = False
