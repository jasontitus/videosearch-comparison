"""Pipeline C — Jina v4 with 2×2 frame-grid compositing (temporal fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import get_shared_model, register


@register
class JinaGridPipeline(BaseEmbeddingPipeline):
    name = "jina_grid"
    display_name = "Jina v4 (Grid Composite)"
    MODEL_ID = "jinaai/jina-embeddings-v4"
    CHUNK_SIZE = 4  # 4 frames → one 2×2 grid

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        cache_key = f"st:{self.MODEL_ID}:{self.device}"

        def _loader() -> SentenceTransformer:
            print(f"[jina-st] Loading {self.MODEL_ID} via sentence-transformers …")
            return SentenceTransformer(
                self.MODEL_ID,
                trust_remote_code=True,
                device=self.device,
            )

        self._model = get_shared_model(cache_key, _loader)
        print(f"[{self.name}] Ready on {self.device}")

    @staticmethod
    def _make_grid(frames: List[Image.Image], cell_size: int = 384) -> Image.Image:
        """Stitch up to 4 frames into a 2×2 composite image."""
        resized = [f.resize((cell_size, cell_size)) for f in frames]
        # Pad to 4 by repeating the last frame
        while len(resized) < 4:
            resized.append(resized[-1].copy())

        grid = Image.new("RGB", (cell_size * 2, cell_size * 2))
        grid.paste(resized[0], (0, 0))
        grid.paste(resized[1], (cell_size, 0))
        grid.paste(resized[2], (0, cell_size))
        grid.paste(resized[3], (cell_size, cell_size))
        return grid

    def embed_frames(
        self, frame_paths: List[Path], timestamps: List[float]
    ) -> List[EmbeddingResult]:
        self.ensure_loaded()
        results: List[EmbeddingResult] = []

        for i in range(0, len(frame_paths), self.CHUNK_SIZE):
            chunk_paths = frame_paths[i : i + self.CHUNK_SIZE]
            ts_start = timestamps[i]
            ts_end = timestamps[min(i + self.CHUNK_SIZE - 1, len(timestamps) - 1)] + 1.0

            chunk_frames = [Image.open(p).convert("RGB") for p in chunk_paths]
            grid = self._make_grid(chunk_frames)
            emb = self._model.encode([grid], task="retrieval")
            results.append(EmbeddingResult(ts_start, ts_end, np.asarray(emb[0])))

            del chunk_frames, grid

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
