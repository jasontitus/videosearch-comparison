"""GGUF-backed Jina v4 text encoder for CPU-only search.

Uses llama-cpp-python to load the jinaai/jina-embeddings-v4-text-retrieval-GGUF
model (Q8_0) in-process. The retrieval LoRA is pre-merged and vision components
are stripped, giving a 3.09B param text-only model.

Only supports embed_text — embed_frames raises NotImplementedError since
video frame embedding requires the full model on GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from app.config import GGUF_MODEL_PATH
from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register

# Shared llama model instance (loaded once, used by all GGUF pipelines)
_llama_model = None


def _get_llama_model():
    global _llama_model
    if _llama_model is None:
        from llama_cpp import Llama

        print(f"[gguf] Loading GGUF model from {GGUF_MODEL_PATH} ...")
        _llama_model = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=512,
            n_threads=4,
            embedding=True,
            pooling_type=1,  # LLAMA_POOLING_TYPE_MEAN
            verbose=True,
        )
        print("[gguf] GGUF model loaded")
    return _llama_model


class JinaGGUFTextPipeline(BaseEmbeddingPipeline):
    """Base class for GGUF-backed Jina text encoding."""

    MODEL_ID = "jinaai/jina-embeddings-v4"

    def _load_model(self) -> None:
        _get_llama_model()

    def embed_frames(
        self, frame_paths: List[Path], timestamps: List[float]
    ) -> List[EmbeddingResult]:
        raise NotImplementedError(
            f"{self.name}: frame embedding requires GPU. "
            "Use the GPU ingest jobs instead."
        )

    def embed_text(self, text: str) -> np.ndarray:
        model = _get_llama_model()

        # The GGUF retrieval model requires "Query: " prefix
        prefixed = f"Query: {text}"

        result = model.embed(prefixed)

        # model.embed() returns list of lists — flatten to 1D
        emb = np.array(result, dtype=np.float32).flatten()

        # Log dimension on first call for debugging
        if not hasattr(self, '_dim_logged'):
            print(f"[gguf] embed_text returned {emb.shape} (raw type: {type(result)}, len: {len(result)})")
            self._dim_logged = True

        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def unload(self) -> None:
        self._loaded = False


@register
class JinaSingleGGUF(JinaGGUFTextPipeline):
    name = "jina_single"
    display_name = "Jina v4 (Single Frame)"


@register
class JinaGridGGUF(JinaGGUFTextPipeline):
    name = "jina_grid"
    display_name = "Jina v4 (Grid Composite)"


@register
class JinaNative3DGGUF(JinaGGUFTextPipeline):
    name = "jina_native_3d"
    display_name = "Jina v4 Native 3D (MRoPE)"
