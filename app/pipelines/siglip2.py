"""Pipeline A — SigLIP 2 baseline vision embeddings (single-frame)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register


@register
class SigLIP2Pipeline(BaseEmbeddingPipeline):
    name = "siglip2"
    display_name = "SigLIP 2 (Baseline Vision)"
    MODEL_ID = "google/siglip2-so400m-patch14-384"
    BATCH_SIZE = 16

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        print(f"[{self.name}] Loading {self.MODEL_ID} …")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
        self._model = (
            AutoModel.from_pretrained(self.MODEL_ID, dtype=torch.float32)
            .to(self.device)
            .eval()
        )
        print(f"[{self.name}] Ready on {self.device}")

    def embed_frames(
        self, frame_paths: List[Path], timestamps: List[float]
    ) -> List[EmbeddingResult]:
        import torch

        self.ensure_loaded()
        results: List[EmbeddingResult] = []

        for i in range(0, len(frame_paths), self.BATCH_SIZE):
            batch_paths = frame_paths[i : i + self.BATCH_SIZE]
            batch_ts = timestamps[i : i + self.BATCH_SIZE]

            batch_frames = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self._processor(
                images=batch_frames, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

            embeddings = feats.cpu().numpy()
            for ts, emb in zip(batch_ts, embeddings):
                results.append(EmbeddingResult(ts, ts + 1.0, emb))

            del batch_frames, inputs, feats

        return results

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        self.ensure_loaded()
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

        return feats.squeeze(0).cpu().numpy()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
