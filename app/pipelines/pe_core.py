"""Pipeline — Meta PE-Core (Perception Encoder) CLIP embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register


@register
class PECorePipeline(BaseEmbeddingPipeline):
    name = "pe_core"
    display_name = "Meta PE-Core L14"
    MODEL_NAME = "PE-Core-L14-336"
    BATCH_SIZE = 16

    def _load_model(self) -> None:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        import torch

        print(f"[{self.name}] Loading {self.MODEL_NAME} …")
        model = pe.CLIP.from_config(self.MODEL_NAME, pretrained=True)

        # Handle meta tensor loading — newer PyTorch may use meta tensors
        # for model init, requiring to_empty + weight reload
        try:
            model = model.half().to(self.device).eval()
        except NotImplementedError:
            # Meta tensors can't be copied — materialize on target device
            model = model.to_empty(device=self.device, dtype=torch.float16)
            checkpoint = pe.CLIP.from_config(self.MODEL_NAME, pretrained=True)
            model.load_state_dict(checkpoint.state_dict(), assign=True)
            model = model.half().to(self.device).eval()
            del checkpoint

        self._model = model
        self._preprocess = pe_transforms.get_image_transform(model.image_size)
        self._tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
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
            tensors = torch.stack([self._preprocess(img) for img in batch_frames]).half().to(
                self.device
            )

            with torch.no_grad():
                feats = self._model.encode_image(tensors)
                feats = torch.nn.functional.normalize(feats, dim=-1)

            embeddings = feats.cpu().float().numpy()
            for ts, emb in zip(batch_ts, embeddings):
                results.append(EmbeddingResult(ts, ts + 1.0, emb))

            del batch_frames, tensors, feats

        return results

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        self.ensure_loaded()
        tokens = self._tokenizer([text]).to(self.device)

        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=-1)

        return feats.squeeze(0).cpu().float().numpy()

    def unload(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
