"""Pipeline D — Jina v4 with native 3D video processing via Qwen2.5-VL MRoPE.

This pipeline bypasses the standard image API. It uses qwen_vl_utils and
Qwen2_5_VLProcessor to build `pixel_values_videos` / `video_grid_thw` tensors,
injects them into the model's forward pass, and mean-pools the hidden states
to produce a single embedding vector per video chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register


@register
class JinaNative3DPipeline(BaseEmbeddingPipeline):
    name = "jina_native_3d"
    display_name = "Jina v4 Native 3D (MRoPE)"
    MODEL_ID = "jinaai/jina-embeddings-v4"
    CHUNK_SIZE = 4  # 4-second chunks (within the 3–5 s range from the plan)

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        print(f"[{self.name}] Loading {self.MODEL_ID} (raw transformers) …")
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            .to(self.device)
            .eval()
        )
        print(f"[{self.name}] Ready on {self.device}")

    # ------------------------------------------------------------------
    # Video chunk → embedding
    # ------------------------------------------------------------------
    def _encode_video_chunk(self, frames: List[Image.Image]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        # Build the message structure that qwen_vl_utils expects
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 1.0,
                    }
                ],
            }
        ]

        image_inputs, video_inputs = process_vision_info(messages)

        # Use the processor's chat template to produce the token sequence
        # that includes <|vision_start|><|video_pad|><|vision_end|> placeholders
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

            # Mean pooling with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = F.normalize(pooled, p=2, dim=-1)

        return emb.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
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
            emb = self._encode_video_chunk(chunk_frames)
            results.append(EmbeddingResult(ts_start, ts_end, emb))

            del chunk_frames

        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text query using the same raw model + mean pooling.

        This keeps Pipeline D fully siloed — it never shares encoders with
        Pipelines B/C.  The trade-off is that we do not activate task-specific
        LoRA adapters (if any); the embedding is the raw mean-pooled hidden
        state, identical in approach to the video encoding above.
        """
        import torch
        import torch.nn.functional as F

        self.ensure_loaded()

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = F.normalize(pooled, p=2, dim=-1)

        return emb.squeeze(0).cpu().numpy()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
