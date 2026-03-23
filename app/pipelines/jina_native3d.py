"""Pipeline D — Jina v4 with native 3D video processing via Qwen2.5-VL MRoPE.

Uses the Jina v4 model (which wraps Qwen2.5-VL) with video inputs to leverage
3D Multi-Resolution Rotary Position Embeddings for temporal understanding.

Key fixes over the initial WIP version:
- Correct torch_dtype (float16, MPS-compatible) instead of wrong kwarg name
- Monkey-patch Jina's get_last_hidden_states to pass video_grid_thw to
  get_rope_index, enabling proper 3D MRoPE for video frames
- Pass task_label="retrieval" through the Jina model's forward for LoRA adapters
- Use Jina's encode_text for text queries (proper LoRA-adapted embeddings)
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.pipelines.base import BaseEmbeddingPipeline, EmbeddingResult
from app.pipelines.registry import register


def _patch_jina_video_rope(model) -> None:
    """Patch the Qwen2.5-VL backbone's get_rope_index so that when Jina's
    get_last_hidden_states calls it (without video_grid_thw), we supply
    the video_grid_thw from a thread-local stash.

    This avoids touching get_last_hidden_states itself, sidestepping
    recursion and duplicate-kwarg issues entirely.
    """
    # Navigate through PeftModel → LoraModel → JinaEmbeddingsV4Model
    jina_model = model
    if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
        jina_model = jina_model.base_model.model

    # The inner Qwen2.5-VL model that has get_rope_index
    qwen_model = jina_model.model

    _orig_get_rope_index = qwen_model.get_rope_index

    # Stash for video_grid_thw — set before forward, cleared after
    qwen_model._video_grid_thw_stash = None

    def _patched_get_rope_index(self, input_ids, image_grid_thw=None,
                                 video_grid_thw=None, attention_mask=None):
        # If caller didn't pass video_grid_thw, use the stash
        if video_grid_thw is None and self._video_grid_thw_stash is not None:
            video_grid_thw = self._video_grid_thw_stash
        return _orig_get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    qwen_model.get_rope_index = types.MethodType(_patched_get_rope_index, qwen_model)
    print("[jina_native_3d] Patched get_rope_index for video_grid_thw support")


@register
class JinaNative3DPipeline(BaseEmbeddingPipeline):
    name = "jina_native_3d"
    display_name = "Jina v4 Native 3D (MRoPE)"
    MODEL_ID = "jinaai/jina-embeddings-v4"
    CHUNK_SIZE = 4  # 4-second chunks (within the 3–5 s range from the plan)

    def _load_model(self) -> None:
        import os

        import torch
        from transformers import AutoModel, AutoProcessor

        dtype_name = os.getenv("JINA_DTYPE", "float16")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

        print(f"[{self.name}] Loading {self.MODEL_ID} (dtype={dtype_name}) …")
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                dtype=torch_dtype,
            )
            .to(self.device)
            .eval()
        )

        # Patch the Jina model so video_grid_thw flows into get_rope_index
        _patch_jina_video_rope(self._model)

        print(f"[{self.name}] Ready on {self.device}")

    # ------------------------------------------------------------------
    # Video chunk → embedding
    # ------------------------------------------------------------------
    def _encode_video_chunk(self, frames: List[Image.Image]) -> np.ndarray:
        import torch
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

        # Match the prompt format used by Jina's process_images — include
        # the "Describe the image." instruction so the LoRA adapter knows
        # to produce retrieval-oriented embeddings from the visual content.
        text_prompt = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe the image.<|im_end|>"

        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Stash video_grid_thw so the patched get_rope_index can use it
        jina_model = self._model
        if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
            jina_model = jina_model.base_model.model
        qwen_model = jina_model.model
        qwen_model._video_grid_thw_stash = inputs.get("video_grid_thw", None)

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self._model(task_label="retrieval", **inputs)
                emb = outputs.single_vec_emb

        qwen_model._video_grid_thw_stash = None
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

            # Skip chunks that produced NaN (float16 MPS numerical instability)
            if np.isnan(emb).any():
                print(f"        [jina_native_3d] NaN at t={ts_start:.0f}s, skipping", flush=True)
            else:
                results.append(EmbeddingResult(ts_start, ts_end, emb))

            del chunk_frames

        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text query using Jina's encode_text with LoRA adapters.

        This uses the model's built-in encode_text which activates the
        task-specific LoRA adapter for 'retrieval' queries, producing
        embeddings that are properly aligned with the video embeddings.
        """
        self.ensure_loaded()
        emb = self._model.encode_text(
            [text], task="retrieval", prompt_name="query", return_numpy=True
        )
        return emb[0]

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
