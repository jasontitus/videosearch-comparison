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
    """Patch JinaEmbeddingsV4Model.get_last_hidden_states to pass
    video_grid_thw to get_rope_index, enabling 3D MRoPE for video.

    The upstream Jina code only passes image_grid_thw, which means video
    frames get flat 1-D position IDs instead of proper temporal-spatial 3-D
    positions.  This one-line addition is the minimal fix.
    """
    import torch

    # Navigate through PeftModel → LoraModel → JinaEmbeddingsV4Model
    jina_model = model
    if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
        jina_model = jina_model.base_model.model

    # Resolve the parent class (Qwen2_5_VLForConditionalGeneration) for super() calls
    parent_cls = type(jina_model).__mro__[1]

    def _patched_get_last_hidden_states(
        self, task_label, input_ids, attention_mask, **kwargs
    ):
        # Original pixel_values trimming logic (unchanged)
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0
            )

        # --- THE FIX: also pass video_grid_thw for 3D MRoPE ---
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=kwargs.get("video_grid_thw", None),
            attention_mask=attention_mask,
        )

        kwargs["output_hidden_states"] = True
        outputs = parent_cls.forward(
            self,
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise ValueError("Hidden states not found in model output")
        return hidden_states[-1]

    jina_model.get_last_hidden_states = types.MethodType(
        _patched_get_last_hidden_states, jina_model
    )
    print("[jina_native_3d] Patched get_last_hidden_states for video_grid_thw support")


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
                torch_dtype=torch.float16,
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
            # Pass task_label for LoRA adapter selection; the patched
            # get_last_hidden_states handles video_grid_thw for 3D MRoPE.
            outputs = self._model(task_label="retrieval", **inputs)
            emb = outputs.single_vec_emb  # Already mean-pooled & normalized

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
