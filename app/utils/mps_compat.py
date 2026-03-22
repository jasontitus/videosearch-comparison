"""Patches for PyTorch MPS compatibility with Jina v4 / Qwen2.5-VL models.

Addresses:
- Jina v4 hardcodes torch.autocast with bfloat16, which can hang on MPS.
  We redirect to float16 (computation precision only, model weights unchanged).
- PyTorch 2.8 SDPA MPS regression with non-contiguous tensors from torch.split
  in the Qwen2.5-VL vision encoder.  We make Q/K/V contiguous before SDPA.
"""

from __future__ import annotations

_patched = False


def apply_mps_patches() -> None:
    """Monkey-patch torch for MPS compatibility. Safe to call multiple times."""
    global _patched
    if _patched:
        return

    import torch
    import torch.nn.functional as F

    # --- Patch 1: disable autocast on MPS (not fully supported) ---
    _OrigAutocast = torch.autocast

    class _MpsAutocast(_OrigAutocast):
        def __init__(self, device_type, dtype=None, **kwargs):
            if device_type == "mps":
                kwargs["enabled"] = False
            super().__init__(device_type, dtype=dtype, **kwargs)

    torch.autocast = _MpsAutocast

    # --- Patch 2: make SDPA tensors contiguous on MPS ---
    _orig_sdpa = F.scaled_dot_product_attention

    def _mps_safe_sdpa(query, key, value, *args, **kwargs):
        if query.device.type == "mps":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
        return _orig_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _mps_safe_sdpa

    _patched = True
    print("[mps_compat] Applied MPS patches (autocast disabled, SDPA contiguous)")
