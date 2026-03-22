from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class EmbeddingResult:
    """Single embedding result with timestamp range."""

    __slots__ = ("timestamp_start", "timestamp_end", "embedding")

    def __init__(
        self, timestamp_start: float, timestamp_end: float, embedding: np.ndarray
    ):
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.embedding = embedding


class BaseEmbeddingPipeline(ABC):
    """Abstract base class for all embedding pipelines.

    Subclasses MUST set `name` and `display_name` class attributes and implement
    `_load_model`, `embed_frames`, and `embed_text`.
    """

    name: str = ""
    display_name: str = ""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = False

    def ensure_loaded(self):
        if not self._loaded:
            self._load_model()
            self._loaded = True

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights. Called lazily on first use."""

    @abstractmethod
    def embed_frames(
        self, frame_paths: List[Path], timestamps: List[float]
    ) -> List[EmbeddingResult]:
        """Embed video frames (loaded from disk in batches) and return EmbeddingResults."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query and return a 1-D vector."""

    def unload(self) -> None:
        """Release model weights to free memory. Subclasses should override."""
        self._loaded = False
