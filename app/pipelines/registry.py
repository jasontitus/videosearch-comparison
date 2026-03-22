from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from app.pipelines.base import BaseEmbeddingPipeline

_registry: Dict[str, Type[BaseEmbeddingPipeline]] = {}
_instances: Dict[str, BaseEmbeddingPipeline] = {}
_shared_models: Dict[str, Any] = {}


def register(cls: Type[BaseEmbeddingPipeline]) -> Type[BaseEmbeddingPipeline]:
    """Class decorator — registers a pipeline by its `name` attribute."""
    _registry[cls.name] = cls
    return cls


def get_pipeline(name: str, device: str = "cpu") -> BaseEmbeddingPipeline:
    if name not in _instances:
        if name not in _registry:
            raise ValueError(f"Unknown pipeline: {name}")
        _instances[name] = _registry[name](device=device)
    return _instances[name]


def get_all_pipelines(device: str = "cpu") -> Dict[str, BaseEmbeddingPipeline]:
    return {name: get_pipeline(name, device=device) for name in _registry}


def list_pipeline_names() -> List[str]:
    return list(_registry.keys())


def list_pipeline_info() -> List[Dict[str, str]]:
    return [
        {"name": name, "display_name": cls.display_name}
        for name, cls in _registry.items()
    ]


def get_shared_model(key: str, loader_fn: Callable[[], Any]) -> Any:
    """Return a cached model instance, creating it via *loader_fn* on first call."""
    if key not in _shared_models:
        model = loader_fn()  # let exceptions propagate without caching
        _shared_models[key] = model
    return _shared_models[key]


def clear_shared_models() -> None:
    """Remove all shared model references so they can be garbage-collected."""
    _shared_models.clear()
