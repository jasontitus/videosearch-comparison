# Import every pipeline module so that the @register decorators execute.
import os as _os

if _os.getenv("SEARCH_BACKEND") == "gguf":
    from app.pipelines import jina_gguf, pe_core  # noqa: F401
else:
    from app.pipelines import jina_grid, jina_native3d, jina_single, pe_core  # noqa: F401
