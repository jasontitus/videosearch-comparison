# Import every pipeline module so that the @register decorators execute.
from app.pipelines import jina_grid, jina_native3d, jina_single, pe_core  # noqa: F401
