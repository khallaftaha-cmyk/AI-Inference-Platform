"""
models.py — Administrative API endpoints for Model Lifecycle Management.

Endpoints:
  - GET  /api/models                -> List all models (disk + memory status)
  - POST /api/models/{version}/load     -> Pre-warm / load a model into RAM
  - POST /api/models/{version}/activate -> Hot-swap: switch active production model
  - POST /api/models/{version}/unload   -> Evict model from RAM to free resources
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Model Lifecycle & Registry"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    version: str = Field(description="Model identifier or filename stem.")
    filename: str = Field(description="File name on disk.")
    file_size_mb: float = Field(description="Size on disk in megabytes.")
    modified_at: str = Field(description="ISO timestamp of last modification.")
    is_loaded: bool = Field(description="True if model is currently loaded in RAM.")
    is_active: bool = Field(description="True if model is the active production model.")


class ModelListResponse(BaseModel):
    total_models: int
    active_version: Optional[str]
    loaded_versions: List[str]
    models: List[ModelInfo]


class ModelActionResponse(BaseModel):
    status: str
    message: str
    active_version: Optional[str]
    loaded_versions: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=ModelListResponse)
async def list_models(request: Request):
    """
    List all models found in the models directory, showing which are loaded in RAM
    and which one is currently serving production traffic.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not initialized.")

    models_data = registry.discover_models()
    return ModelListResponse(
        total_models=len(models_data),
        active_version=registry.active_version,
        loaded_versions=registry.loaded_versions,
        models=[ModelInfo(**m) for m in models_data],
    )


@router.post("/{version}/load", response_model=ModelActionResponse)
async def load_model(version: str, request: Request):
    """
    Pre-warm a model into RAM without switching active traffic.
    Useful for testing a candidate model before routing production traffic to it.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not initialized.")

    try:
        registry.load_model(version)
        return ModelActionResponse(
            status="ok",
            message=f"Model '{version}' loaded into memory successfully.",
            active_version=registry.active_version,
            loaded_versions=registry.loaded_versions,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading model {version}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/{version}/activate", response_model=ModelActionResponse)
async def activate_model(version: str, request: Request):
    """
    Zero-Downtime Hot Swap:
    Instantly switch active production traffic to the specified model version.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not initialized.")

    try:
        registry.activate_model(version)
        return ModelActionResponse(
            status="ok",
            message=f"Model '{version}' is now the active production model.",
            active_version=registry.active_version,
            loaded_versions=registry.loaded_versions,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error activating model {version}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")


@router.post("/{version}/unload", response_model=ModelActionResponse)
async def unload_model(version: str, request: Request):
    """
    Unload an inactive model from RAM to reclaim memory.
    Cannot unload the currently active production model.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not initialized.")

    try:
        unloaded = registry.unload_model(version)
        if not unloaded:
            return ModelActionResponse(
                status="skipped",
                message=f"Model '{version}' was not loaded in memory.",
                active_version=registry.active_version,
                loaded_versions=registry.loaded_versions,
            )
        return ModelActionResponse(
            status="ok",
            message=f"Model '{version}' unloaded from memory.",
            active_version=registry.active_version,
            loaded_versions=registry.loaded_versions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error unloading model {version}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")
