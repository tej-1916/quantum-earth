"""Health and System Telemetry Endpoint."""

from fastapi import APIRouter
import torch
import sys
from backend.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def get_health():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": sys.version.split()[0],
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "endpoints": {
            "gibs_layers": "/api/gibs/layers",
            "cmr_search": "/api/cmr/search",
            "eurosat_classes": "/api/dataset/eurosat/classes",
            "eurosat_split": "/api/dataset/eurosat/split-manifest",
            "models_status": "/api/models/status",
        },
    }
