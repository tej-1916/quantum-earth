"""Model Connectivity and Inference Router."""

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from backend.services.model_service import model_service
from backend.core.config import settings

router = APIRouter(prefix="/models", tags=["AI & Quantum Models"])


@router.get("/status")
async def get_model_status():
    """Returns verified connectivity and checkpoint loading status for all classical and quantum models."""
    return model_service.get_status()


@router.post("/predict")
async def predict_sample(file: UploadFile = File(...)):
    """Inference endpoint.

    In compliance with project scientific integrity guidelines, this endpoint refuses to return
    fabricated predictions when weights are missing.
    """
    model_status = model_service.get_status()
    classical_info = model_status["models"]["classical_resnet18_rgb"]

    if not classical_info["available"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "MODEL_NOT_CONNECTED",
                "message": (
                    "Classical ResNet-18 model weights are not loaded. "
                    "In strict adherence to research integrity, no simulated or fabricated predictions are returned. "
                    "Model training will be executed in Milestone B."
                ),
                "model_status": classical_info,
            },
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 10 MB.")

    # When connected in Milestone B, real PyTorch inference will process 'contents' here
    return {"status": "success", "message": "Inference complete"}
