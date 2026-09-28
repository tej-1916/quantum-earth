"""Model Connectivity and Inference Router."""

from fastapi import APIRouter, HTTPException, Request, status
from typing import Dict, Any, Optional
import base64
from backend.services.model_service import model_service
from backend.core.config import settings

router = APIRouter(prefix="/models", tags=["AI & Quantum Models"])


@router.get("/status")
async def get_model_status():
    """Returns verified connectivity and checkpoint loading status for all classical and quantum models."""
    return model_service.get_status()


@router.post("/reload")
async def reload_checkpoints():
    """Forces reload of disk checkpoints."""
    return model_service.reload_checkpoints()


@router.get("/classical/metrics")
async def get_classical_metrics():
    """Returns genuine EuroSAT ResNet-18 test set evaluation metrics."""
    return model_service.get_classical_metrics()


@router.get("/classical/history")
async def get_classical_training_history():
    """Returns training loss and accuracy curve history."""
    return model_service.get_training_history()


@router.post("/classical/predict")
async def predict_classical_sample(request: Request):
    """Genuine EuroSAT ResNet-18 classical inference endpoint.

    Accepts:
    - Multipart form file upload (`file`), OR
    - JSON payload with base64 image data (`{"image_base64": "..."}`)

    Refuses with HTTP 503 if model weights are not loaded.
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
                    "In strict adherence to research integrity, no simulated or fabricated predictions are returned."
                ),
                "model_status": classical_info,
            },
        )

    content_type = request.headers.get("content-type", "")
    image_bytes: Optional[bytes] = None

    if "application/json" in content_type:
        payload = await request.json()
        if not payload or "image_base64" not in payload:
            raise HTTPException(status_code=400, detail="Missing 'image_base64' in JSON body.")
        raw_b64 = payload["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")
    else:
        # Multipart form data
        form = await request.form()
        upload = form.get("file")
        if not upload or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="No 'file' provided in multipart form data.")
        image_bytes = await upload.read()

    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds maximum allowed size of 10 MB.")

    try:
        result = model_service.predict_rgb_bytes(image_bytes)
        return {
            "success": True,
            "inference": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict")
async def predict_sample(request: Request):
    """Legacy alias pointing to classical ResNet-18 inference."""
    return await predict_classical_sample(request=request)


