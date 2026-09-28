"""FastAPI Router for Quantum Simulation & Hybrid Machine Learning.

Exposes real quantum inference and experiment endpoints:
- POST /api/models/quantum/predict
- GET  /api/models/quantum/circuit
- GET  /api/models/quantum/experiments
- GET  /api/models/quantum/status
"""

import base64
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Query, UploadFile, File

from backend.core.config import settings
from backend.services.quantum_service import quantum_service

router = APIRouter()


@router.get("/status")
def get_quantum_status() -> Dict[str, Any]:
    """Returns quantum simulator status and loaded checkpoint metadata."""
    return quantum_service.get_status()


@router.get("/circuit")
def get_quantum_circuit_definition(
    qubits: int = Query(8, ge=4, le=12, description="Number of qubits (4, 8, 12)"),
    depth: int = Query(2, ge=1, le=4, description="Variational circuit depth (1, 2, 4)"),
    entanglement: str = Query("ring", description="Entanglement mode ('none', 'ring', 'full')"),
    encoding: str = Query("angle", description="Feature encoding ('angle', 'data_reuploading')"),
) -> Dict[str, Any]:
    """Returns dynamic Qiskit ASCII diagram, QASM string, and gate specifications."""
    try:
        details = quantum_service.get_circuit_details(
            n_qubits=qubits,
            depth=depth,
            entanglement=entanglement,
            encoding=encoding,
        )
        return {"success": True, "circuit": details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments")
def get_quantum_experiment_results() -> Dict[str, Any]:
    """Returns systematic ablations and fair classical vs quantum comparison metrics."""
    data = quantum_service.get_experiments()
    return {"success": True, "experiments": data}


@router.post("/predict")
async def predict_quantum_image(request: Request) -> Dict[str, Any]:
    """Executes genuine hybrid quantum inference on an input satellite image.
    
    Supports:
    1. Multipart file upload: -F "file=@image.jpg"
    2. JSON payload: {"image_base64": "data:image/jpeg;base64,..."}
    3. Benchmark sample: {"sample_class": "Forest"}
    """
    if not quantum_service.is_connected:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model Not Connected",
                "message": (
                    "Quantum VQC model is not connected. A trained checkpoint at 'eurosat_hybrid_vqc.pt' "
                    "is required to execute genuine simulator inference. No predictions will be mocked."
                ),
            },
        )

    content_type = request.headers.get("content-type", "")
    image_bytes = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        if "sample_class" in body and body["sample_class"]:
            cname = body["sample_class"]
            sample_file = settings.BASE_DIR / "data" / "eurosat" / "2750" / cname / f"{cname}_1.jpg"
            if not sample_file.exists():
                raise HTTPException(status_code=404, detail=f"Sample file for class {cname} not found on disk.")
            with open(sample_file, "rb") as f:
                image_bytes = f.read()
        elif "image_base64" in body and body["image_base64"]:
            b64_str = body["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64_str)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")
        else:
            raise HTTPException(status_code=400, detail="JSON payload must contain 'image_base64' or 'sample_class'")
    else:
        # Multipart form data
        form = await request.form()
        file_obj = form.get("file")
        if file_obj is not None and hasattr(file_obj, "read"):
            image_bytes = await file_obj.read()
        else:
            raise HTTPException(status_code=400, detail="Please upload a file via multipart form or provide image_base64")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No valid image data received.")

    try:
        result = quantum_service.predict_image_bytes(image_bytes)
        return {"success": True, "inference": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum inference execution failure: {e}")
