"""Test Suite for Quantum Earth Lab Backend (Milestone A).

Tests cover:
1. Health and PyTorch runtime diagnostics
2. NASA GIBS layer discovery and WMTS URL template validity
3. NASA CMR query construction, bbox validation, and error safety
4. EuroSAT class taxonomy, band configurations, and preprocessing
5. EuroSAT zero-leakage stratified train/val/test split verification
6. Model registry honest connectivity and refusal to mock predictions
7. PyTorch model instantiation for RGB and 13-band multispectral architectures
"""

import pytest
from fastapi.testclient import TestClient
import torch

from backend.main import app
from backend.services.gibs_service import gibs_service
from backend.services.dataset_service import dataset_service
from backend.services.model_service import (
    EuroSATResNetRGB,
    EuroSATResNetMultispectral,
    HybridQuantumModelPlaceholder,
)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pytorch" in data
    assert data["pytorch"]["version"] == torch.__version__
    assert "gibs_layers" in data["endpoints"]
    assert "eurosat_split" in data["endpoints"]


def test_gibs_layers_catalog():
    response = client.get("/api/gibs/layers")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] >= 4
    layer_ids = [l["id"] for l in data["layers"]]
    assert "MODIS_Terra_CorrectedReflectance_TrueColor" in layer_ids
    assert "VIIRS_SNPP_CorrectedReflectance_TrueColor" in layer_ids
    assert "MODIS_Terra_CorrectedReflectance_Bands721" in layer_ids

    # Verify tile template structure
    for layer in data["layers"]:
        assert "{z}" in layer["tile_url_template"]
        assert "{y}" in layer["tile_url_template"]
        assert "{x}" in layer["tile_url_template"]
        assert layer["max_zoom"] > 0


def test_gibs_layer_detail_and_404():
    response = client.get("/api/gibs/layer/MODIS_Terra_CorrectedReflectance_TrueColor")
    assert response.status_code == 200
    data = response.json()
    assert data["layer"]["sensor"] == "MODIS"

    err_response = client.get("/api/gibs/layer/NON_EXISTENT_LAYER")
    assert err_response.status_code == 404


def test_gibs_geocode_endpoint():
    response = client.get("/api/gibs/geocode?q=Cairo")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data



def test_cmr_bbox_validation():
    # Invalid bounding box: west > east
    resp = client.get("/api/cmr/search?west=10&east=5&south=0&north=20")
    assert resp.status_code == 400
    assert "West longitude must be strictly less than East longitude" in resp.json()["detail"]

    # Invalid bounding box: south > north
    resp2 = client.get("/api/cmr/search?west=0&east=10&south=20&north=10")
    assert resp2.status_code == 400
    assert "South latitude must be strictly less than North latitude" in resp2.json()["detail"]


def test_eurosat_classes():
    response = client.get("/api/dataset/eurosat/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 10
    class_names = [c["name"] for c in data["classes"]]
    assert "AnnualCrop" in class_names
    assert "Forest" in class_names
    assert "Residential" in class_names
    assert "SeaLake" in class_names


def test_eurosat_band_specifications():
    rgb_resp = client.get("/api/dataset/eurosat/bands?pipeline=rgb")
    assert rgb_resp.status_code == 200
    rgb_data = rgb_resp.json()
    assert len(rgb_data["bands"]) == 3

    ms_resp = client.get("/api/dataset/eurosat/bands?pipeline=multispectral")
    assert ms_resp.status_code == 200
    ms_data = ms_resp.json()
    assert len(ms_data["bands"]) == 13
    assert ms_data["bands"][0]["name"] == "B01"
    assert ms_data["bands"][12]["name"] == "B12"


def test_eurosat_zero_data_leakage_splits():
    # Test split generation with 100 samples per class
    manifest = dataset_service.generate_stratified_split(
        total_samples_per_class=100,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert manifest["total_samples"] == 1000
    assert manifest["split_counts"]["train"] == 700
    assert manifest["split_counts"]["val"] == 150
    assert manifest["split_counts"]["test"] == 150

    # Rigorous zero-leakage check
    train_set = set(manifest["splits"]["train"])
    val_set = set(manifest["splits"]["val"])
    test_set = set(manifest["splits"]["test"])

    assert len(train_set.intersection(val_set)) == 0, "Train and Val must be disjoint"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test must be disjoint"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test must be disjoint"
    assert manifest["leakage_audit"]["disjoint_verified"] is True

    # Reproducibility check: same seed must yield same hash
    manifest2 = dataset_service.generate_stratified_split(
        total_samples_per_class=100,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert manifest["manifest_hash"] == manifest2["manifest_hash"]


def test_model_status_honest_unconnected():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["research_integrity"]["fabricated_metrics"] is False

    models = data["models"]
    assert models["classical_resnet18_rgb"]["status"] == "not_connected"
    assert models["classical_resnet18_multispectral"]["status"] == "not_connected"
    assert models["hybrid_quantum_vqc"]["status"] == "not_connected"


def test_inference_refusal_when_unconnected():
    # Post a dummy image file when model is not connected
    file_content = b"fake image bytes"
    files = {"file": ("tile.png", file_content, "image/png")}
    response = client.post("/api/models/predict", files=files)

    # Must return HTTP 503 with honest error explanation, never fake predictions
    assert response.status_code == 503
    err_detail = response.json()["detail"]
    assert err_detail["error"] == "MODEL_NOT_CONNECTED"
    assert "no simulated or fabricated predictions are returned" in err_detail["message"]


def test_pytorch_architectures_instantiation():
    # RGB Classical ResNet-18 (batch_size=2, channels=3, height=64, width=64)
    rgb_model = EuroSATResNetRGB(num_classes=10, pretrained=False)
    dummy_rgb = torch.randn(2, 3, 64, 64)
    out_rgb = rgb_model(dummy_rgb)
    assert out_rgb.shape == (2, 10)

    # 13-Band Multispectral ResNet-18 (batch_size=2, channels=13, height=64, width=64)
    ms_model = EuroSATResNetMultispectral(num_classes=10)
    dummy_ms = torch.randn(2, 13, 64, 64)
    out_ms = ms_model(dummy_ms)
    assert out_ms.shape == (2, 10)

    # Hybrid Quantum architecture placeholder (batch_size=2, channels=3, height=64, width=64)
    hybrid_model = HybridQuantumModelPlaceholder(num_classes=10, num_qubits=8)
    out_hybrid = hybrid_model(dummy_rgb)
    assert out_hybrid.shape == (2, 10)
