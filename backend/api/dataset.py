"""EuroSAT Dataset Pipeline & Split Router."""

from fastapi import APIRouter, Query
from backend.services.dataset_service import dataset_service

router = APIRouter(prefix="/dataset/eurosat", tags=["EuroSAT Dataset"])


@router.get("/classes")
async def get_classes():
    """Returns the 10 EuroSAT Land Use / Land Cover ground truth classes with descriptions."""
    return {
        "success": True,
        "dataset": "EuroSAT (Sentinel-2 L2A)",
        "num_classes": len(dataset_service.get_classes()),
        "classes": dataset_service.get_classes(),
    }


@router.get("/bands")
async def get_bands(pipeline: str = Query("rgb", pattern="^(rgb|multispectral)$")):
    """Returns band specifications for RGB (3 bands) or Multispectral (13 bands) pipeline."""
    return {
        "success": True,
        "pipeline": pipeline,
        "bands": dataset_service.get_band_specs(pipeline),
    }


@router.get("/preprocessing")
async def get_preprocessing(pipeline: str = Query("rgb", pattern="^(rgb|multispectral)$")):
    """Returns standard preprocessing constants (mean, std, DN scaling factor, target spatial size)."""
    return {
        "success": True,
        "pipeline": pipeline,
        "config": dataset_service.get_preprocessing(pipeline),
    }


@router.get("/split-manifest")
async def get_split_manifest(
    samples_per_class: int = Query(2700, ge=10, le=2700, description="Patches per class (EuroSAT total is 2,700/class)"),
    seed: int = Query(42, description="Random seed for reproducible split generation"),
):
    """Generates and returns reproducible, zero-leakage 70/15/15 train/val/test split configuration.

    Includes SHA-256 integrity hash confirming mutually disjoint sets.
    """
    manifest = dataset_service.generate_stratified_split(
        total_samples_per_class=samples_per_class,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )
    # Return summary counts + metadata (avoid returning 27k strings in standard response)
    return {
        "success": True,
        "seed": manifest["seed"],
        "ratios": manifest["ratios"],
        "class_count": manifest["class_count"],
        "samples_per_class": manifest["samples_per_class"],
        "total_samples": manifest["total_samples"],
        "split_counts": manifest["split_counts"],
        "leakage_audit": manifest["leakage_audit"],
        "manifest_hash": manifest["manifest_hash"],
        "sample_manifest": {
            "train_sample": manifest["splits"]["train"][:5],
            "val_sample": manifest["splits"]["val"][:5],
            "test_sample": manifest["splits"]["test"][:5],
        },
    }
