"""EuroSAT Dataset Pipeline & Split Specification Service.

The EuroSAT benchmark (Helber et al., 2019) consists of 27,000 Sentinel-2 satellite image patches
(64x64 pixels) across 10 Land Use and Land Cover (LULC) classes.

This service manages:
1. Ground-truth class taxonomy and descriptions.
2. Band configurations for RGB baseline (3 bands) vs Multispectral pipeline (13 bands).
3. Deterministic 70% Train / 15% Validation / 15% Test split generation with fixed seed (42).
4. Rigorous verification that train, val, and test splits are strictly disjoint (Zero Data Leakage).
5. Preprocessing parameters (band normalization statistics and reflectance scaling).
"""

import hashlib
import json
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel

EUROSAT_CLASSES: List[Dict[str, Any]] = [
    {"id": 0, "name": "AnnualCrop", "label": "Annual Cropland", "description": "Cultivated fields with annual seasonal crops (cereals, oilseeds, legumes)."},
    {"id": 1, "name": "Forest", "label": "Forest & Woodland", "description": "Continuous broadleaved and coniferous tree canopies."},
    {"id": 2, "name": "HerbaceousVegetation", "label": "Herbaceous Vegetation", "description": "Natural grasslands, prairies, and open vegetative pastures."},
    {"id": 3, "name": "Highway", "label": "Highway & Major Roads", "description": "Asphalt and concrete transportation corridors with high spatial line structures."},
    {"id": 4, "name": "Industrial", "label": "Industrial Buildings", "description": "Manufacturing complexes, logistics warehouses, and commercial impervious surfaces."},
    {"id": 5, "name": "Pasture", "label": "Pastureland", "description": "Grazing areas and agricultural meadow systems."},
    {"id": 6, "name": "PermanentCrop", "label": "Permanent Cropland", "description": "Orchards, vineyards, olive groves, and perennial fruit plantations."},
    {"id": 7, "name": "Residential", "label": "Residential Buildings", "description": "Urban and suburban residential housing, neighborhood grids, and roads."},
    {"id": 8, "name": "River", "label": "River & Watercourse", "description": "Linear freshwater watercourses, canals, and riverbanks with turbidity."},
    {"id": 9, "name": "SeaLake", "label": "Sea & Inland Lake", "description": "Deep open marine waters, estuaries, and inland lake reservoirs."},
]

EUROSAT_BANDS_RGB: List[Dict[str, Any]] = [
    {"index": 0, "name": "B04", "color": "Red", "wavelength_nm": 665, "resolution_m": 10},
    {"index": 1, "name": "B03", "color": "Green", "wavelength_nm": 560, "resolution_m": 10},
    {"index": 2, "name": "B02", "color": "Blue", "wavelength_nm": 490, "resolution_m": 10},
]

EUROSAT_BANDS_MULTISPECTRAL: List[Dict[str, Any]] = [
    {"index": 0, "name": "B01", "name_long": "Coastal Aerosol", "wavelength_nm": 443, "resolution_m": 60},
    {"index": 1, "name": "B02", "name_long": "Blue", "wavelength_nm": 490, "resolution_m": 10},
    {"index": 2, "name": "B03", "name_long": "Green", "wavelength_nm": 560, "resolution_m": 10},
    {"index": 3, "name": "B04", "name_long": "Red", "wavelength_nm": 665, "resolution_m": 10},
    {"index": 4, "name": "B05", "name_long": "Vegetation Red Edge 1", "wavelength_nm": 705, "resolution_m": 20},
    {"index": 5, "name": "B06", "name_long": "Vegetation Red Edge 2", "wavelength_nm": 740, "resolution_m": 20},
    {"index": 6, "name": "B07", "name_long": "Vegetation Red Edge 3", "wavelength_nm": 783, "resolution_m": 20},
    {"index": 7, "name": "B08", "name_long": "NIR (Near Infrared)", "wavelength_nm": 842, "resolution_m": 10},
    {"index": 8, "name": "B08A", "name_long": "Narrow NIR", "wavelength_nm": 865, "resolution_m": 20},
    {"index": 9, "name": "B09", "name_long": "Water Vapour", "wavelength_nm": 945, "resolution_m": 60},
    {"index": 10, "name": "B10", "name_long": "SWIR - Cirrus", "wavelength_nm": 1375, "resolution_m": 60},
    {"index": 11, "name": "B11", "name_long": "SWIR 1", "wavelength_nm": 1610, "resolution_m": 20},
    {"index": 12, "name": "B12", "name_long": "SWIR 2", "wavelength_nm": 2190, "resolution_m": 20},
]

# Normalization constants derived from EuroSAT training corpus
PREPROCESSING_CONFIG = {
    "rgb_baseline": {
        "spatial_size": [64, 64],
        "channels": 3,
        "mean": [0.3444, 0.3803, 0.4078],
        "std": [0.2037, 0.1366, 0.1148],
        "input_dtype": "float32",
        "description": "RGB values scaled [0, 1] then normalized with EuroSAT channel-wise statistics.",
    },
    "multispectral_pipeline": {
        "spatial_size": [64, 64],
        "channels": 13,
        "dn_scale_factor": 10000.0,  # Sentinel-2 L2A digital numbers to physical surface reflectance
        "mean": [
            0.1354, 0.1118, 0.1042, 0.0947, 0.1197, 0.2009,
            0.2374, 0.2303, 0.2469, 0.0722, 0.0121, 0.1821, 0.1119
        ],
        "std": [
            0.0653, 0.0541, 0.0528, 0.0564, 0.0519, 0.0752,
            0.0898, 0.0886, 0.0917, 0.0381, 0.0097, 0.0763, 0.0612
        ],
        "input_dtype": "float32",
        "description": "13-band Level-2A surface reflectance divided by 10,000 and standardized.",
    },
}


class DatasetService:
    @staticmethod
    def get_classes() -> List[Dict[str, Any]]:
        return EUROSAT_CLASSES

    @staticmethod
    def get_band_specs(pipeline_type: str = "rgb") -> List[Dict[str, Any]]:
        if pipeline_type == "multispectral":
            return EUROSAT_BANDS_MULTISPECTRAL
        return EUROSAT_BANDS_RGB

    @staticmethod
    def get_preprocessing(pipeline_type: str = "rgb") -> Dict[str, Any]:
        if pipeline_type == "multispectral":
            return PREPROCESSING_CONFIG["multispectral_pipeline"]
        return PREPROCESSING_CONFIG["rgb_baseline"]

    @classmethod
    def generate_stratified_split(
        cls,
        total_samples_per_class: int = 2700,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates reproducible, class-stratified train/val/test splits.

        Enforces zero data leakage: train, val, and test index sets are mutually exclusive.
        """
        import random

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        rng = random.Random(seed)
        splits = {"train": [], "val": [], "test": []}

        for cls_info in EUROSAT_CLASSES:
            cls_name = cls_info["name"]
            # Generate deterministic sample IDs for this class
            sample_ids = [f"{cls_name}_{i:04d}" for i in range(1, total_samples_per_class + 1)]
            rng.shuffle(sample_ids)

            n_total = len(sample_ids)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_subset = sample_ids[:n_train]
            val_subset = sample_ids[n_train : n_train + n_val]
            test_subset = sample_ids[n_train + n_val :]

            splits["train"].extend(train_subset)
            splits["val"].extend(val_subset)
            splits["test"].extend(test_subset)

        # Verify strict disjointness (Zero Data Leakage)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        overlap_tv = train_set.intersection(val_set)
        overlap_tt = train_set.intersection(test_set)
        overlap_vt = val_set.intersection(test_set)

        if overlap_tv or overlap_tt or overlap_vt:
            raise RuntimeError("CRITICAL INTEGRITY FAILURE: Data leakage detected between dataset splits!")

        # Calculate cryptographic hash of the split manifest
        manifest_bytes = json.dumps(
            {"seed": seed, "train_count": len(splits["train"]), "val_count": len(splits["val"]), "test_count": len(splits["test"])},
            sort_keys=True
        ).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        return {
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "class_count": len(EUROSAT_CLASSES),
            "samples_per_class": total_samples_per_class,
            "total_samples": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
            "split_counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            },
            "leakage_audit": {
                "disjoint_verified": True,
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
            },
            "manifest_hash": manifest_hash,
            "splits": splits,
        }


dataset_service = DatasetService()
