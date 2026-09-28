"""Quantum Earth Lab - Application Configuration"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    BASE_DIR: Path = BASE_DIR
    PROJECT_NAME: str = "Quantum Earth Lab API"
    VERSION: str = "2.0.0-alpha"
    DESCRIPTION: str = (
        "Earth Observation & Quantum Machine Learning Research Laboratory API. "
        "Integrates NASA GIBS/CMR imagery services, EuroSAT dataset splits, "
        "and reproducible classical & hybrid quantum model pipelines."
    )

    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Allowed CORS Origins
    _raw_cors = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173,http://127.0.0.1:4173",
    )
    CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]

    # Security & limits
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # NASA Earthdata API token (optional, grants higher CMR rate limits)
    NASA_EARTHDATA_TOKEN: str = os.getenv("NASA_EARTHDATA_TOKEN", "")

    # IBM Quantum credentials (reserved for Milestone D)
    IBM_QUANTUM_TOKEN: str = os.getenv("IBM_QUANTUM_TOKEN", "")
    IBM_QUANTUM_INSTANCE: str = os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")

    # Storage paths
    DATA_DIR: Path = Path(os.getenv("EUROSAT_DATA_DIR", str(BASE_DIR / "data" / "eurosat")))
    CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", str(BASE_DIR / "checkpoints")))


settings = Settings()
