"""NASA CMR / Scientific Data Discovery Router."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.services.cmr_service import cmr_service
from backend.services.gibs_service import gibs_service

router = APIRouter(prefix="/cmr", tags=["NASA CMR Discovery"])


@router.get("/search")
async def search_scientific_granules(
    west: float = Query(-180.0, ge=-180.0, le=180.0, description="Bounding box West longitude"),
    south: float = Query(-90.0, ge=-90.0, le=90.0, description="Bounding box South latitude"),
    east: float = Query(180.0, ge=-180.0, le=180.0, description="Bounding box East longitude"),
    north: float = Query(90.0, ge=-90.0, le=90.0, description="Bounding box North latitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(5, ge=1, le=20, description="Max granules to discover"),
    provider: Optional[str] = Query(None, description="Optional provider filter (e.g., LPCLOUD)"),
):
    """Searches NASA CMR for calibrated scientific data granules covering an ROI.

    Enables researchers to bridge visual web exploration with calibrated 13-band surface reflectance products.
    """
    if west >= east:
        raise HTTPException(status_code=400, detail="West longitude must be strictly less than East longitude.")
    if south >= north:
        raise HTTPException(status_code=400, detail="South latitude must be strictly less than North latitude.")

    default_date = gibs_service.get_default_date()
    use_end = end_date or default_date
    use_start = start_date or (
        # default to 7-day acquisition window before end date
        f"{use_end[:8]}01" if len(use_end) >= 10 else "2024-01-01"
    )

    bbox = [west, south, east, north]
    result = await cmr_service.search_granules(
        bounding_box=bbox,
        start_date=use_start,
        end_date=use_end,
        limit=limit,
        provider=provider,
    )

    return result
