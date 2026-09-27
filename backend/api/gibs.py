"""NASA GIBS Layer Catalog and Tile Configuration Router."""

from fastapi import APIRouter, HTTPException, Query
from backend.services.gibs_service import gibs_service

router = APIRouter(prefix="/gibs", tags=["NASA GIBS"])


@router.get("/layers")
async def list_gibs_layers():
    """Lists available NASA Earthdata GIBS satellite imagery layers.

    Includes layer metadata, spatial resolution, and ready-to-use WMTS tile URL templates.
    """
    layers = gibs_service.list_layers()
    return {
        "success": True,
        "count": len(layers),
        "default_date": gibs_service.get_default_date(),
        "layers": layers,
        "notice": (
            "GIBS provides 8-bit visual composites in Web Mercator (EPSG:3857). "
            "For calibrated 13-band surface reflectance products, use the /api/cmr/search endpoint."
        ),
    }


@router.get("/layer/{layer_id}")
async def get_gibs_layer(layer_id: str, date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Retrieves metadata and configured tile template for a specific GIBS layer."""
    layer = gibs_service.get_layer(layer_id)
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found in GIBS catalog.")

    use_date = date or gibs_service.get_default_date()
    data = layer.model_dump()
    data["selected_date"] = use_date
    data["tile_url"] = gibs_service.build_tile_url(layer_id, use_date)
    return {"success": True, "layer": data}
