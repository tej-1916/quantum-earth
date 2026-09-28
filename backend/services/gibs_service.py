"""NASA Earthdata GIBS (Global Imagery Browse Services) Integration Service.

GIBS provides full-resolution, rapid-response visual imagery from NASA EOSDIS satellites
projected in EPSG:3857 (Web Mercator).
Note: These 8-bit web tiles are designed for planetary visualization and reconnaissance,
not as calibrated 13-band surface reflectance inputs for scientific ML models.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel


class GIBSLayer(BaseModel):
    id: str
    title: str
    sensor: str
    satellite: str
    resolution: str
    format: str
    tile_matrix_set: str
    max_zoom: int
    min_zoom: int
    start_date: str
    description: str
    wmts_endpoint: str
    type: str  # "true_color", "false_color", "atmospheric"


GIBS_LAYERS: Dict[str, GIBSLayer] = {
    "MODIS_Terra_CorrectedReflectance_TrueColor": GIBSLayer(
        id="MODIS_Terra_CorrectedReflectance_TrueColor",
        title="MODIS Terra True Color (CR)",
        sensor="MODIS",
        satellite="Terra",
        resolution="250m",
        format="jpg",
        tile_matrix_set="GoogleMapsCompatible_Level9",
        max_zoom=9,
        min_zoom=1,
        start_date="2000-02-24",
        description="Daily natural color surface composite from the Terra morning orbital pass.",
        wmts_endpoint="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        type="true_color",
    ),
    "VIIRS_SNPP_CorrectedReflectance_TrueColor": GIBSLayer(
        id="VIIRS_SNPP_CorrectedReflectance_TrueColor",
        title="VIIRS SNPP True Color",
        sensor="VIIRS",
        satellite="Suomi NPP",
        resolution="250m",
        format="jpg",
        tile_matrix_set="GoogleMapsCompatible_Level9",
        max_zoom=9,
        min_zoom=1,
        start_date="2012-05-02",
        description="High-fidelity day-night radiometer true color composite with sharp atmospheric correction.",
        wmts_endpoint="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        type="true_color",
    ),
    "MODIS_Aqua_CorrectedReflectance_TrueColor": GIBSLayer(
        id="MODIS_Aqua_CorrectedReflectance_TrueColor",
        title="MODIS Aqua True Color (CR)",
        sensor="MODIS",
        satellite="Aqua",
        resolution="250m",
        format="jpg",
        tile_matrix_set="GoogleMapsCompatible_Level9",
        max_zoom=9,
        min_zoom=1,
        start_date="2002-07-04",
        description="Afternoon equatorial crossing providing complementary cloud-filtered surface observation.",
        wmts_endpoint="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        type="true_color",
    ),
    "MODIS_Terra_CorrectedReflectance_Bands721": GIBSLayer(
        id="MODIS_Terra_CorrectedReflectance_Bands721",
        title="MODIS Terra False Color (Bands 7-2-1)",
        sensor="MODIS",
        satellite="Terra",
        resolution="250m",
        format="jpg",
        tile_matrix_set="GoogleMapsCompatible_Level9",
        max_zoom=9,
        min_zoom=1,
        start_date="2000-02-24",
        description="Shortwave infrared false color composite (SWIR 2.1µm, NIR 0.86µm, Red 0.65µm) for burn scar and flood detection.",
        wmts_endpoint="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_Bands721/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        type="false_color",
    ),
    "AIRS_Precipitation_Day": GIBSLayer(
        id="AIRS_Precipitation_Day",
        title="AIRS Daily Precipitation",
        sensor="AIRS",
        satellite="Aqua",
        resolution="2km",
        format="png",
        tile_matrix_set="GoogleMapsCompatible_Level6",
        max_zoom=6,
        min_zoom=1,
        start_date="2002-09-01",
        description="Atmospheric Infrared Sounder daily global precipitation estimate.",
        wmts_endpoint="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/AIRS_Precipitation_Day/default/{time}/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png",
        type="atmospheric",
    ),
}


class GIBSService:
    @staticmethod
    def get_default_date() -> str:
        """Returns yesterday's date in YYYY-MM-DD format as standard safe GIBS date."""
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")

    @classmethod
    def list_layers(cls) -> List[Dict[str, Any]]:
        default_date = cls.get_default_date()
        layers = []
        for layer in GIBS_LAYERS.values():
            data = layer.model_dump()
            data["default_date"] = default_date
            data["tile_url_template"] = data["wmts_endpoint"].replace("{time}", default_date)
            layers.append(data)
        return layers

    @classmethod
    def get_layer(cls, layer_id: str) -> Optional[GIBSLayer]:
        return GIBS_LAYERS.get(layer_id)

    @classmethod
    def build_tile_url(cls, layer_id: str, date_str: str) -> Optional[str]:
        layer = cls.get_layer(layer_id)
        if not layer:
            return None
        return layer.wmts_endpoint.replace("{time}", date_str)

    @classmethod
    async def geocode_place(cls, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Geocodes a place name query using standard geocoding service."""
        if not query or len(query.strip()) < 2:
            return []
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query.strip(), "format": "json", "limit": min(limit, 10)}
        headers = {"User-Agent": "QuantumEarthLab/2.0 (Scientific Research Client)"}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.get(url, params=params, headers=headers)
                if res.status_code != 200:
                    return []
                items = res.json()
                results = []
                for it in items:
                    bbox = it.get("boundingbox", [])
                    parsed_bbox = None
                    if len(bbox) == 4:
                        parsed_bbox = {
                            "south": float(bbox[0]),
                            "north": float(bbox[1]),
                            "west": float(bbox[2]),
                            "east": float(bbox[3]),
                        }
                    results.append({
                        "name": it.get("name") or it.get("display_name", "").split(",")[0],
                        "display_name": it.get("display_name", ""),
                        "lat": float(it["lat"]),
                        "lng": float(it["lon"]),
                        "type": it.get("type", "location"),
                        "boundingbox": parsed_bbox,
                    })
                return results
        except Exception:
            return []


gibs_service = GIBSService()
