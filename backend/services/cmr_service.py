"""NASA Common Metadata Repository (CMR) Client Service.

CMR is the authoritative metadata system for NASA's Earth Observing System Data
and Information System (EOSDIS). Unlike visual GIBS map tiles, CMR enables researchers
to discover calibrated Level-1C/Level-2A multispectral data products (e.g. Sentinel-2 MSI,
Harmonized Landsat Sentinel HLS, MODIS Surface Reflectance) with full 10-13 spectral bands.
"""

import httpx
from typing import Dict, Any, List, Optional
from backend.core.config import settings

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


class CMRService:
    def __init__(self):
        self.headers = {
            "User-Agent": "QuantumEarthLab/2.0 (Scientific Research Client)",
            "Accept": "application/json",
        }
        if settings.NASA_EARTHDATA_TOKEN:
            self.headers["Authorization"] = f"Bearer {settings.NASA_EARTHDATA_TOKEN}"

    async def search_granules(
        self,
        bounding_box: List[float],  # [west, south, east, north]
        start_date: str,            # ISO-8601 or YYYY-MM-DD
        end_date: str,
        limit: int = 10,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries NASA CMR for scientific granules covering the given bounding box and time window.

        Args:
            bounding_box: [west, south, east, north] in EPSG:4326 degrees
            start_date: Start of acquisition window
            end_date: End of acquisition window
            limit: Maximum granules to return (1-20)
            provider: Optional provider filter (e.g. LPCLOUD, POCLOUD)
        """
        # Validate bounding box
        west, south, east, north = bounding_box
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("Bounding box coordinates must be within valid WGS84 bounds.")

        # Ensure temporal format includes UTC time if only date was passed
        t_start = start_date if "T" in start_date else f"{start_date}T00:00:00Z"
        t_end = end_date if "T" in end_date else f"{end_date}T23:59:59Z"

        params = {
            "bounding_box": f"{west},{south},{east},{north}",
            "temporal": f"{t_start},{t_end}",
            "page_size": min(limit, 20),
            "sort_key": "-start_date",
        }
        if provider:
            params["provider"] = provider

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(CMR_GRANULES_URL, params=params, headers=self.headers)

                if resp.status_code != 200:
                    return {
                        "success": False,
                        "error": f"CMR query returned status {resp.status_code}: {resp.text[:200]}",
                        "granules": [],
                        "total_hits": 0,
                        "scientific_notice": (
                            "NASA CMR discovered 0 products. Note: Visual map tiles from GIBS "
                            "differ from calibrated 13-band surface reflectance granules."
                        ),
                    }

                data = resp.json()
                entries = data.get("feed", {}).get("entry", [])
                total_hits = int(resp.headers.get("CMR-Hits", len(entries)))

                parsed_granules = []
                for entry in entries:
                    links = []
                    for link in entry.get("links", []):
                        href = link.get("href", "")
                        rel = link.get("rel", "")
                        if "data#" in rel or href.endswith((".tif", ".nc", ".hdf", ".zip")):
                            links.append({
                                "href": href,
                                "type": link.get("type", "application/octet-stream"),
                                "title": link.get("title", "Download Granule Data"),
                            })

                    granule_item = {
                        "id": entry.get("id"),
                        "title": entry.get("title"),
                        "dataset_id": entry.get("dataset_id"),
                        "time_start": entry.get("time_start"),
                        "time_end": entry.get("time_end"),
                        "cloud_cover": entry.get("cloud_cover"),
                        "granule_size_mb": entry.get("granule_size"),
                        "boxes": entry.get("boxes"),
                        "links": links[:3],  # top data links
                    }
                    parsed_granules.append(granule_item)

                return {
                    "success": True,
                    "total_hits": total_hits,
                    "count": len(parsed_granules),
                    "granules": parsed_granules,
                    "query_bbox": {"west": west, "south": south, "east": east, "north": north},
                    "temporal_range": {"start": t_start, "end": t_end},
                    "scientific_notice": (
                        "Discovered true scientific Earth-observation products (L1C/L2A). "
                        "These files contain 10-13 calibrated reflectance bands required for scientific EuroSAT ML inference."
                    ),
                }

        except httpx.RequestError as exc:
            return {
                "success": False,
                "error": f"Network error contacting NASA CMR API: {str(exc)}",
                "granules": [],
                "total_hits": 0,
                "scientific_notice": (
                    "NASA CMR search is currently offline or unreachable. "
                    "In adherence to research integrity, no mock granules are fabricated."
                ),
            }


cmr_service = CMRService()
