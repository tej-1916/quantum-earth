import React, { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Supported NASA Earthdata GIBS layers
const GIBS_LAYERS = [
  {
    id: "MODIS_Terra_CorrectedReflectance_TrueColor",
    name: "MODIS Terra True Color",
    sensor: "MODIS / Terra",
    res: "250m",
    format: "jpg",
    matrixSet: "GoogleMapsCompatible_Level9",
    maxZoom: 9,
    desc: "Natural-color daily planetary composite from morning orbital pass.",
    type: "true_color",
  },
  {
    id: "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    name: "VIIRS SNPP True Color",
    sensor: "VIIRS / Suomi NPP",
    res: "250m",
    format: "jpg",
    matrixSet: "GoogleMapsCompatible_Level9",
    maxZoom: 9,
    desc: "Day-night radiometer true color composite with sharp edge detail.",
    type: "true_color",
  },
  {
    id: "MODIS_Aqua_CorrectedReflectance_TrueColor",
    name: "MODIS Aqua True Color",
    sensor: "MODIS / Aqua",
    res: "250m",
    format: "jpg",
    matrixSet: "GoogleMapsCompatible_Level9",
    maxZoom: 9,
    desc: "Afternoon orbital observation for cloud-differential comparison.",
    type: "true_color",
  },
  {
    id: "MODIS_Terra_CorrectedReflectance_Bands721",
    name: "MODIS False Color (7-2-1)",
    sensor: "MODIS / Terra",
    res: "250m",
    format: "jpg",
    matrixSet: "GoogleMapsCompatible_Level9",
    maxZoom: 9,
    desc: "SWIR/NIR false color highlighting vegetation stress and burn scars.",
    type: "false_color",
  },
  {
    id: "AIRS_Precipitation_Day",
    name: "AIRS Daily Precipitation",
    sensor: "AIRS / Aqua",
    res: "2km",
    format: "png",
    matrixSet: "GoogleMapsCompatible_Level6",
    maxZoom: 6,
    desc: "Atmospheric sounding daily precipitation estimate.",
    type: "atmospheric",
  },
];

// Curated geographic interest presets
const LOCATION_PRESETS = [
  { name: "Cape Hatteras, USA", lat: 35.25, lng: -75.52, zoom: 7, desc: "Ocean barrier spits & sediment dynamics" },
  { name: "Nile Delta, Egypt", lat: 30.80, lng: 31.00, zoom: 7, desc: "Dense irrigated agricultural delta & Mediterranean coast" },
  { name: "Tokyo Bay, Japan", lat: 35.50, lng: 139.90, zoom: 7, desc: "High-density maritime infrastructure and urban sprawl" },
  { name: "Finney County, USA", lat: 37.99, lng: -100.87, zoom: 7, desc: "Center-pivot irrigation circles in High Plains" },
  { name: "Richat Structure, Mauritania", lat: 21.12, lng: -11.40, zoom: 8, desc: "40km circular geologic dome in Sahara Desert" },
  { name: "Amazon Basin, Brazil", lat: -2.50, lng: -54.80, zoom: 7, desc: "Dense equatorial rainforest and river channels" },
];

function getYesterdayDateStr() {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().split("T")[0];
}

// Compute spherical surface area of bounding box in square kilometers
function calculateBBoxAreaKm2(west, south, east, north) {
  const r = 6371; // Earth radius in km
  const rad = Math.PI / 180;
  const sinN = Math.sin(north * rad);
  const sinS = Math.sin(south * rad);
  const dLon = Math.abs(east - west);
  const area = Math.abs(sinN - sinS) * (dLon * rad) * (r * r);
  return Math.round(area);
}

export default function EarthExplorer() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const gibsTileLayerRef = useRef(null);
  const roiLayerRef = useRef(null);

  const [selectedLayerId, setSelectedLayerId] = useState(GIBS_LAYERS[0].id);
  const [selectedDate, setSelectedDate] = useState(getYesterdayDateStr());
  const [activePreset, setActivePreset] = useState(LOCATION_PRESETS[0].name);

  // Manual coordinate jump
  const [inputLat, setInputLat] = useState("35.25");
  const [inputLng, setInputLng] = useState("-75.52");

  // Region of Interest (ROI) State
  const [isDrawingRoi, setIsDrawingRoi] = useState(false);
  const [roiBox, setRoiBox] = useState({
    west: -76.5,
    south: 34.5,
    east: -74.5,
    north: 36.0,
  });

  // CMR Scientific Data Discovery State
  const [cmrLoading, setCmrLoading] = useState(false);
  const [cmrResults, setCmrResults] = useState(null);
  const [cmrError, setCmrError] = useState(null);
  const [copiedGeoJson, setCopiedGeoJson] = useState(false);

  const activeLayer = GIBS_LAYERS.find((l) => l.id === selectedLayerId) || GIBS_LAYERS[0];

  // Helper to build GIBS WMTS URL template
  const getGibsUrl = useCallback((layerId, dateStr) => {
    const layer = GIBS_LAYERS.find((l) => l.id === layerId) || GIBS_LAYERS[0];
    return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer.id}/default/${dateStr}/${layer.matrixSet}/{z}/{y}/{x}.${layer.format}`;
  }, []);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapRef.current) return; // already initialized

    const initialPreset = LOCATION_PRESETS[0];
    const map = L.map(mapContainerRef.current, {
      center: [initialPreset.lat, initialPreset.lng],
      zoom: initialPreset.zoom,
      minZoom: 1,
      maxZoom: 9,
      attributionControl: false,
      zoomControl: true,
    });

    // High-resolution satellite backdrop base layer (clean, no watermark)
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 18,
      attribution: "Esri, Maxar, Earthstar Geographics",
    }).addTo(map);

    // Initial NASA GIBS tile layer
    const gibsUrl = getGibsUrl(selectedLayerId, selectedDate);
    const gibsLayer = L.tileLayer(gibsUrl, {
      maxZoom: activeLayer.maxZoom,
      minZoom: 1,
      opacity: 0.95,
      tileSize: 256,
    }).addTo(map);
    gibsTileLayerRef.current = gibsLayer;

    // ROI Rectangle Layer
    const bounds = [
      [roiBox.south, roiBox.west],
      [roiBox.north, roiBox.east],
    ];
    const roiRect = L.rectangle(bounds, {
      color: "#0df2c9",
      weight: 2,
      dashArray: "6, 6",
      fillColor: "#0df2c9",
      fillOpacity: 0.12,
    }).addTo(map);
    roiLayerRef.current = roiRect;

    // Event listener for map moves to sync coordinates and viewport
    map.on("moveend", () => {
      const c = map.getCenter();
      setInputLat(c.lat.toFixed(4));
      setInputLng(c.lng.toFixed(4));
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update GIBS Tile Layer when layer or date changes
  useEffect(() => {
    if (!mapRef.current) return;

    if (gibsTileLayerRef.current) {
      mapRef.current.removeLayer(gibsTileLayerRef.current);
    }

    const gibsUrl = getGibsUrl(selectedLayerId, selectedDate);
    const newGibsLayer = L.tileLayer(gibsUrl, {
      maxZoom: activeLayer.maxZoom,
      minZoom: 1,
      opacity: 0.95,
      tileSize: 256,
    }).addTo(mapRef.current);

    gibsTileLayerRef.current = newGibsLayer;

    // Bring ROI to front
    if (roiLayerRef.current) {
      roiLayerRef.current.bringToFront();
    }
  }, [selectedLayerId, selectedDate, getGibsUrl, activeLayer.maxZoom]);

  // Update ROI Rectangle on map when state changes
  useEffect(() => {
    if (!roiLayerRef.current) return;
    const bounds = [
      [roiBox.south, roiBox.west],
      [roiBox.north, roiBox.east],
    ];
    roiLayerRef.current.setBounds(bounds);
  }, [roiBox]);

  // Handle ROI Click & Drag Drawing
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (!isDrawingRoi) {
      map.dragging.enable();
      return;
    }

    map.dragging.disable();
    let startLatLng = null;

    const onMouseDown = (e) => {
      startLatLng = e.latlng;
    };

    const onMouseMove = (e) => {
      if (!startLatLng) return;
      const currentLatLng = e.latlng;
      const west = Math.min(startLatLng.lng, currentLatLng.lng);
      const east = Math.max(startLatLng.lng, currentLatLng.lng);
      const south = Math.min(startLatLng.lat, currentLatLng.lat);
      const north = Math.max(startLatLng.lat, currentLatLng.lat);

      setRoiBox({
        west: Number(west.toFixed(4)),
        south: Number(south.toFixed(4)),
        east: Number(east.toFixed(4)),
        north: Number(north.toFixed(4)),
      });
    };

    const onMouseUp = () => {
      if (startLatLng) {
        startLatLng = null;
        setIsDrawingRoi(false);
        map.dragging.enable();
      }
    };

    map.on("mousedown", onMouseDown);
    map.on("mousemove", onMouseMove);
    map.on("mouseup", onMouseUp);

    return () => {
      map.off("mousedown", onMouseDown);
      map.off("mousemove", onMouseMove);
      map.off("mouseup", onMouseUp);
      map.dragging.enable();
    };
  }, [isDrawingRoi]);

  // Jump to location preset
  const handleSelectPreset = (preset) => {
    setActivePreset(preset.name);
    setInputLat(preset.lat.toFixed(4));
    setInputLng(preset.lng.toFixed(4));
    if (mapRef.current) {
      mapRef.current.flyTo([preset.lat, preset.lng], preset.zoom, { duration: 1.2 });
    }

    // Auto update ROI box around the preset
    const dLat = 0.8;
    const dLng = 1.0;
    setRoiBox({
      west: Number((preset.lng - dLng).toFixed(4)),
      south: Number((preset.lat - dLat).toFixed(4)),
      east: Number((preset.lng + dLng).toFixed(4)),
      north: Number((preset.lat + dLat).toFixed(4)),
    });
  };

  // Jump to custom coordinates
  const handleCoordinateJump = (e) => {
    e.preventDefault();
    const lat = parseFloat(inputLat);
    const lng = parseFloat(inputLng);
    if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
      if (mapRef.current) {
        mapRef.current.flyTo([lat, lng], 7, { duration: 1.2 });
      }
      setActivePreset("Custom Coordinates");
    }
  };

  // Adjust date +/- 1 day
  const adjustDate = (days) => {
    const d = new Date(selectedDate);
    d.setUTCDate(d.getUTCDate() + days);
    const today = new Date();
    if (d > today) return; // Don't go into the future
    setSelectedDate(d.toISOString().split("T")[0]);
  };

  // Fit current map viewport to ROI box
  const setRoiToViewport = () => {
    if (!mapRef.current) return;
    const b = mapRef.current.getBounds();
    setRoiBox({
      west: Number(b.getWest().toFixed(4)),
      south: Number(b.getSouth().toFixed(4)),
      east: Number(b.getEast().toFixed(4)),
      north: Number(b.getNorth().toFixed(4)),
    });
  };

  // Generate GeoJSON Feature
  const generateGeoJson = () => {
    return {
      type: "Feature",
      properties: {
        layer: activeLayer.id,
        satellite: activeLayer.satellite,
        sensor: activeLayer.sensor,
        acquisition_date: selectedDate,
        area_km2: calculateBBoxAreaKm2(roiBox.west, roiBox.south, roiBox.east, roiBox.north),
        spatial_resolution: activeLayer.res,
        generator: "Quantum Earth Lab V2 (NASA GIBS / CMR Explorer)",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [roiBox.west, roiBox.north],
            [roiBox.east, roiBox.north],
            [roiBox.east, roiBox.south],
            [roiBox.west, roiBox.south],
            [roiBox.west, roiBox.north],
          ],
        ],
      },
    };
  };

  // Download GeoJSON
  const handleDownloadGeoJson = () => {
    const data = generateGeoJson();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `quantum_earth_roi_${selectedDate}_${activeLayer.id}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Copy GeoJSON to clipboard
  const handleCopyGeoJson = () => {
    const data = generateGeoJson();
    navigator.clipboard.writeText(JSON.stringify(data, null, 2)).then(() => {
      setCopiedGeoJson(true);
      setTimeout(() => setCopiedGeoJson(false), 2500);
    });
  };

  // Query NASA CMR for Scientific Granules
  const handleDiscoverGranules = async () => {
    setCmrLoading(true);
    setCmrError(null);
    setCmrResults(null);

    try {
      const url = `http://127.0.0.1:8000/api/cmr/search?west=${roiBox.west}&south=${roiBox.south}&east=${roiBox.east}&north=${roiBox.north}&start_date=${selectedDate}&end_date=${selectedDate}&limit=6`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`CMR Service error: HTTP ${res.status}`);
      }
      const data = await res.json();
      setCmrResults(data);
    } catch (err) {
      setCmrError(err.message || "Failed to contact NASA CMR backend service.");
    } finally {
      setCmrLoading(false);
    }
  };

  const calculatedArea = calculateBBoxAreaKm2(roiBox.west, roiBox.south, roiBox.east, roiBox.north);

  return (
    <div className="explorer-container">
      {/* Top Explorer Control Bar */}
      <div className="explorer-toolbar">
        {/* Layer Selector */}
        <div className="toolbar-group">
          <label htmlFor="gibs-layer-select" className="toolbar-label">
            NASA GIBS LAYER:
          </label>
          <select
            id="gibs-layer-select"
            className="explorer-select"
            value={selectedLayerId}
            onChange={(e) => setSelectedLayerId(e.target.value)}
          >
            {GIBS_LAYERS.map((layer) => (
              <option key={layer.id} value={layer.id}>
                {layer.name} ({layer.res})
              </option>
            ))}
          </select>
        </div>

        {/* Date / Time Controls */}
        <div className="toolbar-group">
          <label htmlFor="gibs-date-input" className="toolbar-label">
            ACQUISITION DATE:
          </label>
          <div className="date-control-wrap">
            <button
              type="button"
              className="date-nav-btn"
              onClick={() => adjustDate(-1)}
              title="Previous Day"
            >
              ◀ -1D
            </button>
            <input
              id="gibs-date-input"
              type="date"
              className="explorer-date-input"
              value={selectedDate}
              max={getYesterdayDateStr()}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
            <button
              type="button"
              className="date-nav-btn"
              onClick={() => adjustDate(1)}
              title="Next Day"
            >
              +1D ▶
            </button>
          </div>
        </div>

        {/* ROI Bounding Box Action Buttons */}
        <div className="toolbar-group toolbar-group--actions">
          <button
            type="button"
            className={`button button--sm ${isDrawingRoi ? "button--light" : "button--outline"}`}
            onClick={() => setIsDrawingRoi(!isDrawingRoi)}
          >
            {isDrawingRoi ? "Drawing Active (Drag Box)" : "Draw ROI Box ⛶"}
          </button>
          <button
            type="button"
            className="button button--sm button--ghost"
            onClick={setRoiToViewport}
            title="Snap ROI box to current map boundaries"
          >
            Fit ROI to View
          </button>
        </div>
      </div>

      {/* Preset Location Quick Pills */}
      <div className="explorer-presets-bar">
        <span className="presets-label">OBSERVATION PRESETS:</span>
        <div className="presets-list">
          {LOCATION_PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              className={`preset-pill ${activePreset === preset.name ? "is-active" : ""}`}
              onClick={() => handleSelectPreset(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Map Canvas */}
      <div className="map-view-wrapper">
        <div
          ref={mapContainerRef}
          className={`leaflet-map-canvas ${isDrawingRoi ? "cursor-crosshair" : ""}`}
          aria-label="Interactive NASA Earthdata GIBS satellite imagery map"
        />

        {/* Floating Telemetry HUD on Map */}
        <div className="map-hud-overlay">
          <div className="hud-badge">
            <span className="hud-dot" />
            <span>{activeLayer.sensor}</span>
          </div>
          <div className="hud-detail">
            <span>RES: {activeLayer.res}</span>
            <span>·</span>
            <span>DATE: {selectedDate}</span>
            <span>·</span>
            <span>GSD: 250M</span>
          </div>
        </div>

        {isDrawingRoi && (
          <div className="drawing-banner" role="status">
            <span>⛶ Drag your mouse on the map to define the Region of Interest (ROI)</span>
            <button type="button" className="drawing-cancel-btn" onClick={() => setIsDrawingRoi(false)}>
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* Bottom Metadata & Scientific Discovery Panel */}
      <div className="explorer-bottom-grid">
        {/* ROI Metrics & GeoJSON Export */}
        <div className="explorer-panel roi-telemetry-panel">
          <div className="panel-header">
            <h4>REGION OF INTEREST (ROI)</h4>
            <span className="telemetry-tag">EPSG:4326 WGS84</span>
          </div>

          <div className="roi-bounds-grid">
            <div className="roi-coord-item">
              <span className="roi-coord-label">NORTH:</span>
              <strong className="roi-coord-val">{roiBox.north}° N</strong>
            </div>
            <div className="roi-coord-item">
              <span className="roi-coord-label">SOUTH:</span>
              <strong className="roi-coord-val">{roiBox.south}° S</strong>
            </div>
            <div className="roi-coord-item">
              <span className="roi-coord-label">WEST:</span>
              <strong className="roi-coord-val">{roiBox.west}° W</strong>
            </div>
            <div className="roi-coord-item">
              <span className="roi-coord-label">EAST:</span>
              <strong className="roi-coord-val">{roiBox.east}° E</strong>
            </div>
          </div>

          <div className="roi-area-row">
            <span>ESTIMATED SPHERICAL AREA:</span>
            <strong>{calculatedArea.toLocaleString()} km²</strong>
          </div>

          <div className="roi-actions-row">
            <button
              type="button"
              className="button button--sm button--light"
              onClick={handleDownloadGeoJson}
            >
              Export GeoJSON ↓
            </button>
            <button
              type="button"
              className="button button--sm button--outline"
              onClick={handleCopyGeoJson}
            >
              {copiedGeoJson ? "✓ Copied to Clipboard" : "Copy GeoJSON"}
            </button>
          </div>

          {/* Coordinate Search Form */}
          <form className="coord-jump-form" onSubmit={handleCoordinateJump}>
            <span className="jump-title">JUMP TO COORDINATES:</span>
            <div className="jump-inputs">
              <input
                type="number"
                step="any"
                min="-90"
                max="90"
                placeholder="Lat"
                value={inputLat}
                onChange={(e) => setInputLat(e.target.value)}
                className="jump-input"
                aria-label="Latitude"
              />
              <input
                type="number"
                step="any"
                min="-180"
                max="180"
                placeholder="Lng"
                value={inputLng}
                onChange={(e) => setInputLng(e.target.value)}
                className="jump-input"
                aria-label="Longitude"
              />
              <button type="submit" className="button button--sm button--ghost">
                Fly To →
              </button>
            </div>
          </form>
        </div>

        {/* NASA CMR / STAC Scientific Granule Discovery */}
        <div className="explorer-panel cmr-discovery-panel">
          <div className="panel-header">
            <h4>NASA CMR SCIENTIFIC PRODUCT DISCOVERY</h4>
            <span className="telemetry-tag telemetry-tag--science">13-BAND MULTISPECTRAL</span>
          </div>

          <p className="cmr-description">
            Map tiles above are 8-bit web composites (GIBS EPSG:3857). Scientific machine learning
            requires calibrated <strong>13-band surface reflectance products (Level-2A)</strong>.
            Search NASA CMR to discover research granules covering this bounding box.
          </p>

          <button
            type="button"
            className="button button--sm button--light cmr-search-btn"
            onClick={handleDiscoverGranules}
            disabled={cmrLoading}
          >
            {cmrLoading ? "Querying NASA CMR API..." : "Discover Scientific Granules (CMR) ↗"}
          </button>

          {cmrError && (
            <div className="cmr-error-box" role="alert">
              <span>⚠️ {cmrError}</span>
              <small>Ensure the local backend is running at <code>http://127.0.0.1:8000</code>.</small>
            </div>
          )}

          {cmrResults && (
            <div className="cmr-results-area">
              <div className="cmr-meta-bar">
                <span>FOUND <strong>{cmrResults.total_hits}</strong> MATCHING GRANULES</span>
                <span className="cmr-badge">L1C / L2A</span>
              </div>

              {cmrResults.granules.length === 0 ? (
                <p className="cmr-empty">
                  No scientific granules cataloged for this exact day in the current bounding box.
                  Expand the date range or select an active orbital pass preset.
                </p>
              ) : (
                <ul className="cmr-granule-list">
                  {cmrResults.granules.map((g) => (
                    <li key={g.id || g.title} className="cmr-granule-item">
                      <div className="granule-header">
                        <strong className="granule-title">{g.title || g.id}</strong>
                        {g.cloud_cover !== null && (
                          <span className="granule-cloud">Cloud: {g.cloud_cover}%</span>
                        )}
                      </div>
                      <div className="granule-specs">
                        <span>Acquisition: {g.time_start?.slice(0, 16).replace("T", " ")} UTC</span>
                        {g.granule_size_mb && <span>· Size: {g.granule_size_mb} MB</span>}
                      </div>
                      {g.links && g.links.length > 0 && (
                        <div className="granule-links">
                          {g.links.map((link, idx) => (
                            <a
                              key={idx}
                              href={link.href}
                              target="_blank"
                              rel="noreferrer"
                              className="granule-link"
                            >
                              Download Data Product ({link.href.split(".").pop()?.toUpperCase() || "HDF/TIF"}) ↗
                            </a>
                          ))}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}

              <p className="scientific-guarantee">
                🔬 <strong>Scientific Grounding:</strong> {cmrResults.scientific_notice}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
