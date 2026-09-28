import React, { useState, useEffect, useRef, useCallback } from "react";

// EuroSAT 10-Class Ground Truth Taxonomy
const EUROSAT_CLASSES = [
  { id: 0, name: "AnnualCrop", label: "Annual Cropland", description: "Cultivated fields with annual seasonal crops (cereals, legumes, oilseeds)." },
  { id: 1, name: "Forest", label: "Forest & Woodland", description: "Continuous broadleaved and coniferous tree canopies." },
  { id: 2, name: "HerbaceousVegetation", label: "Herbaceous Vegetation", description: "Natural grasslands, prairies, and open vegetative pastures." },
  { id: 3, name: "Highway", label: "Highway & Major Roads", description: "Asphalt and concrete transportation corridors with high spatial line structures." },
  { id: 4, name: "Industrial", label: "Industrial Buildings", description: "Manufacturing complexes, logistics warehouses, and commercial impervious surfaces." },
  { id: 5, name: "Pasture", label: "Pastureland", description: "Grazing areas and agricultural meadow systems." },
  { id: 6, name: "PermanentCrop", label: "Permanent Cropland", description: "Orchards, vineyards, olive groves, and perennial fruit plantations." },
  { id: 7, name: "Residential", label: "Residential Buildings", description: "Urban and suburban residential housing, neighborhood grids, and roads." },
  { id: 8, name: "River", label: "River & Watercourse", description: "Linear freshwater watercourses, canals, and riverbanks with turbidity." },
  { id: 9, name: "SeaLake", label: "Sea & Inland Lake", description: "Deep open marine waters, estuaries, and inland lake reservoirs." },
];

export default function AiAnalysis({ backendTelemetry, stagedRoi, onClearStagedRoi }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [activeBenchmark, setActiveBenchmark] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [spectralMode, setSpectralMode] = useState("rgb"); // rgb | cir | ndvi
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  // Inference state
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [inferenceError, setInferenceError] = useState("");

  // Benchmark metrics modal state
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [metricsData, setMetricsData] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  const fileInputRef = useRef(null);

  const classicalModelStatus = backendTelemetry?.models?.classical_resnet18_rgb;
  const isModelConnected = classicalModelStatus?.status === "connected" && classicalModelStatus?.available;

  // Handle file selection
  const handleFileReceive = useCallback((candidate) => {
    if (!candidate) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(candidate.type)) {
      setErrorMessage("Please provide a valid PNG, JPEG, or WebP image.");
      return;
    }
    if (candidate.size > 10 * 1024 * 1024) {
      setErrorMessage("Image exceeds the maximum allowed size of 10 MB.");
      return;
    }

    setActiveBenchmark(null);
    setFile(candidate);
    setErrorMessage("");
    setInferenceResult(null);
    setInferenceError("");
  }, []);

  // Sync preview URL when file changes
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setFileMeta({
      name: file.name,
      size: (file.size / (1024 * 1024)).toFixed(2) + " MB",
      type: file.type || "image/jpeg",
      source: "Uploaded Local Image",
    });
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Clipboard paste support (Ctrl+V)
  useEffect(() => {
    function handlePaste(e) {
      if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        const pasted = e.clipboardData.files[0];
        if (pasted.type.startsWith("image/")) {
          handleFileReceive(pasted);
        }
      }
    }
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [handleFileReceive]);

  // Load genuine EuroSAT benchmark sample from backend
  const loadEuroSatBenchmark = async (className) => {
    setErrorMessage("");
    setInferenceResult(null);
    setInferenceError("");
    setFile(null);

    const targetUrl = `http://127.0.0.1:8000/api/dataset/eurosat/sample-image?class_name=${className}&index=1`;
    try {
      const resp = await fetch(targetUrl);
      if (!resp.ok) throw new Error(`Failed fetching ${className} sample from server.`);
      const blob = await resp.blob();
      const syntheticFile = new File([blob], `eurosat_${className}_sample.jpg`, { type: "image/jpeg" });
      setFile(syntheticFile);
      setActiveBenchmark(className);
      setFileMeta({
        name: `EuroSAT Benchmark: ${className}`,
        size: "64 × 64 px · Sentinel-2 MSI",
        type: "image/jpeg",
        source: `EuroSAT Ground Truth: ${className}`,
      });
    } catch (err) {
      setErrorMessage(`Could not load sample: ${err.message}`);
    }
  };

  // Reset workspace
  const handleReset = () => {
    setFile(null);
    setPreviewUrl("");
    setActiveBenchmark(null);
    setInferenceResult(null);
    setInferenceError("");
    setErrorMessage("");
    setFileMeta(null);
    if (onClearStagedRoi) onClearStagedRoi();
  };

  // Run Real PyTorch ResNet-18 Inference
  const handleRunInference = async () => {
    if (!file && !previewUrl) {
      setInferenceError("Please upload an image or choose a benchmark sample first.");
      return;
    }
    if (!isModelConnected) {
      setInferenceError("Classical model is not connected. A genuine checkpoint is required to execute inference.");
      return;
    }

    setIsInferring(true);
    setInferenceError("");

    try {
      const formData = new FormData();
      if (file) {
        formData.append("file", file);
      } else {
        // Fetch preview blob if file reference was reset
        const blobResp = await fetch(previewUrl);
        const blob = await blobResp.blob();
        formData.append("file", blob, "tile.jpg");
      }

      const response = await fetch("http://127.0.0.1:8000/api/models/classical/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail?.message || errJson.detail || `Inference HTTP Error ${response.status}`);
      }

      const data = await response.json();
      setInferenceResult(data.inference);
    } catch (err) {
      setInferenceError(err.message || "Failed to execute inference.");
      setInferenceResult(null);
    } finally {
      setIsInferring(false);
    }
  };

  // Fetch Evaluation Metrics
  const handleOpenMetrics = async () => {
    setShowMetricsModal(true);
    if (metricsData) return;

    setLoadingMetrics(true);
    try {
      const resp = await fetch("http://127.0.0.1:8000/api/models/classical/metrics");
      if (!resp.ok) throw new Error("Failed fetching evaluation metrics.");
      const data = await resp.json();
      setMetricsData(data);
    } catch (err) {
      setMetricsData({ error: err.message });
    } finally {
      setLoadingMetrics(false);
    }
  };

  return (
    <div className="ai-analysis-workspace">
      {/* ROI Integrity Callout */}
      {stagedRoi && (
        <div className="staged-roi-alert-card" role="region" aria-label="Staged Region of Interest">
          <div className="staged-roi-badge">
            <span className="hud-dot" />
            <span>ORBITAL FOOTPRINT STAGED FROM EARTH EXPLORER</span>
          </div>
          <div className="staged-roi-meta">
            <strong>{stagedRoi.name}</strong> · {stagedRoi.specs}
            <br />
            <span>
              Bounding Box: [{stagedRoi.roiBox.west}° W, {stagedRoi.roiBox.south}° S, {stagedRoi.roiBox.east}° E, {stagedRoi.roiBox.north}° N] · {stagedRoi.areaKm2.toLocaleString()} km² Surface Area
            </span>
          </div>
          <div className="roi-integrity-notice">
            <span className="notice-icon">⚠️</span>
            <span className="notice-text">
              <strong>Scientific Data Separation:</strong> NASA GIBS tiles are 8-bit visual web representations (EPSG:3857), distinct from calibrated 13-band Level-2A surface reflectance scientific data. The classical ResNet-18 model accepts 64×64 calibrated imagery.
            </span>
          </div>
        </div>
      )}

      {/* Benchmark Sample Quick Picker */}
      <div className="preset-bar">
        <span className="preset-bar-title">EUROSAT BENCHMARK SAMPLES:</span>
        <div className="preset-chips">
          {EUROSAT_CLASSES.map((cls) => (
            <button
              key={cls.name}
              type="button"
              className={`preset-chip ${activeBenchmark === cls.name ? "is-active" : ""}`}
              onClick={() => loadEuroSatBenchmark(cls.name)}
              title={cls.description}
            >
              <span className="preset-chip-dot" />
              {cls.label}
            </button>
          ))}
          {(previewUrl || file) && (
            <button type="button" className="preset-chip preset-chip--clear" onClick={handleReset}>
              ✕ Reset
            </button>
          )}
        </div>
      </div>

      <div className="lab-grid">
        {/* Upload & Imagery Dropzone */}
        <div
          className={`upload-area ${isDragging ? "is-dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFileReceive(e.dataTransfer.files?.[0]);
          }}
          tabIndex={0}
          role="region"
          aria-label="Satellite imagery dropzone and preview"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          {previewUrl ? (
            <div className={`preview-wrapper spectral-filter--${spectralMode}`}>
              <img className="upload-preview" src={previewUrl} alt="Satellite observation tile" />
              {activeBenchmark && (
                <div className="benchmark-watermark">
                  <span>EUROSAT GROUND TRUTH: {activeBenchmark.toUpperCase()}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="upload-orbit" aria-hidden="true">
              <span className="orbit-glyph">＋</span>
            </div>
          )}

          <div className="upload-copy">
            <p className="upload-kicker">
              {previewUrl ? "ORBITAL TILE LOADED" : "INPUT / SATELLITE IMAGERY"}
            </p>
            <h3>{previewUrl ? fileMeta?.name : "Drop an Earth satellite tile."}</h3>
            <p>
              {previewUrl
                ? "Image staged for neural inference. Click the inference action below to execute classification."
                : "Drag & drop a satellite image, paste from clipboard (Ctrl+V), or select an authentic EuroSAT benchmark sample above."}
            </p>

            {/* Spectral band simulation */}
            {previewUrl && (
              <div className="spectral-toggle-bar">
                <span className="spectral-label">SPECTRAL SIMULATION:</span>
                <div className="spectral-buttons">
                  <button
                    type="button"
                    className={`spectral-btn ${spectralMode === "rgb" ? "is-active" : ""}`}
                    onClick={() => setSpectralMode("rgb")}
                  >
                    RGB Natural
                  </button>
                  <button
                    type="button"
                    className={`spectral-btn ${spectralMode === "cir" ? "is-active" : ""}`}
                    onClick={() => setSpectralMode("cir")}
                  >
                    CIR False Color
                  </button>
                  <button
                    type="button"
                    className={`spectral-btn ${spectralMode === "ndvi" ? "is-active" : ""}`}
                    onClick={() => setSpectralMode("ndvi")}
                  >
                    NDVI Vegetation
                  </button>
                </div>
              </div>
            )}

            <div className="upload-actions">
              <button
                type="button"
                className="button button--light"
                onClick={() => fileInputRef.current?.click()}
              >
                {previewUrl ? "Change Image" : "Upload Local Tile"} <span aria-hidden="true">↗</span>
              </button>
              {previewUrl && (
                <button type="button" className="button button--outline" onClick={handleReset}>
                  Clear
                </button>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              aria-label="Upload satellite image file"
              hidden
              onChange={(e) => {
                handleFileReceive(e.target.files?.[0]);
                e.target.value = "";
              }}
            />

            {errorMessage && (
              <p className="file-error" role="alert">
                ⚠️ {errorMessage}
              </p>
            )}

            {fileMeta && (
              <div className="file-meta-box">
                <div><span>Source:</span> <strong>{fileMeta.source}</strong></div>
                <div><span>Format:</span> <strong>{fileMeta.size}</strong></div>
                <div><span>MIME:</span> <strong>{fileMeta.type}</strong></div>
              </div>
            )}

            {/* Inference Action Trigger */}
            <div className="inference-cta-box">
              <button
                type="button"
                className={`button button--primary inference-btn ${isInferring ? "is-loading" : ""}`}
                disabled={!previewUrl || isInferring || !isModelConnected}
                onClick={handleRunInference}
              >
                {isInferring ? (
                  <>
                    <span className="spinner" />
                    <span>Executing ResNet-18 Forward Pass...</span>
                  </>
                ) : (
                  <>
                    <span>⚡ Run Classical Inference (ResNet-18)</span>
                    <span className="inference-shortcut">64×64 RGB</span>
                  </>
                )}
              </button>

              {!isModelConnected && (
                <p className="inference-disabled-note">
                  🔒 Classical model is not connected. A trained checkpoint is required to execute genuine inference.
                </p>
              )}

              {inferenceError && (
                <div className="inference-error-card" role="alert">
                  <strong>Inference Error:</strong> {inferenceError}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Results & Live Telemetry Panel */}
        <div className="results-area">
          <div className="results-top">
            <span className="results-badge">REAL INFERENCE & MODEL TELEMETRY</span>
            <span className={`live-tag ${isModelConnected ? "live-tag--online" : "live-tag--offline"}`}>
              {isModelConnected ? "ONLINE · GENUINE WEIGHTS" : "NOT CONNECTED"}
            </span>
          </div>

          {/* Real Inference Classification Output */}
          {inferenceResult ? (
            <div className="inference-result-card" role="region" aria-label="Classification Result">
              <div className="inference-result-header">
                <span className="result-kicker">PREDICTED LAND USE / LAND COVER</span>
                <span className="inference-runtime-tag">
                  ⚡ {inferenceResult.inference_runtime_ms} ms · {inferenceResult.execution_device.toUpperCase()}
                </span>
              </div>

              <div className="inference-class-hero">
                <div className="class-hero-left">
                  <h3 className="hero-class-name">{inferenceResult.predicted_label}</h3>
                  <span className="hero-class-id">Taxonomy: {inferenceResult.predicted_class}</span>
                  <p className="hero-class-desc">{inferenceResult.predicted_description}</p>
                </div>
                <div className="class-hero-right">
                  <div className="confidence-dial">
                    <span className="dial-value">{inferenceResult.confidence_percentage}%</span>
                    <span className="dial-label">CONFIDENCE</span>
                  </div>
                </div>
              </div>

              {/* Probability Distribution Bar Chart */}
              <div className="probability-distribution-panel">
                <div className="dist-header">
                  <span className="dist-title">PREDICTION PROBABILITY DISTRIBUTION (ALL 10 CLASSES)</span>
                  <span className="dist-subtitle">Softmax Output</span>
                </div>

                <div className="probability-bars-list">
                  {inferenceResult.ranked_predictions.map((p) => {
                    const isTop = p.class === inferenceResult.predicted_class;
                    return (
                      <div key={p.class} className={`prob-row ${isTop ? "is-top-rank" : ""}`}>
                        <div className="prob-label-col">
                          <span className="prob-class-name">{p.label}</span>
                          <span className="prob-class-tag">{p.class}</span>
                        </div>
                        <div className="prob-bar-col">
                          <div className="prob-track">
                            <div
                              className={`prob-fill ${isTop ? "prob-fill--top" : ""}`}
                              style={{ width: `${Math.max(p.percentage, 0.5)}%` }}
                            />
                          </div>
                        </div>
                        <div className="prob-value-col">
                          <span className="prob-pct">{p.percentage.toFixed(2)}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Model Provenance & Checkpoint Metadata */}
              <div className="provenance-footer">
                <div className="prov-item">
                  <span>Architecture:</span>
                  <strong>{inferenceResult.model_version}</strong>
                </div>
                <div className="prov-item">
                  <span>Preprocessing:</span>
                  <strong>{inferenceResult.preprocessing_version}</strong>
                </div>
                <div className="prov-item">
                  <span>Runtime:</span>
                  <strong>{inferenceResult.inference_runtime_ms} ms</strong>
                </div>
                <div className="prov-item">
                  <span>Device:</span>
                  <strong>{inferenceResult.execution_device}</strong>
                </div>
              </div>
            </div>
          ) : (
            <div className="inference-placeholder-card">
              <div className="placeholder-icon">◉</div>
              <h4>Awaiting Inference Execution</h4>
              <p>
                Select an image on the left and click <strong>Run Classical Inference</strong> to evaluate the satellite tile through the trained ResNet-18 model.
              </p>
              <div className="placeholder-specs">
                <span>Dataset: EuroSAT 10-Class (RGB Baseline)</span>
                <span>Pretrained: ImageNet-1K Transfer Learning</span>
                <span>Preprocessing: Standardized (Mean/Std)</span>
              </div>
            </div>
          )}

          {/* Model Pipeline Status Overview */}
          <div className="model-status-overview">
            <div className="overview-row">
              <div className="result-symbol result-symbol--classical">◉</div>
              <div className="result-content">
                <p className="result-title">RESNET-18 (EUROSAT RGB BASELINE)</p>
                <div className="result-status-row">
                  <span className={`status-dot ${isModelConnected ? "status-dot--connected" : "status-dot--pending"}`} />
                  <strong className="status-text">{isModelConnected ? "Connected (Online)" : "Not connected"}</strong>
                </div>
                <small className="result-hint">
                  {classicalModelStatus?.reason || "Awaiting trained weights from Milestone B."}
                </small>
              </div>
              {isModelConnected && (
                <button
                  type="button"
                  className="metrics-modal-btn"
                  onClick={handleOpenMetrics}
                  title="View test accuracy, confusion matrix, and per-class metrics"
                >
                  Metrics 📊
                </button>
              )}
            </div>

            <div className="overview-row">
              <div className="result-symbol result-symbol--classical">▦</div>
              <div className="result-content">
                <p className="result-title">MULTISPECTRAL RESNET-18 (13 BANDS)</p>
                <div className="result-status-row">
                  <span className="status-dot status-dot--pending" />
                  <strong className="status-text">Not connected</strong>
                </div>
                <small className="result-hint">
                  13-band Sentinel-2 Level-2A surface reflectance pipeline. Separated from 3-band RGB.
                </small>
              </div>
            </div>

            <div className="overview-row">
              <div className="result-symbol result-symbol--quantum">✳</div>
              <div className="result-content">
                <p className="result-title">HYBRID QUANTUM CLASSIFIER (8-QUBIT VQC)</p>
                <div className="result-status-row">
                  <span className="status-dot status-dot--pending" />
                  <strong className="status-text">Not connected</strong>
                </div>
                <small className="result-hint">
                  Awaiting quantum circuit simulation in Milestone C. No quantum advantage claimed without empirical proof.
                </small>
              </div>
            </div>
          </div>

          <div className="integrity-box">
            <h4>🔬 Scientific Integrity Policy</h4>
            <p>
              In accordance with research integrity standards, zero model predictions, inference latencies, or accuracy percentages are fabricated or mocked. All inference outputs are computed genuinely by PyTorch on the host runtime.
            </p>
          </div>
        </div>
      </div>

      {/* Evaluation Metrics & Confusion Matrix Modal */}
      {showMetricsModal && (
        <div className="metrics-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="modal-title">
          <div className="metrics-modal-content">
            <div className="modal-header">
              <h3 id="modal-title">EuroSAT ResNet-18 Classical Evaluation Report</h3>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setShowMetricsModal(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              {loadingMetrics && <p className="modal-loading">Loading verified evaluation metrics...</p>}

              {metricsData && metricsData.metrics && (
                <div className="metrics-summary-view">
                  {/* Summary Metric Pills */}
                  <div className="metrics-cards-row">
                    <div className="metric-kpi-card">
                      <span className="kpi-label">TEST ACCURACY (TOP-1)</span>
                      <strong className="kpi-value">{(metricsData.metrics.overall_accuracy * 100).toFixed(2)}%</strong>
                      <span className="kpi-sub">{metricsData.total_test_samples.toLocaleString()} Test Patches</span>
                    </div>
                    <div className="metric-kpi-card">
                      <span className="kpi-label">MACRO F1-SCORE</span>
                      <strong className="kpi-value">{(metricsData.metrics.macro_f1_score * 100).toFixed(2)}%</strong>
                      <span className="kpi-sub">Unweighted Class Average</span>
                    </div>
                    <div className="metric-kpi-card">
                      <span className="kpi-label">WEIGHTED F1-SCORE</span>
                      <strong className="kpi-value">{(metricsData.metrics.weighted_f1_score * 100).toFixed(2)}%</strong>
                      <span className="kpi-sub">Support Weighted</span>
                    </div>
                  </div>

                  {/* Per-Class Performance Table */}
                  <h4 className="metrics-subheading">Per-Class Performance (Unseen Test Set)</h4>
                  <div className="table-responsive">
                    <table className="metrics-table">
                      <thead>
                        <tr>
                          <th>Class</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1-Score</th>
                          <th>Test Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(metricsData.per_class).map(([clsName, stats]) => (
                          <tr key={clsName}>
                            <td><strong>{clsName}</strong></td>
                            <td>{(stats.precision * 100).toFixed(1)}%</td>
                            <td>{(stats.recall * 100).toFixed(1)}%</td>
                            <td><strong>{(stats.f1_score * 100).toFixed(1)}%</strong></td>
                            <td>{stats.support}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Confusion Matrix Section */}
                  <h4 className="metrics-subheading">Confusion Matrix (10 × 10 Classes)</h4>
                  <div className="cm-preview-wrapper">
                    <img
                      src="/experiments/confusion_matrix.png"
                      alt="EuroSAT Confusion Matrix Heatmap"
                      className="cm-image"
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                    <p className="cm-caption">
                      Verified confusion matrix computed over all 4,050 unseen EuroSAT test split images.
                    </p>
                  </div>
                </div>
              )}

              {metricsData && metricsData.error && (
                <div className="modal-error">
                  <p>Error loading metrics: {metricsData.error}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
