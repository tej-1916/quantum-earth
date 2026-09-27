import React, { useEffect, useRef, useState, useCallback } from "react";

// Approved project asset paths (will display when uploaded to /images/)
const imagePaths = {
  earth: "/images/earth-hero.webp",
  coast: "/images/coastline.webp",
  farm: "/images/farmland.webp",
  city: "/images/urban.webp",
  quantum: "/images/quantum-hardware.webp",
};

// Built-in synthetic satellite observation samples for instant lab testing
const sampleTiles = [
  {
    id: "sample-coast",
    name: "Cape Hatteras Inlet (Coastline)",
    category: "Coast & Estuary",
    specs: "Sentinel-2 MSI · 10m GSD · Bands 2, 3, 4, 8",
    description: "Dynamic barrier spit, shallow tidal shoals, and oceanic sediment plumes.",
    svgType: "coast",
  },
  {
    id: "sample-farm",
    name: "Finney County (Agriculture)",
    category: "Cropland & Irrigation",
    specs: "Sentinel-2 MSI · 10m GSD · Bands 4, 8A, 11",
    description: "Center-pivot irrigation circles and structured geometric crop parcels.",
    svgType: "farm",
  },
  {
    id: "sample-urban",
    name: "Tokyo Bay Port (Urban Infrastructure)",
    category: "Built Environment",
    specs: "Sentinel-2 MSI · 10m GSD · Bands 2, 3, 4, 12",
    description: "High-density maritime harbor, shipping berths, and grid road networks.",
    svgType: "city",
  },
];

// Rich cinematic SVG fallbacks rendered when webp files are not present on disk
function VisualSvg({ type }) {
  if (type === "earth") {
    return (
      <svg className="visual-vector" viewBox="0 0 800 800" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <radialGradient id="earthGlow" cx="42%" cy="38%" r="60%">
            <stop offset="0%" stopColor="#8cfbe8" stopOpacity="0.9" />
            <stop offset="22%" stopColor="#258296" stopOpacity="0.85" />
            <stop offset="48%" stopColor="#123b56" stopOpacity="0.95" />
            <stop offset="72%" stopColor="#08182b" stopOpacity="1" />
            <stop offset="100%" stopColor="#030810" stopOpacity="1" />
          </radialGradient>
          <radialGradient id="atmosphereCorona" cx="50%" cy="50%" r="50%">
            <stop offset="80%" stopColor="#76f3de" stopOpacity="0" />
            <stop offset="92%" stopColor="#53d6c7" stopOpacity="0.25" />
            <stop offset="98%" stopColor="#9bfff2" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#41b4a6" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="nightTerminator" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="45%" stopColor="transparent" />
            <stop offset="78%" stopColor="#02060b" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#010408" stopOpacity="0.98" />
          </linearGradient>
          <filter id="coronaBlur">
            <feGaussianBlur stdDeviation="16" />
          </filter>
        </defs>

        {/* Ambient atmosphere outer haze */}
        <circle cx="400" cy="400" r="340" fill="#329792" opacity="0.18" filter="url(#coronaBlur)" />
        <circle cx="400" cy="400" r="310" fill="url(#atmosphereCorona)" />

        {/* Planet Sphere */}
        <circle cx="400" cy="400" r="290" fill="url(#earthGlow)" />

        {/* Continents & Landmasses */}
        <g opacity="0.72">
          {/* North America / Eurasia vector contours */}
          <path d="M 280,240 Q 320,200 370,220 T 430,210 Q 480,230 470,270 T 420,310 Q 360,330 330,290 Z" fill="#2d6e64" opacity="0.85" />
          <path d="M 450,230 Q 520,220 560,260 T 540,320 Q 500,340 470,300 Z" fill="#275e55" opacity="0.8" />
          <path d="M 310,340 Q 350,330 360,380 T 320,440 Q 280,420 290,370 Z" fill="#387a6d" opacity="0.75" />
          {/* Archipelago & coastal shelves */}
          <ellipse cx="490" cy="380" rx="35" ry="18" fill="#449887" opacity="0.7" transform="rotate(-15 490 380)" />
          <ellipse cx="530" cy="420" rx="28" ry="12" fill="#367d6f" opacity="0.65" transform="rotate(25 530 420)" />
          <ellipse cx="260" cy="310" rx="18" ry="9" fill="#58ad9c" opacity="0.6" />
        </g>

        {/* Cloud swirl patterns */}
        <g opacity="0.45" stroke="#ffffff" strokeWidth="6" strokeLinecap="round" fill="none">
          <path d="M 240,280 Q 280,250 340,260 T 420,240" />
          <path d="M 320,360 Q 380,340 440,370 T 520,350" strokeWidth="4" />
          <path d="M 260,420 Q 300,450 380,440 T 470,470" strokeWidth="5" />
          <path d="M 460,270 Q 500,290 560,280" strokeWidth="3" />
        </g>

        {/* Day / Night Terminator Shadow */}
        <circle cx="400" cy="400" r="290" fill="url(#nightTerminator)" />

        {/* Latitude / Longitude Graticule lines */}
        <g opacity="0.22" stroke="#8ce9d8" strokeWidth="1" fill="none">
          <ellipse cx="400" cy="400" rx="290" ry="80" />
          <ellipse cx="400" cy="400" rx="290" ry="170" />
          <ellipse cx="400" cy="400" rx="290" ry="250" />
          <line x1="400" y1="110" x2="400" y2="690" />
          <ellipse cx="400" cy="400" rx="140" ry="290" />
          <ellipse cx="400" cy="400" rx="220" ry="290" />
        </g>

        {/* Orbital Track Rings */}
        <g stroke="#9bf6e7" fill="none">
          <ellipse cx="400" cy="400" rx="360" ry="130" strokeWidth="1" strokeDasharray="6 8" opacity="0.4" transform="rotate(-28 400 400)" />
          <ellipse cx="400" cy="400" rx="380" ry="160" strokeWidth="1.5" opacity="0.25" transform="rotate(35 400 400)" />
          {/* Satellite Beacon */}
          <circle cx="650" cy="270" r="5" fill="#a4ffea" />
          <circle cx="650" cy="270" r="14" stroke="#a4ffea" strokeWidth="1.5" opacity="0.5" className="orbit-pulse" />
        </g>
      </svg>
    );
  }

  if (type === "coast") {
    return (
      <svg className="visual-vector" viewBox="0 0 600 500" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <linearGradient id="oceanDeep" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0c2333" />
            <stop offset="40%" stopColor="#143c4c" />
            <stop offset="70%" stopColor="#1b5a6c" />
            <stop offset="100%" stopColor="#257f8d" />
          </linearGradient>
          <linearGradient id="reefShallows" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1f7178" />
            <stop offset="60%" stopColor="#3caea3" />
            <stop offset="100%" stopColor="#68d8cb" />
          </linearGradient>
          <linearGradient id="coastalLand" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#23382c" />
            <stop offset="50%" stopColor="#3d5843" />
            <stop offset="100%" stopColor="#677f59" />
          </linearGradient>
        </defs>
        <rect width="600" height="500" fill="url(#oceanDeep)" />
        {/* Shallow marine bathymetry zones */}
        <path d="M 0,220 Q 150,190 260,260 T 480,310 Q 550,320 600,280 L 600,0 L 0,0 Z" fill="url(#reefShallows)" opacity="0.45" />
        {/* Coastal spit and dune ridge */}
        <path d="M 0,160 Q 120,130 220,190 T 420,240 Q 520,250 600,200 L 600,0 L 0,0 Z" fill="url(#coastalLand)" />
        {/* Tidal channels and sandbars */}
        <path d="M 80,180 Q 130,220 170,190 T 260,210" stroke="#d5d3b3" strokeWidth="4" strokeLinecap="round" opacity="0.8" fill="none" />
        <path d="M 280,230 Q 340,260 390,220" stroke="#eedda8" strokeWidth="5" strokeLinecap="round" opacity="0.75" fill="none" />
        <path d="M 430,245 Q 490,280 540,250" stroke="#f2e2be" strokeWidth="3.5" strokeLinecap="round" opacity="0.7" fill="none" />
        {/* Bathymetry depth contours */}
        <path d="M 0,300 Q 160,270 290,340 T 600,380" stroke="#5eead4" strokeWidth="1" strokeDasharray="4 6" opacity="0.5" fill="none" />
        <path d="M 0,380 Q 200,340 360,420 T 600,450" stroke="#5eead4" strokeWidth="1" strokeDasharray="3 5" opacity="0.3" fill="none" />
        {/* Satellite sensor HUD grid */}
        <g stroke="#ffffff" strokeWidth="0.8" opacity="0.2">
          <line x1="40" y1="40" x2="100" y2="40" />
          <line x1="40" y1="40" x2="40" y2="100" />
          <line x1="560" y1="460" x2="500" y2="460" />
          <line x1="560" y1="460" x2="560" y2="400" />
          <circle cx="300" cy="250" r="6" stroke="#5eead4" strokeWidth="1.5" />
          <line x1="280" y1="250" x2="320" y2="250" stroke="#5eead4" />
          <line x1="300" y1="230" x2="300" y2="270" stroke="#5eead4" />
        </g>
      </svg>
    );
  }

  if (type === "farm") {
    return (
      <svg className="visual-vector" viewBox="0 0 600 500" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <linearGradient id="soilBase" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#1f2d24" />
            <stop offset="100%" stopColor="#141d18" />
          </linearGradient>
        </defs>
        <rect width="600" height="500" fill="url(#soilBase)" />
        {/* Patchwork agricultural field polygons */}
        <rect x="30" y="30" width="160" height="130" fill="#3b5a38" opacity="0.85" />
        <rect x="200" y="30" width="180" height="80" fill="#587948" opacity="0.9" />
        <rect x="390" y="30" width="180" height="150" fill="#324933" opacity="0.8" />
        <rect x="200" y="120" width="180" height="120" fill="#758f55" opacity="0.85" />
        <rect x="30" y="170" width="160" height="140" fill="#4d6f46" opacity="0.8" />
        <rect x="30" y="320" width="220" height="150" fill="#2d422e" opacity="0.9" />
        <rect x="260" y="250" width="120" height="220" fill="#506e40" opacity="0.75" />
        <rect x="390" y="190" width="180" height="170" fill="#69834d" opacity="0.8" />
        <rect x="390" y="370" width="180" height="100" fill="#38523a" opacity="0.85" />

        {/* Center-pivot circular crop fields */}
        <circle cx="110" cy="95" r="55" fill="#7ba55a" opacity="0.85" />
        <circle cx="110" cy="95" r="28" fill="#4d6f46" opacity="0.9" />
        <circle cx="480" cy="105" r="65" fill="#88b562" opacity="0.9" />
        <circle cx="480" cy="275" r="70" fill="#5d8544" opacity="0.85" />
        <circle cx="140" cy="395" r="60" fill="#759e51" opacity="0.8" />

        {/* Irrigation access roads & ditches */}
        <g stroke="#8da08e" strokeWidth="2" opacity="0.4">
          <line x1="25" y1="165" x2="575" y2="165" />
          <line x1="25" y1="315" x2="575" y2="315" />
          <line x1="195" y1="25" x2="195" y2="475" />
          <line x1="385" y1="25" x2="385" y2="475" />
        </g>
        {/* Furrow line textures */}
        <g stroke="#ffffff" strokeWidth="1" opacity="0.12">
          <line x1="205" y1="135" x2="375" y2="135" strokeDasharray="3 4" />
          <line x1="205" y1="150" x2="375" y2="150" strokeDasharray="3 4" />
          <line x1="205" y1="165" x2="375" y2="165" strokeDasharray="3 4" />
          <line x1="205" y1="180" x2="375" y2="180" strokeDasharray="3 4" />
        </g>
      </svg>
    );
  }

  if (type === "city") {
    return (
      <svg className="visual-vector" viewBox="0 0 600 500" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <linearGradient id="cityWater" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#091823" />
            <stop offset="100%" stopColor="#0e2434" />
          </linearGradient>
        </defs>
        <rect width="600" height="500" fill="#151e26" />
        {/* Harbor bay water basin */}
        <path d="M 0,280 Q 180,260 280,340 T 600,320 L 600,500 L 0,500 Z" fill="url(#cityWater)" />
        {/* Pier docks & wharf structures */}
        <g fill="#273846" stroke="#486377" strokeWidth="1.5">
          <rect x="70" y="275" width="25" height="110" />
          <rect x="130" y="270" width="25" height="130" />
          <rect x="190" y="285" width="28" height="95" />
          <rect x="360" y="325" width="30" height="110" />
          <rect x="430" y="315" width="28" height="125" />
          <rect x="500" y="320" width="26" height="90" />
        </g>
        {/* Dense urban building blocks */}
        <g fill="#2b3b47" stroke="#3d5262" strokeWidth="1">
          <rect x="35" y="40" width="70" height="50" />
          <rect x="120" y="40" width="90" height="50" />
          <rect x="225" y="40" width="130" height="50" />
          <rect x="370" y="40" width="80" height="50" />
          <rect x="465" y="40" width="95" height="50" />

          <rect x="35" y="105" width="80" height="60" fill="#374b5a" />
          <rect x="130" y="105" width="105" height="60" />
          <rect x="250" y="105" width="70" height="60" fill="#374b5a" />
          <rect x="335" y="105" width="115" height="60" />
          <rect x="465" y="105" width="95" height="60" fill="#3f5566" />

          <rect x="35" y="180" width="115" height="70" fill="#3f5566" />
          <rect x="165" y="180" width="95" height="70" />
          <rect x="275" y="180" width="125" height="70" fill="#324654" />
          <rect x="415" y="180" width="145" height="70" />
        </g>
        {/* Road network arteries */}
        <g stroke="#7e95a5" strokeWidth="3" opacity="0.6">
          <line x1="20" y1="97" x2="580" y2="97" />
          <line x1="20" y1="172" x2="580" y2="172" />
          <line x1="20" y1="260" x2="580" y2="260" />
          <line x1="112" y1="30" x2="112" y2="270" />
          <line x1="237" y1="30" x2="237" y2="280" />
          <line x1="362" y1="30" x2="362" y2="330" />
          <line x1="457" y1="30" x2="457" y2="320" />
        </g>
        {/* Active vehicle / radar reflectance blips */}
        <g fill="#93e6d8">
          <circle cx="160" cy="97" r="2.5" />
          <circle cx="290" cy="172" r="2.5" />
          <circle cx="480" cy="172" r="2.5" />
          <circle cx="237" cy="140" r="2.5" />
          <circle cx="362" cy="220" r="2.5" />
        </g>
      </svg>
    );
  }

  if (type === "quantum") {
    return (
      <svg className="visual-vector" viewBox="0 0 600 500" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <radialGradient id="cryoChamber" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#2e224d" />
            <stop offset="60%" stopColor="#17122b" />
            <stop offset="100%" stopColor="#0a0815" />
          </radialGradient>
          <linearGradient id="goldStage" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#8d773e" />
            <stop offset="50%" stopColor="#e5c875" />
            <stop offset="100%" stopColor="#968045" />
          </linearGradient>
        </defs>
        <rect width="600" height="500" fill="url(#cryoChamber)" />
        {/* Cryostat circular thermal plates */}
        <g stroke="url(#goldStage)" strokeWidth="3" fill="none" opacity="0.75">
          <ellipse cx="300" cy="100" rx="220" ry="40" />
          <ellipse cx="300" cy="200" rx="180" ry="32" />
          <ellipse cx="300" cy="300" rx="140" ry="25" />
          <ellipse cx="300" cy="380" rx="90" ry="18" />
        </g>
        {/* Coaxial cryogenic signal lines */}
        <g stroke="#b8a7db" strokeWidth="1.5" opacity="0.5" fill="none">
          <path d="M 160,100 L 160,200 L 190,300 L 240,380" />
          <path d="M 220,100 L 220,200 L 240,300 L 265,380" />
          <path d="M 380,100 L 380,200 L 360,300 L 335,380" />
          <path d="M 440,100 L 440,200 L 410,300 L 360,380" />
        </g>
        {/* Quantum Chip Mount Die at 15 mK Base Stage */}
        <rect x="255" y="365" width="90" height="50" rx="4" fill="#1b1735" stroke="#cbb9ff" strokeWidth="2" />
        <circle cx="300" cy="390" r="12" fill="#503d82" stroke="#a78bfa" strokeWidth="1.5" />
        <circle cx="300" cy="390" r="4" fill="#e9d5ff" />
        {/* Quantum entanglement pulses */}
        <circle cx="300" cy="390" r="28" stroke="#c084fc" strokeWidth="1" strokeDasharray="3 4" opacity="0.6" className="orbit-pulse" />
      </svg>
    );
  }

  return null;
}

function Visual({ name, className = "", children }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className={`visual visual--${name} ${className}`}>
      <div className="visual-art" aria-hidden="true">
        <VisualSvg type={name} />
      </div>
      <img
        src={imagePaths[name]}
        alt=""
        loading={name === "earth" ? "eager" : "lazy"}
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(false)}
        className={loaded ? "visual-photo is-loaded" : "visual-photo"}
      />
      {children}
    </div>
  );
}

function Arrow({ diagonal = false }) {
  return <span aria-hidden="true" className="inline-arrow">{diagonal ? "↗" : "→"}</span>;
}

function SectionHeading({ eyebrow, title, body, children }) {
  return (
    <div className="section-heading reveal">
      <p className="eyebrow">
        <span className="eyebrow-mark" />
        {eyebrow}
      </p>
      <h2>{title}</h2>
      {body && <p className="section-description">{body}</p>}
      {children}
    </div>
  );
}

// Quantum circuit diagram interactive component
function QuantumCircuitDiagram() {
  return (
    <div className="circuit-container" role="figure" aria-label="Conceptual 4-qubit variational circuit architecture">
      <div className="circuit-header">
        <span className="circuit-tag">VQC LAYER / ANGLE EMBEDDING & ENTANGLEMENT</span>
        <span className="circuit-status">PENNYLANE / QISKIT SPEC</span>
      </div>
      <div className="circuit-board">
        {/* Qubit wire 0 */}
        <div className="circuit-row">
          <span className="qubit-label">|q₀⟩</span>
          <div className="circuit-wire">
            <span className="gate gate--ry">Ry(θ₀)</span>
            <span className="cnot-control" />
            <span className="gate gate--rz">Rz(w₀)</span>
            <span className="gate-meter">⟨Z₀⟩</span>
          </div>
        </div>
        {/* Vertical CNOT connection 0-1 */}
        <div className="circuit-cnot-line" style={{ left: "38%", top: "18%", height: "26%" }} />
        {/* Qubit wire 1 */}
        <div className="circuit-row">
          <span className="qubit-label">|q₁⟩</span>
          <div className="circuit-wire">
            <span className="gate gate--ry">Ry(θ₁)</span>
            <span className="cnot-target" />
            <span className="gate gate--rz">Rz(w₁)</span>
            <span className="gate-meter">⟨Z₁⟩</span>
          </div>
        </div>
        {/* Qubit wire 2 */}
        <div className="circuit-row">
          <span className="qubit-label">|q₂⟩</span>
          <div className="circuit-wire">
            <span className="gate gate--ry">Ry(θ₂)</span>
            <span className="cnot-control" />
            <span className="gate gate--rz">Rz(w₂)</span>
            <span className="gate-meter">⟨Z₂⟩</span>
          </div>
        </div>
        {/* Vertical CNOT connection 2-3 */}
        <div className="circuit-cnot-line" style={{ left: "38%", top: "66%", height: "26%" }} />
        {/* Qubit wire 3 */}
        <div className="circuit-row">
          <span className="qubit-label">|q₃⟩</span>
          <div className="circuit-wire">
            <span className="gate gate--ry">Ry(θ₃)</span>
            <span className="cnot-target" />
            <span className="gate gate--rz">Rz(w₃)</span>
            <span className="gate-meter">⟨Z₃⟩</span>
          </div>
        </div>
      </div>
      <div className="circuit-caption">
        <span>Encoding: Classical feature map → Rotations (Ry)</span>
        <span>Entanglement: Circular CNOT Ring</span>
        <span>Output: Expectation Values</span>
      </div>
    </div>
  );
}

function ModelCard({ type }) {
  const quantum = type === "quantum";
  return (
    <article className={`model-card ${quantum ? "model-card--quantum" : ""}`}>
      <div className="model-card-top">
        <span className="model-glyph">{quantum ? "✳" : "◉"}</span>
        <span className="model-index">{quantum ? "MODEL 02 · HYBRID QUANTUM" : "MODEL 01 · CLASSICAL DL"}</span>
      </div>
      <h3>{quantum ? "Hybrid Quantum-Classical" : "Classical Neural Network"}</h3>
      <p>
        {quantum
          ? "A classical ResNet front-end reduces multispectral bands into a compact latent vector, followed by a parameterized quantum circuit (PQC) exploring non-linear spectral state space."
          : "A standard deep convolutional neural network (e.g. ResNet-18) trained end-to-end on Sentinel-2 multispectral reflectance patches to learn spatial and spectral filters."}
      </p>

      <div className="model-specs-list">
        <div className="spec-item">
          <span>Input Shape:</span>
          <strong>64×64 × 12 Bands</strong>
        </div>
        <div className="spec-item">
          <span>{quantum ? "Quantum Substrate:" : "Backbone:"}</span>
          <strong>{quantum ? "8 Qubits · Parameterized (PQC)" : "ResNet-18 · 11.2M Weights"}</strong>
        </div>
        <div className="spec-item">
          <span>Integration Status:</span>
          <span className="status-badge status-badge--pending">Not Connected</span>
        </div>
      </div>

      <div className="model-flow" aria-label="Processing pipeline">
        <span>SATELLITE TILE</span>
        <Arrow />
        <span>{quantum ? "LATENT ENCODER" : "CONV LAYERS"}</span>
        <Arrow />
        <span>{quantum ? "PQC ENTANGLER" : "AVG POOLING"}</span>
        <Arrow />
        <span>{quantum ? "PAULI-Z CLASSIFIER" : "SOFTMAX"}</span>
      </div>

      <div className="model-card-footer">
        <span className="status-dot status-dot--pending" />
        <span>Awaiting weights from {quantum ? "models/quantum/" : "models/classical/"}</span>
        <span className="baseline-tag">{quantum ? "Baseline B" : "Baseline A"}</span>
      </div>
    </article>
  );
}

// Interactive Image Lab with presets, drag-and-drop, clipboard paste, and spectral filters
function Prototype() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [fileMeta, setFileMeta] = useState(null);
  const [activePreset, setActivePreset] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [spectralMode, setSpectralMode] = useState("rgb"); // rgb, cir, ndvi
  const inputRef = useRef(null);

  // Clear preview URL on unmount or file change
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    setFileMeta({
      name: file.name,
      size: (file.size / (1024 * 1024)).toFixed(2) + " MB",
      type: file.type || "image/unknown",
      source: "Uploaded file",
    });
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const receiveFile = useCallback((candidate) => {
    if (!candidate) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(candidate.type)) {
      setError("Please choose a valid PNG, JPEG, or WebP image.");
      return;
    }
    if (candidate.size > 10 * 1024 * 1024) {
      setError("Image file exceeds the 10 MB limit.");
      return;
    }
    setActivePreset(null);
    setFile(candidate);
    setError("");
  }, []);

  // Clipboard paste support (Ctrl+V)
  useEffect(() => {
    function handlePaste(e) {
      if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        const pastedFile = e.clipboardData.files[0];
        if (pastedFile.type.startsWith("image/")) {
          receiveFile(pastedFile);
        }
      }
    }
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [receiveFile]);

  function loadPreset(preset) {
    setFile(null);
    setPreview("");
    setActivePreset(preset);
    setError("");
    setFileMeta({
      name: preset.name,
      size: "Synthetic Benchmark Tile",
      type: preset.specs,
      source: preset.category,
    });
  }

  function clearImage() {
    setFile(null);
    setPreview("");
    setActivePreset(null);
    setError("");
    setFileMeta(null);
  }

  return (
    <div className="lab-container">
      {/* Sample preset selector bar */}
      <div className="preset-bar">
        <span className="preset-bar-title">QUICK BENCHMARK SAMPLES:</span>
        <div className="preset-chips">
          {sampleTiles.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`preset-chip ${activePreset?.id === preset.id ? "is-active" : ""}`}
              onClick={() => loadPreset(preset)}
            >
              <span className="preset-chip-dot" />
              {preset.name}
            </button>
          ))}
          {(preview || activePreset) && (
            <button type="button" className="preset-chip preset-chip--clear" onClick={clearImage}>
              ✕ Reset Lab
            </button>
          )}
        </div>
      </div>

      <div className="lab-grid">
        {/* Upload / Preview Area */}
        <div
          className={`upload-area ${dragging ? "is-dragging" : ""} ${preview || activePreset ? "has-preview" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            receiveFile(e.dataTransfer.files?.[0]);
          }}
          tabIndex={0}
          role="region"
          aria-label="Satellite imagery dropzone and preview"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
        >
          {preview ? (
            <div className={`preview-wrapper spectral-filter--${spectralMode}`}>
              <img className="upload-preview" src={preview} alt="User-selected satellite preview" />
            </div>
          ) : activePreset ? (
            <div className={`preview-wrapper spectral-filter--${spectralMode}`}>
              <div className="preset-svg-wrap">
                <VisualSvg type={activePreset.svgType} />
              </div>
            </div>
          ) : (
            <div className="upload-orbit" aria-hidden="true">
              <span className="orbit-glyph">＋</span>
            </div>
          )}

          <div className="upload-copy">
            <p className="upload-kicker">
              {preview || activePreset ? "ORBITAL TILE LOADED" : "INPUT / SATELLITE IMAGERY"}
            </p>
            <h3>{preview ? fileMeta?.name : activePreset ? activePreset.name : "Drop an image of Earth."}</h3>
            <p>
              {preview || activePreset
                ? "Image preview stays local in memory. No model inference has been executed."
                : "Drag & drop a satellite image, paste from clipboard (Ctrl+V), or select a benchmark sample above."}
            </p>

            {/* Filter buttons if an image or sample is selected */}
            {(preview || activePreset) && (
              <div className="spectral-toggle-bar">
                <span className="spectral-label">BAND SIMULATION:</span>
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
                onClick={() => inputRef.current?.click()}
              >
                {preview || activePreset ? "Change local image" : "Choose local image"} <Arrow diagonal />
              </button>
              {(preview || activePreset) && (
                <button type="button" className="button button--outline" onClick={clearImage}>
                  Remove
                </button>
              )}
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              aria-label="Choose a satellite image from file system"
              hidden
              onChange={(e) => {
                receiveFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />

            {error && (
              <p className="file-error" role="alert">
                ⚠️ {error}
              </p>
            )}

            {fileMeta && (
              <div className="file-meta-box">
                <div><span>File:</span> <strong>{fileMeta.name}</strong></div>
                <div><span>Size:</span> <strong>{fileMeta.size}</strong></div>
                <div><span>Type:</span> <strong>{fileMeta.type}</strong></div>
              </div>
            )}

            <p className="upload-note">Supports PNG, JPEG, WebP · Up to 10 MB · Client-side preview only</p>
          </div>
        </div>

        {/* Results / Status Area */}
        <div className="results-area">
          <div className="results-top">
            <span className="results-badge">MODEL PIPELINE TELEMETRY</span>
            <span className="live-tag live-tag--offline">OFFLINE PROTOTYPE</span>
          </div>

          <div className="result-row">
            <div className="result-symbol result-symbol--classical">◉</div>
            <div className="result-content">
              <p className="result-title">CLASSICAL MODEL (RESNET-18)</p>
              <div className="result-status-row">
                <span className="status-dot status-dot--pending" />
                <strong className="status-text">Not connected</strong>
              </div>
              <small className="result-hint">
                Awaiting trained weights from <code>models/classical/</code>. Classification output and feature map activations will appear here upon backend verification.
              </small>
            </div>
            <span className="result-arrow">↗</span>
          </div>

          <div className="result-row">
            <div className="result-symbol result-symbol--quantum">✳</div>
            <div className="result-content">
              <p className="result-title">HYBRID QUANTUM MODEL (HQNN / VQC)</p>
              <div className="result-status-row">
                <span className="status-dot status-dot--pending" />
                <strong className="status-text">Not connected</strong>
              </div>
              <small className="result-hint">
                Awaiting quantum circuit backend from <code>models/quantum/</code>. Quantum state expectations and entangled spectral probabilities will be computed on verified hardware/simulator.
              </small>
            </div>
            <span className="result-arrow">↗</span>
          </div>

          <div className="integrity-box">
            <h4>🔬 Research Integrity Guarantee</h4>
            <p>
              No predictions, class probabilities, accuracy percentages, or processing latencies are fabricated or mocked in this interface. Both classical and quantum models will be evaluated rigorously against an identical, documented benchmark test split.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.05, rootMargin: "0px 0px 60px 0px" }
    );

    document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));

    // Ensure elements initially in viewport become visible immediately
    const checkInitialVisibility = () => {
      document.querySelectorAll(".reveal").forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
          node.classList.add("is-visible");
        }
      });
    };
    checkInitialVisibility();

    return () => observer.disconnect();
  }, []);

  const closeMenu = () => setMenuOpen(false);

  return (
    <>
      <a href="#main-content" className="skip-link">Skip to main content</a>

      <header className="site-header">
        <a className="brand" href="#top" aria-label="Quantum Earth home" onClick={closeMenu}>
          <span className="brand-icon">✳</span>
          <span className="brand-text">QUANTUM<span className="brand-muted">EARTH</span></span>
          <span className="brand-badge">LEO ORBIT</span>
        </a>

        <button
          type="button"
          className="menu-toggle"
          aria-expanded={menuOpen}
          aria-controls="primary-navigation"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          <span>{menuOpen ? "Close" : "Menu"}</span>
          <span className="menu-toggle-icon">{menuOpen ? "✕" : "☰"}</span>
        </button>

        <nav id="primary-navigation" className={menuOpen ? "nav is-open" : "nav"} aria-label="Main navigation">
          <a href="#mission" onClick={closeMenu}>01 Mission</a>
          <a href="#signals" onClick={closeMenu}>02 Signals</a>
          <a href="#approach" onClick={closeMenu}>03 Approaches</a>
          <a href="#compare" onClick={closeMenu}>04 Comparison</a>
          <a className="nav-lab" href="#lab" onClick={closeMenu}>
            Explore Lab <Arrow diagonal />
          </a>
        </nav>
      </header>

      <main id="main-content">
        {/* SECTION 1: HERO */}
        <section id="top" className="hero">
          <div className="hero-grid" aria-hidden="true" />
          <Visual name="earth" className="hero-visual" />
          <div className="hero-shade" />

          <div className="hero-content">
            <p className="eyebrow hero-eyebrow">
              <span className="eyebrow-mark" />
              EARTH OBSERVATION / EXPERIMENT 001
            </p>
            <h1>
              See Earth.<br />
              <em>Think beyond.</em>
            </h1>
            <p className="hero-copy">
              One planet. Two computational paradigms. Investigating classical deep neural networks and hybrid quantum machine learning through multispectral satellite imagery.
            </p>
            <div className="hero-actions">
              <a className="button button--outline" href="#mission">
                Explore the mission <span>↓</span>
              </a>
              <a className="button button--ghost" href="#lab">
                Open interactive lab <Arrow diagonal />
              </a>
            </div>
          </div>

          <div className="hero-bottom">
            <span>01 / 06 · QUANTUM EARTH</span>
            <span className="hero-telemetry">SENTINEL-2 MSI · 13 SPECTRAL BANDS · 10M GSD</span>
            <span>SCROLL TO EXPLORE ↓</span>
          </div>
        </section>

        {/* SECTION 2: MISSION */}
        <section id="mission" className="section mission">
          <div className="container mission-layout">
            <SectionHeading
              eyebrow="01 / THE MISSION"
              title={
                <>
                  The planet is speaking.<br />
                  <em>Can AI understand it?</em>
                </>
              }
              body="Every orbital pass over Earth records millions of multispectral pixels—reflectance signatures of coastal estuaries, agricultural canopies, urban sprawl, and shifting ecosystems. Our research investigates how classical and quantum algorithms extract spatial-spectral representations from this planetary data."
            >
              <div className="mission-highlights">
                <div className="highlight-item">
                  <span className="highlight-num">13</span>
                  <span className="highlight-desc">Multispectral bands from Coastal Aerosol to SWIR</span>
                </div>
                <div className="highlight-item">
                  <span className="highlight-num">2</span>
                  <span className="highlight-desc">Independent model families benchmarked side-by-side</span>
                </div>
                <div className="highlight-item">
                  <span className="highlight-num">0</span>
                  <span className="highlight-desc">Unverified claims: evidence-based metrics only</span>
                </div>
              </div>
            </SectionHeading>

            <div className="mission-visual reveal">
              <Visual name="coast" className="mission-photo">
                <span className="image-label image-label--top">OBSERVATION / 001 · COASTLINE</span>
                <span className="image-label image-label--bottom">LAND · WATER · TURBIDITY · 10M GSD</span>
              </Visual>
              <span className="corner-decoration" aria-hidden="true">+</span>
            </div>
          </div>

          <div className="container data-strip reveal">
            <span>MULTISPECTRAL SATELLITE TILE</span>
            <Arrow />
            <span>SPECTRAL-SPATIAL REPRESENTATION</span>
            <Arrow />
            <span>CLASSICAL CNN vs HYBRID QUANTUM VQC</span>
          </div>
        </section>

        {/* SECTION 3: LANDSCAPES & SIGNALS */}
        <section id="signals" className="section landscapes">
          <div className="container">
            <SectionHeading
              eyebrow="02 / THE SIGNALS"
              title={
                <>
                  Different landscapes.<br />
                  <em>Different clues.</em>
                </>
              }
              body="Satellite pixels are not just RGB pictures—they are physical reflectance measurements. Different terrestrial land cover types present distinct spatial correlations and spectral absorption peaks."
            />

            <div className="landscape-grid">
              <article className="landscape-card reveal">
                <Visual name="farm">
                  <span className="image-label">01 / AGRICULTURE · CROPLAND</span>
                </Visual>
                <div className="landscape-card-info">
                  <div>
                    <h4>Photosynthetic Canopy</h4>
                    <p>Chlorophyll absorption in red (665 nm) and intense reflectance in near-infrared (842 nm) form strong vegetative clues.</p>
                  </div>
                  <div className="landscape-card-link">
                    <span>NDVI SIGNATURES</span>
                    <Arrow diagonal />
                  </div>
                </div>
              </article>

              <article className="landscape-card landscape-card--offset reveal">
                <Visual name="city">
                  <span className="image-label">02 / URBAN INFRASTRUCTURE</span>
                </Visual>
                <div className="landscape-card-info">
                  <div>
                    <h4>Built Structures & Concrete</h4>
                    <p>High-spatial-frequency road networks, impervious surfaces, and shortwave infrared reflectance patterns.</p>
                  </div>
                  <div className="landscape-card-link">
                    <span>SPATIAL DENSITY</span>
                    <Arrow diagonal />
                  </div>
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* SECTION 4: APPROACHES */}
        <section id="approach" className="section approach">
          <div className="container">
            <SectionHeading
              eyebrow="03 / TWO APPROACHES"
              title={
                <>
                  Same Earth.<br />
                  <em>Different computation.</em>
                </>
              }
              body="To discover whether quantum algorithms offer genuine utility in Earth observation, we compare two distinct architectures trained under identical conditions on shared benchmark datasets."
            />

            <div className="model-grid reveal">
              <ModelCard type="classical" />
              <ModelCard type="quantum" />
            </div>
          </div>

          <div className="quantum-feature container reveal">
            <div className="quantum-art-col">
              <Visual name="quantum" className="quantum-art" />
              <QuantumCircuitDiagram />
            </div>

            <div className="quantum-feature-copy">
              <p className="eyebrow">
                <span className="eyebrow-mark" />
                INSIDE THE HYBRID MODEL
              </p>
              <h3>A new way to connect the clues.</h3>
              <p>
                In our hybrid quantum approach, a compact classical feature extractor generates an embedding vector from multispectral pixels. This vector is encoded into the quantum amplitudes and rotation angles of an 8-qubit register.
              </p>
              <p>
                Parameterized entangling gates then explore complex correlated states across the Hilbert space before projective measurement yields classification observables.
              </p>
              <div className="feature-note">
                <strong>Scientific Stance:</strong> This is an active research question investigating feature expressibility and parameter efficiency, not an unsubstantiated claim of immediate quantum advantage.
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 5: COMPARISON */}
        <section id="compare" className="section compare">
          <div className="container">
            <SectionHeading
              eyebrow="04 / THE COMPARISON"
              title={
                <>
                  Not a race for hype.<br />
                  <em>A test for evidence.</em>
                </>
              }
              body="A rigorous scientific comparison controls dataset splits, preprocessing pipelines, and evaluation metrics. A quantum model earns its place through verified, reproducible experimental data."
            />

            <div className="comparison-table reveal" role="table" aria-label="Scientific model comparison matrix">
              <div className="comparison-row comparison-row--head" role="row">
                <span role="columnheader">EVALUATION METRIC</span>
                <span role="columnheader">CLASSICAL BASELINE (CNN)</span>
                <span role="columnheader">HYBRID QUANTUM (HQNN)</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Architecture Type</strong>
                </span>
                <span role="cell">ResNet-18 Deep Convolutional</span>
                <span role="cell">ResNet Feature Extractor + 8-Qubit VQC</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Input Representation</strong>
                </span>
                <span role="cell">64×64 Sentinel-2 Multispectral (12 Bands)</span>
                <span role="cell">Compact Latent State + Angle Embedding</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Model Parameters</strong>
                </span>
                <span role="cell">~11.2M Classical Weights</span>
                <span role="cell">~1.2M Classical + 48 Variational Quantum Gates</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Validation Accuracy</strong>
                </span>
                <span role="cell" className="metric-pending">Not measured · Awaiting run</span>
                <span role="cell" className="metric-pending">Not measured · Awaiting run</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Inference Latency</strong>
                </span>
                <span role="cell" className="metric-pending">Not measured · GPU</span>
                <span role="cell" className="metric-pending">Not measured · Simulator/QPU</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Shared Benchmark Split</strong>
                </span>
                <span role="cell" className="metric-planned">Planned · EuroSAT / BigEarthNet</span>
                <span role="cell" className="metric-planned">Planned · EuroSAT / BigEarthNet</span>
              </div>
              <div className="comparison-row" role="row">
                <span role="cell">
                  <strong>Integration Status</strong>
                </span>
                <span role="cell">
                  <span className="status-badge status-badge--pending">Not Connected</span>
                </span>
                <span role="cell">
                  <span className="status-badge status-badge--pending">Not Connected</span>
                </span>
              </div>
            </div>

            <p className="table-footnote">
              * Note: In compliance with project integrity principles, all evaluation rows remain strictly marked as &quot;Not measured&quot; until physical training checkpoints and automated test harnesses are verified.
            </p>
          </div>
        </section>

        {/* SECTION 6: INTERACTIVE LAB */}
        <section id="lab" className="section lab">
          <div className="container">
            <SectionHeading
              eyebrow="05 / INTERACTIVE LAB"
              title={
                <>
                  Your view from orbit.<br />
                  <em>Our next experiment.</em>
                </>
              }
              body="Test client-side satellite tile inspection and band previews below. Real model inference will activate as soon as classical and quantum training checkpoints are integrated."
            />

            <Prototype />
          </div>
        </section>

        {/* SECTION 7: CLOSING & ROADMAP */}
        <section className="closing">
          <div className="closing-orbit" aria-hidden="true" />
          <div className="container closing-content reveal">
            <p className="eyebrow">
              <span className="eyebrow-mark" />
              06 / WHAT COMES NEXT
            </p>
            <h2>
              From observation<br />
              to <em>understanding.</em>
            </h2>
            <p>
              Next research milestones: train the classical baseline on multispectral EuroSAT, design a structured quantum feature map for non-linear band correlations, and validate on physical quantum hardware.
            </p>

            <div className="roadmap-grid">
              <div className="roadmap-step">
                <span className="step-num">PHASE 01</span>
                <h4>Classical Baseline</h4>
                <p>Implement ResNet-18 pipeline in <code>models/classical/</code> on standardized Sentinel-2 patches.</p>
              </div>
              <div className="roadmap-step">
                <span className="step-num">PHASE 02</span>
                <h4>Quantum Circuit</h4>
                <p>Develop PennyLane variational circuit in <code>models/quantum/</code> with angle embedding & expressibility tests.</p>
              </div>
              <div className="roadmap-step">
                <span className="step-num">PHASE 03</span>
                <h4>Hardware Deployment</h4>
                <p>Evaluate noise mitigation and execution on physical superconducting / ion-trap QPUs.</p>
              </div>
            </div>

            <div className="closing-actions">
              <a href="#top" className="button button--outline">
                Back to Earth ↑
              </a>
              <a href="#lab" className="button button--light">
                Try the interactive lab <Arrow diagonal />
              </a>
            </div>
          </div>

          <footer className="container footer">
            <div className="footer-brand">
              <span className="brand-icon">✳</span>
              <strong>QUANTUM<span className="brand-muted">EARTH</span></strong>
              <span className="footer-tag">RESEARCH PROTOTYPE / V1</span>
            </div>
            <div className="footer-links">
              <a href="#mission">Mission</a>
              <a href="#approach">Approaches</a>
              <a href="#compare">Comparison</a>
              <a href="#lab">Lab</a>
              <a href="https://github.com/tej-1916/quantum-earth" target="_blank" rel="noreferrer">
                GitHub Repository <Arrow diagonal />
              </a>
            </div>
          </footer>
        </section>
      </main>
    </>
  );
}
