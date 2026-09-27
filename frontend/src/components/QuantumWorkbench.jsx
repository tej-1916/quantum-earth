import React, { useState } from "react";

export default function QuantumWorkbench() {
  const [qubits, setQubits] = useState(8);
  const [depth, setDepth] = useState(2);
  const [topology, setTopology] = useState("linear"); // linear, circular, all_to_all
  const [execMode, setExecMode] = useState("simulator"); // simulator, ibm_hardware

  // Calculate circuit metrics
  const totalParams = qubits + 2 * depth * qubits; // Embedding + Rotations per layer
  const hilbertDim = Math.pow(2, qubits);

  return (
    <div className="quantum-workbench-container">
      {/* Top Controls & Configuration Bar */}
      <div className="workbench-toolbar">
        <div className="workbench-group">
          <label className="workbench-label">QUBIT REGISTER:</label>
          <div className="segmented-control">
            {[4, 6, 8].map((q) => (
              <button
                key={q}
                type="button"
                className={`segment-btn ${qubits === q ? "is-active" : ""}`}
                onClick={() => setQubits(q)}
              >
                {q} Qubits
              </button>
            ))}
          </div>
        </div>

        <div className="workbench-group">
          <label className="workbench-label">VARIATIONAL DEPTH:</label>
          <div className="segmented-control">
            {[1, 2, 4].map((d) => (
              <button
                key={d}
                type="button"
                className={`segment-btn ${depth === d ? "is-active" : ""}`}
                onClick={() => setDepth(d)}
              >
                {d} {d === 1 ? "Layer" : "Layers"}
              </button>
            ))}
          </div>
        </div>

        <div className="workbench-group">
          <label className="workbench-label">ENTANGLEMENT TOPOLOGY:</label>
          <div className="segmented-control">
            {[
              { id: "linear", label: "Linear" },
              { id: "circular", label: "Circular" },
              { id: "all_to_all", label: "All-to-All" },
            ].map((t) => (
              <button
                key={t.id}
                type="button"
                className={`segment-btn ${topology === t.id ? "is-active" : ""}`}
                onClick={() => setTopology(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="workbench-group">
          <label className="workbench-label">EXECUTION TARGET:</label>
          <div className="segmented-control">
            <button
              type="button"
              className={`segment-btn ${execMode === "simulator" ? "is-active" : ""}`}
              onClick={() => setExecMode("simulator")}
            >
              Simulator (Default)
            </button>
            <button
              type="button"
              className={`segment-btn ${execMode === "ibm_hardware" ? "is-active" : ""}`}
              onClick={() => setExecMode("ibm_hardware")}
            >
              IBM Quantum QPU
            </button>
          </div>
        </div>
      </div>

      {/* Main Interactive Circuit Schematic Visualizer */}
      <div className="circuit-canvas-card">
        <div className="circuit-card-header">
          <div>
            <h4>PARAMETERIZED QUANTUM CIRCUIT (PQC / VQC)</h4>
            <span className="circuit-subtitle">
              Satellite Embedding → Angle Encoding → Entangling Layers → Expectation Measurement
            </span>
          </div>
          <span className="telemetry-tag telemetry-tag--quantum">
            PENNYLANE + QISKIT BACKEND
          </span>
        </div>

        <div className="circuit-schematic-wrap">
          <svg
            className="circuit-svg-diagram"
            viewBox={`0 0 920 ${qubits * 48 + 40}`}
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-label="Quantum circuit gate diagram"
          >
            {/* Horizontal Qubit Wires */}
            {Array.from({ length: qubits }).map((_, i) => {
              const y = 30 + i * 48;
              return (
                <g key={`wire-${i}`}>
                  {/* Qubit Label */}
                  <text x="24" y={y + 5} fill="#8ef0dd" fontSize="12" fontFamily="monospace">
                    |q_{i}⟩
                  </text>
                  {/* Wire line */}
                  <line x1="60" y1={y} x2="890" y2={y} stroke="#214e4b" strokeWidth="1.5" />

                  {/* Feature Encoding Gate: Rx(x_i) */}
                  <rect
                    x="90"
                    y={y - 14}
                    width="44"
                    height="28"
                    rx="3"
                    fill="#0f2e2f"
                    stroke="#0df2c9"
                    strokeWidth="1.5"
                  />
                  <text x="112" y={y + 4} fill="#8ef0dd" fontSize="10" fontFamily="monospace" textAnchor="middle">
                    Rx(θ_{i})
                  </text>

                  {/* Variational Layers */}
                  {Array.from({ length: depth }).map((_, layerIdx) => {
                    const startX = 170 + layerIdx * 340;
                    return (
                      <g key={`var-layer-${layerIdx}-${i}`}>
                        {/* Ry Gate */}
                        <rect
                          x={startX}
                          y={y - 14}
                          width="36"
                          height="28"
                          rx="3"
                          fill="#14343d"
                          stroke="#38bdf8"
                          strokeWidth="1.2"
                        />
                        <text x={startX + 18} y={y + 4} fill="#bae6fd" fontSize="10" fontFamily="monospace" textAnchor="middle">
                          Ry
                        </text>

                        {/* Rz Gate */}
                        <rect
                          x={startX + 48}
                          y={y - 14}
                          width="36"
                          height="28"
                          rx="3"
                          fill="#14343d"
                          stroke="#38bdf8"
                          strokeWidth="1.2"
                        />
                        <text x={startX + 66} y={y + 4} fill="#bae6fd" fontSize="10" fontFamily="monospace" textAnchor="middle">
                          Rz
                        </text>

                        {/* CNOT Control & Target */}
                        {i < qubits - 1 && (
                          <g>
                            <line
                              x1={startX + 120}
                              y1={y}
                              x2={startX + 120}
                              y2={y + 48}
                              stroke="#0df2c9"
                              strokeWidth="1.5"
                            />
                            {/* Control dot */}
                            <circle cx={startX + 120} cy={y} r="4" fill="#0df2c9" />
                            {/* Target circle */}
                            <circle cx={startX + 120} cy={y + 48} r="8" stroke="#0df2c9" strokeWidth="1.5" fill="#030b12" />
                            <line x1={startX + 114} y1={y + 48} x2={startX + 126} y2={y + 48} stroke="#0df2c9" strokeWidth="1.5" />
                            <line x1={startX + 120} y1={y + 42} x2={startX + 120} y2={y + 54} stroke="#0df2c9" strokeWidth="1.5" />
                          </g>
                        )}

                        {/* Circular boundary connection */}
                        {topology === "circular" && i === qubits - 1 && (
                          <g>
                            <line
                              x1={startX + 160}
                              y1={y}
                              x2={startX + 160}
                              y2={30}
                              stroke="#a855f7"
                              strokeWidth="1.2"
                              strokeDasharray="4 3"
                            />
                            <circle cx={startX + 160} cy={y} r="4" fill="#a855f7" />
                            <circle cx={startX + 160} cy={30} r="7" stroke="#a855f7" strokeWidth="1.2" fill="#030b12" />
                          </g>
                        )}
                      </g>
                    );
                  })}

                  {/* Projective Measurement Gate: M_z */}
                  <rect
                    x="840"
                    y={y - 14}
                    width="36"
                    height="28"
                    rx="3"
                    fill="#182334"
                    stroke="#818cf8"
                    strokeWidth="1.2"
                  />
                  <path
                    d={`M 848,${y + 7} A 10,10 0 0,1 868,${y + 7} M 858,${y + 7} L 866,${y - 4}`}
                    stroke="#818cf8"
                    strokeWidth="1.2"
                    fill="none"
                  />
                </g>
              );
            })}
          </svg>
        </div>

        {/* Circuit Telemetry Metric Strips */}
        <div className="circuit-metrics-grid">
          <div className="metric-box">
            <span className="metric-label">QUBITS (n):</span>
            <strong className="metric-val">{qubits}</strong>
            <small>Active quantum register</small>
          </div>
          <div className="metric-box">
            <span className="metric-label">HILBERT SPACE:</span>
            <strong className="metric-val">{hilbertDim.toLocaleString()}</strong>
            <small>2^{qubits} complex basis states</small>
          </div>
          <div className="metric-box">
            <span className="metric-label">TRAINABLE WEIGHTS:</span>
            <strong className="metric-val">{totalParams}</strong>
            <small>Angle + Variational gates</small>
          </div>
          <div className="metric-box">
            <span className="metric-label">ENGINE STATUS:</span>
            <strong className="metric-val text-pending">Not Connected</strong>
            <small>Awaiting Milestone C Simulator</small>
          </div>
        </div>
      </div>

      {/* Execution Target Banner & Scientific Integrity */}
      <div className="workbench-status-card">
        {execMode === "simulator" ? (
          <div className="status-mode-box">
            <div className="mode-header">
              <span className="status-dot status-dot--pending" />
              <strong>SIMULATOR MODE (DEFAULT)</strong>
            </div>
            <p>
              PennyLane state-vector simulator (`default.qubit`) provides exact analytical quantum expectations
              without hardware queue delays or sampling shot noise. Scheduled for activation in <strong>Milestone C</strong>.
            </p>
          </div>
        ) : (
          <div className="status-mode-box">
            <div className="mode-header">
              <span className="status-dot status-dot--pending" />
              <strong>IBM QUANTUM HARDWARE ADAPTER</strong>
            </div>
            <p>
              Direct QPU job submission via <code>qiskit-ibm-runtime</code> targeting eligible Open Plan QPUs
              (e.g., <code>ibm_kyoto</code>, <code>ibm_osaka</code>). Scheduled for activation in <strong>Milestone D</strong>.
              Requires valid user-provided <code>IBM_QUANTUM_TOKEN</code>.
            </p>
          </div>
        )}

        <div className="scientific-notice-card">
          <h4>🔬 Quantum Advantage Evidence Standard</h4>
          <p>
            In strict compliance with our research principles: We do not claim quantum advantage without verified
            empirical evidence. The hybrid VQC will be benchmarked on identical, frozen EuroSAT test splits against
            the classical ResNet-18 baseline under documented conditions.
          </p>
        </div>
      </div>
    </div>
  );
}
