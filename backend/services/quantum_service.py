"""Quantum Simulation & Inference Service.

Powers the Quantum Earth Lab V2 Quantum Workbench using PennyLane and Qiskit.
Adheres strictly to research integrity:
- Zero fabricated predictions or latencies
- Refuses inference if genuine checkpoint is absent
- Honest reporting of classical vs quantum performance
"""

import io
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torchvision import transforms

from backend.core.config import settings
from backend.services.dataset_service import EUROSAT_CLASSES
from models.quantum.circuit import (
    build_qiskit_circuit,
    get_qiskit_diagram_ascii,
    get_qasm_string,
    EntanglementMode,
    EncodingType,
)
from models.quantum.vqc_model import HybridEuroSATVQC


class QuantumService:
    """Manages PennyLane VQC simulation, inference, and Qiskit circuit generation."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.checkpoints_dir = settings.BASE_DIR / "checkpoints"
        self.experiments_dir = settings.BASE_DIR / "experiments"
        self.vqc_checkpoint_path = self.checkpoints_dir / "eurosat_hybrid_vqc.pt"

        self._vqc_model: Optional[HybridEuroSATVQC] = None
        self._checkpoint_metadata: Dict[str, Any] = {}

        self._check_and_load_checkpoint()

    def _check_and_load_checkpoint(self) -> None:
        """Loads hybrid VQC model if trained weights exist."""
        if not self.vqc_checkpoint_path.exists():
            self._vqc_model = None
            self._checkpoint_metadata = {}
            return

        try:
            backbone_path = self.checkpoints_dir / "eurosat_resnet18_rgb.pt"
            self._vqc_model, self._checkpoint_metadata = HybridEuroSATVQC.load_from_checkpoint(
                self.vqc_checkpoint_path,
                backbone_checkpoint=backbone_path if backbone_path.exists() else None,
                device=self.device,
            )
            print(f"Loaded EuroSAT Hybrid Quantum VQC checkpoint from: {self.vqc_checkpoint_path}")
        except Exception as e:
            print(f"Failed loading VQC checkpoint {self.vqc_checkpoint_path}: {e}")
            self._vqc_model = None
            self._checkpoint_metadata = {}

    @property
    def is_connected(self) -> bool:
        """Returns True if genuine trained weights are loaded."""
        if self._vqc_model is None:
            self._check_and_load_checkpoint()
        return self._vqc_model is not None

    def get_status(self) -> Dict[str, Any]:
        """Returns honest simulator connection status."""
        connected = self.is_connected
        acc = self._checkpoint_metadata.get("metrics", {}).get("val_acc")
        acc_text = f" (Best Val Acc: {acc*100:.2f}%)" if acc else ""

        circuit_meta = {}
        if connected and self._vqc_model:
            circuit_meta = self._vqc_model.quantum_layer.get_circuit_metadata()

        return {
            "status": "connected" if connected else "not_connected",
            "available": connected,
            "engine": "PennyLane Simulator (default.qubit)",
            "execution_device": str(self.device),
            "checkpoint_file": "eurosat_hybrid_vqc.pt",
            "checkpoint_metadata": self._checkpoint_metadata,
            "circuit_metadata": circuit_meta,
            "reason": f"Genuine checkpoint loaded{acc_text}. Ready for quantum simulation." if connected else "Awaiting quantum VQC checkpoint.",
            "research_integrity": {
                "fabricated_metrics": False,
                "quantum_advantage_claimed": False,
                "policy": "Never fabricate quantum predictions. Clearly report matched classical baseline comparison.",
            },
        }

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Preprocesses raw satellite imagery and executes genuine hybrid quantum inference."""
        if not self.is_connected or self._vqc_model is None:
            self._check_and_load_checkpoint()
        if self._vqc_model is None:
            raise RuntimeError(
                "Quantum VQC model is NOT CONNECTED: No trained weights found at 'eurosat_hybrid_vqc.pt'. "
                "Per research integrity guidelines, inference is refused rather than simulated."
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                pil_img = pil_img.convert("RGB")
                original_dims = [pil_img.width, pil_img.height]

                if pil_img.size != (64, 64):
                    pil_img = pil_img.resize((64, 64), Image.Resampling.BILINEAR)

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3444, 0.3803, 0.4078],
                        std=[0.2037, 0.1366, 0.1148],
                    ),
                ])
                tensor = transform(pil_img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid image format or decoding failure: {e}")

        telemetry = self._vqc_model.forward_with_telemetry(tensor)

        top_idx = telemetry["predicted_index"]
        class_info = EUROSAT_CLASSES[top_idx]
        probs = telemetry["probabilities"]

        ranked = []
        for i, c in enumerate(EUROSAT_CLASSES):
            p = float(probs[i])
            ranked.append({
                "id": c["id"],
                "class": c["name"],
                "label": c["label"],
                "description": c["description"],
                "probability": round(p, 5),
                "percentage": round(p * 100, 2),
            })
        ranked.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predicted_class": class_info["name"],
            "predicted_label": class_info["label"],
            "predicted_description": class_info["description"],
            "confidence": round(float(probs[top_idx]), 5),
            "confidence_percentage": round(float(probs[top_idx]) * 100, 2),
            "all_probabilities": {
                EUROSAT_CLASSES[i]["name"]: round(float(probs[i]), 5) for i in range(len(EUROSAT_CLASSES))
            },
            "ranked_predictions": ranked,
            "expectation_values": telemetry["expectation_values"],
            "latent_angles": telemetry["latent_angles"],
            "timings_ms": telemetry["timings_ms"],
            "circuit_metadata": telemetry["circuit_metadata"],
            "execution_device": f"PennyLane Simulator ({self.device})",
            "model_version": "Hybrid EuroSAT VQC (8-Qubit, Depth 2)",
            "original_dimensions": original_dims,
            "processed_dimensions": [64, 64, 3],
        }

    def get_circuit_details(
        self,
        n_qubits: int = 8,
        depth: int = 2,
        entanglement: EntanglementMode = "ring",
        encoding: EncodingType = "angle",
    ) -> Dict[str, Any]:
        """Generates dynamic Qiskit circuit representation, ASCII diagram, and QASM code."""
        # Validate inputs
        n_qubits = max(4, min(12, n_qubits))
        depth = max(1, min(4, depth))
        if entanglement not in ["none", "ring", "full"]:
            entanglement = "ring"
        if encoding not in ["angle", "data_reuploading"]:
            encoding = "angle"

        ascii_diagram = get_qiskit_diagram_ascii(
            n_qubits=n_qubits,
            depth=depth,
            entanglement=entanglement,
            encoding=encoding,
        )
        qasm_code = get_qasm_string(
            n_qubits=n_qubits,
            depth=depth,
            entanglement=entanglement,
            encoding=encoding,
        )

        # Gate counts
        ry_gates = n_qubits if encoding == "angle" else (n_qubits * (depth + 1))
        ry_gates += n_qubits * depth  # trainable
        rz_gates = n_qubits * depth  # trainable
        if entanglement == "ring":
            cnot_gates = n_qubits * depth
        elif entanglement == "full":
            cnot_gates = ((n_qubits * (n_qubits - 1)) // 2) * depth
        else:
            cnot_gates = 0

        # Build visual layers structure for SVG / React diagram
        layers_spec = []
        # Layer 0: Encoding
        layers_spec.append({
            "type": "encoding",
            "name": "Feature Encoding",
            "gates": [{"qubit": i, "gate": "RY", "param": f"x_{i}"} for i in range(n_qubits)],
        })

        for d in range(depth):
            if encoding == "data_reuploading" and d > 0:
                layers_spec.append({
                    "type": "reuploading",
                    "name": f"Re-uploading (D{d+1})",
                    "gates": [{"qubit": i, "gate": "RY", "param": f"x_{i}"} for i in range(n_qubits)],
                })

            layers_spec.append({
                "type": "rotations",
                "name": f"Trainable Rotations (D{d+1})",
                "gates": [
                    {"qubit": i, "gates": ["RY", "RZ"], "params": [f"θ_{d}_{i}", f"φ_{d}_{i}"]}
                    for i in range(n_qubits)
                ],
            })

            if entanglement == "ring":
                cnots = [{"control": i, "target": (i + 1) % n_qubits} for i in range(n_qubits)]
            elif entanglement == "full":
                cnots = [{"control": i, "target": j} for i in range(n_qubits) for j in range(i + 1, n_qubits)]
            else:
                cnots = []

            layers_spec.append({
                "type": "entanglement",
                "name": f"Entanglement ({entanglement.title()} D{d+1})",
                "cnots": cnots,
            })

        # Measurement layer
        layers_spec.append({
            "type": "measurement",
            "name": "Pauli-Z Expectations",
            "measurements": [f"<Z_{i}>" for i in range(n_qubits)],
        })

        return {
            "n_qubits": n_qubits,
            "depth": depth,
            "entanglement": entanglement,
            "encoding": encoding,
            "gate_counts": {
                "ry_rotations": ry_gates,
                "rz_rotations": rz_gates,
                "cnot_entanglers": cnot_gates,
                "total_gates": ry_gates + rz_gates + cnot_gates,
            },
            "trainable_parameters": (depth * n_qubits * 2),
            "ascii_diagram": ascii_diagram,
            "qasm": qasm_code,
            "layers_spec": layers_spec,
            "measurements": [f"<Z_{i}>" for i in range(n_qubits)],
        }

    def get_experiments(self) -> Dict[str, Any]:
        """Returns ablation metrics, low-data scaling curves, and classical vs quantum comparison."""
        ablation_file = self.experiments_dir / "quantum_ablation_results.json"
        comparison_file = self.experiments_dir / "classical_vs_quantum_comparison.json"
        metrics_file = self.experiments_dir / "quantum_test_metrics.json"

        ablations = {}
        comparison = {}
        metrics = {}

        if ablation_file.exists():
            with open(ablation_file) as f:
                ablations = json.load(f)
        if comparison_file.exists():
            with open(comparison_file) as f:
                comparison = json.load(f)
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

        return {
            "available": bool(ablations or comparison),
            "ablations": ablations,
            "comparison": comparison,
            "test_metrics": metrics,
        }


quantum_service = QuantumService()
