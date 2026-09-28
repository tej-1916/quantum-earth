"""Hybrid Quantum-Classical Neural Network and Matched Classical MLP Models.

Architecture Pipeline:
Satellite Image (64x64x3)
  --> ResNet-18 Backbone (Frozen ImageNet/EuroSAT Embeddings, 512-dim)
  --> Compact Linear Projection (512 -> n_qubits, bounded in [-pi/2, pi/2])
  --> Parameterized Quantum Circuit (Angle Encoding, Variational RY/RZ, CNOT Entanglement)
  --> Pauli-Z Expectation Value Measurements (<Z_0>, ..., <Z_{N-1}> in [-1.0, 1.0])
  --> Linear Classification Head (n_qubits -> 10 EuroSAT classes)
  --> Softmax Probability Distribution

Matched Classical MLP Baseline:
Replaces the quantum circuit with an identically dimensioned classical hidden layer:
512 -> 8 (projection) -> 8 (ReLU hidden layer) -> 10 (head).
Both models have ~4,200 trainable parameters, ensuring fair scientific comparison.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import math
import time
import numpy as np
import torch
import torch.nn as nn

from models.classical.resnet18 import EuroSATResNet18
from models.quantum.circuit import QuantumCircuitLayer, EntanglementMode, EncodingType


class HybridEuroSATVQC(nn.Module):
    """Hybrid Classical-Quantum Classifier for EuroSAT Land Cover Analysis."""

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 2,
        entanglement: EntanglementMode = "ring",
        encoding: EncodingType = "angle",
        num_classes: int = 10,
        backbone_checkpoint: Optional[Path | str] = "checkpoints/eurosat_resnet18_rgb.pt",
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.num_classes = num_classes
        self.device = device

        torch.manual_seed(seed)

        # 1. Classical Backbone (Optional in-memory load for image inference)
        self.backbone: Optional[EuroSATResNet18] = None
        if backbone_checkpoint and Path(backbone_checkpoint).exists():
            self.backbone, _ = EuroSATResNet18.load_from_checkpoint(
                backbone_checkpoint, device=device
            )
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 2. Linear Projection: 512 -> n_qubits
        self.projection = nn.Linear(512, n_qubits)
        self.angle_scale = math.pi / 2.0

        # 3. Variational Quantum Circuit Layer
        self.quantum_layer = QuantumCircuitLayer(
            n_qubits=n_qubits,
            depth=depth,
            entanglement=entanglement,
            encoding=encoding,
            device_name="default.qubit",
            seed=seed,
        )

        # 4. Final Classification Head: n_qubits expectation values -> 10 classes
        self.classifier = nn.Linear(n_qubits, num_classes)

        self.to(device)

    def forward_from_embeddings(self, embeddings_512: torch.Tensor) -> torch.Tensor:
        """Fast forward pass taking pre-extracted 512-dimensional representations.
        
        Args:
            embeddings_512: Tensor of shape (B, 512).
        Returns:
            Logits of shape (B, 10).
        """
        # Compress 512 to n_qubits angle features in [-pi/2, pi/2]
        projected = self.projection(embeddings_512.to(self.device))
        angles = torch.tanh(projected) * self.angle_scale

        # Quantum circuit forward pass -> Pauli-Z expectations in [-1, 1]
        expvals = self.quantum_layer(angles).to(dtype=self.classifier.weight.dtype)

        # Final classification head
        logits = self.classifier(expvals)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass taking raw satellite image tensor (B, 3, 64, 64)."""
        if self.backbone is None:
            raise RuntimeError("Backbone is not loaded. Cannot run forward on raw image.")

        with torch.no_grad():
            embeddings = self.backbone.extract_features(x.to(self.device))

        return self.forward_from_embeddings(embeddings)

    def forward_with_telemetry(self, x: torch.Tensor) -> Dict[str, Any]:
        """Runs end-to-end inference and returns expectation values, latencies, and circuit metrics."""
        start_time = time.perf_counter()

        if self.backbone is None:
            raise RuntimeError("Backbone is not loaded. Cannot run forward on raw image.")

        t_enc0 = time.perf_counter()
        with torch.no_grad():
            embeddings = self.backbone.extract_features(x.to(self.device))
            projected = self.projection(embeddings)
            angles = torch.tanh(projected) * self.angle_scale
        encoding_time_ms = round((time.perf_counter() - t_enc0) * 1000, 2)

        t_q0 = time.perf_counter()
        with torch.no_grad():
            expvals = self.quantum_layer(angles).to(dtype=self.classifier.weight.dtype)
        quantum_sim_time_ms = round((time.perf_counter() - t_q0) * 1000, 2)

        t_head0 = time.perf_counter()
        with torch.no_grad():
            logits = self.classifier(expvals)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        head_time_ms = round((time.perf_counter() - t_head0) * 1000, 2)

        total_runtime_ms = round((time.perf_counter() - start_time) * 1000, 2)
        top_idx = int(torch.argmax(logits, dim=1).item())

        expectation_list = expvals.squeeze().cpu().tolist()
        if isinstance(expectation_list, float):
            expectation_list = [expectation_list]

        return {
            "predicted_index": top_idx,
            "probabilities": probs,
            "confidence": probs[top_idx] if isinstance(probs, list) else probs,
            "expectation_values": [round(float(v), 5) for v in expectation_list],
            "latent_angles": [round(float(v), 5) for v in angles.squeeze().cpu().tolist()],
            "timings_ms": {
                "classical_encoder": encoding_time_ms,
                "quantum_simulator": quantum_sim_time_ms,
                "classification_head": head_time_ms,
                "total_inference": total_runtime_ms,
            },
            "circuit_metadata": self.quantum_layer.get_circuit_metadata(),
            "model_type": "Hybrid Classical-Quantum VQC (PennyLane Simulator)",
        }

    def count_parameters(self) -> Dict[str, int]:
        """Returns breakdown of trainable parameters."""
        proj_params = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        quantum_params = self.quantum_layer.weights_ry.numel() + self.quantum_layer.weights_rz.numel()
        head_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return {
            "projection_parameters": proj_params,
            "quantum_variational_parameters": quantum_params,
            "classifier_head_parameters": head_params,
            "total_trainable_parameters": proj_params + quantum_params + head_params,
            "frozen_backbone_parameters": 11_176_512,  # ResNet-18
        }

    def save_checkpoint(
        self,
        path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves VQC weights and experiment metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_architecture": "HybridEuroSATVQC",
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "entanglement": self.entanglement,
            "encoding": self.encoding,
            "num_classes": self.num_classes,
            "projection_state_dict": self.projection.state_dict(),
            "weights_ry": self.quantum_layer.weights_ry.detach().cpu(),
            "weights_rz": self.quantum_layer.weights_rz.detach().cpu(),
            "classifier_state_dict": self.classifier.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "parameter_counts": self.count_parameters(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Path | str,
        backbone_checkpoint: Optional[Path | str] = "checkpoints/eurosat_resnet18_rgb.pt",
        device: torch.device = torch.device("cpu"),
    ) -> Tuple["HybridEuroSATVQC", Dict[str, Any]]:
        """Loads hybrid VQC model from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VQC checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        n_qubits = checkpoint.get("n_qubits", 8)
        depth = checkpoint.get("depth", 2)
        entanglement = checkpoint.get("entanglement", "ring")
        encoding = checkpoint.get("encoding", "angle")
        num_classes = checkpoint.get("num_classes", 10)

        model = cls(
            n_qubits=n_qubits,
            depth=depth,
            entanglement=entanglement,
            encoding=encoding,
            num_classes=num_classes,
            backbone_checkpoint=backbone_checkpoint,
            device=device,
        )

        model.projection.load_state_dict(checkpoint["projection_state_dict"])
        with torch.no_grad():
            model.quantum_layer.weights_ry.copy_(checkpoint["weights_ry"].to(device))
            model.quantum_layer.weights_rz.copy_(checkpoint["weights_rz"].to(device))
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        model.eval()

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "hyperparameters": checkpoint.get("hyperparameters", {}),
            "parameter_counts": checkpoint.get("parameter_counts", model.count_parameters()),
            "circuit_metadata": model.quantum_layer.get_circuit_metadata(),
        }
        return model, metadata


class MatchedEuroSATMLP(nn.Module):
    """Matched Classical Baseline receiving the identical compressed features.
    
    Architecture:
    512 -> n_latent (projection) -> n_latent (ReLU hidden layer) -> 10 (head).
    Directly matched in capacity and dimensionality to the hybrid quantum model.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 8,
        num_classes: int = 10,
        backbone_checkpoint: Optional[Path | str] = "checkpoints/eurosat_resnet18_rgb.pt",
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device

        torch.manual_seed(seed)

        self.backbone: Optional[EuroSATResNet18] = None
        if backbone_checkpoint and Path(backbone_checkpoint).exists():
            self.backbone, _ = EuroSATResNet18.load_from_checkpoint(
                backbone_checkpoint, device=device
            )
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Projection matching VQC input compression
        self.projection = nn.Linear(512, latent_dim)
        self.angle_scale = math.pi / 2.0

        # Hidden classical layer matching VQC transformation
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),  # Tanh matches bounded [-1, 1] output range of Pauli-Z expectations
        )

        # Classification head matching VQC classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.to(device)

    def forward_from_embeddings(self, embeddings_512: torch.Tensor) -> torch.Tensor:
        """Fast forward pass from pre-extracted features."""
        projected = self.projection(embeddings_512.to(self.device))
        angles = torch.tanh(projected) * self.angle_scale
        h = self.hidden(angles)
        return self.classifier(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass on raw satellite images."""
        if self.backbone is None:
            raise RuntimeError("Backbone is not loaded.")
        with torch.no_grad():
            embeddings = self.backbone.extract_features(x.to(self.device))
        return self.forward_from_embeddings(embeddings)

    def count_parameters(self) -> Dict[str, int]:
        proj = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        hidden = sum(p.numel() for p in self.hidden.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return {
            "projection_parameters": proj,
            "hidden_parameters": hidden,
            "classifier_head_parameters": head,
            "total_trainable_parameters": proj + hidden + head,
            "frozen_backbone_parameters": 11_176_512,
        }

    def save_checkpoint(
        self,
        path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_architecture": "MatchedEuroSATMLP",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "parameter_counts": self.count_parameters(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Path | str,
        backbone_checkpoint: Optional[Path | str] = "checkpoints/eurosat_resnet18_rgb.pt",
        device: torch.device = torch.device("cpu"),
    ) -> Tuple["MatchedEuroSATMLP", Dict[str, Any]]:
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            latent_dim=checkpoint.get("latent_dim", 8),
            hidden_dim=checkpoint.get("hidden_dim", 8),
            num_classes=checkpoint.get("num_classes", 10),
            backbone_checkpoint=backbone_checkpoint,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "hyperparameters": checkpoint.get("hyperparameters", {}),
            "parameter_counts": checkpoint.get("parameter_counts", model.count_parameters()),
        }
        return model, metadata
