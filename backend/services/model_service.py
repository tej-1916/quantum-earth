"""PyTorch Model Definitions & Honest Connectivity Registry Service.

In strict adherence to the project's scientific integrity principles:
1. No predictions, confidence scores, accuracy percentages, or processing latencies are fabricated.
2. If trained weights are not loaded from disk, the model status is explicitly 'not_connected'.
3. Classical ResNet-18 (RGB and 13-band multispectral) and Hybrid Quantum architectures are maintained separately.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from backend.core.config import settings
from backend.services.dataset_service import EUROSAT_CLASSES


class EuroSATResNetRGB(nn.Module):
    """Classical baseline: ResNet-18 architecture for 3-channel EuroSAT RGB images."""
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EuroSATResNetMultispectral(nn.Module):
    """Multispectral baseline: ResNet-18 with 13-channel input layer for all Sentinel-2 L2A bands."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        # Adapt conv1 from 3 to 13 input channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=13,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class HybridQuantumModelPlaceholder(nn.Module):
    """Hybrid Quantum Architecture: Classical CNN feature extractor (8-dim) + Variational Quantum Circuit."""
    def __init__(self, num_classes: int = 10, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        # Feature reducer to 8 quantum angles
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, num_qubits),
        )
        # Quantum circuit projection layer (Milestone C integration)
        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        # Quantum expectation measurements will be evaluated via PennyLane in Milestone C
        return self.classifier(feat)


class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = settings.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._rgb_model: Optional[EuroSATResNetRGB] = None
        self._ms_model: Optional[EuroSATResNetMultispectral] = None
        self._quantum_model: Optional[HybridQuantumModelPlaceholder] = None

        self._check_and_load_checkpoints()

    def _check_and_load_checkpoints(self):
        rgb_path = self.checkpoint_dir / "eurosat_resnet18_rgb.pt"
        if rgb_path.exists():
            try:
                model = EuroSATResNetRGB(num_classes=10)
                model.load_state_dict(torch.load(rgb_path, map_location=self.device, weights_only=True))
                model.eval()
                self._rgb_model = model
            except Exception:
                self._rgb_model = None

        ms_path = self.checkpoint_dir / "eurosat_resnet18_ms.pt"
        if ms_path.exists():
            try:
                model = EuroSATResNetMultispectral(num_classes=10)
                model.load_state_dict(torch.load(ms_path, map_location=self.device, weights_only=True))
                model.eval()
                self._ms_model = model
            except Exception:
                self._ms_model = None

    def get_status(self) -> Dict[str, Any]:
        """Returns honest connection status for all models."""
        return {
            "device": str(self.device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "models": {
                "classical_resnet18_rgb": {
                    "name": "Classical ResNet-18 (RGB Baseline)",
                    "type": "classical_cnn",
                    "input_channels": 3,
                    "status": "connected" if self._rgb_model is not None else "not_connected",
                    "available": self._rgb_model is not None,
                    "target_dataset": "EuroSAT RGB",
                    "milestone": "Milestone B (Training Pipeline)",
                    "checkpoint_file": "eurosat_resnet18_rgb.pt",
                    "reason": "Ready for inference" if self._rgb_model is not None else "Awaiting trained weights from Milestone B. No predictions will be mocked.",
                },
                "classical_resnet18_multispectral": {
                    "name": "Classical ResNet-18 (13-Band Multispectral)",
                    "type": "multispectral_cnn",
                    "input_channels": 13,
                    "status": "connected" if self._ms_model is not None else "not_connected",
                    "available": self._ms_model is not None,
                    "target_dataset": "EuroSAT Multispectral (Sentinel-2 L2A)",
                    "milestone": "Milestone B (Training Pipeline)",
                    "checkpoint_file": "eurosat_resnet18_ms.pt",
                    "reason": "Ready for inference" if self._ms_model is not None else "Awaiting trained weights from Milestone B. No predictions will be mocked.",
                },
                "hybrid_quantum_vqc": {
                    "name": "Hybrid Quantum VQC Classifier (8-Qubit)",
                    "type": "hybrid_quantum_neural_network",
                    "input_channels": 3,
                    "qubits": 8,
                    "status": "not_connected",
                    "available": False,
                    "target_dataset": "EuroSAT (Shared Benchmark Split)",
                    "milestone": "Milestone C (Quantum Simulator Engine)",
                    "reason": "Awaiting quantum circuit simulation engine in Milestone C. No quantum advantage or fake metrics claimed.",
                },
            },
            "research_integrity": {
                "fabricated_metrics": False,
                "policy": "Never fabricate model predictions, accuracy, or processing times. Clearly mark unavailable models as not connected.",
            },
        }

    def predict_rgb(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Runs genuine inference if weights are loaded. Otherwise raises RuntimeError."""
        if self._rgb_model is None:
            raise RuntimeError(
                "Model is NOT CONNECTED: No trained weights found at 'eurosat_resnet18_rgb.pt'. "
                "Per research integrity guidelines, inference is refused rather than simulated."
            )

        with torch.no_grad():
            outputs = self._rgb_model(image_tensor.to(self.device))
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()

            top_idx = int(torch.argmax(outputs, dim=1).item())
            class_info = EUROSAT_CLASSES[top_idx]

            return {
                "predicted_class": class_info["name"],
                "predicted_label": class_info["label"],
                "confidence": probs[top_idx],
                "all_probabilities": {
                    EUROSAT_CLASSES[i]["name"]: probs[i] for i in range(len(EUROSAT_CLASSES))
                },
            }


model_service = ModelService()
