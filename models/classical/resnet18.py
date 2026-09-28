"""EuroSAT ResNet-18 Classical Classifier.

Transfers ImageNet representations to Earth-observation satellite patch classification (64x64 RGB).
Adheres strictly to scientific integrity:
- Reproducible random seeds
- Explicit architecture parameters
- Checkpoint persistence with complete training metadata
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class EuroSATResNet18(nn.Module):
    """ResNet-18 model fine-tuned for 10-class EuroSAT Land Use / Land Cover classification."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout

        # Backbone: ImageNet-pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace classification head
        in_features = self.backbone.fc.in_features  # 512
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (B, 3, 64, 64) normalized with EuroSAT statistics.
        Returns:
            Logits of shape (B, 10).
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dimensional pen-ultimate feature embedding before classification head."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def save_checkpoint(
        self,
        path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves weights and complete training metadata into a reproducible checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "model_architecture": "resnet18",
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "pretrained_backbone": self.pretrained,
            "epoch": epoch,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Path | str,
        device: torch.device = torch.device("cpu"),
    ) -> tuple["EuroSATResNet18", Dict[str, Any]]:
        """Loads model weights and returns model instance along with checkpoint metadata."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle raw state_dict vs packaged checkpoint dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 10)
            dropout = checkpoint.get("dropout", 0.2)
            metadata = {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "hyperparameters": checkpoint.get("hyperparameters", {}),
            }
        else:
            state_dict = checkpoint
            num_classes = 10
            dropout = 0.0
            metadata = {"epoch": 0, "metrics": {}, "hyperparameters": {}}

        model = cls(num_classes=num_classes, pretrained=False, dropout=dropout)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model, metadata
