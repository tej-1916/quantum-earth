"""EuroSAT ResNet-18 Classical Feature Encoder for Quantum Circuits.

Extracts deep 512-dimensional visual representations from satellite imagery using
the fine-tuned ResNet-18 checkpoint, and projects them into compact latent
vectors (e.g. 4, 8, 12 dimensions) scaled for quantum angle encoding.

Adheres strictly to research integrity:
- Reproducible random seeds
- Exact mathematical transformations without fabrication
- Deterministic data splits matching Milestone B
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.classical.resnet18 import EuroSATResNet18
from models.classical.dataset import (
    create_deterministic_splits,
    EuroSATRGBDataset,
    get_transforms,
)


class EuroSATFeatureEncoder(nn.Module):
    """Classical CNN feature extractor with trainable or fixed linear dimensionality reduction.
    
    Transforms 64x64x3 satellite imagery into compact n-dimensional latent vectors
    suitably scaled for quantum state preparation (e.g. Angle Embedding in [-pi/2, pi/2]).
    """

    def __init__(
        self,
        checkpoint_path: Path | str = "checkpoints/eurosat_resnet18_rgb.pt",
        latent_dim: int = 8,
        freeze_backbone: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.freeze_backbone = freeze_backbone
        self.device = device

        # Load trained ResNet-18 from Milestone B
        self.backbone, self.checkpoint_meta = EuroSATResNet18.load_from_checkpoint(
            checkpoint_path, device=device
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Compact latent projection: 512 -> latent_dim
        self.projection = nn.Linear(512, latent_dim)
        # Bounded scaling for angle encoding: [-pi/2, pi/2]
        self.angle_scale = math.pi / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and compresses visual features.
        
        Args:
            x: Satellite image tensor of shape (B, 3, 64, 64).
        Returns:
            Compact latent feature vector of shape (B, latent_dim) scaled for angle encoding.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features_512 = self.backbone.extract_features(x)
        else:
            features_512 = self.backbone.extract_features(x)

        projected = self.projection(features_512)
        # Non-linear squashing into valid rotation angle range [-pi/2, pi/2]
        angles = torch.tanh(projected) * self.angle_scale
        return angles

    def encode_512_features(self, features_512: torch.Tensor) -> torch.Tensor:
        """Projects pre-extracted 512-dim features directly to latent angles."""
        projected = self.projection(features_512)
        return torch.tanh(projected) * self.angle_scale


class CachedEmbeddingDataset(Dataset):
    """In-memory PyTorch Dataset for pre-extracted ResNet-18 embeddings."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def precompute_split_embeddings(
    dataset_dir: Path | str = "data/eurosat/2750",
    checkpoint_path: Path | str = "checkpoints/eurosat_resnet18_rgb.pt",
    cache_file: Path | str = "data/eurosat_resnet_embeddings.pt",
    batch_size: int = 128,
    num_workers: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Pre-extracts 512-dimensional features for the deterministic 70/15/15 EuroSAT splits.
    
    Caching the 512-dim embeddings allows rapid, exact training and evaluation of
    quantum variational circuits and matched MLPs across various qubit counts,
    circuit depths, and low-data regimes without redundant CNN forward passes.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        print(f"Loading cached EuroSAT ResNet-18 embeddings from: {cache_path}")
        cache_data = torch.load(cache_path, map_location=device, weights_only=False)
        return {
            "train": (cache_data["train_embeddings"], cache_data["train_labels"]),
            "val": (cache_data["val_embeddings"], cache_data["val_labels"]),
            "test": (cache_data["test_embeddings"], cache_data["test_labels"]),
        }

    print(f"Pre-extracting EuroSAT representations using checkpoint: {checkpoint_path}")
    model, _ = EuroSATResNet18.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    manifest = create_deterministic_splits(data_dir=dataset_dir, seed=42)

    # For canonical feature extraction, use deterministic transforms (no random flips/rotations)
    eval_transform = get_transforms(is_train=False)
    train_dataset = EuroSATRGBDataset(manifest["train_samples"], transform=eval_transform)
    val_dataset = EuroSATRGBDataset(manifest["val_samples"], transform=eval_transform)
    test_dataset = EuroSATRGBDataset(manifest["test_samples"], transform=eval_transform)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    results = {}
    for split_name in ["train", "val", "test"]:
        loader = loaders[split_name]
        emb_list = []
        lbl_list = []

        print(f"  Extracting representations for {split_name} split ({len(loader.dataset)} samples)...")
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                features = model.extract_features(inputs)
                emb_list.append(features.cpu())
                lbl_list.append(labels.cpu())

        split_embeddings = torch.cat(emb_list, dim=0)
        split_labels = torch.cat(lbl_list, dim=0)
        results[split_name] = (split_embeddings, split_labels)
        print(f"    Done {split_name}: embeddings shape {split_embeddings.shape}")

    # Save to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_embeddings": results["train"][0],
        "train_labels": results["train"][1],
        "val_embeddings": results["val"][0],
        "val_labels": results["val"][1],
        "test_embeddings": results["test"][0],
        "test_labels": results["test"][1],
        "split_manifest_hash": manifest["manifest_hash"],
        "checkpoint_source": str(checkpoint_path),
        "embedding_dim": 512,
    }
    torch.save(payload, cache_path)
    print(f"Saved pre-extracted embeddings cache ({cache_path.stat().st_size / (1024*1024):.2f} MB) to: {cache_path}")
    return results
