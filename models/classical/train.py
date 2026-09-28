"""Reproducible EuroSAT Classical Training Pipeline (ResNet-18 RGB).

Features:
- Deterministic random seed initialization
- Transfer learning with ImageNet pretrained backbone
- Differential learning rates / AdamW optimizer + Cosine Annealing
- Checkpoint persistence with full validation metrics & hyperparameters
- Zero dataset leakage guarantee
- Training loss and accuracy curves plotted and saved to experiments/
"""

import os
import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.classical.resnet18 import EuroSATResNet18
from models.classical.dataset import get_eurosat_dataloaders, CLASS_NAMES


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one training epoch and returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for numerical stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates on validation split and returns (mean_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_training_curves(history: Dict[str, List[float]], output_path: Path | str) -> None:
    """Plots and saves loss and accuracy curves over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#38bdf8", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#f43f5e", linewidth=2)
    ax1.set_title("EuroSAT ResNet-18 Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#0df2c9", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="#a855f7", linewidth=2)
    ax2.set_title("EuroSAT Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_training(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 4,
    data_dir: str = "data/eurosat/2750",
    save_dir: str = "checkpoints",
    exp_dir: str = "experiments",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Executes complete training loop with model checkpointing and telemetry logging."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"EuroSAT Classical ResNet-18 Training Pipeline")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Learning Rate: {lr} | Weight Decay: {weight_decay}")
    print(f"==================================================")

    smoke_samples = 256 if smoke_test else None
    train_loader, val_loader, test_loader, manifest = get_eurosat_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        smoke_test_samples=smoke_samples,
    )

    print(f"Dataset Split Manifest:")
    print(f"  Train samples: {len(train_loader.dataset):,} | Batches: {len(train_loader)}")
    print(f"  Val samples:   {len(val_loader.dataset):,} | Batches: {len(val_loader)}")
    print(f"  Test samples:  {len(test_loader.dataset):,} | Batches: {len(test_loader)}")
    print(f"  Disjoint Sets (Zero Leakage): {manifest['leakage_audit']['disjoint_verified']}")
    print(f"  Split SHA-256 Hash: {manifest['manifest_hash'][:16]}...")

    # Instantiate model with pretrained ImageNet weights
    model = EuroSATResNet18(num_classes=10, pretrained=True, dropout=0.2)
    model.to(device)

    # Differential learning rate: slightly lower for pretrained backbone, higher for classification head
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.fc")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("backbone.fc")]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.2},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = 0

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = save_path / "eurosat_resnet18_rgb.pt"
    backup_best_path = save_path / "eurosat_resnet18_rgb_best.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[1]["lr"]
        history["lr"].append(current_lr)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
            f"LR: {current_lr:.1e} | Time: {epoch_duration:4.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"  >>> Best validation accuracy updated: {best_val_acc*100:.2f}%. Saving checkpoint...")

            metrics_meta = {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            hyper_meta = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "split_hash": manifest["manifest_hash"],
            }
            model.save_checkpoint(
                best_checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics_meta,
                hyperparameters=hyper_meta,
            )
            model.save_checkpoint(
                backup_best_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics_meta,
                hyperparameters=hyper_meta,
            )

    total_training_time = time.time() - start_time
    print(f"==================================================")
    print(f"Training Complete in {total_training_time:.1f}s!")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Saved Checkpoint: {best_checkpoint_path}")
    print(f"==================================================")

    # Plot & save curves
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, exp_path / "classical_training_curves.png")

    training_summary = {
        "architecture": "EuroSATResNet18",
        "total_training_time_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "final_train_accuracy": round(history["train_acc"][-1], 4),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_val_loss": round(history["val_loss"][-1], 4),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": str(device),
        },
        "history": history,
    }

    with open(exp_path / "classical_training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Classical Training Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--data-dir", type=str, default="data/eurosat/2750", help="Data directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--exp-dir", type=str, default="experiments", help="Experiment log directory")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run with subset of samples")

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        exp_dir=args.exp_dir,
        smoke_test=args.smoke_test,
    )
