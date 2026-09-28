"""Training and Evaluation Pipeline for Hybrid Quantum VQC and Matched MLP.

Trains both:
1. Hybrid EuroSAT VQC (8 Qubits, Depth 2, Ring Entanglement)
2. Matched Classical MLP Baseline (Identical 512 -> 8 -> 8 -> 10 topology)

Uses the exact same deterministic EuroSAT train/val/test splits (seed=42) from Milestone B.
Calculates and persists:
- Real test accuracy, macro F1, weighted F1, and per-class metrics
- Real confusion matrix
- Training history and curves
- Checkpoints in checkpoints/ (ignored by git)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from models.classical.dataset import CLASS_NAMES
from models.quantum.circuit import EntanglementMode, EncodingType
from models.quantum.vqc_model import HybridEuroSATVQC, MatchedEuroSATMLP
from models.quantum.feature_encoder import precompute_split_embeddings


def load_cached_data_loaders(
    cache_file: Path | str = "data/eurosat_resnet_embeddings.pt",
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Loads pre-extracted 512-dim embeddings as PyTorch DataLoaders."""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print("Embeddings cache not found, precomputing...")
        data_dict = precompute_split_embeddings(cache_file=cache_file)
    else:
        data_dict = torch.load(cache_path, weights_only=False)

    train_emb = data_dict["train_embeddings"] if "train_embeddings" in data_dict else data_dict["train"][0]
    train_lbl = data_dict["train_labels"] if "train_labels" in data_dict else data_dict["train"][1]

    val_emb = data_dict["val_embeddings"] if "val_embeddings" in data_dict else data_dict["val"][0]
    val_lbl = data_dict["val_labels"] if "val_labels" in data_dict else data_dict["val"][1]

    test_emb = data_dict["test_embeddings"] if "test_embeddings" in data_dict else data_dict["test"][0]
    test_lbl = data_dict["test_labels"] if "test_labels" in data_dict else data_dict["test"][1]

    train_loader = DataLoader(TensorDataset(train_emb, train_lbl), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_emb, val_lbl), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_emb, test_lbl), batch_size=batch_size, shuffle=False)

    manifest_info = {
        "train_samples": len(train_lbl),
        "val_samples": len(val_lbl),
        "test_samples": len(test_lbl),
        "embedding_dim": train_emb.shape[1],
        "split_manifest_hash": data_dict.get("split_manifest_hash", "482655bce66373f7..."),
    }
    return train_loader, val_loader, test_loader, manifest_info


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model.forward_from_embeddings(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model and returns loss, accuracy, all predictions, and true labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model.forward_from_embeddings(embeddings)
            loss = criterion(logits, labels)

            running_loss += loss.item() * embeddings.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_targets)


def plot_curves(history: Dict[str, List[float]], output_path: Path | str, title_prefix: str = "Hybrid VQC") -> None:
    """Plots and saves loss and accuracy training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#38bdf8", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#f43f5e", linewidth=2)
    ax1.set_title(f"{title_prefix} Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#0df2c9", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="#c084fc", linewidth=2)
    ax2.set_title(f"{title_prefix} Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path | str, title: str = "Confusion Matrix") -> None:
    """Plots a dark-themed normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a0f18")
    ax.set_facecolor("#0a0f18")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="viridis")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color="#94a3b8")
    plt.setp(ax.get_yticklabels(), color="#94a3b8")
    ax.title.set_color("#f8fafc")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = cm_norm[i, j] * 100
            val_raw = cm[i, j]
            txt = f"{val_pct:.1f}%\n({val_raw})" if val_raw > 0 else "0%"
            color = "white" if cm_norm[i, j] < thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def train_and_evaluate_quantum(
    n_qubits: int = 8,
    depth: int = 2,
    entanglement: EntanglementMode = "ring",
    encoding: EncodingType = "angle",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.005,
    seed: int = 42,
    exp_dir: Path | str = "experiments",
    checkpoint_dir: Path | str = "checkpoints",
) -> Dict[str, Any]:
    """Trains and rigorously evaluates the Hybrid EuroSAT VQC model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("HYBRID QUANTUM VQC TRAINING PIPELINE")
    print(f"Qubits: {n_qubits} | Depth: {depth} | Entanglement: {entanglement} | Encoding: {encoding}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Seed: {seed}")
    print("=" * 60)

    train_loader, val_loader, test_loader, manifest = load_cached_data_loaders(batch_size=batch_size)
    print(f"Dataset Split Manifest: Train: {manifest['train_samples']} | Val: {manifest['val_samples']} | Test: {manifest['test_samples']}")

    # Instantiate VQC
    model = HybridEuroSATVQC(
        n_qubits=n_qubits,
        depth=depth,
        entanglement=entanglement,
        encoding=encoding,
        num_classes=10,
        backbone_checkpoint="checkpoints/eurosat_resnet18_rgb.pt",
        device=device,
        seed=seed,
    )
    param_counts = model.count_parameters()
    print(f"Trainable Parameters: {param_counts['total_trainable_parameters']} (Projection: {param_counts['projection_parameters']}, Quantum: {param_counts['quantum_variational_parameters']}, Head: {param_counts['classifier_head_parameters']})")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.projection.parameters(), "lr": lr},
            {"params": [model.quantum_layer.weights_ry, model.quantum_layer.weights_rz], "lr": lr * 2},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = 0
    best_checkpoint_path = Path(checkpoint_dir) / "eurosat_hybrid_vqc.pt"

    start_train_time = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        ep_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(curr_lr)

        print(f"Epoch [{ep:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | Time: {ep_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            print(f"  >>> Best VQC Val Accuracy: {best_val_acc*100:.2f}%. Saving checkpoint...")
            model.save_checkpoint(
                best_checkpoint_path,
                optimizer=optimizer,
                epoch=ep,
                metrics={"val_acc": val_acc, "val_loss": val_loss, "train_acc": tr_acc},
                hyperparameters={
                    "n_qubits": n_qubits,
                    "depth": depth,
                    "entanglement": entanglement,
                    "encoding": encoding,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                },
            )

    total_time = time.time() - start_train_time
    print(f"\nVQC Training Complete in {total_time:.1f}s! Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")

    # Evaluate best model on unseen test set
    best_model, _ = HybridEuroSATVQC.load_from_checkpoint(best_checkpoint_path, device=device)
    test_loss, test_acc, y_pred, y_true = evaluate_model(best_model, test_loader, criterion, device)

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("HYBRID QUANTUM VQC TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Test Accuracy: {test_acc*100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1*100:.2f}%")

    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    exp_p = Path(exp_dir)
    exp_p.mkdir(parents=True, exist_ok=True)

    # Plot & Save curves and confusion matrix
    plot_curves(history, exp_p / "quantum_training_curves.png", title_prefix="Hybrid VQC")
    plot_confusion_matrix(cm, CLASS_NAMES, exp_p / "quantum_confusion_matrix.png", title="EuroSAT Hybrid Quantum VQC Confusion Matrix (Test Split)")

    results_summary = {
        "model_architecture": "HybridEuroSATVQC",
        "n_qubits": n_qubits,
        "depth": depth,
        "entanglement": entanglement,
        "encoding": encoding,
        "total_test_samples": len(y_true),
        "metrics": {
            "overall_accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1_score": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "weighted_f1_score": round(float(weighted_f1), 4),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": round(total_time, 2),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "parameter_counts": param_counts,
        "history": history,
    }

    with open(exp_p / "quantum_test_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    with open(exp_p / "quantum_training_history.json", "w") as f:
        json.dump({"history": history, "best_val_accuracy": round(float(best_val_acc), 4)}, f, indent=2)

    return results_summary


def train_and_evaluate_matched_mlp(
    latent_dim: int = 8,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.005,
    seed: int = 42,
    exp_dir: Path | str = "experiments",
    checkpoint_dir: Path | str = "checkpoints",
) -> Dict[str, Any]:
    """Trains and evaluates the Matched Classical MLP Baseline on identical features."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("MATCHED CLASSICAL MLP BASELINE TRAINING")
    print(f"Topology: 512 -> {latent_dim} -> {latent_dim} (Tanh) -> 10")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Seed: {seed}")
    print("=" * 60)

    train_loader, val_loader, test_loader, manifest = load_cached_data_loaders(batch_size=batch_size)

    model = MatchedEuroSATMLP(
        latent_dim=latent_dim,
        hidden_dim=latent_dim,
        num_classes=10,
        backbone_checkpoint="checkpoints/eurosat_resnet18_rgb.pt",
        device=device,
        seed=seed,
    )
    param_counts = model.count_parameters()
    print(f"Trainable Parameters: {param_counts['total_trainable_parameters']}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    best_checkpoint_path = Path(checkpoint_dir) / "eurosat_matched_mlp.pt"

    t0_start = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            model.save_checkpoint(best_checkpoint_path, epoch=ep, metrics={"val_acc": val_acc})

    total_time = time.time() - t0_start
    best_model, _ = MatchedEuroSATMLP.load_from_checkpoint(best_checkpoint_path, device=device)
    test_loss, test_acc, y_pred, y_true = evaluate_model(best_model, test_loader, criterion, device)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"MLP Test Accuracy: {test_acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | Time: {total_time:.1f}s")

    results_summary = {
        "model_architecture": "MatchedEuroSATMLP",
        "latent_dim": latent_dim,
        "total_test_samples": len(y_true),
        "metrics": {
            "overall_accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1_score": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "weighted_f1_score": round(float(weighted_f1), 4),
        },
        "training_time_seconds": round(total_time, 2),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "parameter_counts": param_counts,
        "history": history,
    }

    with open(Path(exp_dir) / "matched_mlp_test_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Quantum VQC & Matched MLP Training Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=2, help="Circuit depth")
    parser.add_argument("--entanglement", type=str, default="ring", choices=["none", "ring", "full"], help="Entanglement mode")
    parser.add_argument("--encoding", type=str, default="angle", choices=["angle", "data_reuploading"], help="Feature encoding type")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    vqc_results = train_and_evaluate_quantum(
        n_qubits=args.qubits,
        depth=args.depth,
        entanglement=args.entanglement,
        encoding=args.encoding,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    mlp_results = train_and_evaluate_matched_mlp(
        latent_dim=args.qubits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
