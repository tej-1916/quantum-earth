"""EuroSAT Classical Model Evaluation on Unseen Test Split.

Calculates genuine, non-fabricated metrics:
- Overall Top-1 Accuracy
- Macro and Weighted Precision, Recall, and F1-Score
- Class-level performance metrics (Precision, Recall, F1, Support)
- Full 10x10 Confusion Matrix
- Exports confusion matrix heatmap to experiments/confusion_matrix.png
- Exports evaluation report to experiments/classical_test_metrics.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from models.classical.resnet18 import EuroSATResNet18
from models.classical.dataset import get_eurosat_dataloaders, CLASS_NAMES


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path | str) -> None:
    """Plots and saves normalized confusion matrix heatmap with dark scientific aesthetic."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#070c13")
    ax.set_facecolor("#070c13")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8")
    cbar.set_label("Normalized Frequency", color="#94a3b8", fontsize=10)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="#e2e8f0", fontsize=9)
    ax.set_yticklabels(class_names, color="#e2e8f0", fontsize=9)

    ax.set_title("EuroSAT ResNet-18 Confusion Matrix (Test Split)", color="#f8fafc", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Ground Truth Class", color="#94a3b8", fontsize=10)
    ax.set_xlabel("Predicted Class", color="#94a3b8", fontsize=10)

    # Annotate matrix cells with percentage and raw count
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            raw = cm[i, j]
            color = "#070c13" if val > thresh else "#f8fafc"
            ax.text(
                j,
                i,
                f"{val*100:.1f}%\n({raw})",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
                fontweight="normal",
            )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def run_evaluation(
    checkpoint_path: str = "checkpoints/eurosat_resnet18_rgb.pt",
    data_dir: str = "data/eurosat/2750",
    exp_dir: str = "experiments",
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs rigorous test set inference, computes metrics, and produces visual artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, metadata = EuroSATResNet18.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    print(f"Instantiating test DataLoader (seed={seed})...")
    _, _, test_loader, manifest = get_eurosat_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    all_probs: List[List[float]] = []

    print(f"Evaluating {len(test_loader.dataset):,} test samples...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Core metrics
    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr)

    # Per-class metrics
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(y_true_arr, y_pred_arr, average=None, zero_division=0)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_metrics[name] = {
            "precision": round(float(class_prec[idx]), 4),
            "recall": round(float(class_rec[idx]), 4),
            "f1_score": round(float(class_f1[idx]), 4),
            "support": int(class_supp[idx]),
        }

    report_text = classification_report(y_true_arr, y_pred_arr, target_names=CLASS_NAMES, digits=4, zero_division=0)

    results = {
        "dataset": "EuroSAT RGB",
        "split": "test",
        "total_test_samples": len(y_true),
        "metrics": {
            "overall_accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1_score": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "weighted_f1_score": round(float(weighted_f1), 4),
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "checkpoint_used": str(Path(checkpoint_path).resolve()),
        "checkpoint_metadata": metadata,
        "classification_report_raw": report_text,
    }

    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_file = exp_path / "classical_test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot confusion matrix
    cm_plot_file = exp_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_file)

    print("\n" + "=" * 60)
    print("EUROSAT RESNET-18 TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Test Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1*100:.2f}%")
    print("-" * 60)
    print(report_text)
    print("=" * 60)
    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved confusion matrix plot to: {cm_plot_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Classical Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/eurosat_resnet18_rgb.pt")
    parser.add_argument("--data-dir", type=str, default="data/eurosat/2750")
    parser.add_argument("--exp-dir", type=str, default="experiments")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        exp_dir=args.exp_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
