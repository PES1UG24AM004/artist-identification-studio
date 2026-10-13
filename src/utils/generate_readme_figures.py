"""
Generate publication-quality visualization figures for the GitHub repository README.

Produces 4 standalone figures in docs/images/:
1. docs/images/model-performance-comparison.png:
   Paired-panel comparison of held-out test accuracy and test macro F1 across SVM,
   original frozen ResNet-18, and fine-tuned ResNet-18.
2. docs/images/test-confusion-matrices.png:
   Side-by-side normalized confusion matrices on the 200 held-out test paintings
   comparing Classical SVM (53.0%) vs Fine-Tuned ResNet-18 (82.0%).
3. docs/images/per-artist-f1-comparison.png:
   Per-artist F1-score comparison across all 10 artist classes comparing SVM Baseline,
   original frozen ResNet-18, and fine-tuned ResNet-18.
4. docs/images/training-validation-curves.png:
   Paired loss and accuracy learning curves across all 10 fine-tuning epochs from
   genuine saved training history, highlighting early stopping and best epoch selection.
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOCS_IMG = PROJECT_ROOT / "docs" / "images"
DOCS_IMG.mkdir(parents=True, exist_ok=True)

# Set consistent matplotlib aesthetic
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "text.color": "#0f172a",
    "axes.labelcolor": "#1e293b",
    "xtick.color": "#334155",
    "ytick.color": "#334155"
})


def load_verified_data():
    """Load verified experimental and benchmark artifacts."""
    with open(PROJECT_ROOT / "outputs" / "experiments" / "test_evaluation_comparison.json", "r", encoding="utf-8") as f:
        test_eval = json.load(f)

    with open(PROJECT_ROOT / "outputs" / "model_comparison.json", "r", encoding="utf-8") as f:
        model_comp = json.load(f)

    with open(PROJECT_ROOT / "outputs" / "experiments" / "cnn_finetune_val_metrics.json", "r", encoding="utf-8") as f:
        val_metrics = json.load(f)

    return test_eval, model_comp, val_metrics


def generate_model_performance_comparison(test_eval, model_comp):
    """Figure 1: Held-out test performance comparison (Accuracy & Macro F1)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    models = [
        "Classical SVM\n(Baseline)",
        "ResNet-18\n(Frozen Backbone)",
        "ResNet-18\n(Layer 4 Fine-Tuned)"
    ]

    accuracies = [
        model_comp["summary"]["svm_baseline"]["test_accuracy"] * 100,
        test_eval["summary"]["baseline_model"]["accuracy"] * 100,
        test_eval["summary"]["finetuned_model"]["accuracy"] * 100
    ]

    macro_f1s = [
        model_comp["summary"]["svm_baseline"]["test_macro_f1"],
        test_eval["summary"]["baseline_model"]["macro_f1"],
        test_eval["summary"]["finetuned_model"]["macro_f1"]
    ]

    colors = ["#64748b", "#475569", "#0f766e"]
    edge_color = "#1e293b"

    # Panel A: Held-out Test Accuracy
    bars1 = axes[0].bar(models, accuracies, color=colors, edgecolor=edge_color, width=0.55, linewidth=1.2)
    axes[0].set_title("A. Held-Out Test Accuracy (%)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="semibold")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.7)

    for bar, val in zip(bars1, accuracies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.8,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#0f172a"
        )

    # Highlight delta on accuracy
    axes[0].annotate(
        "+11.5% vs Base CNN\n(+29.0% vs SVM)",
        xy=(2, accuracies[2]),
        xytext=(1.6, 88),
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ccfbf1", ec="#14b8a6", lw=1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#0f766e", lw=1.2)
    )

    # Panel B: Held-out Test Macro F1
    bars2 = axes[1].bar(models, macro_f1s, color=colors, edgecolor=edge_color, width=0.55, linewidth=1.2)
    axes[1].set_title("B. Held-Out Test Macro F1-Score", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Macro F1-Score (0.0 – 1.0)", fontsize=11, fontweight="semibold")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.7)

    for bar, val in zip(bars2, macro_f1s):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.018,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#0f172a"
        )

    # Highlight delta on Macro F1
    axes[1].annotate(
        "+0.1212 vs Base CNN\n(+0.2994 vs SVM)",
        xy=(2, macro_f1s[2]),
        xytext=(1.6, 0.88),
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ccfbf1", ec="#14b8a6", lw=1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#0f766e", lw=1.2)
    )

    fig.suptitle("Model Evaluation on Fixed Held-Out Test Split (N=200, 20 Paintings / Artist)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = DOCS_IMG / "model-performance-comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def generate_test_confusion_matrices(test_eval, model_comp):
    """Figure 2: Side-by-side normalized test confusion matrices (SVM vs Fine-Tuned CNN)."""
    class_labels = test_eval["class_labels"]
    clean_labels = [c.replace("_", " ") for c in class_labels]

    # Ground truth and CNN predictions from test evaluation record
    preds_data = test_eval["per_image_predictions"]
    y_true = [p["true_artist"] for p in preds_data]
    y_pred_cnn = [p["finetuned_pred"] for p in preds_data]

    # Obtain SVM predictions deterministically
    from src.features.extract_features import extract_image_features
    svm_data = joblib.load(PROJECT_ROOT / "models" / "svm_baseline.joblib")
    svm_pipe = svm_data["model_pipeline"]

    manifest_p = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
    with open(manifest_p, "r", encoding="utf-8") as f:
        test_recs = [r for r in csv.DictReader(f) if r["split"] == "test"]

    y_pred_svm = []
    for r in test_recs:
        p = PROJECT_ROOT / r["rel_path"]
        f_vec = extract_image_features(p)
        pred = svm_pipe.predict(f_vec.reshape(1, -1))[0]
        y_pred_svm.append(str(pred))

    cm_svm = confusion_matrix(y_true, y_pred_svm, labels=class_labels)
    cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=class_labels)

    # Normalize by row (True Artist Recall)
    cm_svm_norm = np.nan_to_num(cm_svm.astype(float) / cm_svm.sum(axis=1, keepdims=True), nan=0.0)
    cm_cnn_norm = np.nan_to_num(cm_cnn.astype(float) / cm_cnn.sum(axis=1, keepdims=True), nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), dpi=300)

    # Left: SVM Baseline
    sns.heatmap(
        cm_svm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[0],
        cbar_kws={"label": "Normalized Recall (Per Class)"}
    )
    axes[0].set_title(
        f"A. Classical ML (SVM Baseline)\nTest Accuracy: 53.0% (106 / 200 Correct) | Macro F1: 0.5206",
        fontsize=12, fontweight="bold", pad=12
    )
    axes[0].set_xlabel("Predicted Artist", fontsize=10.5, fontweight="semibold", labelpad=8)
    axes[0].set_ylabel("True Artist", fontsize=10.5, fontweight="semibold", labelpad=8)
    axes[0].tick_params(axis="x", rotation=40, labelsize=9.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9.5)

    # Right: Fine-Tuned ResNet-18
    sns.heatmap(
        cm_cnn_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[1],
        cbar_kws={"label": "Normalized Recall (Per Class)"}
    )
    axes[1].set_title(
        f"B. Deep Learning (Fine-Tuned ResNet-18)\nTest Accuracy: 82.0% (164 / 200 Correct) | Macro F1: 0.8200",
        fontsize=12, fontweight="bold", pad=12
    )
    axes[1].set_xlabel("Predicted Artist", fontsize=10.5, fontweight="semibold", labelpad=8)
    axes[1].set_ylabel("True Artist", fontsize=10.5, fontweight="semibold", labelpad=8)
    axes[1].tick_params(axis="x", rotation=40, labelsize=9.5)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9.5)

    fig.suptitle(
        "Side-by-Side Test Confusion Matrices: Classical SVM vs. Fine-Tuned ResNet-18\n(Held-Out Test Split, N=200, 20 Paintings / Artist)",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    out_path = DOCS_IMG / "test-confusion-matrices.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def generate_per_artist_f1_comparison(test_eval, model_comp):
    """Figure 3: Per-artist F1 comparison across all 3 models for all 10 artist classes."""
    class_labels = test_eval["class_labels"]
    clean_labels = [c.replace("_", " ") for c in class_labels]

    f1_svm = [model_comp["per_class_test_results"][c]["svm"]["f1_score"] for c in class_labels]
    f1_base_cnn = [test_eval["per_artist_comparison"][c]["baseline"]["f1_score"] for c in class_labels]
    f1_ft_cnn = [test_eval["per_artist_comparison"][c]["finetuned"]["f1_score"] for c in class_labels]

    x = np.arange(len(class_labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=300)

    bars1 = ax.bar(x - width, f1_svm, width, label="SVM Baseline (Macro F1: 0.5206)",
                   color="#94a3b8", edgecolor="#334155", linewidth=0.9, alpha=0.95)
    bars2 = ax.bar(x, f1_base_cnn, width, label="ResNet-18 Frozen Head (Macro F1: 0.6988)",
                   color="#475569", edgecolor="#1e293b", linewidth=0.9, alpha=0.95)
    bars3 = ax.bar(x + width, f1_ft_cnn, width, label="Fine-Tuned ResNet-18 (Macro F1: 0.8200)",
                   color="#0f766e", edgecolor="#042f2e", linewidth=1.1, alpha=0.95)

    ax.set_title("Per-Artist Held-Out Test F1-Score Comparison (20 Paintings per Artist)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Authoritative Artist Class", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_ylabel("F1-Score (Test Set)", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=30, ha="right", fontsize=10.5, fontweight="medium")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)

    # Annotate values above bars
    for bars, c_color in [(bars1, "#475569"), (bars2, "#1e293b"), (bars3, "#042f2e")]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=7.8, fontweight="bold", color=c_color
            )

    plt.tight_layout()
    out_path = DOCS_IMG / "per-artist-f1-comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def generate_training_validation_curves(val_metrics):
    """Figure 4: Genuine epoch-wise training and validation curves across 10 fine-tuning epochs."""
    history = val_metrics.get("training_history", val_metrics.get("history", {}))
    epochs = list(range(1, val_metrics["epochs_trained"] + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = [acc * 100 for acc in history["train_acc"]]
    val_acc = [acc * 100 for acc in history["val_acc"]]
    best_epoch = val_metrics["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Loss Curves
    axes[0].plot(epochs, train_loss, marker="o", color="#0284c7", linewidth=2.0, label="Training Loss")
    axes[0].plot(epochs, val_loss, marker="s", color="#dc2626", linewidth=2.0, linestyle="--", label="Validation Loss")
    axes[0].axvline(best_epoch, color="#059669", linestyle=":", linewidth=1.5,
                    label=f"Best Checkpoint (Epoch {best_epoch})")
    axes[0].set_title("A. Cross-Entropy Loss Across Fine-Tuning Epochs", fontsize=12.5, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("Loss", fontsize=11, fontweight="semibold")
    axes[0].set_xticks(epochs)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.7)
    axes[0].legend(fontsize=10, loc="upper right")

    # Panel B: Accuracy Curves
    axes[1].plot(epochs, train_acc, marker="o", color="#0284c7", linewidth=2.0, label="Training Accuracy")
    axes[1].plot(epochs, val_acc, marker="s", color="#059669", linewidth=2.0, linestyle="--", label="Validation Accuracy")
    axes[1].axvline(best_epoch, color="#059669", linestyle=":", linewidth=1.5,
                    label=f"Best Val Accuracy ({val_metrics['validation_accuracy']*100:.1f}%)")
    axes[1].set_title("B. Classification Accuracy (%) Across Fine-Tuning Epochs", fontsize=12.5, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    axes[1].set_ylabel("Accuracy (%)", fontsize=11, fontweight="semibold")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(70, 102)
    axes[1].grid(True, alpha=0.7)
    axes[1].legend(fontsize=10, loc="lower right")

    fig.suptitle("ResNet-18 Layer 4 + Linear Head Fine-Tuning Dynamics (10 Epochs, AdamW, Differential LR)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = DOCS_IMG / "training-validation-curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("=== Generating Documentation Figures for README.md ===")
    test_eval, model_comp, val_metrics = load_verified_data()

    generate_model_performance_comparison(test_eval, model_comp)
    generate_test_confusion_matrices(test_eval, model_comp)
    generate_per_artist_f1_comparison(test_eval, model_comp)
    generate_training_validation_curves(val_metrics)

    print("=== All 4 Figures Successfully Generated in docs/images/ ===")


if __name__ == "__main__":
    main()
