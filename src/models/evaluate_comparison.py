"""
Comparative Evaluation and Benchmarking Suite for Artist Identification.

Evaluates both frozen models:
1. Classical Machine Learning: SVM Baseline (Feature Extraction + Scaler + PCA + RBF SVM)
2. Deep Learning: Fine-tuned ResNet-18 Transfer Learning

Strict Protocol:
- Evaluates exclusively on the 200 held-out test paintings from data/processed/dataset_manifest.csv
- Both models predict on the identical 200 test paintings using the exact same class label order
- Zero retraining, fine-tuning, or parameter updates occur
- End-to-end and model-only inference latencies measured with warm-up cycles
- Generates side-by-side confusion matrices and per-artist F1 comparison plots in outputs/
- Generates model_comparison.json and per_class_metrics_comparison.csv
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Torch imports
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

from src.features.extract_features import extract_image_features
from src.utils.evaluate import compute_classification_metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_test_records(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["split"] == "test"]


def benchmark_svm_test(
    records: List[Dict[str, str]],
    workspace_root: Path,
    model_pipeline: Any,
    class_labels: List[str]
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Run inference on test set with SVM pipeline and benchmark latency.
    """
    y_true = [r["artist"] for r in records]
    y_pred = []

    # Warm-up on first 5 samples
    for r in records[:5]:
        p = workspace_root / r["rel_path"]
        f = extract_image_features(p)
        _ = model_pipeline.predict(f.reshape(1, -1))

    end_to_end_latencies = []
    feature_extraction_latencies = []
    classifier_latencies = []

    for r in records:
        img_p = workspace_root / r["rel_path"]

        t0 = time.perf_counter()
        feat = extract_image_features(img_p)
        t1 = time.perf_counter()

        feat_2d = feat.reshape(1, -1)
        pred_label = model_pipeline.predict(feat_2d)[0]
        t2 = time.perf_counter()

        y_pred.append(str(pred_label))
        feature_extraction_latencies.append((t1 - t0) * 1000)
        classifier_latencies.append((t2 - t1) * 1000)
        end_to_end_latencies.append((t2 - t0) * 1000)

    timing_summary = {
        "mean_end_to_end_ms": float(np.mean(end_to_end_latencies)),
        "median_end_to_end_ms": float(np.median(end_to_end_latencies)),
        "p95_end_to_end_ms": float(np.percentile(end_to_end_latencies, 95)),
        "mean_feature_extraction_ms": float(np.mean(feature_extraction_latencies)),
        "mean_model_only_ms": float(np.mean(classifier_latencies))
    }

    return y_true, y_pred, timing_summary


def benchmark_cnn_test(
    records: List[Dict[str, str]],
    workspace_root: Path,
    model: torch.nn.Module,
    idx_to_label: Dict[int, str]
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Run inference on test set with PyTorch ResNet-18 model and benchmark latency.
    """
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # Warm-up on first 5 samples
    for r in records[:5]:
        p = workspace_root / r["rel_path"]
        with Image.open(p) as img:
            t = val_tf(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(t)

    y_true = [r["artist"] for r in records]
    y_pred = []

    end_to_end_latencies = []
    image_preprocessing_latencies = []
    forward_pass_latencies = []

    with torch.no_grad():
        for r in records:
            img_p = workspace_root / r["rel_path"]

            t0 = time.perf_counter()
            with Image.open(img_p) as img:
                tensor = val_tf(img.convert("RGB")).unsqueeze(0).to(device)
            t1 = time.perf_counter()

            logits = model(tensor)
            _, pred_idx = torch.max(logits, 1)
            t2 = time.perf_counter()

            pred_label = idx_to_label[pred_idx.item()]
            y_pred.append(pred_label)

            image_preprocessing_latencies.append((t1 - t0) * 1000)
            forward_pass_latencies.append((t2 - t1) * 1000)
            end_to_end_latencies.append((t2 - t0) * 1000)

    timing_summary = {
        "mean_end_to_end_ms": float(np.mean(end_to_end_latencies)),
        "median_end_to_end_ms": float(np.median(end_to_end_latencies)),
        "p95_end_to_end_ms": float(np.percentile(end_to_end_latencies, 95)),
        "mean_preprocessing_ms": float(np.mean(image_preprocessing_latencies)),
        "mean_model_only_ms": float(np.mean(forward_pass_latencies))
    }

    return y_true, y_pred, timing_summary


def compute_extended_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_labels: List[str]
) -> Dict[str, Any]:
    """Compute base and weighted classification metrics."""
    base = compute_classification_metrics(y_true, y_pred, class_labels)
    base["precision_weighted"] = float(precision_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0))
    base["recall_weighted"] = float(recall_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0))
    base["f1_weighted"] = float(f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0))

    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total_count = len(y_true)
    base["correct_count"] = correct_count
    base["incorrect_count"] = total_count - correct_count
    base["error_rate"] = float((total_count - correct_count) / total_count)

    return base


def plot_side_by_side_confusion_matrices(
    cm_svm: np.ndarray,
    cm_cnn: np.ndarray,
    class_labels: List[str],
    acc_svm: float,
    acc_cnn: float,
    output_path: Path
):
    """Render side-by-side normalized test confusion matrices."""
    clean_labels = [c.replace('_', ' ') for c in class_labels]

    # Normalize row-wise (recall)
    cm_svm_norm = np.nan_to_num(cm_svm.astype(float) / cm_svm.sum(axis=1, keepdims=True), nan=0.0)
    cm_cnn_norm = np.nan_to_num(cm_cnn.astype(float) / cm_cnn.sum(axis=1, keepdims=True), nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # SVM Heatmap
    sns.heatmap(
        cm_svm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[0],
        cbar_kws={"label": "Normalized Recall"}
    )
    axes[0].set_title(f"A. Classical ML (SVM Baseline)\nTest Accuracy: {acc_svm * 100:.1f}%", fontsize=13, pad=12)
    axes[0].set_xlabel("Predicted Artist", fontsize=11, labelpad=8)
    axes[0].set_ylabel("True Artist", fontsize=11, labelpad=8)
    axes[0].tick_params(axis='x', rotation=45)

    # CNN Heatmap
    sns.heatmap(
        cm_cnn_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[1],
        cbar_kws={"label": "Normalized Recall"}
    )
    axes[1].set_title(f"B. Deep Learning (ResNet-18 Transfer Learning)\nTest Accuracy: {acc_cnn * 100:.1f}%", fontsize=13, pad=12)
    axes[1].set_xlabel("Predicted Artist", fontsize=11, labelpad=8)
    axes[1].set_ylabel("True Artist", fontsize=11, labelpad=8)
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle("Side-by-Side Test Confusion Matrices: Classical SVM vs. Deep Learning CNN", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_artist_f1_comparison(
    per_class_svm: Dict[str, Dict[str, Any]],
    per_class_cnn: Dict[str, Dict[str, Any]],
    class_labels: List[str],
    output_path: Path
):
    """Bar chart comparing per-artist F1 scores."""
    clean_labels = [c.replace('_', ' ') for c in class_labels]
    f1_svm = [per_class_svm[c]["f1_score"] for c in class_labels]
    f1_cnn = [per_class_cnn[c]["f1_score"] for c in class_labels]

    x = np.arange(len(class_labels))
    width = 0.38

    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(x - width/2, f1_svm, width, label="SVM Baseline", color="#4a7ebb", edgecolor="black", alpha=0.9)
    bars2 = plt.bar(x + width/2, f1_cnn, width, label="CNN (ResNet-18)", color="#2ca02c", edgecolor="black", alpha=0.9)

    plt.title("Per-Artist F1-Score Comparison on Held-Out Test Set (20 Test Images / Class)", fontsize=14, pad=12)
    plt.xlabel("Artist Class", fontsize=12, labelpad=10)
    plt.ylabel("F1-Score", fontsize=12, labelpad=10)
    plt.xticks(x, clean_labels, rotation=35, ha="right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle=":", alpha=0.7)
    plt.legend(fontsize=11, loc="upper right")

    # Add numeric annotations on top of bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print("=== Phase 5: Final SVM vs. CNN Comparative Evaluation ===")
    print(f"Project Workspace: {PROJECT_ROOT}\n")

    manifest_path = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
    svm_path = PROJECT_ROOT / "models" / "svm_baseline.joblib"
    cnn_path = PROJECT_ROOT / "models" / "cnn_best.pth"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test records
    test_records = load_test_records(manifest_path)
    print(f"[1/5] Loaded {len(test_records)} held-out test paintings (20 per artist).")

    # 2. Evaluate SVM Baseline
    print("\n[2/5] Evaluating SVM Baseline on Test Set...")
    svm_bundle = joblib.load(svm_path)
    svm_model = svm_bundle["model_pipeline"]
    class_labels = [str(c) for c in svm_bundle["class_labels"]]

    y_true_svm, y_pred_svm, svm_timing = benchmark_svm_test(
        test_records, PROJECT_ROOT, svm_model, class_labels
    )
    svm_metrics = compute_extended_metrics(y_true_svm, y_pred_svm, class_labels)
    svm_metrics["timing"] = svm_timing

    print(f"      SVM Test Accuracy:      {svm_metrics['accuracy'] * 100:.2f}% ({svm_metrics['correct_count']}/200)")
    print(f"      SVM Macro F1:          {svm_metrics['f1_macro']:.4f}")
    print(f"      SVM Macro Precision:   {svm_metrics['precision_macro']:.4f}")
    print(f"      SVM Macro Recall:      {svm_metrics['recall_macro']:.4f}")
    print(f"      SVM Mean Latency:      {svm_timing['mean_end_to_end_ms']:.2f} ms "
          f"({svm_timing['mean_feature_extraction_ms']:.2f} ms feat + {svm_timing['mean_model_only_ms']:.2f} ms model)")

    # 3. Evaluate CNN Transfer Learning
    print("\n[3/5] Evaluating CNN Transfer Learning on Test Set...")
    cnn_ckpt = torch.load(cnn_path, map_location="cpu")
    cnn_model = resnet18()
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 10)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    idx_to_label = cnn_ckpt["idx_to_label"]

    y_true_cnn, y_pred_cnn, cnn_timing = benchmark_cnn_test(
        test_records, PROJECT_ROOT, cnn_model, idx_to_label
    )
    cnn_metrics = compute_extended_metrics(y_true_cnn, y_pred_cnn, class_labels)
    cnn_metrics["timing"] = cnn_timing

    print(f"      CNN Test Accuracy:      {cnn_metrics['accuracy'] * 100:.2f}% ({cnn_metrics['correct_count']}/200)")
    print(f"      CNN Macro F1:          {cnn_metrics['f1_macro']:.4f}")
    print(f"      CNN Macro Precision:   {cnn_metrics['precision_macro']:.4f}")
    print(f"      CNN Macro Recall:      {cnn_metrics['recall_macro']:.4f}")
    print(f"      CNN Mean Latency:      {cnn_timing['mean_end_to_end_ms']:.2f} ms "
          f"({cnn_timing['mean_preprocessing_ms']:.2f} ms prep + {cnn_timing['mean_model_only_ms']:.2f} ms forward)")

    # 4. Generate Comparative Visualizations
    print("\n[4/5] Generating Comparative Plots...")
    cm_plot_path = outputs_dir / "svm_vs_cnn_confusion_matrices.png"
    plot_side_by_side_confusion_matrices(
        cm_svm=np.array(svm_metrics["confusion_matrix"]),
        cm_cnn=np.array(cnn_metrics["confusion_matrix"]),
        class_labels=class_labels,
        acc_svm=svm_metrics["accuracy"],
        acc_cnn=cnn_metrics["accuracy"],
        output_path=cm_plot_path
    )
    print(f"      Saved: {cm_plot_path}")

    f1_plot_path = outputs_dir / "per_artist_f1_comparison.png"
    plot_per_artist_f1_comparison(
        per_class_svm=svm_metrics["per_class"],
        per_class_cnn=cnn_metrics["per_class"],
        class_labels=class_labels,
        output_path=f1_plot_path
    )
    print(f"      Saved: {f1_plot_path}")

    # 5. Export Summary JSON and CSV
    print("\n[5/5] Exporting Comparison Data and Metrics Tables...")
    # Load validation metrics for comprehensive contrast
    with open(outputs_dir / "svm_val_metrics.json", "r", encoding="utf-8") as f:
        svm_val = json.load(f)
    with open(outputs_dir / "cnn_val_metrics.json", "r", encoding="utf-8") as f:
        cnn_val = json.load(f)

    comparison_manifest = {
        "dataset": "ikarus777/best-artworks-of-all-time (10 Artists)",
        "total_test_samples": len(test_records),
        "test_samples_per_artist": 20,
        "class_labels": class_labels,
        "summary": {
            "svm_baseline": {
                "paradigm": "Traditional ML (HSV Hist + Hu + GLCM -> PCA -> RBF SVM)",
                "val_accuracy": svm_val["validation_accuracy"],
                "val_macro_f1": svm_val["validation_macro_f1"],
                "test_accuracy": svm_metrics["accuracy"],
                "test_macro_f1": svm_metrics["f1_macro"],
                "test_macro_precision": svm_metrics["precision_macro"],
                "test_macro_recall": svm_metrics["recall_macro"],
                "test_weighted_f1": svm_metrics["f1_weighted"],
                "test_correct": svm_metrics["correct_count"],
                "test_incorrect": svm_metrics["incorrect_count"],
                "mean_latency_ms": svm_timing["mean_end_to_end_ms"],
                "median_latency_ms": svm_timing["median_end_to_end_ms"]
            },
            "cnn_transfer_learning": {
                "paradigm": "Deep Learning (ResNet-18 ImageNet Transfer Learning)",
                "val_accuracy": cnn_val["validation_accuracy"],
                "val_macro_f1": cnn_val["validation_macro_f1"],
                "test_accuracy": cnn_metrics["accuracy"],
                "test_macro_f1": cnn_metrics["f1_macro"],
                "test_macro_precision": cnn_metrics["precision_macro"],
                "test_macro_recall": cnn_metrics["recall_macro"],
                "test_weighted_f1": cnn_metrics["f1_weighted"],
                "test_correct": cnn_metrics["correct_count"],
                "test_incorrect": cnn_metrics["incorrect_count"],
                "mean_latency_ms": cnn_timing["mean_end_to_end_ms"],
                "median_latency_ms": cnn_timing["median_end_to_end_ms"]
            },
            "relative_performance": {
                "accuracy_diff_percent": round((cnn_metrics["accuracy"] - svm_metrics["accuracy"]) * 100, 2),
                "macro_f1_diff": round(cnn_metrics["f1_macro"] - svm_metrics["f1_macro"], 4),
                "winner": "CNN Transfer Learning" if cnn_metrics["accuracy"] > svm_metrics["accuracy"] else "SVM Baseline"
            }
        },
        "per_class_test_results": {
            c: {
                "svm": svm_metrics["per_class"][c],
                "cnn": cnn_metrics["per_class"][c]
            }
            for c in class_labels
        }
    }

    comparison_json_path = outputs_dir / "model_comparison.json"
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_manifest, f, indent=2)
    print(f"      Saved: {comparison_json_path}")

    # Export per-class CSV
    comparison_csv_path = outputs_dir / "per_class_metrics_comparison.csv"
    with open(comparison_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "artist", "support",
            "svm_precision", "svm_recall", "svm_f1",
            "cnn_precision", "cnn_recall", "cnn_f1",
            "f1_advantage"
        ])
        for c in class_labels:
            s_f1 = svm_metrics["per_class"][c]["f1_score"]
            c_f1 = cnn_metrics["per_class"][c]["f1_score"]
            writer.writerow([
                c,
                svm_metrics["per_class"][c]["support"],
                round(svm_metrics["per_class"][c]["precision"], 4),
                round(svm_metrics["per_class"][c]["recall"], 4),
                round(s_f1, 4),
                round(cnn_metrics["per_class"][c]["precision"], 4),
                round(cnn_metrics["per_class"][c]["recall"], 4),
                round(c_f1, 4),
                "CNN" if c_f1 > s_f1 else ("SVM" if s_f1 > c_f1 else "Tie")
            ])
    print(f"      Saved: {comparison_csv_path}")

    print("\n=== Comparative Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
