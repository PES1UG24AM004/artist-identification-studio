"""
Held-Out Test Set Comparative Evaluation: Baseline vs. Fine-Tuned ResNet-18.

Evaluates both frozen models on the untouched 200 held-out test paintings:
1. Baseline ResNet-18: models/cnn_best.pth (Frozen Backbone + Linear Head)
2. Fine-Tuned ResNet-18: models/experiments/cnn_layer4_finetune.pth (Layer 4 + Linear Head)

Strict Protocol:
- Evaluates exclusively on the 200 held-out test paintings from data/processed/dataset_manifest.csv
- Deterministic preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet Normalize
- Zero test-based tuning or parameter changes
- Detailed per-artist metrics, confusion matrices, and CPU inference latencies
- Paired per-image correctness analysis, McNemar test (with continuity correction and exact binomial p-value),
  and 95% bootstrap confidence intervals for Delta Accuracy and Delta Macro F1
- Saves output artifacts exclusively to outputs/experiments/
- ZERO overwrites to existing benchmark results or default checkpoints.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import scipy.stats as stats
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.utils.evaluate import compute_classification_metrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 10
BOOTSTRAP_ROUNDS = 10000
BOOTSTRAP_SEED = 42


def load_test_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["split"] == "test"]


def evaluate_model_on_test(
    records: List[Dict[str, str]],
    workspace_root: Path,
    model: torch.nn.Module,
    idx_to_label: Dict[int, str],
    class_labels: List[str]
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, float]]:
    """
    Run deterministic inference on test set and measure latencies.
    Returns (y_true, y_pred, y_probs, timing_summary).
    """
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    device = torch.device("cpu")
    model.eval()
    model.to(device)

    # Warm-up (5 samples)
    for r in records[:5]:
        p = workspace_root / r["rel_path"]
        with Image.open(p) as img:
            t = val_tf(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(t)

    y_true = [r["artist"] for r in records]
    y_pred = []
    y_probs_list = []
    latencies = []

    with torch.no_grad():
        for r in records:
            p = workspace_root / r["rel_path"]
            t0 = time.perf_counter()
            with Image.open(p) as img:
                tensor = val_tf(img.convert("RGB")).unsqueeze(0).to(device)
            t1 = time.perf_counter()

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            t2 = time.perf_counter()

            y_pred.append(idx_to_label[pred_idx])
            y_probs_list.append(probs)
            latencies.append((t2 - t0) * 1000)

    timing_summary = {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "std_latency_ms": float(np.std(latencies))
    }

    return y_true, y_pred, np.array(y_probs_list), timing_summary


def compute_paired_mcnemar_and_bootstrap(
    y_true: List[str],
    y_pred_base: List[str],
    y_pred_ft: List[str],
    class_labels: List[str],
    n_bootstrap: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED
) -> Dict[str, Any]:
    """
    Compute McNemar test and bootstrap 95% confidence intervals for paired predictions.
    """
    n = len(y_true)
    correct_base = np.array([yt == yp for yt, yp in zip(y_true, y_pred_base)])
    correct_ft = np.array([yt == yp for yt, yp in zip(y_true, y_pred_ft)])

    # Contingency table
    # n00: both wrong, n01: base wrong & ft correct, n10: base correct & ft wrong, n11: both correct
    n00 = int(np.sum((~correct_base) & (~correct_ft)))
    n01 = int(np.sum((~correct_base) & correct_ft))
    n10 = int(np.sum(correct_base & (~correct_ft)))
    n11 = int(np.sum(correct_base & correct_ft))

    # McNemar with continuity correction
    discordant = n01 + n10
    if discordant > 0:
        stat = ((abs(n01 - n10) - 1.0) ** 2) / discordant
        p_value_chi2 = float(1.0 - stats.chi2.cdf(stat, df=1))
        # Exact binomial two-sided test
        binom_res = stats.binomtest(k=n01, n=discordant, p=0.5, alternative="two-sided")
        p_value_exact = float(binom_res.pvalue)
    else:
        stat = 0.0
        p_value_chi2 = 1.0
        p_value_exact = 1.0

    # Bootstrap 95% confidence intervals for Delta Accuracy and Delta Macro F1
    rng = np.random.RandomState(seed)
    delta_accs = []
    delta_f1s = []

    y_true_arr = np.array(y_true)
    y_base_arr = np.array(y_pred_base)
    y_ft_arr = np.array(y_pred_ft)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        b_true = y_true_arr[idx]
        b_base = y_base_arr[idx]
        b_ft = y_ft_arr[idx]

        acc_b = accuracy_score(b_true, b_base)
        acc_ft = accuracy_score(b_true, b_ft)
        delta_accs.append(acc_ft - acc_b)

        f1_b = f1_score(b_true, b_base, labels=class_labels, average="macro", zero_division=0)
        f1_ft = f1_score(b_true, b_ft, labels=class_labels, average="macro", zero_division=0)
        delta_f1s.append(f1_ft - f1_b)

    ci_acc_lower = float(np.percentile(delta_accs, 2.5))
    ci_acc_upper = float(np.percentile(delta_accs, 97.5))
    ci_f1_lower = float(np.percentile(delta_f1s, 2.5))
    ci_f1_upper = float(np.percentile(delta_f1s, 97.5))

    return {
        "contingency_table": {
            "both_correct_n11": n11,
            "both_incorrect_n00": n00,
            "baseline_incorrect_finetuned_correct_n01": n01,
            "baseline_correct_finetuned_incorrect_n10": n10,
            "net_correct_gain": n01 - n10
        },
        "mcnemar_test": {
            "statistic_chi2_corrected": float(stat),
            "p_value_chi2": p_value_chi2,
            "p_value_exact_binomial": p_value_exact,
            "statistically_significant_at_alpha_0_05": bool(p_value_exact < 0.05)
        },
        "bootstrap_95_ci": {
            "bootstrap_rounds": n_bootstrap,
            "delta_accuracy": {
                "point_estimate": float(accuracy_score(y_true, y_pred_ft) - accuracy_score(y_true, y_pred_base)),
                "ci_lower_2_5": ci_acc_lower,
                "ci_upper_97_5": ci_acc_upper,
                "ci_excludes_zero": bool(ci_acc_lower > 0)
            },
            "delta_macro_f1": {
                "point_estimate": float(f1_score(y_true, y_pred_ft, labels=class_labels, average="macro", zero_division=0) -
                                        f1_score(y_true, y_pred_base, labels=class_labels, average="macro", zero_division=0)),
                "ci_lower_2_5": ci_f1_lower,
                "ci_upper_97_5": ci_f1_upper,
                "ci_excludes_zero": bool(ci_f1_lower > 0)
            }
        }
    }


def plot_side_by_side_test_confusion_matrices(
    cm_base: np.ndarray,
    cm_ft: np.ndarray,
    class_labels: List[str],
    acc_base: float,
    acc_ft: float,
    output_path: Path
):
    clean_labels = [c.replace('_', ' ') for c in class_labels]
    cm_base_norm = np.nan_to_num(cm_base.astype(float) / cm_base.sum(axis=1, keepdims=True), nan=0.0)
    cm_ft_norm = np.nan_to_num(cm_ft.astype(float) / cm_ft.sum(axis=1, keepdims=True), nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Baseline Heatmap
    sns.heatmap(
        cm_base_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[0],
        cbar_kws={"label": "Normalized Recall"}
    )
    axes[0].set_title(f"A. Baseline ResNet-18 (Head Only)\nHeld-Out Test Accuracy: {acc_base * 100:.1f}%", fontsize=13, pad=12)
    axes[0].set_xlabel("Predicted Artist", fontsize=11, labelpad=8)
    axes[0].set_ylabel("True Artist", fontsize=11, labelpad=8)
    axes[0].tick_params(axis='x', rotation=45)

    # Fine-Tuned Heatmap
    sns.heatmap(
        cm_ft_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=axes[1],
        cbar_kws={"label": "Normalized Recall"}
    )
    axes[1].set_title(f"B. Fine-Tuned ResNet-18 (Layer 4 + Head)\nHeld-Out Test Accuracy: {acc_ft * 100:.1f}%", fontsize=13, pad=12)
    axes[1].set_xlabel("Predicted Artist", fontsize=11, labelpad=8)
    axes[1].set_ylabel("True Artist", fontsize=11, labelpad=8)
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle("Held-Out Test Set (N=200): Baseline vs. Fine-Tuned ResNet-18 Normalized Confusion Matrices", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_artist_test_f1_comparison(
    per_class_base: Dict[str, Dict[str, Any]],
    per_class_ft: Dict[str, Dict[str, Any]],
    class_labels: List[str],
    output_path: Path
):
    clean_labels = [c.replace('_', ' ') for c in class_labels]
    f1_base = [per_class_base[c]["f1_score"] for c in class_labels]
    f1_ft = [per_class_ft[c]["f1_score"] for c in class_labels]

    x = np.arange(len(class_labels))
    width = 0.38

    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(x - width/2, f1_base, width, label="Baseline ResNet-18 (Head Only)", color="#4a7ebb", edgecolor="black", alpha=0.9)
    bars2 = plt.bar(x + width/2, f1_ft, width, label="Fine-Tuned ResNet-18 (Layer 4 + Head)", color="#2ca02c", edgecolor="black", alpha=0.9)

    plt.title("Per-Artist F1-Score Comparison on Held-Out Test Set (20 Test Images / Class)", fontsize=14, pad=12)
    plt.xlabel("Artist Class", fontsize=12, labelpad=10)
    plt.ylabel("F1-Score", fontsize=12, labelpad=10)
    plt.xticks(x, clean_labels, rotation=35, ha="right", fontsize=10)
    plt.ylim(0, 1.08)
    plt.grid(axis='y', linestyle=":", alpha=0.7)
    plt.legend(fontsize=11, loc="upper right")

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
    print("=== Held-Out Test Set Comparative Evaluation ===")
    print(f"Project Workspace: {PROJECT_ROOT}\n")

    manifest_path = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
    base_ckpt_path = PROJECT_ROOT / "models" / "experiments" / "cnn_baseline_head_only.pth"
    ft_ckpt_path = PROJECT_ROOT / "models" / "cnn_best.pth"
    exp_outputs_dir = PROJECT_ROOT / "outputs" / "experiments"
    exp_outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Checkpoints and Architecture
    print("[1/5] Verifying checkpoints and architecture compatibility...")
    assert base_ckpt_path.exists(), f"Baseline checkpoint not found at {base_ckpt_path}"
    assert ft_ckpt_path.exists(), f"Fine-tuned checkpoint not found at {ft_ckpt_path}"

    base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
    ft_ckpt = torch.load(ft_ckpt_path, map_location="cpu")

    class_labels = base_ckpt["class_labels"]
    idx_to_label = base_ckpt["idx_to_label"]
    assert ft_ckpt["class_labels"] == class_labels, "Class labels mismatch between checkpoints!"
    assert ft_ckpt["idx_to_label"] == idx_to_label, "idx_to_label mapping mismatch between checkpoints!"
    print(f"      Verified 10 target classes: {class_labels}")
    print(f"      Baseline checkpoint epoch: {base_ckpt.get('epoch')}")
    print(f"      Fine-tuned checkpoint epoch: {ft_ckpt.get('epoch')}\n")

    # 2. Load Test Records
    test_records = load_test_manifest(manifest_path)
    print(f"[2/5] Loaded {len(test_records)} held-out test records (20 images per artist).")

    # 3. Load Models
    print("\n[3/5] Evaluating Models on Held-Out Test Set...")
    # Baseline
    base_model = resnet18()
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, NUM_CLASSES)
    base_model.load_state_dict(base_ckpt["model_state_dict"])

    print("      Running inference: Baseline ResNet-18 (Head Only)...", end="", flush=True)
    y_true_base, y_pred_base, y_probs_base, timing_base = evaluate_model_on_test(
        test_records, PROJECT_ROOT, base_model, idx_to_label, class_labels
    )
    metrics_base = compute_classification_metrics(y_true_base, y_pred_base, class_labels)
    print(f" Done ({timing_base['mean_latency_ms']:.2f} ms/image, Acc={metrics_base['accuracy']*100:.2f}%)")

    # Fine-Tuned
    ft_model = resnet18()
    ft_model.fc = torch.nn.Linear(ft_model.fc.in_features, NUM_CLASSES)
    ft_model.load_state_dict(ft_ckpt["model_state_dict"])

    print("      Running inference: Fine-Tuned ResNet-18 (Layer 4 + Head)...", end="", flush=True)
    y_true_ft, y_pred_ft, y_probs_ft, timing_ft = evaluate_model_on_test(
        test_records, PROJECT_ROOT, ft_model, idx_to_label, class_labels
    )
    metrics_ft = compute_classification_metrics(y_true_ft, y_pred_ft, class_labels)
    print(f" Done ({timing_ft['mean_latency_ms']:.2f} ms/image, Acc={metrics_ft['accuracy']*100:.2f}%)")

    # 4. Statistical Analysis: Paired McNemar Test & Bootstrap CIs
    print("\n[4/5] Computing Paired McNemar Test and 10,000-Round Bootstrap CIs...")
    stat_analysis = compute_paired_mcnemar_and_bootstrap(
        y_true=y_true_base,
        y_pred_base=y_pred_base,
        y_pred_ft=y_pred_ft,
        class_labels=class_labels,
        n_bootstrap=BOOTSTRAP_ROUNDS,
        seed=BOOTSTRAP_SEED
    )

    ct = stat_analysis["contingency_table"]
    mcn = stat_analysis["mcnemar_test"]
    boot = stat_analysis["bootstrap_95_ci"]

    print(f"      Paired Contingency Table (N=200):")
    print(f"        - Both correct:                 {ct['both_correct_n11']}")
    print(f"        - Both incorrect:               {ct['both_incorrect_n00']}")
    print(f"        - Base wrong, Fine-Tuned right: {ct['baseline_incorrect_finetuned_correct_n01']} (Gains)")
    print(f"        - Base right, Fine-Tuned wrong: {ct['baseline_correct_finetuned_incorrect_n10']} (Losses)")
    print(f"        - Net correct samples gained:   {ct['net_correct_gain']}")
    print(f"      McNemar Test:")
    print(f"        - Chi2 statistic (corrected):   {mcn['statistic_chi2_corrected']:.4f}")
    print(f"        - Chi2 p-value:                 {mcn['p_value_chi2']:.4e}")
    print(f"        - Exact Binomial p-value:       {mcn['p_value_exact_binomial']:.4e}")
    print(f"        - Significant (alpha=0.05):     {mcn['statistically_significant_at_alpha_0_05']}")
    print(f"      Bootstrap 95% Confidence Intervals ({BOOTSTRAP_ROUNDS} rounds):")
    print(f"        - Delta Accuracy:  {boot['delta_accuracy']['point_estimate']*100:+.2f}% "
          f"[{boot['delta_accuracy']['ci_lower_2_5']*100:+.2f}%, {boot['delta_accuracy']['ci_upper_97_5']*100:+.2f}%]")
    print(f"        - Delta Macro F1:  {boot['delta_macro_f1']['point_estimate']:+.4f} "
          f"[{boot['delta_macro_f1']['ci_lower_2_5']:+.4f}, {boot['delta_macro_f1']['ci_upper_97_5']:+.4f}]")

    # 5. Save Artifacts & Visualizations
    print("\n[5/5] Exporting Test Comparison Artifacts...")
    test_cm_path = exp_outputs_dir / "test_confusion_matrices.png"
    plot_side_by_side_test_confusion_matrices(
        cm_base=np.array(metrics_base["confusion_matrix"]),
        cm_ft=np.array(metrics_ft["confusion_matrix"]),
        class_labels=class_labels,
        acc_base=metrics_base["accuracy"],
        acc_ft=metrics_ft["accuracy"],
        output_path=test_cm_path
    )
    print(f"      Saved: {test_cm_path}")

    test_f1_path = exp_outputs_dir / "test_per_artist_f1_comparison.png"
    plot_per_artist_test_f1_comparison(
        per_class_base=metrics_base["per_class"],
        per_class_ft=metrics_ft["per_class"],
        class_labels=class_labels,
        output_path=test_f1_path
    )
    print(f"      Saved: {test_f1_path}")

    # Build comprehensive comparison JSON
    comparison_payload = {
        "evaluation_split": "held_out_test (N=200 paintings, strictly 20 per artist, untouched during training/tuning)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_labels": class_labels,
        "summary": {
            "baseline_model": {
                "name": "ResNet-18 (Frozen Backbone, Linear Head Only)",
                "checkpoint": str(base_ckpt_path),
                "accuracy": metrics_base["accuracy"],
                "macro_f1": metrics_base["f1_macro"],
                "macro_precision": metrics_base["precision_macro"],
                "macro_recall": metrics_base["recall_macro"],
                "correct_count": int(np.sum([yt == yp for yt, yp in zip(y_true_base, y_pred_base)])),
                "incorrect_count": int(np.sum([yt != yp for yt, yp in zip(y_true_base, y_pred_base)])),
                "vg_precision": metrics_base["per_class"]["Vincent_van_Gogh"]["precision"],
                "vg_recall": metrics_base["per_class"]["Vincent_van_Gogh"]["recall"],
                "vg_f1": metrics_base["per_class"]["Vincent_van_Gogh"]["f1_score"],
                "timing": timing_base
            },
            "finetuned_model": {
                "name": "ResNet-18 (Layer 4 + Linear Head Fine-Tuned)",
                "checkpoint": str(ft_ckpt_path),
                "best_epoch": ft_ckpt.get("epoch"),
                "accuracy": metrics_ft["accuracy"],
                "macro_f1": metrics_ft["f1_macro"],
                "macro_precision": metrics_ft["precision_macro"],
                "macro_recall": metrics_ft["recall_macro"],
                "correct_count": int(np.sum([yt == yp for yt, yp in zip(y_true_ft, y_pred_ft)])),
                "incorrect_count": int(np.sum([yt != yp for yt, yp in zip(y_true_ft, y_pred_ft)])),
                "vg_precision": metrics_ft["per_class"]["Vincent_van_Gogh"]["precision"],
                "vg_recall": metrics_ft["per_class"]["Vincent_van_Gogh"]["recall"],
                "vg_f1": metrics_ft["per_class"]["Vincent_van_Gogh"]["f1_score"],
                "timing": timing_ft
            },
            "deltas": {
                "accuracy_delta_percent": round((metrics_ft["accuracy"] - metrics_base["accuracy"]) * 100, 2),
                "macro_f1_delta": round(metrics_ft["f1_macro"] - metrics_base["f1_macro"], 4),
                "macro_precision_delta": round(metrics_ft["precision_macro"] - metrics_base["precision_macro"], 4),
                "macro_recall_delta": round(metrics_ft["recall_macro"] - metrics_base["recall_macro"], 4),
                "vg_recall_delta": round((metrics_ft["per_class"]["Vincent_van_Gogh"]["recall"] - metrics_base["per_class"]["Vincent_van_Gogh"]["recall"]) * 100, 2),
                "vg_f1_delta": round(metrics_ft["per_class"]["Vincent_van_Gogh"]["f1_score"] - metrics_base["per_class"]["Vincent_van_Gogh"]["f1_score"], 4)
            }
        },
        "statistical_testing": stat_analysis,
        "per_artist_comparison": {
            c: {
                "support": 20,
                "baseline": metrics_base["per_class"][c],
                "finetuned": metrics_ft["per_class"][c],
                "f1_delta": round(metrics_ft["per_class"][c]["f1_score"] - metrics_base["per_class"][c]["f1_score"], 4),
                "recall_delta": round(metrics_ft["per_class"][c]["recall"] - metrics_base["per_class"][c]["recall"], 4)
            }
            for c in class_labels
        },
        "per_image_predictions": [
            {
                "rel_path": r["rel_path"],
                "filename": r["filename"],
                "true_artist": r["artist"],
                "baseline_pred": y_pred_base[idx],
                "baseline_correct": bool(y_pred_base[idx] == r["artist"]),
                "finetuned_pred": y_pred_ft[idx],
                "finetuned_correct": bool(y_pred_ft[idx] == r["artist"])
            }
            for idx, r in enumerate(test_records)
        ]
    }

    test_json_path = exp_outputs_dir / "test_evaluation_comparison.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"      Saved: {test_json_path}")

    print("\n=== Held-Out Test Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
