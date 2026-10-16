"""
Validation Experiments for Post-Training Model Improvements.

Evaluates candidate accuracy and confidence improvements on the established
validation split (200 images, 20 per artist) WITHOUT retraining or modifying model weights:
1. Baseline: Single-crop Resize(256) -> CenterCrop(224)
2. TTA (Test-Time Augmentation): Horizontal Flip averaging
3. Aspect-Ratio-Preserving Preprocessing: Letterbox / Pad to square before resize
4. Preprocessing + TTA combinations
5. Temperature Scaling (Calibration) analysis on validation logits
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from scipy.optimize import minimize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_pad_image(img: Image.Image, target_size: int = 224, fill_color: Tuple[int, int, int] = (124, 116, 104)) -> Image.Image:
    """
    Resize image to fit within target_size x target_size while preserving aspect ratio,
    padding the remaining space with neutral fill (ImageNet mean in 0-255 RGB: ~124, 116, 104).
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    padded = Image.new("RGB", (target_size, target_size), fill_color)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))
    return padded


def get_transforms():
    standard_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    flip_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return standard_tf, flip_tf, to_tensor_norm


def evaluate_candidate(
    candidate_name: str,
    predict_fn,
    val_records: List[Dict[str, str]],
    class_labels: List[str]
) -> Dict[str, Any]:
    y_true = [r["artist"] for r in val_records]
    y_pred = []
    latencies = []
    all_logits = []
    all_probs = []

    # Warm-up (5 samples)
    for r in val_records[:5]:
        with Image.open(r["rel_path"]) as img:
            _ = predict_fn(img.convert("RGB"))

    for r in val_records:
        with Image.open(r["rel_path"]) as img:
            rgb_img = img.convert("RGB")
            t0 = time.perf_counter()
            pred_label, logits, probs = predict_fn(rgb_img)
            t1 = time.perf_counter()

        y_pred.append(pred_label)
        latencies.append((t1 - t0) * 1000)
        all_logits.append(logits)
        all_probs.append(probs)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels).tolist()

    per_class = {}
    precisions = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)

    for i, label in enumerate(class_labels):
        per_class[label] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1_score": float(f1s[i]),
            "support": sum(1 for yt in y_true if yt == label)
        }

    return {
        "candidate": candidate_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "correct_count": sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp),
        "total_count": len(y_true),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "per_class": per_class,
        "confusion_matrix": cm,
        "logits": np.array(all_logits),
        "probs": np.array(all_probs),
        "y_true": y_true,
        "y_pred": y_pred
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return float(ece)


def main():
    print("=== Evaluating Accuracy & Preprocessing Improvements on Validation Split ===")
    manifest_p = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
    ckpt_p = PROJECT_ROOT / "models" / "cnn_best.pth"

    with open(manifest_p, "r", encoding="utf-8") as f:
        val_records = [r for r in csv.DictReader(f) if r["split"] == "val"]

    ckpt = torch.load(ckpt_p, map_location="cpu")
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_to_label = ckpt["idx_to_label"]
    label_to_idx = ckpt["label_to_idx"]
    class_labels = ckpt["class_labels"]
    y_true_indices = np.array([label_to_idx[r["artist"]] for r in val_records])

    standard_tf, flip_tf, to_tensor_norm = get_transforms()

    # Define prediction functions
    # 1. Baseline: Standard Resize 256 + CenterCrop 224
    def predict_baseline(img: Image.Image):
        with torch.no_grad():
            t = standard_tf(img).unsqueeze(0)
            logits = model(t)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            return idx_to_label[pred_idx], logits.numpy()[0], probs

    # 2. Candidate A: Horizontal-Flip TTA (Average Probabilities)
    def predict_hflip_tta(img: Image.Image):
        with torch.no_grad():
            t_orig = standard_tf(img).unsqueeze(0)
            t_flip = flip_tf(img).unsqueeze(0)
            batch = torch.cat([t_orig, t_flip], dim=0)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).mean(dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            mean_logits = logits.mean(dim=0).numpy()
            return idx_to_label[pred_idx], mean_logits, probs

    # 3. Candidate B: Letterbox / Aspect-Ratio Padding
    def predict_letterbox(img: Image.Image):
        with torch.no_grad():
            padded = letterbox_pad_image(img, target_size=224)
            t = to_tensor_norm(padded).unsqueeze(0)
            logits = model(t)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            return idx_to_label[pred_idx], logits.numpy()[0], probs

    # 4. Candidate C: Letterbox + Horizontal-Flip TTA
    def predict_letterbox_tta(img: Image.Image):
        with torch.no_grad():
            padded = letterbox_pad_image(img, target_size=224)
            padded_flip = padded.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            t_orig = to_tensor_norm(padded).unsqueeze(0)
            t_flip = to_tensor_norm(padded_flip).unsqueeze(0)
            batch = torch.cat([t_orig, t_flip], dim=0)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).mean(dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            mean_logits = logits.mean(dim=0).numpy()
            return idx_to_label[pred_idx], mean_logits, probs

    candidates = [
        ("1. Original Baseline (Single Crop 224)", predict_baseline),
        ("2. TTA: Horizontal Flip Average", predict_hflip_tta),
        ("3. Preprocessing: Letterbox Aspect-Preserving", predict_letterbox),
        ("4. Combined: Letterbox + Horizontal Flip TTA", predict_letterbox_tta)
    ]

    results = []
    print(f"Loaded {len(val_records)} validation paintings (20 per artist). Starting evaluations...\n")

    for name, fn in candidates:
        print(f"--> Running: {name} ...")
        res = evaluate_candidate(name, fn, val_records, class_labels)
        ece = compute_ece(res["probs"], y_true_indices)
        res["ece"] = ece
        results.append(res)
        print(f"    Val Accuracy:  {res['accuracy']*100:.2f}% ({res['correct_count']}/200)")
        print(f"    Val Macro F1:  {res['macro_f1']:.4f}")
        print(f"    Mean Latency:  {res['mean_latency_ms']:.2f} ms")
        print(f"    ECE:           {ece:.4f}\n")

    # Temperature Scaling on Baseline Logits
    base_logits = results[0]["logits"]
    
    def nll_loss(T):
        scaled = base_logits / T[0]
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp / np.sum(exp, axis=1, keepdims=True)
        # NLL
        chosen_p = p[np.arange(len(y_true_indices)), y_true_indices]
        return -np.mean(np.log(np.clip(chosen_p, 1e-12, 1.0)))

    opt = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="Nelder-Mead")
    opt_T = float(opt.x[0])
    
    scaled_logits = base_logits / opt_T
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    calibrated_probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    calibrated_ece = compute_ece(calibrated_probs, y_true_indices)

    print("=== Temperature Scaling Calibration on Validation Logits ===")
    print(f"Optimal Temperature T: {opt_T:.4f}")
    print(f"Baseline ECE (Uncalibrated, T=1.0): {results[0]['ece']:.4f}")
    print(f"Calibrated ECE (T={opt_T:.4f}):       {calibrated_ece:.4f}")
    print(f"Top-1 Argmax Accuracy after T-scaling: {results[0]['accuracy']*100:.2f}% (strictly unchanged mathematically)")

    # Save summary report
    out_dir = PROJECT_ROOT / "outputs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = {
        "evaluation_split": "validation (N=200 paintings, 20 per artist)",
        "results": [
            {
                "candidate": r["candidate"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "correct_count": r["correct_count"],
                "mean_latency_ms": r["mean_latency_ms"],
                "ece": r["ece"],
                "per_class": r["per_class"]
            }
            for r in results
        ],
        "temperature_scaling": {
            "optimal_temperature": opt_T,
            "uncalibrated_ece": results[0]["ece"],
            "calibrated_ece": calibrated_ece
        }
    }
    
    with open(out_dir / "improvements_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved detailed validation comparison to {out_dir / 'improvements_validation_results.json'}")


if __name__ == "__main__":
    main()
