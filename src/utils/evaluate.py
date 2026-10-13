"""
Evaluation and visualization utilities for artist identification models.

Computes:
- Overall accuracy, macro precision, recall, and macro F1-score
- Detailed per-class precision, recall, and F1 metrics
- Confusion matrix array and normalized confusion matrix
- Formats formatted Markdown and text summary tables
- Renders and saves publication-quality confusion matrix plots using Seaborn
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics across target classes.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))

    # Per-class metrics
    prec_per = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)

    per_class = {}
    for idx, c in enumerate(class_labels):
        per_class[c] = {
            "precision": float(prec_per[idx]),
            "recall": float(rec_per[idx]),
            "f1_score": float(f1_per[idx]),
            "support": int(sum(1 for y in y_true if y == c))
        }

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "classification_report_text": classification_report(y_true, y_pred, labels=class_labels, zero_division=0)
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> Path:
    """
    Render and save confusion matrix heatmap to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        # Normalize row-wise (true label)
        with np.errstate(all='ignore'):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot, nan=0.0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    # Clean display labels (replace underscore with space)
    clean_labels = [c.replace('_', ' ') for c in class_labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Normalized Recall" if normalize else "Painting Count"}
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Artist", fontsize=12, labelpad=10)
    plt.ylabel("True Artist", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path
