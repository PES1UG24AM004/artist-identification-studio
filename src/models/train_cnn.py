"""
PyTorch Transfer Learning Pipeline for Artist Identification.

Trains a ResNet-18 classifier using transfer learning:
- Pretrained weights: ResNet-18 ImageNet-1K (official torchvision weights)
- Local weights cache: .cache/torch (strictly inside authorized workspace)
- Input image resolution: 224x224
- Training set: 1,600 images across 10 balanced classes (160/artist)
- Validation set: 200 images across 10 balanced classes (20/artist)
- Test set: 200 images (STRICTLY UNTOUCHED / PRESERVED)
- Training schedule: 5 initial epochs, batch size 8, AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- Augmentations: RandomResizedCrop, RandomHorizontalFlip, ColorJitter on training data only
- Validation transforms: Deterministic Resize(256) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalize
- Model selection: Tracks best validation accuracy and saves checkpoint to models/cnn_best.pth
- Evaluation: Saves confusion matrix plot and metrics JSON to outputs/
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure torch cache inside project workspace
TORCH_CACHE = PROJECT_ROOT / ".cache" / "torch"
TORCH_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from src.utils.evaluate import compute_classification_metrics, plot_and_save_confusion_matrix


DEFAULT_SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 8
DEFAULT_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = DEFAULT_SEED):
    """Ensure reproducibility across random, numpy, and torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PaintingDataset(Dataset):
    """
    Reads painting images dynamically from disk using relative paths from the manifest.
    Avoids loading all decoded images into memory simultaneously.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        workspace_root: Path,
        label_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None
    ):
        self.records = records
        self.workspace_root = Path(workspace_root)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        image_path = self.workspace_root / record["rel_path"]
        artist_name = record["artist"]
        label = self.label_to_idx[artist_name]

        with Image.open(image_path) as pil_img:
            img = pil_img.convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transforms, val_transforms).
    Train: random crop, flip, and jitter for regularized learning.
    Val: deterministic resize, center crop, and normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def build_artist_resnet18(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True
) -> nn.Module:
    """
    Instantiate ResNet-18 with ImageNet-1K pretrained weights (or uninitialized for tests)
    and customized 10-class head. Freezes all feature extractor parameters by default.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear layer with newly initialized 10-class classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Ensure classification head is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


class CNNTrainer:
    """Manages data loading, training loop, validation tracking, and artifact saving."""

    def __init__(
        self,
        workspace_root: Path,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        seed: int = DEFAULT_SEED
    ):
        self.workspace_root = Path(workspace_root)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.manifest_path = self.workspace_root / "data" / "processed" / "dataset_manifest.csv"
        self.models_dir = self.workspace_root / "models"
        self.outputs_dir = self.workspace_root / "outputs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        """Read training and validation records from dataset_manifest.csv."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        train_records = [r for r in reader if r["split"] == "train"]
        val_records = [r for r in reader if r["split"] == "val"]
        test_records = [r for r in reader if r["split"] == "test"]

        # Ensure consistent class label sorting
        class_labels = sorted(list(set(r["artist"] for r in train_records)))
        return train_records, val_records, class_labels

    def run(self) -> Dict[str, Any]:
        set_seed(self.seed)

        print("=== Artist Identification: PyTorch CNN Transfer Learning Pipeline ===")
        print(f"Workspace:       {self.workspace_root}")
        print(f"Confirmed Device: {self.device}")
        print(f"PyTorch Version:  {torch.__version__}")
        print(f"Model Backbone:   ResNet-18 (ImageNet-1K pretrained, frozen backbone)")
        print(f"Hyperparameters:  Epochs={self.epochs}, Batch Size={self.batch_size}, LR={self.lr}, Seed={self.seed}\n")

        # 1. Prepare Datasets & DataLoaders
        train_records, val_records, class_labels = self.load_records()
        label_to_idx = {c: i for i, c in enumerate(class_labels)}
        idx_to_label = {i: c for i, c in enumerate(class_labels)}

        print(f"[1/5] Loaded dataset records:")
        print(f"      Training samples:   {len(train_records)}")
        print(f"      Validation samples: {len(val_records)}")
        print(f"      Test samples:       200 (STRICTLY PRESERVED / UNTOUCHED)")
        print(f"      Artist Classes (10): {class_labels}\n")

        train_tf, val_tf = get_transforms()
        train_dataset = PaintingDataset(train_records, self.workspace_root, label_to_idx, train_tf)
        val_dataset = PaintingDataset(val_records, self.workspace_root, label_to_idx, val_tf)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # 2. Build Model, Loss, Optimizer
        print("[2/5] Building ResNet-18 model with custom 10-class head...")
        model = build_artist_resnet18(num_classes=NUM_CLASSES, freeze_backbone=True)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)

        # 3. Training Loop with Validation Checkpointing
        print(f"[3/5] Starting Training Loop ({self.epochs} epochs)...")
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_duration_seconds": []
        }

        best_val_acc = 0.0
        best_checkpoint_path = self.models_dir / "cnn_best.pth"
        total_train_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += torch.sum(preds == labels.data).item()
                total_train += labels.size(0)

            epoch_train_loss = running_loss / total_train
            epoch_train_acc = correct_train / total_train

            # --- Validation Phase ---
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_val += torch.sum(preds == labels.data).item()
                    total_val += labels.size(0)

            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = correct_val / total_val
            epoch_time = time.time() - epoch_start

            history["epoch"].append(epoch)
            history["train_loss"].append(float(epoch_train_loss))
            history["train_acc"].append(float(epoch_train_acc))
            history["val_loss"].append(float(epoch_val_loss))
            history["val_acc"].append(float(epoch_val_acc))
            history["epoch_duration_seconds"].append(float(epoch_time))

            print(f"      Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s): "
                  f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc*100:.2f}% | "
                  f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc*100:.2f}%")

            # Save best checkpoint
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": epoch_val_acc,
                    "val_loss": epoch_val_loss,
                    "class_labels": class_labels,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "architecture": "ResNet-18 (Transfer Learning)",
                    "seed": self.seed
                }, best_checkpoint_path)

        total_training_duration = time.time() - total_train_start
        print(f"\n      Total Training Finished in {total_training_duration:.1f}s")
        print(f"      Best Validation Accuracy Achieved: {best_val_acc*100:.2f}%\n")

        # 4. Final Evaluation of Best Checkpoint
        print("[4/5] Evaluating Best Checkpoint on Validation Set (200 images)...")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        all_preds = []
        all_targets = []
        inf_latencies = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                t_b_start = time.time()
                outputs = model(images)
                t_b_end = time.time()

                inf_latencies.append((t_b_end - t_b_start) / images.size(0))
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())

        y_val_str = [idx_to_label[i] for i in all_targets]
        y_pred_str = [idx_to_label[i] for i in all_preds]

        avg_inference_latency_ms = float(np.mean(inf_latencies) * 1000)
        metrics = compute_classification_metrics(y_val_str, y_pred_str, class_labels)

        print(f"      Validation Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"      Validation Macro F1:    {metrics['f1_macro']:.4f}")
        print(f"      Validation Macro Prec:  {metrics['precision_macro']:.4f}")
        print(f"      Validation Macro Rec:   {metrics['recall_macro']:.4f}")
        print(f"      Avg Inference Latency:  {avg_inference_latency_ms:.2f} ms/image\n")

        # 5. Save Metrics, Learning Curves, and Confusion Matrix
        print("[5/5] Saving Evaluation Plots and Metric Logs...")
        metrics_json_path = self.outputs_dir / "cnn_val_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "ResNet18_TransferLearning",
                "backbone": "resnet18 (ImageNet-1K pretrained)",
                "device": str(self.device),
                "epochs_trained": self.epochs,
                "best_epoch": checkpoint["epoch"],
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "random_seed": self.seed,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["f1_macro"],
                "validation_macro_precision": metrics["precision_macro"],
                "validation_macro_recall": metrics["recall_macro"],
                "per_class_metrics": metrics["per_class"],
                "training_history": history,
                "timing": {
                    "total_training_duration_seconds": float(total_training_duration),
                    "avg_epoch_duration_seconds": float(np.mean(history["epoch_duration_seconds"])),
                    "avg_inference_latency_ms": float(avg_inference_latency_ms)
                }
            }, f, indent=2)

        # Plot Confusion Matrix
        cm_plot_path = self.outputs_dir / "cnn_confusion_matrix.png"
        plot_and_save_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_labels=class_labels,
            output_path=cm_plot_path,
            title=f"CNN Transfer Learning (ResNet-18) — Validation Confusion Matrix\nAccuracy: {metrics['accuracy']*100:.1f}%"
        )

        # Plot Learning Curves
        curves_plot_path = self.outputs_dir / "cnn_learning_curves.png"
        self._plot_learning_curves(history, curves_plot_path)

        print(f"      Saved best checkpoint:       {best_checkpoint_path}")
        print(f"      Saved confusion matrix plot: {cm_plot_path}")
        print(f"      Saved learning curves:       {curves_plot_path}")
        print(f"      Saved validation metrics:    {metrics_json_path}")
        print("\n=== CNN Training and Evaluation Completed Successfully ===")

        return {
            "status": "success",
            "best_checkpoint": str(best_checkpoint_path),
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "history": history,
            "metrics_path": str(metrics_json_path)
        }

    def _plot_learning_curves(self, history: Dict[str, List[float]], output_path: Path):
        """Render and save Loss and Accuracy learning curves over epochs."""
        import matplotlib.pyplot as plt
        epochs = history["epoch"]

        plt.figure(figsize=(12, 5))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="royalblue")
        plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="darkorange")
        plt.title("Cross-Entropy Loss vs. Epochs", fontsize=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="forestgreen")
        plt.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="crimson")
        plt.title("Classification Accuracy vs. Epochs", fontsize=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    trainer = CNNTrainer(workspace_root=PROJECT_ROOT)
    trainer.run()


if __name__ == "__main__":
    main()
