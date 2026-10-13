"""
Fine-Tuning ResNet-18 (Layer 4 + Linear Head) for Artist Identification.

Phase 2 Experiment:
- Starts directly from existing production checkpoint: models/cnn_best.pth
- Preserves layer1, layer2, layer3 frozen (frozen feature extractors)
- Unfreezes layer4 and classification head (fc)
- Differential learning rates:
    - layer4: 1e-4
    - fc: 1e-3
- AdamW optimizer with weight decay 1e-4
- CosineAnnealingLR scheduler
- Conservative training augmentations (RandomResizedCrop(0.85-1.0), Flip, mild ColorJitter)
- Deterministic validation transforms (Resize(256), CenterCrop(224), ImageNet Normalize)
- Early stopping with patience 4-5 epochs
- Saves best checkpoint strictly to: models/experiments/cnn_layer4_finetune.pth
- Saves metrics and comparison plots strictly to: outputs/experiments/
- STRICT ISOLATION: models/cnn_best.pth and test split remain completely untouched.
"""

import os
import sys
import time
import json
import csv
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Torch cache
TORCH_CACHE = PROJECT_ROOT / ".cache" / "torch"
TORCH_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

from src.utils.evaluate import compute_classification_metrics, plot_and_save_confusion_matrix


DEFAULT_SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4

# Differential Learning Rates
LR_LAYER4 = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = DEFAULT_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PaintingDataset(Dataset):
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


def get_finetune_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Train: Conservative crops and mild variations to preserve artwork composition.
    Val: Deterministic Resize(256) -> CenterCrop(224) -> ImageNet Normalize.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
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


def build_finetuned_resnet18(
    base_checkpoint_path: Path,
    num_classes: int = NUM_CLASSES
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads model state from existing checkpoint and configures:
    - conv1, bn1, layer1, layer2, layer3: FROZEN (requires_grad = False)
    - layer4: UNFROZEN (requires_grad = True)
    - fc: UNFROZEN (requires_grad = True)
    """
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {base_checkpoint_path}")

    checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze fc head
    for param in model.fc.parameters():
        param.requires_grad = True

    return model, checkpoint


class CNNFineTuner:
    """Manages Layer 4 fine-tuning, validation tracking, early stopping, and artifact saving."""

    def __init__(
        self,
        workspace_root: Path,
        max_epochs: int = MAX_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr_layer4: float = LR_LAYER4,
        lr_head: float = LR_HEAD,
        weight_decay: float = WEIGHT_DECAY,
        patience: int = EARLY_STOPPING_PATIENCE,
        seed: int = DEFAULT_SEED
    ):
        self.workspace_root = Path(workspace_root)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr_layer4 = lr_layer4
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

        self.device = torch.device("cpu")
        self.manifest_path = self.workspace_root / "data" / "processed" / "dataset_manifest.csv"
        self.baseline_checkpoint_path = self.workspace_root / "models" / "experiments" / "cnn_baseline_head_only.pth"

        # Dedicated experiment paths
        self.exp_models_dir = self.workspace_root / "models" / "experiments"
        self.exp_outputs_dir = self.workspace_root / "outputs" / "experiments"
        self.exp_models_dir.mkdir(parents=True, exist_ok=True)
        self.exp_outputs_dir.mkdir(parents=True, exist_ok=True)

        self.target_checkpoint_path = self.exp_models_dir / "cnn_layer4_finetune.pth"

    def load_records(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        """Read train and val records from dataset_manifest.csv. Test split strictly untouched."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        train_records = [r for r in reader if r["split"] == "train"]
        val_records = [r for r in reader if r["split"] == "val"]
        class_labels = sorted(list(set(r["artist"] for r in train_records)))
        return train_records, val_records, class_labels

    def run(self) -> Dict[str, Any]:
        set_seed(self.seed)

        print("=== Artist Identification: ResNet-18 Layer 4 Fine-Tuning Pipeline ===")
        print(f"Workspace:               {self.workspace_root}")
        print(f"Baseline Checkpoint:     {self.baseline_checkpoint_path}")
        print(f"Target Exp Checkpoint:   {self.target_checkpoint_path}")
        print(f"Confirmed Device:        {self.device}")
        print(f"Unfrozen Components:     layer4 (lr={self.lr_layer4}), fc (lr={self.lr_head})")
        print(f"Frozen Components:       conv1, bn1, layer1, layer2, layer3")
        print(f"Optimizer:               AdamW (weight_decay={self.weight_decay})")
        print(f"Scheduler:               CosineAnnealingLR (T_max={self.max_epochs}, eta_min=1e-6)")
        print(f"Schedule:                Max Epochs={self.max_epochs}, Batch Size={self.batch_size}, Patience={self.patience}, Seed={self.seed}\n")

        # 1. Prepare Datasets & DataLoaders
        train_records, val_records, class_labels = self.load_records()
        label_to_idx = {c: i for i, c in enumerate(class_labels)}
        idx_to_label = {i: c for i, c in enumerate(class_labels)}

        print(f"[1/5] Loaded dataset records:")
        print(f"      Training samples:   {len(train_records)}")
        print(f"      Validation samples: {len(val_records)}")
        print(f"      Test samples:       STRICTLY PRESERVED / UNTOUCHED")
        print(f"      Artist Classes (10): {class_labels}\n")

        train_tf, val_tf = get_finetune_transforms()
        train_dataset = PaintingDataset(train_records, self.workspace_root, label_to_idx, train_tf)
        val_dataset = PaintingDataset(val_records, self.workspace_root, label_to_idx, val_tf)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # 2. Build Model from Existing Checkpoint
        print("[2/5] Loading baseline checkpoint and unfreezing layer4 + classification head...")
        model, base_ckpt = build_finetuned_resnet18(self.baseline_checkpoint_path, num_classes=NUM_CLASSES)
        model.to(self.device)

        # Audit trainable parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        layer4_params = sum(p.numel() for p in model.layer4.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)

        print(f"      Total parameters:     {trainable_params + frozen_params:,}")
        print(f"      Frozen parameters:    {frozen_params:,} (conv1, bn1, layer1-3)")
        print(f"      Trainable parameters: {trainable_params:,} ({layer4_params:,} in layer4, {fc_params:,} in fc)\n")

        # 3. Setup Differential Optimizer, Scheduler, Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {"params": model.layer4.parameters(), "lr": self.lr_layer4, "weight_decay": self.weight_decay},
            {"params": model.fc.parameters(), "lr": self.lr_head, "weight_decay": self.weight_decay}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        # 4. Training Loop with Validation Tracking and Early Stopping
        print(f"[3/5] Starting Fine-Tuning Loop (up to {self.max_epochs} epochs)...")
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "val_vg_recall": [],
            "val_vg_f1": [],
            "epoch_duration_seconds": [],
            "lr_layer4": [],
            "lr_head": []
        }

        best_val_score = 0.0  # Track combination of accuracy and macro F1
        best_epoch = 0
        best_val_acc = 0.0
        best_val_macro_f1 = 0.0
        best_model_state = None
        patience_counter = 0

        total_train_start = time.time()

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            curr_lr_layer4 = optimizer.param_groups[0]["lr"]
            curr_lr_head = optimizer.param_groups[1]["lr"]

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

            scheduler.step()

            epoch_train_loss = running_loss / total_train
            epoch_train_acc = correct_train / total_train

            # --- Validation Phase ---
            model.eval()
            running_val_loss = 0.0
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)

                    val_preds_list.extend(preds.cpu().numpy())
                    val_targets_list.extend(labels.cpu().numpy())

            epoch_val_loss = running_val_loss / len(val_targets_list)
            y_val_str = [idx_to_label[i] for i in val_targets_list]
            y_pred_str = [idx_to_label[i] for i in val_preds_list]

            val_metrics = compute_classification_metrics(y_val_str, y_pred_str, class_labels)
            epoch_val_acc = val_metrics["accuracy"]
            epoch_val_f1 = val_metrics["f1_macro"]
            vg_metrics = val_metrics["per_class"]["Vincent_van_Gogh"]

            epoch_time = time.time() - epoch_start

            history["epoch"].append(epoch)
            history["train_loss"].append(float(epoch_train_loss))
            history["train_acc"].append(float(epoch_train_acc))
            history["val_loss"].append(float(epoch_val_loss))
            history["val_acc"].append(float(epoch_val_acc))
            history["val_macro_f1"].append(float(epoch_val_f1))
            history["val_vg_recall"].append(float(vg_metrics["recall"]))
            history["val_vg_f1"].append(float(vg_metrics["f1_score"]))
            history["epoch_duration_seconds"].append(float(epoch_time))
            history["lr_layer4"].append(float(curr_lr_layer4))
            history["lr_head"].append(float(curr_lr_head))

            print(f"      Epoch {epoch:02d}/{self.max_epochs} ({epoch_time:.1f}s): "
                  f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc*100:.2f}% | "
                  f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc*100:.2f}%, Val F1={epoch_val_f1:.4f} | "
                  f"VG Recall={vg_metrics['recall']*100:.1f}%, VG F1={vg_metrics['f1_score']:.3f}")

            # Tracking selection score (validation accuracy + macro F1)
            selection_score = epoch_val_acc + epoch_val_f1

            if selection_score > best_val_score:
                best_val_score = selection_score
                best_val_acc = epoch_val_acc
                best_val_macro_f1 = epoch_val_f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0

                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": epoch_val_acc,
                    "val_loss": epoch_val_loss,
                    "val_macro_f1": epoch_val_f1,
                    "class_labels": class_labels,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "architecture": "ResNet-18 (Layer 4 + Head Fine-Tuned)",
                    "seed": self.seed,
                    "lr_layer4": self.lr_layer4,
                    "lr_head": self.lr_head,
                    "base_checkpoint": str(self.baseline_checkpoint_path)
                }, self.target_checkpoint_path)
                print(f"        -> [*] New Best Validation Score ({selection_score:.4f})! Saved checkpoint.")
            else:
                patience_counter += 1
                print(f"        -> [ ] Patience counter: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\n      Early stopping triggered at Epoch {epoch} (No improvement for {self.patience} epochs).")
                    break

        total_training_duration = time.time() - total_train_start
        print(f"\n      Fine-Tuning Finished in {total_training_duration:.1f}s")
        print(f"      Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc*100:.2f}% | Best Val Macro F1: {best_val_macro_f1:.4f}\n")

        # 5. Final Comprehensive Validation Evaluation of Best Checkpoint
        print("[4/5] Evaluating Best Fine-Tuned Checkpoint on Validation Set (200 images)...")
        best_ckpt = torch.load(self.target_checkpoint_path, map_location=self.device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        y_val_str = [idx_to_label[i] for i in all_targets]
        y_pred_str = [idx_to_label[i] for i in all_preds]

        finetune_metrics = compute_classification_metrics(y_val_str, y_pred_str, class_labels)

        # 6. Load Baseline Metrics for Rigorous Side-by-Side Comparison
        print("[5/5] Generating Comparative Validation Reports and Visualizations...")
        base_model = resnet18()
        base_model.fc = nn.Linear(base_model.fc.in_features, NUM_CLASSES)
        base_model.load_state_dict(base_ckpt["model_state_dict"])
        base_model.eval()

        base_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = base_model(images)
                _, preds = torch.max(outputs, 1)
                base_preds.extend(preds.cpu().numpy())

        y_base_pred_str = [idx_to_label[i] for i in base_preds]
        baseline_metrics = compute_classification_metrics(y_val_str, y_base_pred_str, class_labels)

        # Save Metrics JSON
        metrics_json_path = self.exp_outputs_dir / "cnn_finetune_val_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "ResNet18_Layer4_FineTuning",
                "base_checkpoint": str(self.baseline_checkpoint_path),
                "target_checkpoint": str(self.target_checkpoint_path),
                "device": str(self.device),
                "best_epoch": best_epoch,
                "epochs_trained": len(history["epoch"]),
                "batch_size": self.batch_size,
                "lr_layer4": self.lr_layer4,
                "lr_head": self.lr_head,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "validation_accuracy": finetune_metrics["accuracy"],
                "validation_macro_f1": finetune_metrics["f1_macro"],
                "validation_macro_precision": finetune_metrics["precision_macro"],
                "validation_macro_recall": finetune_metrics["recall_macro"],
                "per_class_metrics": finetune_metrics["per_class"],
                "training_history": history,
                "total_duration_seconds": float(total_training_duration)
            }, f, indent=2)

        # Save Side-by-Side Comparison JSON
        comparison_manifest_path = self.exp_outputs_dir / "finetune_vs_baseline_comparison.json"
        comp_data = {
            "evaluation_split": "validation (N=200 paintings, strictly disjoint from training)",
            "summary": {
                "baseline_cnn": {
                    "architecture": "ResNet-18 (Frozen Backbone, FC Head Only)",
                    "val_accuracy": baseline_metrics["accuracy"],
                    "val_macro_f1": baseline_metrics["f1_macro"],
                    "val_macro_precision": baseline_metrics["precision_macro"],
                    "val_macro_recall": baseline_metrics["recall_macro"],
                    "vg_precision": baseline_metrics["per_class"]["Vincent_van_Gogh"]["precision"],
                    "vg_recall": baseline_metrics["per_class"]["Vincent_van_Gogh"]["recall"],
                    "vg_f1": baseline_metrics["per_class"]["Vincent_van_Gogh"]["f1_score"]
                },
                "finetuned_cnn": {
                    "architecture": "ResNet-18 (Layer 4 + FC Head Fine-Tuned)",
                    "best_epoch": best_epoch,
                    "val_accuracy": finetune_metrics["accuracy"],
                    "val_macro_f1": finetune_metrics["f1_macro"],
                    "val_macro_precision": finetune_metrics["precision_macro"],
                    "val_macro_recall": finetune_metrics["recall_macro"],
                    "vg_precision": finetune_metrics["per_class"]["Vincent_van_Gogh"]["precision"],
                    "vg_recall": finetune_metrics["per_class"]["Vincent_van_Gogh"]["recall"],
                    "vg_f1": finetune_metrics["per_class"]["Vincent_van_Gogh"]["f1_score"]
                },
                "delta": {
                    "accuracy_delta_percent": round((finetune_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100, 2),
                    "macro_f1_delta": round(finetune_metrics["f1_macro"] - baseline_metrics["f1_macro"], 4),
                    "vg_recall_delta": round((finetune_metrics["per_class"]["Vincent_van_Gogh"]["recall"] - baseline_metrics["per_class"]["Vincent_van_Gogh"]["recall"]) * 100, 2),
                    "vg_f1_delta": round(finetune_metrics["per_class"]["Vincent_van_Gogh"]["f1_score"] - baseline_metrics["per_class"]["Vincent_van_Gogh"]["f1_score"], 4)
                }
            },
            "per_class_comparison": {
                c: {
                    "support": 20,
                    "baseline": baseline_metrics["per_class"][c],
                    "finetuned": finetune_metrics["per_class"][c],
                    "f1_delta": round(finetune_metrics["per_class"][c]["f1_score"] - baseline_metrics["per_class"][c]["f1_score"], 4)
                }
                for c in class_labels
            }
        }
        with open(comparison_manifest_path, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=2)

        # Save Confusion Matrix Heatmap
        cm_plot_path = self.exp_outputs_dir / "cnn_finetune_confusion_matrix.png"
        plot_and_save_confusion_matrix(
            cm=np.array(finetune_metrics["confusion_matrix"]),
            class_labels=class_labels,
            output_path=cm_plot_path,
            title=f"Fine-Tuned ResNet-18 (Layer 4 + Head) — Validation Confusion Matrix\nAccuracy: {finetune_metrics['accuracy']*100:.1f}%, Macro F1: {finetune_metrics['f1_macro']:.4f}"
        )

        # Plot Learning Curves
        curves_plot_path = self.exp_outputs_dir / "cnn_finetune_learning_curves.png"
        self._plot_learning_curves(history, curves_plot_path)

        print(f"      Saved fine-tuned checkpoint: {self.target_checkpoint_path}")
        print(f"      Saved validation metrics:    {metrics_json_path}")
        print(f"      Saved comparison manifest:   {comparison_manifest_path}")
        print(f"      Saved confusion matrix plot: {cm_plot_path}")
        print(f"      Saved learning curves:       {curves_plot_path}")
        print("\n=== ResNet-18 Layer 4 Fine-Tuning Completed Successfully ===")

        return {
            "status": "success",
            "best_checkpoint": str(self.target_checkpoint_path),
            "best_epoch": best_epoch,
            "finetune_metrics": finetune_metrics,
            "baseline_metrics": baseline_metrics,
            "delta": comp_data["summary"]["delta"],
            "comp_data": comp_data
        }

    def _plot_learning_curves(self, history: Dict[str, List[float]], output_path: Path):
        import matplotlib.pyplot as plt
        epochs = history["epoch"]

        plt.figure(figsize=(14, 5))

        # Loss subplot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="royalblue")
        plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="darkorange")
        plt.title("Cross-Entropy Loss vs. Epochs", fontsize=11)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=9)

        # Accuracy subplot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="forestgreen")
        plt.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="crimson")
        plt.title("Classification Accuracy vs. Epochs", fontsize=11)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Accuracy (%)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=9)

        # Van Gogh Tracking subplot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [r * 100 for r in history["val_vg_recall"]], "o-", label="Van Gogh Recall (%)", color="purple")
        plt.plot(epochs, [f * 100 for f in history["val_vg_f1"]], "s--", label="Van Gogh F1 (%)", color="teal")
        plt.title("Van Gogh Validation Recall & F1", fontsize=11)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Percentage (%)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    trainer = CNNFineTuner(workspace_root=PROJECT_ROOT)
    trainer.run()


if __name__ == "__main__":
    main()
