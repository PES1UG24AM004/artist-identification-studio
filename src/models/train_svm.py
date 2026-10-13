"""
SVM Baseline Training and Validation Pipeline for Artist Identification.

Trains a leakage-safe Machine Learning baseline:
1. Loads 1,600 training images from data/processed/dataset_manifest.csv
2. Extracts 559-dimensional visual features (HSV color hist + Hu moments + GLCM)
3. Fits feature scaling (StandardScaler) and PCA (100 components) strictly on training data
4. Runs stratified 3-fold cross-validation on the training set to optimize RBF SVM hyperparameters:
   - C in [0.1, 1.0, 10.0, 50.0]
   - gamma in ['scale', 'auto', 0.01, 0.001]
5. Evaluates the best estimator strictly on the 200 validation images
6. Generates metrics report and confusion matrix plot in outputs/
7. Serializes the fitted Pipeline and metadata to models/svm_baseline.joblib

STRICT ENFORCEMENT: The test split (200 images) remains 100% untouched.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.features.extract_features import extract_image_features, TOTAL_FEATURE_DIM, FEATURE_NAMES
from src.utils.evaluate import compute_classification_metrics, plot_and_save_confusion_matrix

DEFAULT_RANDOM_SEED = 42
TARGET_PCA_COMPONENTS = 100


def load_split_manifest(manifest_csv_path: Path, split_name: str) -> List[Dict[str, str]]:
    """Load image records matching a specific split from manifest."""
    with open(manifest_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["split"] == split_name]


def extract_features_for_records(
    records: List[Dict[str, str]],
    workspace_root: Path
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract features for a list of records.
    Returns (X_features, y_labels, duration_seconds).
    """
    X_list = []
    y_list = []

    t0 = time.time()
    for idx, r in enumerate(records):
        full_path = workspace_root / r["rel_path"]
        feat = extract_image_features(full_path)
        X_list.append(feat)
        y_list.append(r["artist"])

    duration = time.time() - t0
    return np.array(X_list, dtype=np.float32), np.array(y_list), duration


class SVMBaselineTrainer:
    """Orchestrates training, CV grid search, validation evaluation, and artifact saving."""

    def __init__(
        self,
        workspace_root: Path,
        seed: int = DEFAULT_RANDOM_SEED,
        pca_components: int = TARGET_PCA_COMPONENTS
    ):
        self.workspace_root = Path(workspace_root)
        self.seed = seed
        self.pca_components = pca_components

        self.manifest_path = self.workspace_root / "data" / "processed" / "dataset_manifest.csv"
        self.models_dir = self.workspace_root / "models"
        self.outputs_dir = self.workspace_root / "outputs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("=== Artist Identification: SVM Baseline Pipeline ===")
        print(f"Workspace:    {self.workspace_root}")
        print(f"Random Seed:  {self.seed}")
        print(f"Manifest:     {self.manifest_path}\n")

        # 1. Load Train and Validation Records
        train_records = load_split_manifest(self.manifest_path, "train")
        val_records = load_split_manifest(self.manifest_path, "val")
        test_records = load_split_manifest(self.manifest_path, "test")

        print(f"[1/6] Loaded split manifests:")
        print(f"      Training samples:   {len(train_records)}")
        print(f"      Validation samples: {len(val_records)}")
        print(f"      Test samples:       {len(test_records)} (STRICTLY PRESERVED / UNTOUCHED)\n")

        # 2. Extract Features
        print(f"[2/6] Extracting visual features ({TOTAL_FEATURE_DIM} dims per image)...")
        print("      Extracting training features (1,600 images)...", end="", flush=True)
        X_train, y_train, train_feat_time = extract_features_for_records(train_records, self.workspace_root)
        print(f" Done ({train_feat_time:.2f}s, {train_feat_time / len(train_records) * 1000:.2f} ms/image)")

        print("      Extracting validation features (200 images)...", end="", flush=True)
        X_val, y_val, val_feat_time = extract_features_for_records(val_records, self.workspace_root)
        print(f" Done ({val_feat_time:.2f}s, {val_feat_time / len(val_records) * 1000:.2f} ms/image)\n")

        class_labels = sorted(list(set(y_train)))

        # 3. Define Pipeline and Bounded Grid Search
        print(f"[3/6] Configuring Leakage-Safe Scikit-Learn Pipeline:")
        print(f"      - Step 1: StandardScaler()")
        print(f"      - Step 2: PCA(n_components={self.pca_components}, random_state={self.seed})")
        print(f"      - Step 3: SVC(kernel='rbf', probability=True, random_state={self.seed})\n")

        base_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=self.pca_components, random_state=self.seed)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=self.seed))
        ])

        param_grid = {
            "svm__C": [0.5, 1.0, 5.0, 10.0],
            "svm__gamma": ["scale", "auto", 0.01, 0.005]
        }
        total_candidates = len(param_grid["svm__C"]) * len(param_grid["svm__gamma"])
        cv_folds = 3
        print(f"      Parameter Grid: {param_grid}")
        print(f"      Total candidates: {total_candidates} x {cv_folds} folds = {total_candidates * cv_folds} fits")

        # 4. Execute 3-Fold Stratified Cross-Validation on Training Split Only
        print(f"\n[4/6] Executing 3-Fold Stratified Cross-Validation on Training Set...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        train_start = time.time()
        grid_search.fit(X_train, y_train)
        svm_train_time = time.time() - train_start

        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        print(f"\n      Grid Search Finished in {svm_train_time:.2f}s")
        print(f"      Best Cross-Validation Accuracy: {best_cv_score * 100:.2f}%")
        print(f"      Best Parameters: {best_params}\n")

        # 5. Evaluate on Validation Split
        print("[5/6] Evaluating Best SVM Pipeline on Validation Set (200 images)...")
        # Measure inference latency
        t_inf_start = time.time()
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)
        pure_inf_time = time.time() - t_inf_start

        # Latency including single-image feature extraction
        avg_feat_ms = (val_feat_time / len(val_records)) * 1000
        avg_inf_ms = (pure_inf_time / len(val_records)) * 1000
        total_single_image_latency_ms = avg_feat_ms + avg_inf_ms

        val_metrics = compute_classification_metrics(y_val, y_val_pred, class_labels)

        print(f"      Validation Accuracy:    {val_metrics['accuracy'] * 100:.2f}%")
        print(f"      Validation Macro F1:    {val_metrics['f1_macro']:.4f}")
        print(f"      Validation Macro Prec:  {val_metrics['precision_macro']:.4f}")
        print(f"      Validation Macro Rec:   {val_metrics['recall_macro']:.4f}")
        print(f"      Single Image Latency:   {total_single_image_latency_ms:.2f} ms "
              f"({avg_feat_ms:.2f} ms feature extraction + {avg_inf_ms:.2f} ms classifier)\n")

        # Save Metrics and Confusion Matrix Plot
        metrics_json_path = self.outputs_dir / "svm_val_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "SVM_Baseline_RBF",
                "random_seed": self.seed,
                "pca_components": self.pca_components,
                "feature_dimension": TOTAL_FEATURE_DIM,
                "best_hyperparameters": best_params,
                "cv_best_accuracy": float(best_cv_score),
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["f1_macro"],
                "validation_macro_precision": val_metrics["precision_macro"],
                "validation_macro_recall": val_metrics["recall_macro"],
                "per_class_metrics": val_metrics["per_class"],
                "timing": {
                    "train_feature_extraction_seconds": float(train_feat_time),
                    "val_feature_extraction_seconds": float(val_feat_time),
                    "svm_training_and_tuning_seconds": float(svm_train_time),
                    "avg_feature_extraction_ms": float(avg_feat_ms),
                    "avg_inference_ms": float(avg_inf_ms),
                    "total_single_image_latency_ms": float(total_single_image_latency_ms)
                }
            }, f, indent=2)

        cm_plot_path = self.outputs_dir / "svm_confusion_matrix.png"
        plot_and_save_confusion_matrix(
            cm=np.array(val_metrics["confusion_matrix"]),
            class_labels=class_labels,
            output_path=cm_plot_path,
            title=f"SVM Baseline (RBF) — Validation Confusion Matrix\nAccuracy: {val_metrics['accuracy']*100:.1f}%"
        )
        print(f"      Saved confusion matrix plot: {cm_plot_path}")
        print(f"      Saved validation metrics:    {metrics_json_path}")

        # 6. Save Model Artifact
        print("\n[6/6] Serializing Trained Model and Metadata...")
        model_artifact_path = self.models_dir / "svm_baseline.joblib"
        artifact_payload = {
            "model_pipeline": best_model,
            "class_labels": class_labels,
            "feature_config": {
                "color_hist": "HSV 8x8x8 normalized (512 dims)",
                "hu_moments": "7 invariant moments with log scaling (7 dims)",
                "glcm_texture": "GLCM 5 properties x 2 dist x 4 angles (40 dims)",
                "total_dim": TOTAL_FEATURE_DIM
            },
            "best_params": best_params,
            "pca_components": self.pca_components,
            "seed": self.seed,
            "trained_samples": len(train_records),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(artifact_payload, model_artifact_path)
        print(f"      Saved complete model artifact: {model_artifact_path}")
        print("\n=== Pipeline Execution Completed Successfully ===")

        return {
            "status": "success",
            "model_path": str(model_artifact_path),
            "metrics_path": str(metrics_json_path),
            "cm_plot_path": str(cm_plot_path),
            "cv_score": best_cv_score,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "timing": {
                "train_feature_time": train_feat_time,
                "svm_train_time": svm_train_time,
                "total_latency_ms": total_single_image_latency_ms
            }
        }


def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent
    trainer = SVMBaselineTrainer(workspace_root=workspace_root)
    trainer.run()


if __name__ == "__main__":
    main()
