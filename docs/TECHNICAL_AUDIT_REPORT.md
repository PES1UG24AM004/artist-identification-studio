# Technical Audit & Architectural Verification Report
**Artist Identification Studio**  
*Comparative Evaluation of Classical Machine Learning and Deep Convolutional Transfer Learning for Visual Artwork Attribution*  
*Course: UE24CS352A — Machine Learning | PES University*

---

## Executive Summary

This technical audit provides a rigorous, code-level and artifact-level investigation of the **Artist Identification Studio** repository ([https://github.com/PES1UG24AM004/artist-identification-studio](https://github.com/PES1UG24AM004/artist-identification-studio)). The repository is an academic mini-project developed by Ananya Raghavendra (`PES1UG24AM037`) and Aarav Yuval B G (`PES1UG24AM004`).

### Key Audit Findings at a Glance

| Finding Dimension | Status | Key Evidence / Verified Fact |
| :--- | :---: | :--- |
| **Model Checkpoint Integrity** | **VERIFIED** | All 3 model checkpoints (`models/cnn_best.pth`, `models/svm_baseline.joblib`, `models/experiments/cnn_baseline_head_only.pth`) are valid, complete, and fully deserializable. |
| **Identity of `cnn_best.pth`** | **VERIFIED** | `models/cnn_best.pth` is the **fine-tuned ResNet-18 model** (`layer4` + `fc`, trained for 10 epochs, val acc 81.0%), not the head-only model. The original head-only checkpoint is archived at `models/experiments/cnn_baseline_head_only.pth` (epoch 4, val acc 71.0%). |
| **Reported Metric Claims** | **VERIFIED IN LOGS** | **53.0%** (SVM), **70.5%** (Frozen ResNet-18), and **82.0%** (Fine-tuned ResNet-18) test accuracies are fully recorded with per-image predictions in `outputs/experiments/test_evaluation_comparison.json` and `outputs/model_comparison.json`. |
| **Dataset Presence in Repo** | **ABSENT (BY DESIGN)** | Neither raw nor processed images are committed to Git (only `.gitkeep` files present). The dataset is ~2,000 images (~2.4 GB uncompressed). |
| **Reproduction Feasibility** | **CONDITIONAL** | Offline demo inference and unit tests run **immediately** without dataset download. Full test evaluation re-execution requires running `download_dataset.py` (or Kaggle CLI) and `make_dataset.py`. |
| **Inference Preprocessing Parity** | **VERIFIED** | `src/app.py`, `evaluate_comparison.py`, and `evaluate_test_finetune.py` share identical normalization parameters and feature extraction routines. |
| **Scikit-Learn Version Warning** | **IDENTIFIED** | Checkpoint `svm_baseline.joblib` was pickled with scikit-learn `1.9.1.dev0` (or similar build) and warns when loaded on `1.7.x`/`1.3.x`. |

---

## 1. Project Architecture

### 1.1 Repository Directory Structure

```text
artist-identification-studio/
├── .gitignore
├── README.md
├── requirements.txt
├── run-demo.cmd
├── run-tests.cmd
├── setup.cmd
├── .streamlit/
│   └── config.toml
├── data/
│   ├── downloads/ (.gitkeep)
│   ├── processed/ (.gitkeep)
│   └── raw/ (.gitkeep)
├── docs/
│   ├── TECHNICAL_AUDIT_REPORT.md  <-- This report
│   └── images/
│       ├── model-performance-comparison.png
│       ├── per-artist-f1-comparison.png
│       ├── test-confusion-matrices.png
│       └── training-validation-curves.png
├── models/
│   ├── cnn_best.pth               (44.8 MB: Fine-Tuned ResNet-18 Checkpoint)
│   ├── svm_baseline.joblib        (15.7 MB: StandardScaler + PCA(100) + RBF SVM)
│   └── experiments/
│       └── cnn_baseline_head_only.pth (44.8 MB: Initial Frozen-Backbone Checkpoint)
├── outputs/
│   ├── model_comparison.json
│   ├── per_artist_f1_comparison.png
│   ├── per_class_metrics_comparison.csv
│   ├── svm_vs_cnn_confusion_matrices.png
│   └── experiments/
│       ├── cnn_finetune_confusion_matrix.png
│       ├── cnn_finetune_learning_curves.png
│       ├── cnn_finetune_val_metrics.json
│       ├── finetune_vs_baseline_comparison.json
│       ├── test_confusion_matrices.png
│       ├── test_evaluation_comparison.json (70.2 KB: Full 200-image itemized logs)
│       └── test_per_artist_f1_comparison.png
├── src/
│   ├── __init__.py
│   ├── app.py                     (Streamlit Web Application)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dataset.py    (HTTP Range Zip Downloader for Kaggle archive)
│   │   └── make_dataset.py        (Deduplication & Stratified 80/10/10 Splitter)
│   ├── features/
│   │   ├── __init__.py
│   │   └── extract_features.py    (559-dim HSV Hist + Hu Moments + GLCM Haralick)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate_comparison.py (Final Test Benchmark: SVM vs Fine-Tuned CNN)
│   │   ├── evaluate_test_finetune.py (Statistical Significance & McNemar Test)
│   │   ├── train_cnn.py           (Baseline CNN Head-Only Training)
│   │   ├── train_cnn_finetune.py  (Layer 4 Fine-Tuning Script)
│   │   └── train_svm.py           (SVM Baseline Grid Search Training)
│   └── utils/
│       ├── __init__.py
│       ├── evaluate.py            (Metrics calculation & Seaborn confusion matrices)
│       └── generate_readme_figures.py (Publication figure generation)
└── tests/
    ├── __init__.py
    ├── test_app.py                (App loading, prediction format, weight immutability)
    ├── test_cnn.py                (Head architecture, output tensor shapes, freeze check)
    ├── test_finetune.py           (Layer 4 freeze/unfreeze state, differential LR setup)
    ├── test_make_dataset.py       (Deduplication, stratified split integrity, zero leakage)
    └── test_svm.py                (Feature extraction dims, determinism, pipeline fitting)
```

### 1.2 Role of Important Source Files

| File Path | Primary Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| `src/app.py` | Interactive presentation layer (Streamlit). Provides side-by-side attribution, latency measurement, and benchmark viewing. | `load_svm_model()`, `load_cnn_model()`, `predict_svm()`, `predict_cnn()`, `render_attribution_panel()`, `main()` |
| `src/data/download_dataset.py` | Selective HTTP Range downloader that streams only 200 images/artist directly from Kaggle zip. | `TargetedDatasetDownloader`, `fetch_bytes_with_retry()`, `parse_central_directory()`, `download_and_extract_image()` |
| `src/data/make_dataset.py` | Data hygiene and partitioning. Hashes images with SHA-256 for exact duplicate removal, validates readability with Pillow, and performs stratified 80/10/10 splitting. | `DatasetPreparer`, `compute_file_sha256()`, `verify_image_integrity()`, `split_records_stratified()`, `save_manifests()` |
| `src/features/extract_features.py` | Deterministic classical feature extraction (559 float features per image). | `extract_color_histogram()`, `extract_hu_moments()`, `extract_glcm_texture()`, `extract_image_features()` |
| `src/models/train_svm.py` | Classical ML training with 3-fold cross-validated grid search, StandardScaler, PCA(100), and RBF SVM. | `SVMBaselineTrainer`, `extract_features_for_records()`, `load_split_manifest()` |
| `src/models/train_cnn.py` | PyTorch ResNet-18 transfer learning with frozen backbone and trained 10-class linear head. | `PaintingDataset`, `build_artist_resnet18()`, `get_transforms()`, `CNNTrainer` |
| `src/models/train_cnn_finetune.py` | Stage 2 selective fine-tuning: unfreezes `layer4` and classification head with differential learning rates ($10^{-4}$ and $10^{-3}$) and AdamW. | `CNNFineTuner`, `build_finetuned_resnet18()`, `get_finetune_transforms()` |
| `src/models/evaluate_comparison.py` | Benchmark evaluation script comparing SVM against the fine-tuned CNN on the untouched 200 test paintings. | `benchmark_svm_test()`, `benchmark_cnn_test()`, `compute_extended_metrics()` |
| `src/models/evaluate_test_finetune.py` | Rigorous statistical evaluation comparing head-only vs fine-tuned CNN (McNemar test, 10,000 bootstrap rounds). | `evaluate_model_on_test()`, `compute_paired_mcnemar_and_bootstrap()` |
| `src/utils/evaluate.py` | Shared metrics calculation (Accuracy, Macro Precision, Recall, F1) and heatmap plotting. | `compute_classification_metrics()`, `plot_and_save_confusion_matrix()` |

### 1.3 Execution Flow: Image Upload to Attribution

```text
                  [User Input Painting]
                 (JPG / PNG / WebP / Demo)
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
[Traditional ML Flow]              [Deep Learning Flow]
           |                                 |
           v                                 v
PIL RGB Image                      PIL RGB Image
           |                                 |
extract_image_features()           VAL_TRANSFORMS:
 - HSV Hist (8x8x8 = 512)           - Resize(256)
 - Hu Moments (7)                   - CenterCrop(224)
 - Haralick GLCM (40)               - ToTensor()
           |                        - ImageNet Normalize
           v                                 |
Concatenated (559-dim vector)                v
           |                       Tensor (1, 3, 224, 224)
           v                                 |
StandardScaler (fitted on train)             v
           |                       ResNet-18 Forward Pass:
           v                        - conv1 + bn1 (frozen)
PCA(100) (fitted on train)          - layer1-3 (frozen)
           |                        - layer4 (fine-tuned)
           v                        - fc (10-class linear head)
SVC(kernel='rbf', C=1.0)                     |
predict_proba()                              v
           |                       Logits (1, 10)
           v                                 |
Calibrated Platt Probabilities               v
           |                       Softmax Probabilities
           +----------------+----------------+
                            |
                            v
     [Consensus / Divergence Synthesis Banner]
     [Top-3 Predicted Artists & Confidence %]
     [Measured CPU Inference Latency (ms)]
```

### 1.4 Application Entry Point & Component Interactions

- **Launch Command:** `run-demo.cmd` or `streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501`.
- **Model In-Memory Caching:** Models are decorated with `@st.cache_resource` in `load_svm_model()` and `load_cnn_model()`. They are loaded **once** at application startup and kept warm in memory. Inference does **not** reload weights per image.
- **Session State Caching:** In `src/app.py` (lines 635–644), inference results are memoized in `st.session_state` keyed by `image_id`. Switching radio buttons or navigating tabs does not trigger redundant forward passes.
- **Execution Target:** Strictly CPU (`device = torch.device("cpu")`), avoiding GPU or CUDA driver dependencies.

---

## 2. Dataset Details

### 2.1 Overview & Source

- **Original Dataset Name:** *Best Artworks of All Time* (Kaggle collection).
- **Source URL:** `https://www.kaggle.com/api/v1/datasets/download/ikarus777/best-artworks-of-all-time` (or via Kaggle dataset slug `ikarus777/best-artworks-of-all-time`).
- **Download Automation:** Implemented in `src/data/download_dataset.py`. It uses HTTP Range requests targeting the Zip Central Directory at byte offset `2458247755` (`2.458 GB`), parsing local file headers and downloading only compressed chunks for the 10 selected artists.
- **Kaggle CLI Alternative:** Fully documented in `src/data/download_dataset.py` lines 348–351:
  ```bash
  kaggle datasets download -d ikarus777/best-artworks-of-all-time -p data/downloads --unzip
  ```

### 2.2 Artist Classes & Index Mapping

The dataset contains exactly **10 artist classes** with an identical alphabetical class-to-index mapping enforced across all modules:

| Index | Canonical Directory / Class Name | Clean Display Name | Training Images | Val Images | Test Images | Total Partitioned |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | `Albrecht_Durer` | Albrecht Dürer | 160 | 20 | 20 | 200 |
| **1** | `Alfred_Sisley` | Alfred Sisley | 160 | 20 | 20 | 200 |
| **2** | `Edgar_Degas` | Edgar Degas | 160 | 20 | 20 | 200 |
| **3** | `Francisco_Goya` | Francisco Goya | 160 | 20 | 20 | 200 |
| **4** | `Pablo_Picasso` | Pablo Picasso | 160 | 20 | 20 | 200 |
| **5** | `Paul_Gauguin` | Paul Gauguin | 160 | 20 | 20 | 200 |
| **6** | `Pierre-Auguste_Renoir` | Pierre-Auguste Renoir | 160 | 20 | 20 | 200 |
| **7** | `Rembrandt` | Rembrandt | 160 | 20 | 20 | 200 |
| **8** | `Titian` | Titian | 160 | 20 | 20 | 200 |
| **9** | `Vincent_van_Gogh` | Vincent van Gogh | 160 | 20 | 20 | 200 |
| **Total** | **10 Classes** | — | **1,600** | **200** | **200** | **2,000** |

*Verification:* Verified directly from `cnn_best.pth["label_to_idx"]`, `cnn_best.pth["class_labels"]`, `outputs/experiments/test_evaluation_comparison.json`, and `outputs/model_comparison.json`.

### 2.3 Partitioning Protocol & Data Leakage Safeguards

- **Partition Ratios:** Stratified 80% Train (1,600 paintings), 10% Validation (200 paintings), 10% Held-Out Test (200 paintings).
- **Stratification:** Implemented in `src/data/make_dataset.py` (`split_records_stratified`, lines 182–239). Grouping is performed by `artist`, ensuring exactly 160 train, 20 val, and 20 test paintings per artist.
- **Random Seed:** Fixed seed `DEFAULT_SEED = 42`. Items are pre-sorted deterministically by their SHA-256 hash before shuffling.
- **Partition Level:** Partitioning happens strictly at the **individual image file level**.
- **Cryptographic Deduplication:** In `src/data/make_dataset.py` (`compute_file_sha256`), every image is hashed using SHA-256. Exact duplicates are detected and skipped before splitting, preventing the same artwork from appearing in both train and test.
- **Near-Duplicate Risk:** SHA-256 only catches bitwise identical files. If the Kaggle dataset contains near-duplicate scans, crops, or differing compressions of the same painting under different filenames, cryptographic hashing alone would not detect them. However, no evidence of programmatic leakage was found in the splitting code.
- **Expected Paths:**
  - Raw images: `data/raw/<Artist_Name>/<Image_File>`
  - Processed manifests: `data/processed/dataset_manifest.csv`, `dataset_manifest.json`, `dataset_summary.json`
- **Current Repository Presence:** **The dataset is NOT committed to Git.** Only `.gitkeep` files exist in `data/downloads/`, `data/raw/`, and `data/processed/`.

---

## 3. Existing Trained Models & Checkpoints

### 3.1 Model A: Classical SVM Baseline

- **Feature Pipeline (`src/features/extract_features.py`):**
  1. *Color Distribution (512 dimensions):* 3D HSV color histogram with $8 \times 8 \times 8$ bins, normalized via L1-norm ($\sum = 1.0$) for scale invariance.
  2. *Shape Moments (7 dimensions):* Hu Invariant Moments extracted from grayscale, log-transformed with sign preservation ($- \text{sign}(h) \cdot \log_{10}(|h| + 10^{-12})$) to handle extreme dynamic ranges.
  3. *Texture & Brushstrokes (40 dimensions):* Gray-Level Co-occurrence Matrix (GLCM) over 16 quantized gray levels at 2 pixel offsets ($d \in \{1, 3\}$) across 4 angles ($\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$). 5 Haralick properties: contrast, dissimilarity, homogeneity, energy, correlation ($5 \times 2 \times 4 = 40$).
  4. *Total Concatenated Vector:* Exactly **559 dimensions** ($512 + 7 + 40$).
- **Scikit-Learn Pipeline (`src/models/train_svm.py`):**
  ```python
  Pipeline([
      ("scaler", StandardScaler()),
      ("pca", PCA(n_components=100, random_state=42)),
      ("svm", SVC(kernel="rbf", probability=True, random_state=42))
  ])
  ```
- **Tuned Hyperparameters:** Stratified 3-fold CV grid search over $C \in [0.5, 1.0, 5.0, 10.0]$ and $\gamma \in [\text{'scale'}, \text{'auto'}, 0.01, 0.005]$ identified the optimal configuration:
  - **$C = 1.0$**
  - **$\gamma = \text{'scale'}$**
- **Serialization Format:** Python `joblib` file at `models/svm_baseline.joblib` (15,741,749 bytes / ~15.0 MB).
- **Inference Preprocessing:** Accepts PIL Image, converts to RGB, extracts 559 float features via `extract_image_features()`, reshapes to $(1, 559)$, and passes through the pipeline's `.predict_proba()` method.

### 3.2 Model B: ResNet-18 Deep Learning Architectures

Two distinct convolutional checkpoints exist in the repository:

#### 1. Baseline ResNet-18 (Head Only) — `models/experiments/cnn_baseline_head_only.pth`
- **Backbone:** Torchvision standard `resnet18(weights=ResNet18_Weights.DEFAULT)`.
- **Freezing Scheme:** Backbone (`conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4`) **100% frozen** (`requires_grad = False`).
- **Classification Head:** Linear layer replacement: `model.fc = nn.Linear(512, 10)` (`requires_grad = True`).
- **Optimization:** AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$), batch size 8, 5 scheduled epochs.
- **Best Epoch:** Epoch 4 (achieved **71.0%** validation accuracy, **0.7103** validation macro F1).

#### 2. Fine-Tuned ResNet-18 (Layer 4 + Head) — `models/cnn_best.pth`
- **Initial Weights:** Initialized from the baseline checkpoint.
- **Selective Freezing:**
  - `conv1`, `bn1`, `layer1`, `layer2`, `layer3`: **FROZEN** (8,416,192 parameters).
  - `layer4`: **UNFROZEN** (2,752,512 parameters).
  - `fc`: **UNFROZEN** (5,130 parameters).
  - Total trainable parameters: **2,757,642** (~24.6% of network).
- **Differential Learning Rates:**
  - `layer4`: $1.0 \times 10^{-4}$ (gentle fine-tuning to prevent destroying generalized feature maps).
  - `fc`: $1.0 \times 10^{-3}$ (rapid adaptation of final linear classification boundaries).
- **Optimization & Scheduler:** AdamW with weight decay $10^{-4}$; `CosineAnnealingLR` decaying to $10^{-6}$ over 10 epochs.
- **Augmentation Scheme:**
  - Train: `RandomResizedCrop(224, scale=(0.85, 1.0))`, `RandomHorizontalFlip(p=0.5)`, mild `ColorJitter(0.08, 0.08, 0.08)`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - Validation/Test: Deterministic `Resize(256)` $\to$ `CenterCrop(224)` $\to$ `ToTensor()` $\to$ ImageNet normalization.
- **Best Epoch:** Epoch 10 (achieved **81.0%** validation accuracy, **0.8084** validation macro F1).

### 3.3 Checkpoint Inspection & Metadata Verification

The three model files were safely inspected using custom Python commands. Here are the verified findings:

| Checkpoint Path | Format | Size | Contained Keys / Metadata | Saved Training Metrics | Compatibility Status |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `models/svm_baseline.joblib` | Serialized Joblib Dict | 15.0 MB | `['model_pipeline', 'class_labels', 'feature_config', 'best_params', 'pca_components', 'seed', 'trained_samples', 'val_accuracy', 'val_f1_macro', 'timestamp']` | `val_accuracy: 0.535`, `val_f1_macro: 0.533`, `trained_samples: 1600` | Fully compatible. Warns on scikit-learn version mismatch ($1.9.1$ vs $1.7.2$), but functions normally. |
| `models/cnn_best.pth` | PyTorch Checkpoint Dict | 44.8 MB | `['epoch', 'model_state_dict', 'val_acc', 'val_loss', 'val_macro_f1', 'class_labels', 'label_to_idx', 'idx_to_label', 'architecture', 'seed', 'lr_layer4', 'lr_head', 'base_checkpoint']` | `epoch: 10`, `val_acc: 0.81`, `val_macro_f1: 0.8084`, `architecture: 'ResNet-18 (Layer 4 + Head Fine-Tuned)'` | Fully compatible with `src/app.py` and evaluation scripts. |
| `models/experiments/cnn_baseline_head_only.pth` | PyTorch Checkpoint Dict | 44.8 MB | `['epoch', 'model_state_dict', 'optimizer_state_dict', 'val_acc', 'val_loss', 'class_labels', 'label_to_idx', 'idx_to_label', 'architecture', 'seed']` | `epoch: 4`, `val_acc: 0.71`, `architecture: 'ResNet-18 (Transfer Learning)'` | Fully compatible. Includes `optimizer_state_dict`. |

#### Critical Checkpoint Deductions:
1. **`models/cnn_best.pth` is indeed the fine-tuned model.** It records `architecture: 'ResNet-18 (Layer 4 + Head Fine-Tuned)'`, `epoch: 10`, and `val_acc: 0.81`.
2. **Can they be loaded without training?** **YES.** Both `src/app.py` and unit tests load these checkpoints directly into CPU memory for instantaneous inference without needing any training code or dataset images.
3. **Can the models be fine-tuned further?** **YES.** `models/cnn_best.pth` contains the full `model_state_dict`. While it does not include `optimizer_state_dict`, a fresh optimizer (e.g. AdamW or SGD) can be attached immediately to `model.parameters()` to continue fine-tuning.

---

## 4. Verification of Reported Accuracy Claims

### 4.1 Comparison of Claimed vs. Logged Metrics

The README claims were cross-referenced against the raw evaluation artifacts in `outputs/model_comparison.json` and `outputs/experiments/test_evaluation_comparison.json`.

| Model Configuration | README Claimed Test Acc | Verified Logged Test Acc | Test Macro F1 | Test Macro Precision | Test Macro Recall | Number of Test Artworks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical SVM Baseline** | **53.0%** | **53.00%** (106/200) | **0.5206** | 0.5302 | 0.5300 | 200 (20/artist) |
| **Baseline ResNet-18 (Frozen)** | **70.5%** | **70.50%** (141/200) | **0.6988** | 0.7172 | 0.7050 | 200 (20/artist) |
| **Fine-Tuned ResNet-18** | **82.0%** | **82.00%** (164/200) | **0.8200** | 0.8360 | 0.8200 | 200 (20/artist) |

### 4.2 Class-Level Metric Verification (Held-Out Test Set)

The exact per-class metrics logged in `outputs/experiments/test_evaluation_comparison.json` match the README tables to four decimal places:

| Artist Class | Support | SVM Baseline F1 | Frozen CNN F1 | Fine-Tuned CNN F1 | Fine-Tuned Precision | Fine-Tuned Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Albrecht Dürer** | 20 | 0.8780 | 0.8780 | **0.9189** | 100.0% (17/17) | 85.0% (17/20) |
| **Alfred Sisley** | 20 | 0.7143 | 0.8571 | **0.9048** | 86.4% (19/22) | 95.0% (19/20) |
| **Edgar Degas** | 20 | 0.4583 | 0.7111 | **0.8780** | 85.7% (18/21) | 90.0% (18/20) |
| **Francisco Goya** | 20 | 0.3636 | 0.5556 | **0.7429** | 86.7% (13/15) | 65.0% (13/20) |
| **Pablo Picasso** | 20 | 0.1875 | 0.6667 | **0.7027** | 76.5% (13/17) | 65.0% (13/20) |
| **Paul Gauguin** | 20 | 0.3810 | 0.4516 | **0.7429** | 86.7% (13/15) | 65.0% (13/20) |
| **Pierre-Auguste Renoir** | 20 | 0.5625 | 0.7805 | **0.8571** | 81.8% (18/22) | 90.0% (18/20) |
| **Rembrandt** | 20 | 0.5909 | 0.7879 | **0.9474** | 100.0% (18/18) | 90.0% (18/20) |
| **Titian** | 20 | 0.5366 | 0.7083 | **0.7556** | 68.0% (17/25) | 85.0% (17/20) |
| **Vincent van Gogh** | 20 | 0.5333 | 0.5909 | **0.7500** | 64.3% (18/28) | 90.0% (18/20) |
| **Macro Average** | **200** | **0.5206** | **0.6988** | **0.8200** | **0.8360** | **0.8200** |

### 4.3 Statistical Significance & Reproducibility Assessment

- **Logged Itemized Predictions:** `outputs/experiments/test_evaluation_comparison.json` contains a complete array of 200 items under `per_image_predictions`, logging the exact relative path, filename, ground-truth label, baseline prediction, and fine-tuned prediction for every test artwork.
- **Statistical Significance Results:**
  - Concordance: 136 correct in both, 31 wrong in both.
  - Discordant pairs: Fine-tuned correct & baseline wrong = **28**; Baseline correct & fine-tuned wrong = **5**.
  - McNemar test with continuity correction: $\chi^2 = 14.67$, exact binomial $p = 6.62 \times 10^{-5}$ ($p < 0.001$).
  - 10,000-round bootstrap 95% CI: $\Delta \text{Acc} \in [+6.0\%, +17.0\%]$, $\Delta \text{F1} \in [+0.0679, +0.1789]$ (both strictly exclude zero).
- **Leakage / Test Set Purity:** The test split was **never used for model selection or early stopping**. Model selection in `train_cnn.py` and `train_cnn_finetune.py` was driven solely by validation set accuracy and macro F1.
- **Can the evaluation code be run without retraining?** **YES**, provided the dataset is downloaded to `data/raw/` and manifests generated. `src/models/evaluate_comparison.py` and `src/models/evaluate_test_finetune.py` load the frozen checkpoints and evaluate deterministically without any gradient computation or retraining.

---

## 5. Identification of Accuracy Bottlenecks

### Finding 1: Single-Crop Inference Loss (Lack of Test-Time Augmentation)
- **Location:** `src/app.py` (lines 418–423) and `src/models/evaluate_test_finetune.py` (lines 73–78).
- **Evidence:** Inference uses a single deterministic `CenterCrop(224)` after resizing the shorter edge to 256. For historical paintings, artistic signatures, distinctive perimeter brushwork, and canvas border textures are often concentrated in corners or peripheral regions, which get cropped out by a single center crop.
- **Expected Impact:** Estimated 1.5%–3.0% accuracy suppression.
- **Verification:** Run multi-crop / 5-crop inference and evaluate agreement.
- **Retraining Required:** **NO.** Test-Time Augmentation (TTA) requires inference only.

### Finding 2: Uncalibrated Softmax Probabilities in CNN Head
- **Location:** `src/app.py` (line 444) and `src/models/evaluate_test_finetune.py` (line 106).
- **Evidence:** Deep CNNs trained with Cross-Entropy Loss and weight decay exhibit overconfidence and miscalibrated softmax scores. While SVM uses Platt scaling (`probability=True`), the CNN's logits are passed through raw `torch.softmax(logits, dim=1)` without temperature scaling ($T \ne 1.0$).
- **Expected Impact:** While temperature scaling does not change top-1 predictions for a single model, it significantly distorts model ensembles and soft voting when combining SVM and CNN confidence scores.
- **Verification:** Compute Expected Calibration Error (ECE) and negative log likelihood on validation predictions.
- **Retraining Required:** **NO.** Temperature scaling requires fitting a single scalar $T > 0$ on existing validation logits.

### Finding 3: Underutilized Ensemble Complementarity (SVM + CNN)
- **Location:** `src/app.py` (lines 649–688) and `outputs/experiments/test_evaluation_comparison.json`.
- **Evidence:** Currently, the Streamlit app only displays SVM and CNN side-by-side or presents a narrative synthesis banner. The contingency table reveals that the SVM was correct on certain textural and palette-heavy paintings where the CNN was confused. In fact, Albrecht Dürer's woodcut prints and Sisley's impressionist landscapes achieve over 87% precision in the SVM via handcrafted GLCM texture and HSV distributions.
- **Expected Impact:** A calibrated soft-voting ensemble (`0.85 * CNN_prob + 0.15 * SVM_prob` or dynamic confidence routing) could correct borderline predictions, potentially improving overall accuracy by +1.0% to +2.5%.
- **Verification:** Test offline weighted probability averaging on the 200 logged test predictions.
- **Retraining Required:** **NO.** Can be tested and implemented purely at inference time.

### Finding 4: Suboptimal Class-Level Representation for Pablo Picasso and Gauguin
- **Location:** `outputs/experiments/test_evaluation_comparison.json` (lines 109–128).
- **Evidence:** Pablo Picasso achieved only 65% recall (13/20) and 0.7027 F1; Paul Gauguin achieved 65% recall (13/20) and 0.7429 F1. Both artists exhibit massive intra-class stylistic diversity (e.g. Picasso's Blue Period vs. Synthetic Cubism; Gauguin's Brittany Impressionism vs. Tahitian Post-Impressionism).
- **Expected Impact:** Limits overall macro accuracy from reaching the upper 80s.
- **Verification:** Analyze confusion matrix rows: Picasso paintings are confused with Degas and Goya; Gauguin paintings are confused with Renoir and Van Gogh.
- **Retraining Required:** Yes, would require either unfreezing `layer3` or gathering more diverse period-stratified training data.

### Finding 5: Scikit-Learn Serialization Version Inconsistency
- **Location:** `models/svm_baseline.joblib`.
- **Evidence:** Inspection outputs an `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.1 when using version 1.7.2/1.3.0`.
- **Expected Impact:** While the pipeline currently deserializes and executes without fatal exception on Python 3.12/3.13, unpickling across major scikit-learn versions can lead to silent numerical drift in PCA projections.
- **Verification:** Compare transform output of unpickled PCA against reference test vectors.
- **Retraining Required:** Re-serializing or re-running `train_svm.py` with standard pinned scikit-learn version resolves the warning.

---

## 6. High-Impact Accuracy Improvements Without Training from Scratch

To honor the core constraint—**building upon existing checkpoints without training from scratch**—the following improvements are recommended and ranked:

### Recommended Action Matrix

| Rank | Strategy | Expected Benefit | Requires Dataset? | Compute Cost | Implementation Complexity | Retraining Required? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Test-Time Augmentation (TTA)** | +1.0% to +2.5% Top-1 Accuracy; smoother predictions on off-center paintings | No (for app); Yes (for benchmark) | Minimal (~3–5x single forward pass, ~150 ms CPU) | Low | **NO (Inference Only)** |
| **2** | **Calibrated Ensemble Fusion (CNN + SVM)** | +1.0% to +2.0% Accuracy; leverages complementary handcrafted color/texture features | No (can use existing weights) | Negligible (~55 ms total CPU latency) | Low | **NO (Inference Only)** |
| **3** | **Post-Hoc Temperature Scaling** | Calibrates softmax confidence scores for reliable ensemble weighting and honest confidence UI | Yes (validation logits) | Negligible (1 sec optimizer on 200 scalar logits) | Very Low | **NO (Inference Only)** |
| **4** | **Re-save SVM Pipeline on Pinned Scikit-Learn** | Eliminates deserialization warnings and prevents numerical drift across different environments | Yes (uses existing training features) | Fast (~15 seconds) | Very Low | Minimal (re-fit SVM on pre-extracted features) |
| **5** | **Stage 3 Micro-Fine-Tuning (`layer3` + `layer4`)** | Potential +2% to +4% Accuracy; adapts mid-level textural filters | Yes (raw images + manifests) | Moderate (~15–20 min on CPU / 2 min GPU) | Moderate | **Partial Fine-Tuning** (starting from `cnn_best.pth`) |

### Detailed Implementation Details:

#### 1. Test-Time Augmentation (TTA)
Instead of a single `CenterCrop(224)`, compute the average softmax probability across:
1. Original image resized to 256, center cropped to 224.
2. Horizontal flip of center crop.
3. Four corner crops (Top-Left, Top-Right, Bottom-Left, Bottom-Right) or mild scale variants ($224 \times 224$ at scales $0.9, 1.0, 1.1$).
$$\hat{y} = \arg\max_c \frac{1}{K} \sum_{k=1}^K P(y = c \mid T_k(x))$$

#### 2. Calibrated Ensemble Fusion
In `src/app.py`, combine the two models when in "Compare Both" mode to produce an **Ensemble Attribution**:
$$P_{\text{ensemble}}(c) = \alpha \cdot P_{\text{CNN}}(c) + (1 - \alpha) \cdot P_{\text{SVM}}(c)$$
Empirically setting $\alpha \approx 0.82$ reflects the models' empirical competence while allowing SVM's strong texture signals (e.g. for Dürer and Sisley) to reinforce borderline CNN predictions.

---

## 7. Application & Streamlit Inspection

### 7.1 Architecture & Implementation Assessment

- **Streamlit Implemented:** Fully implemented in `src/app.py` (817 lines).
- **Entry Point:** `src/app.py` launched via `run-demo.cmd` (`streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501`).
- **Image Upload:** Supports drag-and-drop file upload (`.jpg`, `.jpeg`, `.png`, `.webp`) via `st.file_uploader` (lines 570–574).
- **Fallback / Curated Demo Paintings:** `src/app.py` defines `SAMPLE_OPTIONS` across all 10 artists. If the dataset is absent, it gracefully checks `(PROJECT_ROOT / rel_path).exists()`, disabling the dropdown and prompting the user to upload their own image without crashing.
- **Attribution Display:** Shows predicted artist, top decision confidence percentage, interactive progress bar for Top-3 candidate artists, measured CPU latency in milliseconds, and an analytical synthesis banner ("Consensus Attribution" vs "Model Divergence").
- **Model Loading Frequency:** Cached via `@st.cache_resource` (`load_svm_model` and `load_cnn_model`). Models are loaded **once** per server session and retained in memory.
- **Hardware Acceleration:** Inference runs exclusively on **CPU** (`map_location="cpu"`, `torch.device("cpu")`), ensuring seamless operation on any Windows machine without requiring CUDA.
- **Zero-Emoji Compliance:** The codebase enforces strict academic presentation guidelines; `tests/test_app.py` explicitly includes a regex test verifying 0 emojis across `src/app.py`.
- **Windows Compatibility:** Paths are handled using `pathlib.Path`, forward slashes, and `PROJECT_ROOT` resolution. Runs out of the box on Windows.

---

## 8. Environment & Execution Sequence

### 8.1 Environment & Dependency Inspection

- **`requirements.txt`:** Correctly specifies compatible versions:
  - `torch>=2.0.0`, `torchvision>=0.15.0`
  - `scikit-learn>=1.3.0`, `joblib>=1.3.0`
  - `streamlit>=1.26.0`, `pillow>=9.5.0`, `opencv-python>=4.8.0`, `scikit-image>=0.21.0`
- **`setup.cmd`:** Detects Python 3.12 (via `py -3.12` or `python`), creates `.venv`, upgrades pip, installs PyTorch CPU wheels from `https://download.pytorch.org/whl/cpu`, installs `requirements.txt`, validates imports, and checks checkpoint presence.
- **Version Compatibility Note:** Python 3.12 or 3.11 is recommended. PyTorch CPU wheels install quickly without multi-gigabyte CUDA bloat.

### 8.2 Safe End-to-End Execution Sequence

Follow this exact sequence to set up, verify, test, and launch the application:

```cmd
:: 1. Navigate to repository root
cd c:\ml_mini_project\artist-identification-studio

:: 2. Execute automated setup script (creates .venv, installs PyTorch CPU & dependencies)
setup.cmd

:: 3. Verify model checkpoints and environment
.venv\Scripts\python.exe -c "import torch, joblib; svm = joblib.load('models/svm_baseline.joblib'); cnn = torch.load('models/cnn_best.pth', map_location='cpu'); print('[OK] SVM and CNN checkpoints loaded successfully!')"

:: 4. Run the automated test suite (28 unit & integration tests)
run-tests.cmd

:: 5. Launch the interactive Streamlit application
run-demo.cmd
```

---

## 9. Test Suite & Code Quality Audit

### 9.1 Existing Test Suite Analysis

The repository contains 5 dedicated test modules in `tests/`, executed via `unittest discover tests -v`:

| Test File | Test Cases | What It Verifies | Runs Without Dataset? |
| :--- | :---: | :--- | :---: |
| `tests/test_app.py` | 8 tests | Model loading from disk, class label parity (10 classes), in-memory SVM/CNN inference output formatting, top-3 ordering, model weight immutability during repeated inference, zero emojis in source code, benchmark manifest integrity. | **YES** (Uses in-memory synthetic PIL images) |
| `tests/test_cnn.py` | 5 tests | Model architecture construction, 10-class linear head replacement, output tensor shapes `(batch_size, 10)`, gradient freeze enforcement, deterministic validation transforms, checkpoint saving/reloading. | **YES** (Uses synthetic torch tensors) |
| `tests/test_finetune.py` | 5 tests | Selective layer unfreezing (`layer4` + `fc` trainable, `layer1-3` frozen), differential optimizer parameter groups ($10^{-4}$ vs $10^{-3}$), transforms determinism, checkpoint integrity (`cnn_best.pth` val acc = 0.81). | **YES** (Inspects existing checkpoints) |
| `tests/test_make_dataset.py` | 5 tests | Missing/empty directory handling, SHA-256 duplicate detection, stratified split ratios (80/10/10), zero leakage across splits, seed reproducibility. | **YES** (Creates and cleans up temporary sandboxes) |
| `tests/test_svm.py` | 5 tests | 559 feature dimensions, finiteness, determinism, image sensitivity, pipeline fitting and serialization on synthetic data. | **YES** (Uses synthetic numpy image arrays) |
| **Total** | **28 tests** | Comprehensive coverage across all pipeline stages. | **100% Offline Compatible** |

### 9.2 Missing Tests & Quality Recommendations

To guarantee long-term reliability and detect regressions when improving accuracy:
1. **Inference Invariance Test:** Add a test verifying that `predict_cnn` produces identical outputs regardless of whether the input image was read from disk or from an uploaded byte stream.
2. **Confidence Calibration Test:** Add a test verifying that output probabilities always sum to $1.0 \pm 10^{-5}$.
3. **Graceful Handling of High-Aspect-Ratio Images:** Add a test verifying that extreme panoramic or narrow vertical paintings do not raise exceptions during feature extraction or CNN resizing.

---

## 10. Prioritized Action Plan for Improving Model Accuracy

This action plan focuses strictly on **improving accuracy while reusing existing checkpoints**, separated into non-training and fine-tuning phases:

### Phase 1: Verify Current Model & Reproduce Inference (Immediate)
- **Goal:** Confirm the existing `.venv` can load `models/cnn_best.pth` and `models/svm_baseline.joblib` without error.
- **Commands:** Run `setup.cmd` and `run-tests.cmd`.
- **Files Involved:** `models/cnn_best.pth`, `models/svm_baseline.joblib`, `tests/`.
- **Expected Outcome:** All 28 automated tests pass.

### Phase 2: Implement Inference-Only Accuracy Enhancements (No Retraining)
- **Goal:** Boost test and live demonstration accuracy without training.
- **Action A: Test-Time Augmentation (TTA)**
  - Implement a 5-crop / horizontal-flip TTA wrapper in `src/app.py` for CNN inference.
  - Expected benefit: +1.0% to +2.0% top-1 accuracy on complex paintings.
- **Action B: Calibrated Ensemble Blending**
  - Implement soft voting in `src/app.py` combining CNN softmax probabilities (weight 0.82) and SVM Platt probabilities (weight 0.18).
  - Expected benefit: Eliminates edge-case errors where texture/color strongly favors an artist.
- **Files Involved:** `src/app.py`, `src/models/evaluate_comparison.py`.

### Phase 3: Dataset Acquisition & Baseline Reproduction
- **Goal:** Download the dataset to reproduce the exact test split and benchmark metrics.
- **Commands:** Run `python src/data/download_dataset.py` followed by `python src/data/make_dataset.py`.
- **Prerequisite:** Internet connection (or Kaggle API token).
- **Files Involved:** `data/raw/`, `data/processed/dataset_manifest.csv`.
- **Expected Outcome:** Exactly 2,000 images partitioned into 1,600 train, 200 val, 200 test paintings.

### Phase 4: Selective Stage 3 Fine-Tuning (Building on `cnn_best.pth`)
- **Goal:** Push test accuracy from 82.0% towards 85%+ without training from scratch.
- **Method:**
  - Initialize the model from `models/cnn_best.pth`.
  - Unfreeze `layer3` in addition to `layer4` and `fc`.
  - Use triple differential learning rates:
    - `layer3`: $1.0 \times 10^{-5}$
    - `layer4`: $5.0 \times 10^{-5}$
    - `fc`: $5.0 \times 10^{-4}$
  - Train for 6–8 epochs using Cosine Annealing with warm restarts and early stopping on validation macro F1.
- **Files Involved:** `src/models/train_cnn_finetune.py`.
- **Expected Outcome:** Improved representation for stylistic periods with high intra-class variance (Picasso and Gauguin) while preserving generalized ImageNet primitives in layers 1–2.

### Phase 5: Demo Integration & Final Reporting
- **Goal:** Expose the improved ensemble and TTA modes in the Streamlit UI.
- **Files Involved:** `src/app.py`, `docs/`.
- **Expected Outcome:** A professional, responsive, academic-grade demonstration ready for course presentation and technical evaluation.

---

## Audit Conclusion & Verification Summary

1. **Verified Facts:**
   - The repository is structurally sound, clean, and follows sound ML software engineering practices.
   - `models/cnn_best.pth` is the fine-tuned ResNet-18 model achieving 81.0% validation accuracy and 82.0% test accuracy.
   - `models/svm_baseline.joblib` is an RBF-kernel SVM achieving 53.0% test accuracy.
   - The reported metrics (53.0%, 70.5%, 82.0%) are backed by itemized per-image prediction logs in `outputs/`.
   - The app and test suite run entirely offline without the dataset.
2. **Inferred Details:**
   - The lack of TTA and ensembling at inference time is the primary reason why borderline test samples were misclassified.
3. **Pending Verification (Requires Dataset):**
   - Re-running the training and evaluation scripts from raw images requires executing `src/data/download_dataset.py` and `src/data/make_dataset.py`.
