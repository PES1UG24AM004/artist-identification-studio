# Artist Identification Studio: Machine Learning Mini-Project

An academic machine learning system developed for comparative evaluation of classical computer vision (Support Vector Machine with handcrafted color, shape, and texture features) versus deep convolutional transfer learning (ImageNet-pretrained ResNet-18) on visual artwork attribution across 10 master artists.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Features & Application Workflow](#features--application-workflow)
4. [Dataset Details](#dataset-details)
5. [Technologies & Libraries Used](#technologies--libraries-used)
6. [Repository Structure](#repository-structure)
7. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
8. [Virtual Environment Activation (Windows PowerShell)](#virtual-environment-activation-windows-powershell)
9. [Dependency Installation](#dependency-installation)
10. [Model Checkpoints & Dataset Placement](#model-checkpoints--dataset-placement)
11. [Launching the Streamlit Application](#launching-the-streamlit-application)
12. [Running Automated Tests](#running-automated-tests)
13. [Model Architecture: SVM Baseline vs. ResNet-18 CNN](#model-architecture-svm-baseline-vs-resnet-18-cnn)
14. [Test-Time Augmentation (Horizontal-Flip TTA)](#test-time-augmentation-horizontal-flip-tta)
15. [Validation Results & Benchmarks](#validation-results--benchmarks)
16. [Latency Measurement Methodology](#latency-measurement-methodology)
17. [Limitations & Future Improvements](#limitations--future-improvements)
18. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

**Artist Identification Studio** is an end-to-end machine learning platform that classifies fine art paintings according to their artist. The project directly contrasts two prominent computer vision paradigms:
* A **Classical ML Baseline** relying on domain-specific feature engineering (color histograms, Hu invariant moments, and Haralick texture features) paired with dimensionality reduction and an RBF Support Vector Machine.
* A **Deep Convolutional Transfer Learning Network** utilizing an ImageNet-pretrained ResNet-18 backbone fine-tuned on painting representations.

The project features a full interactive demonstration web app built with Streamlit, comprehensive unit/integration test suites, offline benchmark analytics, and test-time augmentation (TTA) capabilities.

---

## 2. Problem Statement

Attributing historical paintings to their original artist from digital imagery is an arduous task in computer vision due to:
* High **intra-class variance**: An artist's technique, palette, and subject matter often change dramatically across distinct artistic periods (e.g., Vincent van Gogh's dark Nuenen-era studies versus his vibrant Arles-era Post-Impressionism).
* Subtle **inter-class similarity**: Contemporary painters belonging to the same artistic movement (such as Impressionists Claude Monet, Alfred Sisley, and Pierre-Auguste Renoir) employ overlapping palettes, lighting techniques, and stroke textures.
* Resolution and framing discrepancies in digital photographic captures of physical canvases.

The objective of this project is to quantitatively measure whether learned convolutional representations capture higher-order stylistic signatures significantly better than traditional handcrafted color and texture descriptors, and to provide a reproducible, interactive platform to demonstrate this comparison.

---

## 3. Features & Application Workflow

* **Side-by-Side Dual-Model Inference**: Real-time comparative classification showing Top-3 predicted artist probabilities for both the Classical SVM and Deep CNN pipelines simultaneously.
* **Curated Gallery & Custom Uploads**: Built-in sample selector featuring representative artworks across all 10 artist classes (including difficult intra-class diagnostic paintings like Van Gogh's *Weaver at the Loom*) alongside support for drag-and-drop user image uploads (`.jpg`, `.jpeg`, `.png`, `.webp`).
* **Test-Time Augmentation (TTA) Toggle**: Interactive checkbox in the studio UI allowing horizontal-flip test-time augmentation on ResNet-18 to observe prediction stability and confidence smoothing in real time.
* **Integrated Benchmark Explorer**: An interactive benchmark dashboard displaying confusion matrices, per-artist F1-score breakdowns, and execution latency metrics directly inside the web interface.
* **Zero-Download Fresh-Clone Support**: Model weights are included in the repository; the interface gracefully degrades if raw training images are absent by serving in-memory custom uploads and verified offline evaluation metrics.

---

## 4. Dataset Details

The dataset is derived from the *Best Artworks of All Time* collection:
* **Total Images**: 2,000 digital painting images.
* **Classes**: 10 master artists with exactly 200 paintings per artist (balanced class distribution):
  1. Albrecht Dürer
  2. Alfred Sisley
  3. Edgar Degas
  4. Francisco Goya
  5. Pablo Picasso
  6. Paul Gauguin
  7. Pierre-Auguste Renoir
  8. Rembrandt
  9. Titian
  10. Vincent van Gogh
* **Split Partitioning Protocol**:
  * **Training Split**: 1,600 images (80%, 160 images per artist).
  * **Validation Split**: 200 images (10%, 20 images per artist).
  * **Held-Out Test Split**: 200 images (10%, 20 images per artist).
* **Stratification & Reproducibility**: The dataset split is strictly stratified by artist class using a fixed random seed (`seed=42`). Data splits are recorded deterministically in `data/processed/dataset_manifest.csv` and `data/processed/dataset_manifest.json` to prevent data leakage.

---

## 5. Technologies & Libraries Used

* **Language**: Python 3.12 (supported on Python 3.10+)
* **Core Numeric & Data Analysis**: `numpy`, `scipy`, `pandas`
* **Classical Vision & Feature Engineering**: `opencv-python`, `scikit-image`, `pillow`
* **Classical Machine Learning**: `scikit-learn`, `joblib`
* **Deep Learning Framework**: `torch`, `torchvision` (PyTorch CPU runtime)
* **Interactive UI**: `streamlit`
* **Visualization & Plotting**: `matplotlib`, `seaborn`
* **Testing**: Python standard library `unittest`

---

## 6. Repository Structure

```text
artist-identification-studio/
├── .gitignore                     # Git ignore rules for virtualenvs, caches, dataset zips, and logs
├── README.md                      # Original project documentation and test benchmark report
├── README2.md                     # Comprehensive submission setup and execution guide
├── requirements.txt               # Pinned project dependencies
├── setup.cmd                      # One-click Windows setup script (venv, dependencies, sanity check)
├── run-demo.cmd                   # One-click script to launch the Streamlit web demo
├── run-tests.cmd                  # One-click script to execute all automated unit tests
├── data/
│   ├── downloads/                 # Directory for downloaded archives (.gitkeep tracked)
│   ├── raw/                       # Directory for uncompressed raw artist folders (.gitkeep tracked)
│   └── processed/                 # Dataset split manifest files
│       ├── dataset_manifest.csv   # 2,000-image split assignments (train/val/test)
│       ├── dataset_manifest.json  # JSON format manifest
│       └── dataset_summary.json   # Split distribution and metadata summary
├── docs/
│   ├── TECHNICAL_AUDIT_REPORT.md  # Detailed experimental validation audit and code inventory
│   └── images/                    # Evaluation plots and diagrams for documentation
├── models/
│   ├── cnn_best.pth               # Pretrained fine-tuned ResNet-18 model weights (~44.8 MB)
│   ├── svm_baseline.joblib        # Trained StandardScaler + PCA + SVM pipeline (~1.5 MB)
│   └── experiments/
│       └── cnn_baseline_head_only.pth # Archived frozen-backbone baseline model (~44.8 MB)
├── outputs/                       # Verified evaluation metrics and offline benchmark assets
│   ├── model_comparison.json
│   ├── per_artist_f1_comparison.png
│   ├── per_class_metrics_comparison.csv
│   └── svm_vs_cnn_confusion_matrices.png
├── src/
│   ├── app.py                     # Streamlit web demonstration application
│   ├── data/                      # Dataset extraction and stratified splitting utilities
│   ├── features/                  # Handcrafted feature extraction (HSV, Hu moments, Haralick)
│   ├── models/                    # Model training, fine-tuning, and evaluation pipelines
│   └── utils/                     # Metric calculation, plotting, and reproducibility utilities
└── tests/
    ├── test_app.py                # Streamlit interface, prediction logic, and TTA unit tests
    ├── test_cnn.py                # CNN model architecture and inference tests
    ├── test_finetune.py           # Fine-tuning parameter unfreezing and head tests
    ├── test_make_dataset.py       # Dataset stratification and integrity tests
    └── test_svm.py                # SVM feature extraction and pipeline tests
```

---

## 7. Prerequisites & Environment Setup

* **Operating System**: Windows 10/11 (or Linux/macOS with equivalent shell commands).
* **Python Version**: Python 3.12 is recommended (Python 3.10 and 3.11 are also supported). Verify your installation:
  ```powershell
  py -3.12 --version
  # or
  python --version
  ```

---

## 8. Virtual Environment Activation (Windows PowerShell)

In Windows PowerShell, navigate to the project directory and create the virtual environment:

```powershell
cd C:\ml_mini_project\artist-identification-studio

# Create virtual environment named .venv
py -3.12 -m venv .venv
# (or: python -m venv .venv)

# Set execution policy for the current PowerShell process if needed
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

Once activated, your terminal prompt will display `(.venv)`.

---

## 9. Dependency Installation

With the virtual environment activated, install the required packages:

```powershell
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install PyTorch CPU runtime
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies from requirements.txt
python -m pip install -r requirements.txt
```

Alternatively, on Windows, you can simply run the automated batch script:
```cmd
setup.cmd
```

---

## 10. Model Checkpoints & Dataset Placement

### Model Checkpoints (Zero-Download Verification)
The project tracks the necessary trained model checkpoints directly in Git:
* `models/svm_baseline.joblib` (1.58 MB)
* `models/cnn_best.pth` (44.8 MB)

Because these files are tracked in the repository, no external download or retraining step is required to run the demo or unit tests.

### Dataset Placement (Optional / Local Verification)
The 2,000 painting images and raw zip archives are deliberately excluded from Git via `.gitignore` to keep the repository lightweight and adhere to submission constraints.
* To run the application on custom paintings, no dataset is needed — simply upload paintings via the web UI.
* If you have the dataset archive (`raw.zip` or the 10 artist folders), place the unzipped artist folders directly into:
  ```text
  data/raw/
  ├── Albrecht_Dürer/
  ├── Alfred_Sisley/
  ├── Edgar_Degas/
  ├── Francisco_Goya/
  ├── Pablo_Picasso/
  ├── Paul_Gauguin/
  ├── Pierre_Auguste_Renoir/
  ├── Rembrandt/
  ├── Titian/
  └── Vincent_van_Gogh/
  ```
  The manifest files in `data/processed/` will automatically index and locate the images.

---

## 11. Launching the Streamlit Application

Ensure the virtual environment is activated, then launch Streamlit:

```powershell
streamlit run src/app.py
```

Or run the provided Windows script:
```cmd
run-demo.cmd
```

Once started, open your web browser to:
```text
http://127.0.0.1:8501
```

---

## 12. Running Automated Tests

Run the complete 30-test suite across data preparation, feature extraction, SVM, CNN, TTA, and Streamlit serving components:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Or run the provided Windows script:
```cmd
run-tests.cmd
```

All 30 unit tests should pass with `OK`.

---

## 13. Model Architecture: SVM Baseline vs. ResNet-18 CNN

### Classical SVM Baseline Pipeline
1. **Input Preprocessing**: Images are resized to $256 \times 256$ pixels.
2. **Handcrafted Feature Engineering (559 features total)**:
   * **Color**: 3D HSV color histogram with 8 bins per channel ($8 \times 8 \times 8 = 512$ bins), normalized by total pixels.
   * **Shape**: 7 Hu invariant moments computed on grayscale images with log transform applied for scale, translation, and rotation invariance.
   * **Texture**: 40 Haralick texture features extracted from the Gray-Level Co-occurrence Matrix (GLCM) across 4 spatial directions ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) and 10 statistical properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM, etc.).
3. **Dimensionality Reduction**: `StandardScaler` standardizes features to zero mean and unit variance, followed by `PCA(n_components=100, random_state=42)` retaining the primary variance modes while mitigating the curse of dimensionality.
4. **Classification**: Support Vector Classifier (`SVC`) with Radial Basis Function (RBF) kernel (`C=10.0`, `gamma='scale'`, `probability=True`).

### Deep CNN Architecture (Fine-Tuned ResNet-18)
1. **Backbone**: ImageNet-pretrained ResNet-18 convolutional neural network.
2. **Preprocessing**: Images are resized to $256 \times 256$, center-cropped to $224 \times 224$, and normalized with ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).
3. **Selective Fine-Tuning Strategy**:
   * Layers `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` are frozen to preserve generic low-level edge, contour, and mid-level texture primitives.
   * `layer4` (the final residual block) is unfrozen and fine-tuned at learning rate $\eta = 10^{-4}$ to adapt high-level brushstroke impasto and canvas textural representations to fine art.
   * The classification head (`fc`) is replaced with a linear layer ($512 \to 10$) and trained at $\eta = 10^{-3}$.
4. **Optimization**: AdamW optimizer with weight decay $10^{-4}$ and Cosine Annealing learning rate scheduling.

---

## 14. Test-Time Augmentation (Horizontal-Flip TTA)

Test-Time Augmentation (TTA) improves prediction stability at inference time without modifying trained weights or retraining:
1. For an input image $X$, the network computes softmax class probabilities $P(y \mid X)$.
2. A horizontally mirrored copy $X_{\text{flip}}$ is generated using `PIL.Image.FLIP_LEFT_RIGHT` and passed through the identical preprocessing and network forward pass to compute $P(y \mid X_{\text{flip}})$.
3. The final prediction probabilities are computed as the exact ensemble average:
   $$P_{\text{TTA}}(y \mid X) = \frac{P(y \mid X) + P(y \mid X_{\text{flip}})}{2}$$
4. Horizontal-flip TTA is effective for artwork classification because an artist's signature brushwork, chromatic palette, and medium texture remain invariant under horizontal reflection, while slight spatial composition asymmetries are smoothed out.

---

## 15. Validation Results & Benchmarks

All models were evaluated on the fixed **200-image validation split** ($20$ images per artist) from `data/processed/dataset_manifest.csv`:

| Model / Configuration | Validation Accuracy | Validation Macro F1 | Validation Macro Precision | Validation Macro Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Classical SVM Baseline** | 53.50% | 0.5330 | 0.5401 | 0.5350 |
| **ResNet-18 CNN (No TTA)** | 81.00% | 0.8084 | 0.8212 | 0.8100 |
| **ResNet-18 CNN (Horizontal-Flip TTA)** | **81.50%** | **0.8131** | **0.8260** | **0.8150** |

### Per-Artist Validation Performance (F1-Score Comparison)

| Artist Class | SVM Baseline F1 | ResNet-18 (No TTA) F1 | ResNet-18 (With TTA) F1 |
| :--- | :---: | :---: | :---: |
| **Albrecht Dürer** | 0.7778 | 0.9000 | 0.9000 |
| **Alfred Sisley** | 0.7727 | 0.8718 | 0.8780 |
| **Edgar Degas** | 0.4444 | 0.8718 | 0.8718 |
| **Francisco Goya** | 0.3889 | 0.7368 | 0.7368 |
| **Pablo Picasso** | 0.4000 | 0.7895 | 0.8000 |
| **Paul Gauguin** | 0.4286 | 0.7500 | 0.7500 |
| **Pierre-Auguste Renoir** | 0.4255 | 0.7895 | 0.7895 |
| **Rembrandt** | 0.6977 | 0.9268 | 0.9268 |
| **Titian** | 0.4615 | 0.7500 | 0.7805 |
| **Vincent van Gogh** | 0.5333 | 0.6977 | 0.6977 |

> **Important Benchmark Clarification**:
> The metrics presented above represent **validation set performance** on the 200 validation images. Held-out test-set performance has not yet been established for the TTA configuration. Validation results should not be confused with finalized test-set metrics.

---

## 16. Latency Measurement Methodology

Reported inference latency values reflect **end-to-end processing time per image**, measured using high-resolution monotonic clocks (`time.perf_counter()`).

Specifically, the measured latency includes:
1. **Disk I/O** (reading image bytes from disk or cache).
2. **Image decoding and loading** into memory via PIL.
3. **Preprocessing transforms** (RGB conversion, resizing, cropping, tensor normalization, and mirroring if TTA is enabled).
4. **Model forward pass** through the SVM feature pipeline or PyTorch neural network.
5. **Softmax probability calculation and top-3 sorting**.

This comprehensive latency measurement represents actual end-user wall-clock time rather than merely isolated GPU/CPU forward-pass matrix multiplication time.

---

## 17. Limitations & Future Improvements

### Current Limitations
1. **Closed-World Assumption**: The classifier strictly attributes any submitted artwork to one of the 10 trained artist classes. Artworks by other historical or modern artists will be assigned to whichever of the 10 classes has highest relative similarity.
2. **No Historical Provenance Certification**: The system is an academic machine learning experiment detecting visual style and statistical patterns. It does not certify legal authenticity, historical provenance, or physical canvas authorship.
3. **Hardware Latency Variations**: CPU inference times vary depending on the host machine processor, concurrent background tasks, and thermal throttling.

### Potential Future Improvements
* **Open-Set Recognition / Out-of-Distribution (OOD) Detection**: Incorporate energy-based OOD scores or temperature-scaled confidence thresholds to flag paintings that do not belong to the 10 target artists.
* **Higher-Resolution Vision Architectures**: Experiment with Vision Transformers (ViT) or ConvNeXt backbones capable of processing $384 \times 384$ crops for finer brushstroke impasto extraction.
* **Model Explainability**: Integrate Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations in the Streamlit UI to highlight the specific canvas regions influencing artist predictions.

---

## 18. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'torch'` or `'streamlit'`
* **Cause**: Commands are being executed outside the virtual environment.
* **Fix**: Ensure the virtual environment is activated (`.\.venv\Scripts\Activate.ps1`) before executing Python scripts, or invoke python via `.venv\Scripts\python.exe`.

### 2. `Execution of scripts is disabled on this system` (PowerShell)
* **Cause**: Windows PowerShell default execution policy restricts script execution.
* **Fix**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your current PowerShell session, then retry running `.\.venv\Scripts\Activate.ps1`.

### 3. Missing Model Checkpoints
* **Cause**: Files were not pulled completely or git clone was shallow.
* **Fix**: Verify that `models/svm_baseline.joblib` and `models/cnn_best.pth` exist in the `models/` directory. If missing, run `git checkout models/` or re-pull the branch.

### 4. Custom Paintings Return Out-of-Distribution Artist
* **Cause**: The input painting belongs to an artist outside the 10 trained classes.
* **Fix**: Refer to the closed-world limitation above; the model can only assign probabilities distributed among the 10 target classes.
