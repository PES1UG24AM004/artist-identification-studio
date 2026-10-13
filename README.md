# Artist Identification Studio: Comparative Evaluation of Classical Machine Learning and Deep Learning

Artist Identification Studio is an academic research platform for multi-class visual artwork attribution, comparing classical machine learning (handcrafted visual feature engineering with an RBF Support Vector Machine) against deep transfer learning (a fine-tuned ResNet-18 convolutional neural network). Evaluated on 2,000 paintings across 10 historical artists from the Kaggle "Best Artworks of All Time" collection, the project demonstrates how hierarchical neural representations outperform manual color, texture, and shape descriptors on complex artistic styles while contrasting their respective computational trade-offs.

---

## Team Members

- **ANANYA RAGHAVENDRA** — SRN: `PES1UG24AM037`
- **AARAV YUVAL B G** — SRN: `PES1UG24AM004`

**Course:** UE24CS352A — Machine Learning<br>
**Institution:** PES University

---

## Quick Start (No Dataset Download Required)

The repository includes pre-trained model weights (`models/svm_baseline.joblib` [1.51 MB] and `models/cnn_best.pth` [42.77 MB]) and verified evaluation artifacts. **You do NOT need to download the Kaggle dataset or retrain models to run the application or tests.**

### Step 1: Clone the repository

```bash
git clone (https://github.com/PES1UG24AM004/artist-identification-studio.git)
cd Final
```

### Step 2: Set up the environment

Run the automated Windows setup script:

```cmd
setup.cmd
```

This script:
1. Detects Python 3.12 (via `py -3.12` or default `python`).
2. Creates a local virtual environment in `.venv` if not already present.
3. Checks if required dependencies are already functional. If so, skips redundant package installation.
4. Installs lightweight PyTorch CPU runtime wheels (`--index-url https://download.pytorch.org/whl/cpu`) and remaining dependencies from `requirements.txt`.
5. Validates essential imports and confirms model checkpoint presence.

### Step 3: Run the test suite

```cmd
run-tests.cmd
```

Executes all 23 unit and integration tests across data preparation, feature extraction, SVM training, CNN transfer learning, and the demonstration interface.

### Step 4: Launch the interactive studio

```cmd
run-demo.cmd
```

Once initialized, open your browser and navigate to:

```text
http://127.0.0.1:8501
```

To stop the application, press `Ctrl+C` in the terminal.

---

## What the Application Does

Artist Identification Studio features a streamlined dark-mode research interface organized into two core workflows:

### 1. Attribution Analysis Tab
- **Custom Image Upload:** Drag and drop any painting file (JPEG, PNG, WebP) for instant artist attribution.
- **Curated Validation Samples:** If the raw dataset is present, select from 10 pre-configured historical paintings spanning all 10 artists. If running on a fresh clone without the dataset, the selector gracefully prompts for custom image uploads.
- **Multi-Model Attribution View:** Compare predictions side-by-side or inspect either model individually:
  - **Traditional ML:** Evaluates a 559-dimensional feature vector combining 512 HSV color histogram bins, 7 Hu invariant moments, and 40 GLCM texture features passed through PCA dimensionality reduction and an RBF-kernel SVM.
  - **Deep Learning:** Evaluates a 224x224 RGB input through a transfer-learned ResNet-18 convolutional backbone.
- **Top-3 Candidates & Confidence:** Displays top-3 ranked artist predictions with calibrated class probabilities.
- **CPU Inference Latency:** Displays real-time single-image inference latency in milliseconds.
- **Deterministic Caching:** Prediction caching guarantees that switching between model views does not trigger redundant forward passes.

### 2. Model Benchmark & Evaluation Tab
- **Held-Out Test Set Metrics:** Displays verified performance metrics on 200 held-out test paintings (20 per artist) withheld during all training and tuning stages.
- **Side-by-Side Normalized Confusion Matrices:** High-resolution confusion matrices illustrating inter-artist confusions for both models.
- **Per-Artist F1-Score Breakdown:** Comparative bar chart highlighting class-by-class strengths and weaknesses across all 10 artists.
- **Comparative Analysis Table:** Side-by-side breakdown of accuracy, macro F1, macro precision, macro recall, and inference speed.

---

## Measured Experimental Results

Both models were trained on 1,600 images (160/artist), tuned on 200 validation images (20/artist), and evaluated on an unbiased held-out test set of 200 paintings (20/artist).

### Overall Test Performance Summary (N=200 Held-Out Artworks)

| Evaluation Metric | SVM Baseline (RBF + PCA-100) | CNN Transfer Learning (ResNet-18) | Advantage / Delta |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 53.50% | 71.00% | +17.50% (CNN) |
| **Test Accuracy** | **53.00%** (106 / 200) | **70.50%** (141 / 200) | **+17.50% (CNN)** |
| **Test Macro F1-Score** | 0.5206 | **0.6988** | +0.1782 (CNN) |
| **Test Weighted F1-Score** | 0.5206 | **0.6988** | +0.1782 (CNN) |
| **Test Macro Precision** | 0.5302 | **0.7172** | +0.1870 (CNN) |
| **Test Macro Recall** | 0.5300 | **0.7050** | +0.1750 (CNN) |
| **Mean Single-Image Latency (CPU)** | **18.45 ms** | 47.40 ms | +28.95 ms (SVM is 2.6x faster) |
| **Model Checkpoint Size** | **1.51 MB** | 42.77 MB | SVM is ~28x smaller |

### Per-Artist Class Breakdown on Held-Out Test Data

| Target Artist Class | Art Movement / Style | SVM Precision | SVM Recall | SVM F1 | CNN Precision | CNN Recall | CNN F1 | F1 Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Albrecht Durer** | Northern Renaissance / Printmaking | 0.8571 | 0.9000 | **0.8780** | 0.8571 | 0.9000 | **0.8780** | Tie (0.0000) |
| **Alfred Sisley** | Impressionist Landscape | 0.6818 | 0.7500 | 0.7143 | 0.8182 | 0.9000 | **0.8571** | +0.1428 (CNN) |
| **Edgar Degas** | Impressionism / Ballet & Figure | 0.3929 | 0.5500 | 0.4583 | 0.6400 | 0.8000 | **0.7111** | +0.2528 (CNN) |
| **Francisco Goya** | Romanticism / Dark Period | 0.4615 | 0.3000 | 0.3636 | 0.6250 | 0.5000 | **0.5556** | +0.1920 (CNN) |
| **Pablo Picasso** | Cubism / Modernism | 0.2500 | 0.1500 | 0.1875 | 0.6842 | 0.6500 | **0.6667** | +0.4792 (CNN) |
| **Paul Gauguin** | Post-Impressionist Synthetism | 0.3636 | 0.4000 | 0.3810 | 0.6364 | 0.3500 | **0.4516** | +0.0706 (CNN) |
| **Pierre-Auguste Renoir** | Impressionist Portraits | 0.7500 | 0.4500 | 0.5625 | 0.7619 | 0.8000 | **0.7805** | +0.2180 (CNN) |
| **Rembrandt** | Dutch Baroque / Chiaroscuro | 0.5417 | 0.6500 | 0.5909 | 1.0000 | 0.6500 | **0.7879** | +0.1970 (CNN) |
| **Titian** | High Renaissance Venetian School | 0.5238 | 0.5500 | 0.5366 | 0.6071 | 0.8500 | **0.7083** | +0.1717 (CNN) |
| **Vincent van Gogh** | Post-Impressionist Impasto | 0.4800 | 0.6000 | 0.5333 | 0.5417 | 0.6500 | **0.5909** | +0.0576 (CNN) |

### Key Experimental Insights
1. **Representational Hierarchy:** ResNet-18 achieves a substantial +17.50% test accuracy gain over the SVM baseline because convolutional kernels capture structural brushwork, compositional rhythm, and multi-scale texture features that fixed global descriptors compress away.
2. **Distinctive Visual Signatures:** Artists with stark, unique visual signatures (e.g., Albrecht Dürer's high-contrast woodcut engravings) attained high F1-scores on both paradigms (0.8780 F1).
3. **Impressionist and Complex Styles:** Pablo Picasso presented the highest challenge for classical feature engineering (0.1875 SVM F1 vs. 0.6667 CNN F1), where manual color histograms and Hu moments failed to capture multifaceted abstract geometry across Picasso's varied stylistic periods.
4. **Efficiency vs. Accuracy:** The SVM baseline executes single-image feature extraction and classification in 18.45 ms on CPU with a 1.51 MB footprint, making it suitable for edge deployments where deep learning runtimes are constrained.

---

## Repository Directory Structure

```text
Final/
├── .streamlit/
│   └── config.toml                  # Streamlit theme configuration (pure dark mode)
├── data/
│   ├── downloads/                   # Raw archive downloads (.gitkeep tracked)
│   ├── raw/                         # Raw artist images directory (.gitkeep tracked)
│   └── processed/                   # Stratified manifests directory (.gitkeep tracked)
├── models/
│   ├── svm_baseline.joblib          # Trained Classical SVM pipeline checkpoint (1.51 MB)
│   └── cnn_best.pth                 # Trained ResNet-18 model weights checkpoint (42.77 MB)
├── outputs/
│   ├── model_comparison.json        # Unified test benchmark metrics manifest
│   ├── svm_vs_cnn_confusion_matrices.png # Side-by-side normalized test confusion matrices
│   └── per_artist_f1_comparison.png # Per-artist F1-score comparison chart
├── src/
│   ├── __init__.py                  # Package marker
│   ├── app.py                       # Interactive Streamlit studio application
│   ├── data/
│   │   ├── __init__.py              # Subpackage marker
│   │   ├── download_dataset.py      # HTTP Range selective Kaggle acquisition pipeline
│   │   └── make_dataset.py          # Validation, deduplication, and stratified 80/10/10 splitting
│   ├── features/
│   │   ├── __init__.py              # Subpackage marker
│   │   └── extract_features.py      # HSV color histogram, Hu moments, and GLCM descriptors
│   ├── models/
│   │   ├── __init__.py              # Subpackage marker
│   │   ├── train_svm.py             # Feature caching, PCA, CV hyperparameter tuning, and SVM training
│   │   ├── train_cnn.py             # Transfer learning training and validation loop (ResNet-18)
│   │   └── evaluate_comparison.py   # Held-out test evaluation, confusion matrices, and comparisons
│   └── utils/
│       ├── __init__.py              # Subpackage marker
│       └── evaluate.py              # Metrics computation, timing, and plot generation helpers
├── tests/
│   ├── __init__.py                  # Subpackage marker
│   ├── test_make_dataset.py         # Unit tests for image verification and stratified splitting
│   ├── test_svm.py                  # Unit tests for feature extraction and SVM pipeline
│   ├── test_cnn.py                  # Unit tests for ResNet-18 architecture and frozen backbone
│   └── test_app.py                  # Integration tests for Streamlit caching and inference parity
├── .gitignore                       # Git exclusion rules (data, virtualenvs, caches, logs, PDFs)
├── requirements.txt                 # Project dependency specifications
├── setup.cmd                        # Automated Windows environment and dependency setup script
├── run-demo.cmd                     # Windows launch script for the Streamlit demo application
├── run-tests.cmd                    # Windows test runner script with status code propagation
└── README.md                        # Comprehensive project documentation
```

---

## Full Experiment Reproduction (Optional Workflow)

To reproduce the complete machine learning experiment from scratch (downloading raw data, generating splits, extracting features, training both models, and computing test metrics):

### 1. Dataset Acquisition

You can acquire the dataset using either of the following approaches:

**Option A: Official Kaggle CLI (Recommended for Reliability)**
1. Place your Kaggle API key at `~/.kaggle/kaggle.json` or `.kaggle/kaggle.json`.
2. Download and unzip the archive:
   ```cmd
   kaggle datasets download -d ikarus777/best-artworks-of-all-time -p data/downloads --unzip
   ```
3. Copy the 10 target artist folders from `data/downloads/resized/resized/` into `data/raw/`.

**Option B: Selective HTTP Range Script**
The repository includes a custom acquisition pipeline (`src/data/download_dataset.py`) that uses HTTP Range headers to stream only the 200 resized images per target artist directly from the remote archive:
```cmd
.venv\Scripts\python.exe src/data/download_dataset.py
```
*Note:* This script relies on the ZIP central directory offset of Version 1 of the dataset on Kaggle. If the remote archive structure is updated by the author, use Option A above.

### 2. Verify and Split the Dataset
Perform SHA-256 deduplication and generate the deterministic 80/10/10 stratified split:

```cmd
.venv\Scripts\python.exe src/data/make_dataset.py
```

Outputs: `data/processed/dataset_manifest.csv`, `dataset_manifest.json`, and `dataset_summary.json`.

### 3. Train the Classical SVM Baseline
Extract 559-dimensional visual descriptors, tune hyperparameters with 3-fold cross validation, and evaluate on validation data:

```cmd
.venv\Scripts\python.exe src/models/train_svm.py
```

Outputs: `models/svm_baseline.joblib` and `outputs/svm_confusion_matrix.png`.

### 4. Train the CNN Transfer Learning Model
Fine-tune the linear head on frozen ResNet-18 ImageNet backbone:

```cmd
.venv\Scripts\python.exe src/models/train_cnn.py
```

Outputs: `models/cnn_best.pth`, `outputs/cnn_learning_curves.png`, and `outputs/cnn_confusion_matrix.png`.

### 5. Run Comparative Evaluation on Held-Out Test Set
Evaluate both models on the held-out 200 test paintings:

```cmd
.venv\Scripts\python.exe src/models/evaluate_comparison.py
```

Outputs: `outputs/model_comparison.json`, `outputs/svm_vs_cnn_confusion_matrices.png`, and `outputs/per_artist_f1_comparison.png`.

---

## Modifying and Extending Models

The modular architecture allows straightforward customization of features, models, and hyperparameters:

### 1. Extending Handcrafted Features (`src/features/extract_features.py`)
- **Color Descriptors:** Modify `extract_color_histogram` to adjust HSV binning or evaluate alternative color spaces (e.g., CIE Lab).
- **Texture Descriptors:** Modify `extract_glcm_features` to add additional GLCM distances, angles, or Local Binary Patterns (LBP).
- **Edge / Shape Descriptors:** Add Gabor filter responses or Histogram of Oriented Gradients (HOG) by concatenating vectors inside `extract_image_features`.

### 2. Adjusting the SVM Pipeline (`src/models/train_svm.py`)
- **Dimensionality Reduction:** Modify the PCA component count (`PCA(n_components=100)`) or test kernel PCA.
- **Hyperparameter Grid:** Adjust the parameter grid (`PARAM_GRID`) in `tune_and_train_svm` to explore alternate kernels (`linear`, `poly`), degrees, or regularization values `C`.

### 3. Modifying CNN Backbones (`src/models/train_cnn.py`)
- **Backbone Selection:** Swap `resnet18(weights=...)` with alternate lightweight backbones such as `mobilenet_v3_small` or `resnet50`.
- **Fine-Tuning:** Unfreeze specific convolutional blocks (e.g., `layer4`) before training by modifying parameter `requires_grad` flags.
- **Training Parameters:** Adjust learning rates, batch size, weight decay, or learning rate schedulers in `train_model`.

---

## Dataset Attribution & Licensing

- **Dataset Name:** Best Artworks of All Time
- **Curator:** `ikarus777` on Kaggle
- **Source:** [Kaggle Dataset: ikarus777/best-artworks-of-all-time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
- **Dataset License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Target Artists Included:**
  - Albrecht Dürer (Northern Renaissance)
  - Alfred Sisley (Impressionism)
  - Edgar Degas (Impressionism)
  - Francisco Goya (Romanticism)
  - Pablo Picasso (Cubism / Modernism)
  - Paul Gauguin (Post-Impressionism)
  - Pierre-Auguste Renoir (Impressionism)
  - Rembrandt (Dutch Baroque)
  - Titian (Venetian Renaissance)
  - Vincent van Gogh (Post-Impressionism)
- **Licensing Context:** The underlying historical paintings were created between the 15th and early 20th centuries. The digitized reproductions in this collection are shared under CC BY-NC-SA 4.0 for non-commercial research and educational purposes. It remains an unsettled legal question whether trained machine learning weights qualify as adapted material / derivative works under CC BY-NC-SA 4.0 or non-copyrightable statistical parameters. Consequently, retaining these model checkpoints (`models/svm_baseline.joblib` and `models/cnn_best.pth`) in this repository is intended strictly for private academic course evaluation, and any future public or commercial redistribution requires separate licensing review.

---

## Limitations & Troubleshooting

### Known Limitations
- **Stylistic Overlap:** The highest misclassification rate occurs between Impressionist contemporaries (Alfred Sisley, Pierre-Auguste Renoir, Edgar Degas) due to overlapping pastel palettes, outdoor lighting, and rapid brush techniques.
- **Sample Scale:** While 200 paintings per artist provide sufficient data for transfer learning head adaptation, fine-grained differentiation of subtle artist periods (e.g., Picasso's Blue vs. Cubist periods) requires larger, period-annotated corpuses.
- **Color Normalization:** Variations in photographic illumination and museum digitization equipment can influence color histogram features.
- **Experimental Variation:** Training results and exact metric values may vary slightly depending on library versions, CPU architecture, non-deterministic operations, and dataset revisions.

### Troubleshooting Common Issues

1. **Port 8501 is already in use:**
   - If port 8501 is occupied by an earlier session, specify an alternate port:
     ```cmd
     .venv\Scripts\python.exe -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8502
     ```
2. **Python Launcher (`py`) not recognized:**
   - If `py -3.12` fails, `setup.cmd` automatically falls back to `python`. Ensure Python 3.10+ is checked in your system PATH.
3. **PyTorch CUDA warnings on GPU machines:**
   - The setup explicitly pins PyTorch CPU wheels for lightweight, zero-configuration execution. To leverage an NVIDIA GPU, reinstall torch with CUDA support using official PyTorch index instructions.
