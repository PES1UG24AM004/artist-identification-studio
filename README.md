# Artist Identification Studio: Comparative Evaluation of Classical Machine Learning and Deep Learning

Artist Identification Studio is an academic research platform for multi-class visual artwork attribution, comparing classical machine learning (handcrafted visual feature engineering with an RBF Support Vector Machine) against deep transfer learning (a fine-tuned ResNet-18 convolutional neural network). Evaluated on 2,000 paintings across 10 historical artists from the Kaggle "Best Artworks of All Time" collection, the project demonstrates how hierarchical neural representations with selective convolutional fine-tuning outperform manual color, texture, and shape descriptors on complex artistic styles while contrasting their respective computational trade-offs.

---

## Team Members

- **ANANYA RAGHAVENDRA** — SRN: `PES1UG24AM037`
- **AARAV YUVAL B G** — SRN: `PES1UG24AM004`

**Course:** UE24CS352A — Machine Learning<br>
**Institution:** PES University

---

## Quick Start (No Dataset Download Required)

The repository includes pre-trained model checkpoints (`models/svm_baseline.joblib` [1.51 MB] and the promoted production checkpoint `models/cnn_best.pth` [42.78 MB]), an archived baseline checkpoint (`models/experiments/cnn_baseline_head_only.pth` [42.77 MB]), and verified evaluation artifacts. **You do NOT need to download the Kaggle dataset or retrain models to run the demonstration application or unit tests.**

### Step 1: Clone the repository

```bash
git clone https://github.com/PES1UG24AM004/artist-identification-studio.git
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

Executes all 28 unit and integration tests across data preparation, feature extraction, SVM training, CNN transfer learning, fine-tuning gradient verification, and the demonstration interface.

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

Artist Identification Studio features a polished dark-mode academic interface organized into two core workflows:

### 1. Attribution Analysis Tab
- **Custom Image Upload:** Drag and drop any painting file (JPEG, PNG, WebP) for instant artist attribution.
- **Curated Validation Samples:** If the raw dataset is present, select from 10 pre-configured historical paintings spanning all 10 artists. (Note: The curated Van Gogh sample `Vincent_van_Gogh_183.jpg` is accurately labeled as a *"Nuenen-era Monochrome Study"*). If running on a fresh clone without the raw dataset, the selector gracefully prompts for custom image uploads.
- **Multi-Model Attribution View:** Compare predictions side-by-side or inspect either model individually:
  - **Traditional ML (SVM Baseline):** Evaluates a 559-dimensional feature vector combining 512 HSV color histogram bins, 7 Hu invariant moments, and 40 GLCM texture features passed through `StandardScaler`, `PCA(100)`, and an RBF-kernel SVM.
  - **Deep Learning (Fine-Tuned ResNet-18):** Evaluates a 224x224 RGB input through an ImageNet-pretrained ResNet-18 with its final residual block (`layer4`) and 10-class linear classification head fine-tuned for painting attribution.
- **Top-3 Candidates & Confidence:** Displays top-3 ranked artist predictions with decision confidence.
- **CPU Inference Latency:** Displays real-time single-image CPU inference latency in milliseconds.
- **Deterministic Caching:** Streamlit caching guarantees that switching between model views does not trigger redundant forward passes.

### 2. Model Benchmark & Evaluation Tab
- **Held-Out Test Set Metrics:** Displays verified performance metrics on 200 held-out test paintings (20 per artist) withheld during all training and tuning stages.
- **Side-by-Side Normalized Confusion Matrices:** High-resolution confusion matrices illustrating inter-artist confusions for both models.
- **Per-Artist F1-Score Breakdown:** Comparative bar chart highlighting class-by-class strengths and weaknesses across all 10 artists.
- **Comparative Analysis Table:** Side-by-side breakdown of accuracy, macro F1, macro precision, macro recall, and CPU inference speed.

---

## Experimental Dataset and Protocol

The dataset consists of 2,000 paintings spanning 10 balanced artist classes (200 paintings per artist) sampled from the Kaggle *Best Artworks of All Time* collection:

- **Albrecht Dürer** (Northern Renaissance / Printmaking)
- **Alfred Sisley** (Impressionist Landscapes)
- **Edgar Degas** (Impressionism / Ballet & Figure Studies)
- **Francisco Goya** (Romanticism / Dark Period & Etchings)
- **Pablo Picasso** (Cubism / Early Modernism)
- **Paul Gauguin** (Post-Impressionist Synthetism)
- **Pierre-Auguste Renoir** (Impressionist Portraits)
- **Rembrandt** (Dutch Baroque / Chiaroscuro)
- **Titian** (Venetian Renaissance Colorito)
- **Vincent van Gogh** (Post-Impressionism / Impasto & Early Dutch Studies)

### Leakage-Free Stratified Splitting (80 / 10 / 10)
- **Training Split:** 1,600 paintings (160 per artist across 10 balanced classes).
- **Validation Split:** 200 paintings (20 per artist across 10 balanced classes) — used exclusively for hyperparameter tuning, checkpoint tracking, and early stopping.
- **Held-Out Test Split:** 200 paintings (20 per artist across 10 balanced classes) — strictly preserved and evaluated only once per final model evaluation.

---

## Measured Experimental Results

### 1. High-Level Performance Comparison: Validation vs. Held-Out Test Splits

Validation and held-out test metrics are strictly separated below:

| Split | Metric | Classical SVM Baseline | Original CNN (Head Only) | Promoted CNN (`layer4` Fine-Tuned) | Delta (Promoted vs. Original CNN) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation**<br>*(Tuning Split, N=200)* | **Accuracy** | 53.50% | 71.00% | **81.00%** | **+10.00%** |
| | **Macro F1** | 0.5330 | 0.7103 | **0.8084** | **+0.0981** |
| | **Van Gogh Recall** | 60.00% | 55.00% | **75.00%** | **+20.00%** |
| | **Van Gogh F1** | 0.5333 | 0.5366 | **0.6977** | **+0.1611** |
| **Held-Out Test**<br>*(Untouched Split, N=200)* | **Accuracy** | 53.00% (106/200) | 70.50% (141/200) | **82.00% (164/200)** | **+11.50%** |
| | **Macro F1** | 0.5206 | 0.6988 | **0.8200** | **+0.1212** |
| | **Macro Precision** | 0.5302 | 0.7172 | **0.8360** | **+0.1188** |
| | **Macro Recall** | 0.5300 | 0.7050 | **0.8200** | **+0.1150** |
| | **Van Gogh Recall** | 60.00% (12/20) | 65.00% (13/20) | **90.00% (18/20)** | **+25.00%** |
| | **Van Gogh F1** | 0.5333 | 0.5909 | **0.7500** | **+0.1591** |
| | **Mean CPU Latency** | **18.45 ms** | 42.45 ms | **39.56 ms** | Hardware-dependent |

### 2. Per-Artist Breakdown on Held-Out Test Data (N=20 Test Artworks per Artist)

| Target Artist Class | SVM Test F1 | Original CNN Test F1 | Promoted CNN Test F1 | CNN Test Recall | CNN Test Precision | F1 Advantage (vs. Orig CNN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paul Gauguin** | 0.3810 | 0.4516 | **0.7429** | 65.0% | 86.7% | **+0.2912** |
| **Francisco Goya** | 0.3636 | 0.5556 | **0.7429** | 65.0% | 86.7% | **+0.1873** |
| **Edgar Degas** | 0.4583 | 0.7111 | **0.8780** | 90.0% | 85.7% | **+0.1669** |
| **Rembrandt** | 0.5909 | 0.7879 | **0.9474** | 90.0% | 100.0% | **+0.1595** |
| **Vincent van Gogh** | 0.5333 | 0.5909 | **0.7500** | 90.0% | 64.3% | **+0.1591** |
| **Pierre-Auguste Renoir** | 0.5625 | 0.7805 | **0.8571** | 90.0% | 81.8% | **+0.0767** |
| **Alfred Sisley** | 0.7143 | 0.8571 | **0.9048** | 95.0% | 86.4% | **+0.0476** |
| **Titian** | 0.5366 | 0.7083 | **0.7556** | 85.0% | 68.0% | **+0.0472** |
| **Albrecht Dürer** | 0.8780 | 0.8780 | **0.9189** | 85.0% | 100.0% | **+0.0409** |
| **Pablo Picasso** | 0.1875 | 0.6667 | **0.7027** | 65.0% | 76.5% | **+0.0360** |

*Every single one of the 10 artist classes improved in test F1-score.*

### 3. Statistical Significance Analysis (Paired Test Set Evaluation, N=200)

- **Paired Contingency Table:**
  - Both models correct: $136$ paintings
  - Both models incorrect: $31$ paintings
  - Discordant pairs: $33$ paintings ($28$ where fine-tuned won vs. $5$ where baseline won, a **$5.6 : 1$ improvement ratio**).
  - Net gain: $+23$ correctly attributed test paintings.
- **McNemar Test for Paired Proportions:**
  - Continuity-corrected $\chi^2$ statistic: $\mathbf{14.6667}$ ($p = 1.28 \times 10^{-4}$)
  - Exact Binomial two-sided test: $\mathbf{p = 6.62 \times 10^{-5}}$ ($p < 0.0001$)
  - *Result:* Rejects the null hypothesis of equal error rates with high statistical significance ($\alpha = 0.001$).
- **10,000-Round Empirical Bootstrap 95% Confidence Intervals:**
  - $\Delta \text{Accuracy} = \text{Acc}_{\text{FT}} - \text{Acc}_{\text{Base}}$: Point estimate **$+11.50\%$**, $95\%\text{ CI } = [\mathbf{+6.00\%}, \mathbf{+17.00\%}]$ (strictly positive, excludes zero).
  - $\Delta \text{Macro F1} = \text{F1}_{\text{FT}} - \text{F1}_{\text{Base}}$: Point estimate **$+0.1212$**, $95\%\text{ CI } = [\mathbf{+0.0679}, \mathbf{+0.1789}]$ (strictly positive, excludes zero).

---

## Architectural & Engineering Evolution

### Why the Initial CNN Failed on Certain Van Gogh Paintings
During initial diagnostic analysis of `Vincent_van_Gogh_266.jpg` (*Weaver at the Loom*, an 1884 monochrome/sepia study from Van Gogh's early Dutch period in Nuenen), the baseline CNN incorrectly attributed it to Francisco Goya with 67.2% softmax score:
1. **Intra-Class Stylistic Variance:** Van Gogh's oeuvre contains two starkly distinct visual modes: early sombre Dutch realism / monochrome charcoal studies (1881–1885) and late chromatic French Post-Impressionism with vibrant impasto (1886–1890).
2. **Frozen Backbone Bottleneck:** When the ResNet-18 convolutional backbone is 100% frozen, the single linear classification head ($5,130$ weights) must find a single linear hyperplane. It associates Van Gogh primarily with high-chroma Post-Impressionist signatures and misprojects low-chroma chiaroscuro interior studies into Goya's dark Romanticism cluster.

### The Fine-Tuning Solution
1. **Selective Layer Freezing:** Keeps `conv1`, `bn1`, and residual stages `layer1`, `layer2`, and `layer3` frozen ($2.78\text{ M}$ parameters) to preserve generic low-level edge, contour, and corner primitives.
2. **Convolutional Unfreezing:** Unfreezes `layer4` ($8.39\text{ M}$ parameters) alongside the 10-class linear head. This allows higher-level receptive fields to adapt from ImageNet natural objects to domain-specific artistic brushstrokes, line-work, and canvas textures.
3. **Differential Learning Rates:**
   - `layer4`: $\mathbf{1 \times 10^{-4}}$ (conservative rate preventing catastrophic forgetting).
   - classification head (`fc`): $\mathbf{1 \times 10^{-3}}$ (rapid convergence on class boundaries).
4. **Regularization & Schedule:** `AdamW` (weight decay $= 10^{-4}$) with `CosineAnnealingLR` schedule and validation early stopping ($patience=4$).
5. **Conservative Augmentation:** `RandomResizedCrop(scale=(0.85, 1.0))`, horizontal flipping, and mild color jitter ($0.08$) ensure composition and tonal characteristics are not distorted.

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
│   ├── cnn_best.pth                 # Promoted Fine-Tuned ResNet-18 production checkpoint (42.78 MB)
│   └── experiments/
│       └── cnn_baseline_head_only.pth # Archived baseline ResNet-18 frozen-backbone checkpoint (42.77 MB)
├── outputs/
│   ├── model_comparison.json        # Production held-out test benchmark metrics manifest
│   ├── svm_vs_cnn_confusion_matrices.png # Side-by-side normalized test confusion matrices
│   ├── per_artist_f1_comparison.png # Per-artist F1-score comparison chart
│   ├── per_class_metrics_comparison.csv # Production class-by-class metrics table
│   └── experiments/
│       ├── test_evaluation_comparison.json   # Full paired statistical test report (McNemar & Bootstrap)
│       ├── test_confusion_matrices.png       # Test confusion matrices (Baseline vs. Fine-Tuned)
│       ├── test_per_artist_f1_comparison.png # Test F1 bar chart (Baseline vs. Fine-Tuned)
│       ├── cnn_finetune_val_metrics.json     # Validation metrics for fine-tuned checkpoint
│       ├── cnn_finetune_confusion_matrix.png # Validation confusion matrix
│       └── cnn_finetune_learning_curves.png  # Training and validation loss/accuracy curves
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
│   │   ├── train_cnn.py             # Transfer learning baseline training loop (frozen backbone)
│   │   ├── train_cnn_finetune.py    # Layer 4 differential fine-tuning pipeline
│   │   ├── evaluate_comparison.py   # Production held-out test evaluation
│   │   └── evaluate_test_finetune.py # Dedicated comparative evaluation with McNemar & Bootstrap testing
│   └── utils/
│       ├── __init__.py              # Subpackage marker
│       └── evaluate.py              # Metrics computation, timing, and plot generation helpers
├── tests/
│   ├── __init__.py                  # Subpackage marker
│   ├── test_make_dataset.py         # Unit tests for image verification and stratified splitting
│   ├── test_svm.py                  # Unit tests for feature extraction and SVM pipeline
│   ├── test_cnn.py                  # Unit tests for ResNet-18 architecture and frozen backbone
│   ├── test_finetune.py             # Unit tests for Layer 4 gradient unfreezing and checkpoint parity
│   └── test_app.py                  # Integration tests for Streamlit caching, inference, and benchmarks
├── .gitignore                       # Git exclusion rules (data, virtualenvs, caches, logs, PDFs)
├── requirements.txt                 # Project dependency specifications
├── setup.cmd                        # Automated Windows environment and dependency setup script
├── run-demo.cmd                     # Windows launch script for the Streamlit demo application
├── run-tests.cmd                    # Windows test runner script with status code propagation
└── README.md                        # Comprehensive project documentation
```

---

## Full Experiment Reproduction (Optional Workflow)

To reproduce the machine learning pipelines from scratch using the local dataset:

### 1. Dataset Acquisition (If Not Present)
If downloading raw images:
- **Kaggle CLI:**
  ```cmd
  kaggle datasets download -d ikarus777/best-artworks-of-all-time -p data/downloads --unzip
  ```
- Copy the 10 target artist folders from `data/downloads/resized/resized/` into `data/raw/`.

### 2. Verify and Split the Dataset
```cmd
.venv\Scripts\python.exe src/data/make_dataset.py
```
Generates `data/processed/dataset_manifest.csv` with the fixed 80/10/10 split.

### 3. Train the Classical SVM Baseline
```cmd
.venv\Scripts\python.exe src/models/train_svm.py
```
Outputs `models/svm_baseline.joblib` and `outputs/svm_confusion_matrix.png`.

### 4. Train the Baseline CNN (Head Only)
```cmd
.venv\Scripts\python.exe src/models/train_cnn.py
```
Trains the frozen-backbone linear probe, saving checkpoints to `models/cnn_best.pth`.

### 5. Run Layer 4 Fine-Tuning
```cmd
.venv\Scripts\python.exe src/models/train_cnn_finetune.py
```
Starts from `models/experiments/cnn_baseline_head_only.pth`, trains `layer4` and `fc` with differential learning rates, and writes experimental candidate checkpoints strictly under `models/experiments/` without modifying production checkpoints.

### 6. Run Comparative Held-Out Evaluation
```cmd
.venv\Scripts\python.exe src/models/evaluate_test_finetune.py
```
Computes paired test predictions, McNemar tests, bootstrap confidence intervals, and side-by-side matrices under `outputs/experiments/`.

---

## Important Scope, Authenticity, and Academic Disclaimers

1. **Non-Authentication Disclaimer:** **This model does not establish the historical authenticity, provenance, or physical authorship of any artwork.** It is a closed-world, 10-class visual pattern classifier trained on digitized museum reproductions. Given an input image, it outputs normalized similarity scores across the 10 pre-defined artist classes.
2. **Generalization Scope:** The measured 82.0% held-out test accuracy applies specifically to the 200 held-out test paintings from the *Best Artworks of All Time* distribution. It does not imply universal 82% accuracy on outside datasets, unfamiliar art movements, private collections, or non-painting mediums.
3. **Statistical Sample Size:** With 20 test paintings per artist, individual class metrics carry an inherent margin of error ($\pm 5\%$ per painting). While aggregate statistical tests (McNemar $p < 0.0001$, Bootstrap 95% CI $[+6.0\%, +17.0\%]$) demonstrate that the fine-tuning improvement is statistically robust, individual predictions should be interpreted as probabilistic machine indications rather than definitive attributions.
4. **Hardware-Dependent Latency:** CPU inference latencies (mean $\approx 39.6\text{ ms}$ for CNN, $\approx 18.5\text{ ms}$ for SVM) were measured on a modern multi-core x86-64 CPU and will vary across different processors and background operating system loads.

---

## Dataset Attribution & Licensing

- **Dataset Name:** Best Artworks of All Time
- **Curator:** `ikarus777` on Kaggle
- **Source:** [Kaggle Dataset: ikarus777/best-artworks-of-all-time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
- **Dataset License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Target Artists Included:**
  Albrecht Dürer, Alfred Sisley, Edgar Degas, Francisco Goya, Pablo Picasso, Paul Gauguin, Pierre-Auguste Renoir, Rembrandt, Titian, Vincent van Gogh.
- **Licensing Context:** The underlying historical paintings were created between the 15th and early 20th centuries. The digitized reproductions in this collection are shared under CC BY-NC-SA 4.0 for non-commercial research and educational purposes. It remains an unsettled legal question whether trained machine learning weights qualify as adapted material / derivative works under CC BY-NC-SA 4.0 or non-copyrightable statistical parameters. Consequently, retaining these model checkpoints in this repository is intended strictly for private academic course evaluation, and any future public or commercial redistribution requires separate licensing review.
