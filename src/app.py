"""
Artist Identification Studio - Interactive Demonstration Interface.

Streamlit web application for demonstrating and comparing:
1. Classical Machine Learning: SVM Classifier with Handcrafted Visual Descriptors
2. Deep Learning: ResNet-18 ImageNet Transfer Learning Model

Course: UE24CS352A - Machine Learning
Team:
  - ANANYA RAGHAVENDRA (PES1UG24AM037)
  - AARAV YUVAL B G (PES1UG24AM004)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import joblib

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Torch imports
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import streamlit as st

from src.features.extract_features import extract_image_features


# --- Page Configuration (Wide Layout, Zero Emojis) ---
st.set_page_config(
    page_title="Artist Identification Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Theme & Clean Presentation Layer (Pure Dark Mode) ---
def inject_custom_styles() -> None:
    """Inject minimal, academic dark theme CSS variables and component styles."""
    css = """
    <style>
        :root {
            --bg-main: #111214;
            --bg-surface: #17181B;
            --bg-content: #1D1F23;
            --bg-elevated: #24262B;
            --border-color: #34363D;
            --text-primary: #F3F3F4;
            --text-secondary: #B0B2B8;
            --text-muted: #898C94;
            --accent: #D9DADF;
        }

        /* Base typography */
        html, body {
            font-family: Inter, "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, sans-serif !important;
            color: var(--text-primary);
            background-color: var(--bg-main);
        }

        /* App container backgrounds */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stHeader"] {
            background-color: var(--bg-main) !important;
            border-bottom: 1px solid var(--border-color);
        }

        /* Sidebar surface */
        [data-testid="stSidebar"] {
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        /* Main content area expansion to 1150-1350px centered on desktop */
        [data-testid="stMain"],
        section.main {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewBlockContainer"] {
            max-width: 1280px !important;
            width: 100% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
        }

        /* Ensure inner layout blocks expand to 100% of the centered container */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stTabs"],
        [data-testid="stTabContent"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Header typography */
        .studio-main-title {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }

        .studio-subtitle {
            font-size: 15.5px;
            color: var(--text-secondary);
            margin-bottom: 1.15rem;
            line-height: 1.5;
        }

        /* Academic research disclaimer banner */
        .academic-disclaimer {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1.5rem;
            font-size: 14.5px;
            line-height: 1.55;
            color: var(--text-secondary);
        }

        /* Section headings */
        .studio-section-title {
            font-size: 21px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
            margin-bottom: 0.75rem;
            padding-bottom: 0.25rem;
        }

        .studio-subsection-title {
            font-size: 17px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }

        /* Disable automatic heading anchor link icons */
        [data-testid="stHeaderActionElements"],
        .anchor-link,
        a.header-anchor {
            display: none !important;
            visibility: hidden !important;
        }

        /* Tab navigation */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.75rem;
            border-bottom: 1px solid var(--border-color) !important;
            background-color: transparent !important;
            padding-bottom: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border: none !important;
            color: var(--text-muted) !important;
            font-size: 15px !important;
            font-weight: 500 !important;
            padding: 0.5rem 0.25rem !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            border-bottom: 2px solid var(--text-primary) !important;
        }

        /* Form input direct labels */
        .stRadio > label, .stSelectbox > label, .stFileUploader > label {
            font-size: 15px !important;
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            margin-bottom: 0.35rem !important;
        }

        /* Native file uploader dropzone styling */
        [data-testid="stFileUploadDropzone"] {
            background-color: var(--bg-content) !important;
            border: 1px dashed var(--border-color) !important;
            border-radius: 6px !important;
            padding: 1.25rem 1rem !important;
        }

        [data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--accent) !important;
        }

        /* Secondary buttons */
        [data-testid="stBaseButton-secondary"] {
            background-color: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 4px !important;
            font-size: 14px !important;
        }

        [data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--accent) !important;
        }

        /* Painting preview image container constraints */
        [data-testid="stImage"] img {
            max-height: 400px !important;
            width: auto !important;
            object-fit: contain !important;
            border-radius: 4px !important;
            margin: 0 auto !important;
            display: block !important;
        }

        /* Painting empty state container */
        .preview-empty-state {
            background-color: var(--bg-content);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            min-height: 310px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2rem 1.5rem;
        }

        .empty-state-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .empty-state-text {
            font-size: 14px;
            color: var(--text-muted);
            max-width: 340px;
            line-height: 1.55;
        }

        /* Card elements inside native containers */
        .predicted-artist-name {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.25;
            margin-bottom: 0.25rem;
        }

        .top-confidence-line {
            font-size: 15px;
            color: var(--text-secondary);
            margin-bottom: 1.15rem;
        }

        .distribution-header {
            font-size: 13.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.65rem;
        }

        /* Native progress bar styling */
        .stProgress > div > div > div > div {
            background-color: var(--accent) !important;
        }

        /* Synthesis observation banner */
        .synthesis-banner {
            background-color: var(--bg-content);
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
            padding: 0.85rem 1.15rem;
            font-size: 14.5px;
            line-height: 1.55;
            color: var(--text-secondary);
            margin-top: 1rem;
        }

        /* Streamlit metric overrides */
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 26px !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 13.5px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Table styling */
        [data-testid="stTable"] table {
            background-color: var(--bg-content) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            width: 100% !important;
        }

        [data-testid="stTable"] th {
            background-color: var(--bg-surface) !important;
            color: var(--text-muted) !important;
            font-size: 13px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            border-bottom: 1px solid var(--border-color) !important;
            padding: 0.65rem 0.85rem !important;
        }

        [data-testid="stTable"] td {
            border-bottom: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            font-size: 14.5px !important;
            padding: 0.65rem 0.85rem !important;
        }

        /* Sidebar metadata */
        .sidebar-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
        }

        .sidebar-meta {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .sidebar-meta-item {
            margin-bottom: 0.75rem;
        }

        /* Horizontal rule */
        hr {
            border-color: var(--border-color) !important;
            margin: 1.5rem 0 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --- Model Caching ---
@st.cache_resource
def load_svm_model() -> Tuple[Any, List[str]]:
    """Load serialized SVM baseline pipeline without retraining."""
    model_path = PROJECT_ROOT / "models" / "svm_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found at {model_path}")
    data = joblib.load(model_path)
    return data["model_pipeline"], [str(c) for c in data["class_labels"]]


@st.cache_resource
def load_cnn_model() -> Tuple[torch.nn.Module, Dict[int, str], List[str]]:
    """Load ResNet-18 transfer learning checkpoint without re-downloading pretrained weights."""
    model_path = PROJECT_ROOT / "models" / "cnn_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_label = checkpoint["idx_to_label"]
    class_labels = checkpoint["class_labels"]
    return model, idx_to_label, class_labels


@st.cache_data
def load_benchmark_manifest() -> Dict[str, Any]:
    """Load verified evaluation results from model_comparison.json."""
    manifest_p = PROJECT_ROOT / "outputs" / "model_comparison.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Standard ImageNet transform for CNN validation & inference
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_svm(img: Image.Image, pipeline: Any, classes: List[str]) -> Tuple[List[Tuple[str, float]], float]:
    """Run SVM prediction on in-memory PIL image."""
    t0 = time.perf_counter()
    feat = extract_image_features(img)
    proba = pipeline.predict_proba(feat.reshape(1, -1))[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    top_indices = np.argsort(proba)[::-1][:3]
    top3 = [(classes[i].replace('_', ' '), float(proba[i])) for i in top_indices]
    return top3, latency_ms


def predict_cnn(img: Image.Image, model: torch.nn.Module, idx_to_label: Dict[int, str]) -> Tuple[List[Tuple[str, float]], float]:
    """Run CNN prediction on in-memory PIL image."""
    t0 = time.perf_counter()
    tensor = VAL_TRANSFORMS(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    latency_ms = (time.perf_counter() - t0) * 1000

    top_indices = np.argsort(probs)[::-1][:3]
    top3 = [(idx_to_label[i].replace('_', ' '), float(probs[i])) for i in top_indices]
    return top3, latency_ms


def render_attribution_panel(
    model_tag: str,
    top_predictions: List[Tuple[str, float]],
    latency_ms: float,
    architecture_summary: str,
    score_label: str = "Confidence Score",
    calibration_note: str = ""
) -> None:
    """Render attribution results using clean native Streamlit components without raw HTML markup."""
    top_artist, top_score = top_predictions[0]

    with st.container(border=True):
        st.caption(model_tag.upper())
        st.markdown(f'<div class="predicted-artist-name">{top_artist}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="top-confidence-line">{score_label}: <strong>{top_score*100:.1f}%</strong></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="distribution-header">Top 3 Candidates</div>', unsafe_allow_html=True)
        for rank, (artist, score) in enumerate(top_predictions, 1):
            pct = score * 100
            col_name, col_pct = st.columns([3, 1])
            with col_name:
                st.write(f"**{rank}. {artist}**")
            with col_pct:
                st.markdown(
                    f'<div style="text-align: right; font-family: monospace; font-size: 14px; color: #B0B2B8;">{pct:.1f}%</div>',
                    unsafe_allow_html=True
                )
            st.progress(float(np.clip(score, 0.0, 1.0)))

        st.divider()
        col_lat_label, col_lat_val = st.columns([2, 1])
        with col_lat_label:
            st.caption("Inference Latency (CPU)")
        with col_lat_val:
            st.markdown(
                f'<div style="text-align: right; font-family: monospace; font-weight: 600; font-size: 14px; color: #F3F3F4;">{latency_ms:.2f} ms</div>',
                unsafe_allow_html=True
            )

        st.caption(architecture_summary)
        if calibration_note:
            st.caption(calibration_note)


# Curated demo validation samples across all 10 artists
SAMPLE_OPTIONS: Dict[str, str] = {
    "Albrecht Durer - Woodcut / Engraving": "data/raw/Albrecht_Durer/Albrecht_Durer_101.jpg",
    "Alfred Sisley - Impressionist Landscape": "data/raw/Alfred_Sisley/Alfred_Sisley_39.jpg",
    "Edgar Degas - Ballet / Figure Study": "data/raw/Edgar_Degas/Edgar_Degas_147.jpg",
    "Francisco Goya - Dark Romanticism": "data/raw/Francisco_Goya/Francisco_Goya_21.jpg",
    "Pablo Picasso - Cubist / Modernist": "data/raw/Pablo_Picasso/Pablo_Picasso_10.jpg",
    "Paul Gauguin - Post-Impressionist Synthesis": "data/raw/Paul_Gauguin/Paul_Gauguin_123.jpg",
    "Pierre-Auguste Renoir - Impressionist Portrait": "data/raw/Pierre-Auguste_Renoir/Pierre-Auguste_Renoir_136.jpg",
    "Rembrandt - Baroque Chiaroscuro": "data/raw/Rembrandt/Rembrandt_250.jpg",
    "Titian - High Renaissance Colorito": "data/raw/Titian/Titian_42.jpg",
    "Vincent van Gogh - Expressive Impasto": "data/raw/Vincent_van_Gogh/Vincent_van_Gogh_183.jpg"
}


# --- Main Application ---
def main():
    # Inject dark appearance styling
    inject_custom_styles()

    # Header
    st.markdown('<div class="studio-main-title">Artist Identification Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="studio-subtitle">Comparing traditional machine learning and deep learning for painting attribution.</div>', unsafe_allow_html=True)

    # Academic Disclaimer
    st.markdown("""
    <div class="academic-disclaimer">
        <strong>Academic Research Prototype:</strong> This interface evaluates visual attribution across 10 historical artist classes
        using an RBF-kernel SVM (HSV Color, Hu Moments, and GLCM Texture descriptors) alongside a fine-tuned ResNet-18 CNN.
        It is intended for comparative educational analysis, not artwork authentication.
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar (Pure Dark Mode, No Appearance Toggle) ---
    st.sidebar.markdown('<div class="sidebar-title">About the Project</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="sidebar-meta">
        <div class="sidebar-meta-item"><strong>Course:</strong> UE24CS352A - Machine Learning</div>
        <div class="sidebar-meta-item"><strong>Team:</strong><br>
            ANANYA RAGHAVENDRA (PES1UG24AM037)<br>
            AARAV YUVAL B G (PES1UG24AM004)
        </div>
        <div class="sidebar-meta-item"><strong>Dataset:</strong> 10 Artists, 2,000 Paintings</div>
        <div class="sidebar-meta-item"><strong>Splits:</strong> 80% Train / 10% Val / 10% Test</div>
        <div class="sidebar-meta-item"><strong>Evaluation:</strong> 200 Held-Out Test Artworks</div>
        <div class="sidebar-meta-item"><strong>Execution:</strong> Local CPU Inference</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Load Models ---
    try:
        svm_pipe, svm_classes = load_svm_model()
        cnn_model, cnn_idx_map, cnn_classes = load_cnn_model()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # --- Main Interface Tabs (Zero Emojis) ---
    tabs = st.tabs(["Attribution Analysis", "Model Benchmark & Evaluation"])

    # ---------------- TAB 1: Attribution Analysis ----------------
    with tabs[0]:
        col_input, col_preview = st.columns([1, 1], gap="large")

        img_to_process: Optional[Image.Image] = None
        source_caption: str = ""
        image_id: Optional[str] = None

        with col_input:
            st.markdown('<div class="studio-section-title">Input Painting</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload painting file (JPG, PNG, WebP):",
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload a painting image file for classification."
            )

            available_samples = {
                name: rel_path
                for name, rel_path in SAMPLE_OPTIONS.items()
                if (PROJECT_ROOT / rel_path).exists()
            }

            if available_samples:
                sample_choice = st.selectbox(
                    "Or select a curated demo painting:",
                    options=["None (Upload custom image)"] + list(available_samples.keys()),
                    index=0
                )
            else:
                st.caption("Curated demo paintings unavailable (dataset not downloaded). Upload a custom painting file above.")
                sample_choice = "None (Upload custom image)"

            if uploaded_file is not None:
                try:
                    img_to_process = Image.open(uploaded_file).convert("RGB")
                    source_caption = f"Uploaded File: {uploaded_file.name} ({img_to_process.width} x {img_to_process.height} px)"
                    image_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
                except Exception:
                    st.error("Unable to decode uploaded image. Please provide a standard JPEG or PNG file.")
            elif sample_choice != "None (Upload custom image)":
                sample_rel = available_samples.get(sample_choice)
                if sample_rel:
                    sample_p = PROJECT_ROOT / sample_rel
                    if sample_p.exists():
                        img_to_process = Image.open(sample_p).convert("RGB")
                        source_caption = f"Demo Sample: {sample_choice} ({img_to_process.width} x {img_to_process.height} px)"
                        image_id = f"sample_{sample_rel}"

            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            model_mode = st.radio(
                "Classifier Architecture:",
                options=[
                    "Compare Both (Side-by-Side)",
                    "Deep Learning (ResNet-18)",
                    "Traditional ML (SVM Baseline)"
                ],
                index=0
            )

        with col_preview:
            st.markdown('<div class="studio-section-title">Painting Preview</div>', unsafe_allow_html=True)

            if img_to_process is not None:
                st.image(img_to_process, caption=source_caption, width="stretch")
            else:
                st.markdown("""
                <div class="preview-empty-state">
                    <div class="empty-state-title">No Painting Selected</div>
                    <div class="empty-state-text">Drag and drop a painting file on the left or select a curated demo painting to begin attribution analysis.</div>
                </div>
                """, unsafe_allow_html=True)

        # Execution and Results Display
        if img_to_process is not None and image_id is not None:
            # Cache predictions in session state
            if st.session_state.get("cached_image_id") != image_id:
                with st.spinner("Executing models on CPU..."):
                    svm_top3, svm_latency = predict_svm(img_to_process, svm_pipe, svm_classes)
                    cnn_top3, cnn_latency = predict_cnn(img_to_process, cnn_model, cnn_idx_map)
                    st.session_state["cached_image_id"] = image_id
                    st.session_state["cached_svm"] = (svm_top3, svm_latency)
                    st.session_state["cached_cnn"] = (cnn_top3, cnn_latency)
            else:
                svm_top3, svm_latency = st.session_state["cached_svm"]
                cnn_top3, cnn_latency = st.session_state["cached_cnn"]

            st.markdown('<div style="margin-top: 1.75rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="studio-section-title">Attribution Results</div>', unsafe_allow_html=True)

            if model_mode == "Compare Both (Side-by-Side)":
                res_col1, res_col2 = st.columns(2, gap="large")

                with res_col1:
                    render_attribution_panel(
                        model_tag="Traditional ML - SVM Baseline",
                        top_predictions=svm_top3,
                        latency_ms=svm_latency,
                        architecture_summary="559-dim visual descriptors (HSV Hist, Hu Moments, GLCM Texture) standardized, reduced to PCA(100), classified with RBF SVM.",
                        score_label="Decision Confidence (Calibrated)",
                        calibration_note="Note: SVM scores reflect calibrated Platt decision confidence."
                    )

                with res_col2:
                    render_attribution_panel(
                        model_tag="Deep Learning - ResNet-18",
                        top_predictions=cnn_top3,
                        latency_ms=cnn_latency,
                        architecture_summary="ImageNet-pretrained ResNet-18 backbone (frozen) fine-tuned on painting dataset with 10-class linear classification head.",
                        score_label="Softmax Model Score",
                        calibration_note="Note: Softmax output reflects relative normalized logits, not uncalibrated real-world probabilities."
                    )

                # Analytical synthesis note
                if svm_top3[0][0] == cnn_top3[0][0]:
                    st.markdown(f"""
                    <div class="synthesis-banner">
                        <strong>Consensus Attribution:</strong> Both models independently attribute this artwork to
                        <strong>{svm_top3[0][0]}</strong> (SVM: {svm_top3[0][1]*100:.1f}%, CNN: {cnn_top3[0][1]*100:.1f}%).
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="synthesis-banner">
                        <strong>Model Divergence:</strong> SVM attributes this artwork to <strong>{svm_top3[0][0]}</strong> ({svm_top3[0][1]*100:.1f}%)
                        while ResNet-18 attributes it to <strong>{cnn_top3[0][0]}</strong> ({cnn_top3[0][1]*100:.1f}%).
                        This divergence illustrates the operational contrast between global color/texture distributions
                        and deep hierarchical spatial representations.
                    </div>
                    """, unsafe_allow_html=True)

            elif model_mode == "Traditional ML (SVM Baseline)":
                render_attribution_panel(
                    model_tag="Traditional ML - SVM Baseline",
                    top_predictions=svm_top3,
                    latency_ms=svm_latency,
                    architecture_summary="559-dim visual descriptors (HSV Hist, Hu Moments, GLCM Texture) standardized, reduced to PCA(100), classified with RBF SVM.",
                    score_label="Decision Confidence (Calibrated)",
                    calibration_note="Note: SVM scores reflect calibrated Platt decision confidence."
                )

            else:  # Deep Learning Single View
                render_attribution_panel(
                    model_tag="Deep Learning - ResNet-18",
                    top_predictions=cnn_top3,
                    latency_ms=cnn_latency,
                    architecture_summary="ImageNet-pretrained ResNet-18 backbone (frozen) fine-tuned on painting dataset with 10-class linear classification head.",
                    score_label="Softmax Model Score",
                    calibration_note="Note: Softmax output reflects relative normalized logits, not uncalibrated real-world probabilities."
                )

    # ---------------- TAB 2: Benchmarks & Evaluation ----------------
    with tabs[1]:
        st.markdown('<div class="studio-section-title">Held-Out Test Set Evaluation (N=200)</div>', unsafe_allow_html=True)
        st.markdown("""
        Both models were evaluated under strict experimental protocol on **200 held-out test paintings**
        (20 images per artist across 10 balanced classes) that were withheld during training and tuning.
        """)

        bench_data = load_benchmark_manifest()
        if bench_data and "summary" in bench_data:
            s_svm = bench_data["summary"]["svm_baseline"]
            s_cnn = bench_data["summary"]["cnn_transfer_learning"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SVM Test Accuracy", f"{s_svm['test_accuracy']*100:.1f}%")
            m2.metric(
                "CNN Test Accuracy",
                f"{s_cnn['test_accuracy']*100:.1f}%",
                delta=f"+{(s_cnn['test_accuracy']-s_svm['test_accuracy'])*100:.1f}%"
            )
            m3.metric("SVM Macro F1", f"{s_svm['test_macro_f1']:.4f}")
            m4.metric(
                "CNN Macro F1",
                f"{s_cnn['test_macro_f1']:.4f}",
                delta=f"+{s_cnn['test_macro_f1']-s_svm['test_macro_f1']:.4f}"
            )

            st.divider()

            # Side-by-side Confusion Matrix Image
            cm_img_p = PROJECT_ROOT / "outputs" / "svm_vs_cnn_confusion_matrices.png"
            if cm_img_p.exists():
                st.markdown('<div class="studio-subsection-title">Normalized Test Confusion Matrices</div>', unsafe_allow_html=True)
                st.image(
                    str(cm_img_p),
                    caption="Normalized Confusion Matrices: Classical SVM (Left) vs. Deep Learning ResNet-18 (Right)",
                    width="stretch"
                )

            # Per-Artist F1 Comparison Bar Chart
            f1_img_p = PROJECT_ROOT / "outputs" / "per_artist_f1_comparison.png"
            if f1_img_p.exists():
                st.markdown('<div class="studio-subsection-title">Per-Artist F1-Score Comparison</div>', unsafe_allow_html=True)
                st.image(
                    str(f1_img_p),
                    caption="Class-by-Class F1-Score Breakdown on Held-Out Test Data",
                    width="stretch"
                )

            # Detailed Comparison Table
            st.markdown('<div class="studio-subsection-title">Detailed Performance Comparison Table</div>', unsafe_allow_html=True)
            comp_table = [
                {
                    "Metric": "Validation Accuracy (Phase 3/4)",
                    "SVM Baseline": f"{s_svm['val_accuracy']*100:.2f}%",
                    "CNN ResNet-18": f"{s_cnn['val_accuracy']*100:.2f}%",
                    "Delta / Advantage": f"+{(s_cnn['val_accuracy']-s_svm['val_accuracy'])*100:.2f}%"
                },
                {
                    "Metric": "Test Accuracy (Unbiased Phase 5)",
                    "SVM Baseline": f"{s_svm['test_accuracy']*100:.2f}%",
                    "CNN ResNet-18": f"{s_cnn['test_accuracy']*100:.2f}%",
                    "Delta / Advantage": f"+{(s_cnn['test_accuracy']-s_svm['test_accuracy'])*100:.2f}%"
                },
                {
                    "Metric": "Test Macro F1-Score",
                    "SVM Baseline": f"{s_svm['test_macro_f1']:.4f}",
                    "CNN ResNet-18": f"{s_cnn['test_macro_f1']:.4f}",
                    "Delta / Advantage": f"+{s_cnn['test_macro_f1']-s_svm['test_macro_f1']:.4f}"
                },
                {
                    "Metric": "Test Macro Precision",
                    "SVM Baseline": f"{s_svm['test_macro_precision']:.4f}",
                    "CNN ResNet-18": f"{s_cnn['test_macro_precision']:.4f}",
                    "Delta / Advantage": f"+{s_cnn['test_macro_precision']-s_svm['test_macro_precision']:.4f}"
                },
                {
                    "Metric": "Test Macro Recall",
                    "SVM Baseline": f"{s_svm['test_macro_recall']:.4f}",
                    "CNN ResNet-18": f"{s_cnn['test_macro_recall']:.4f}",
                    "Delta / Advantage": f"+{s_cnn['test_macro_recall']-s_svm['test_macro_recall']:.4f}"
                },
                {
                    "Metric": "Mean End-to-End Latency (CPU)",
                    "SVM Baseline": f"{s_svm['mean_latency_ms']:.2f} ms",
                    "CNN ResNet-18": f"{s_cnn['mean_latency_ms']:.2f} ms",
                    "Delta / Advantage": "SVM is 2.57x faster"
                },
                {
                    "Metric": "Median Latency (CPU)",
                    "SVM Baseline": f"{s_svm['median_latency_ms']:.2f} ms",
                    "CNN ResNet-18": f"{s_cnn['median_latency_ms']:.2f} ms",
                    "Delta / Advantage": "SVM is 2.82x faster"
                }
            ]
            st.table(comp_table)

        with st.expander("Methodology & Architectural Details", expanded=False):
            st.markdown(r"""
            - **Classical ML Pipeline:** Extracts a 559-dimensional visual descriptor combining a 3D HSV Color Histogram (512 dimensions), 7 Hu Invariant Shape Moments, and 40 Gray-Level Co-occurrence Matrix (GLCM) texture descriptors. Features are standardized via `StandardScaler`, reduced via `PCA(100)`, and classified using an RBF-kernel Support Vector Machine ($C=1.0$, $\gamma='scale'$).
            - **Deep Learning Pipeline:** Leverages transfer learning on an ImageNet-pretrained `ResNet-18` backbone with frozen convolutional feature extractors, training a 10-class linear classification head using `AdamW` (learning rate $= 10^{-3}$, weight decay $= 10^{-4}$) for 5 epochs on CPU with cross-entropy loss.
            - **Latency vs. Accuracy Tradeoff:** While ResNet-18 outperforms the SVM baseline by **+17.50% test accuracy** and **+0.1782 macro F1**, the classical SVM achieves **2.57x lower mean CPU latency** (18.45 ms vs 47.40 ms), making it suitable for resource-constrained environments where rapid approximate classification is needed.
            """)


if __name__ == "__main__":
    main()
