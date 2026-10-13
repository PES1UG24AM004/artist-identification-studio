"""
Modular Handcrafted Visual Feature Extraction for Artist Identification.

Extracts three complementary visual descriptors from painting images:
1. Color Distribution: 3D HSV Color Histogram (8x8x8 bins = 512 dimensions)
   - Captures color palette, pigment saturation, and tonal distribution.
   - L1-normalized to make the representation invariant to image resolution.
2. Shape Moments: Hu Invariant Moments (7 dimensions)
   - Evaluated on grayscale image to capture global geometric and silhouette properties.
   - Invariant to scale, 2D translation, and rotation. Log-transformed for numerical stability.
3. Texture & Brushstroke Dynamics: Gray-Level Co-occurrence Matrix (GLCM / Haralick)
   - Computed over 2 spatial offsets (1, 3 pixels) and 4 directional angles (0, 45, 90, 135 deg).
   - 5 statistical properties: contrast, dissimilarity, homogeneity, energy, correlation.
   - Total GLCM dimensions = 5 properties * 2 distances * 4 angles = 40 dimensions.

Total concatenated feature vector dimension = 512 + 7 + 40 = 559 dimensions.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from PIL import Image


FEATURE_NAMES = (
    [f"hsv_hist_{i:03d}" for i in range(512)] +
    [f"hu_moment_{i}" for i in range(7)] +
    [f"glcm_{prop}_d{d}_a{a}" for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
     for d in [1, 3] for a in [0, 45, 90, 135]]
)
TOTAL_FEATURE_DIM = 559


def extract_color_histogram(img_rgb: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """
    Compute 3D HSV color histogram normalized by L1 norm.
    Returns 1D numpy array of size 512.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [img_hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(bins),
        ranges=[0, 180, 0, 256, 0, 256]
    )
    # L1 normalization so histogram sums to 1.0
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


def extract_hu_moments(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute 7 Hu invariant shape moments with logarithmic scaling.
    Returns 1D numpy array of size 7.
    """
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform to manage exponential dynamic range: -sign(hu) * log10(abs(hu))
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    # Replace any NaNs or Infs if image is completely uniform
    hu_log = np.nan_to_num(hu_log, nan=0.0, posinf=0.0, neginf=0.0)
    return hu_log.astype(np.float32)


def extract_glcm_texture(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM) texture descriptors.
    Quantizes grayscale image into 16 bins to ensure fast, robust texture analysis.
    Distances: [1, 3], Angles: [0, pi/4, pi/2, 3pi/4].
    Properties: contrast, dissimilarity, homogeneity, energy, correlation.
    Returns 1D numpy array of size 40.
    """
    # Quantize from 256 levels down to 16 levels
    gray_quantized = (img_gray // 16).astype(np.uint8)
    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16,
        symmetric=True,
        normed=True
    )
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats = []
    for prop in properties:
        val = graycoprops(glcm, prop).flatten()
        feats.extend(val)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(feats, dtype=np.float32)


def extract_image_features(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Extract full 559-dimensional feature vector from an image path, PIL image, or numpy array.
    Guarantees deterministic, finite float output.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        with Image.open(img_path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"))
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input
        else:
            raise ValueError(f"Invalid image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Grayscale representation for moments and texture
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Extract 3 feature components
    color_hist = extract_color_histogram(img_rgb)
    hu_moments = extract_hu_moments(img_gray)
    glcm_texture = extract_glcm_texture(img_gray)

    # Concatenate into unified feature vector
    feat_vec = np.concatenate([color_hist, hu_moments, glcm_texture])

    if len(feat_vec) != TOTAL_FEATURE_DIM:
        raise ValueError(f"Feature vector dimension mismatch: expected {TOTAL_FEATURE_DIM}, got {len(feat_vec)}")

    # Ensure all values are finite
    if not np.all(np.isfinite(feat_vec)):
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

    return feat_vec
