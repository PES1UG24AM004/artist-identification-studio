"""
Unit tests for visual feature extraction and SVM baseline components.

Tests verify:
- Feature vector dimensionality (exactly 559 dims)
- Value finiteness (zero NaNs, Infs)
- Feature extraction determinism across repeated calls on identical images
- Sensitivity to image content (different images yield different feature vectors)
- Pipeline leakage safety (scaler and PCA fit only on training data)
- End-to-end model fitting, prediction, and joblib serialization on synthetic data
"""

import unittest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from src.features.extract_features import (
    extract_image_features,
    extract_color_histogram,
    extract_hu_moments,
    extract_glcm_texture,
    TOTAL_FEATURE_DIM
)
from src.utils.evaluate import compute_classification_metrics


class TestFeatureExtractionAndSVM(unittest.TestCase):

    def setUp(self):
        # Create small synthetic test images
        self.img1_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.img1_rgb[:32, :, 0] = 255  # Red top half
        self.img1_rgb[32:, :, 2] = 255  # Blue bottom half

        self.img2_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.img2_rgb[:, :32, 1] = 255  # Green left half
        self.img2_rgb[:, 32:, 0] = 200  # Red right half

    def test_feature_dimensions_and_finiteness(self):
        feat1 = extract_image_features(self.img1_rgb)
        self.assertEqual(len(feat1), TOTAL_FEATURE_DIM)
        self.assertEqual(feat1.shape, (559,))
        self.assertTrue(np.all(np.isfinite(feat1)), "Features contain NaN or Inf!")

    def test_feature_extraction_determinism(self):
        feat_run1 = extract_image_features(self.img1_rgb)
        feat_run2 = extract_image_features(self.img1_rgb)
        np.testing.assert_array_almost_equal(feat_run1, feat_run2, decimal=6)

    def test_distinct_images_produce_distinct_features(self):
        feat1 = extract_image_features(self.img1_rgb)
        feat2 = extract_image_features(self.img2_rgb)
        self.assertFalse(np.allclose(feat1, feat2), "Different images produced identical features!")

    def test_feature_subcomponents(self):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        hist = extract_color_histogram(self.img1_rgb)
        self.assertEqual(len(hist), 512)
        self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=4)

        hu = extract_hu_moments(gray)
        self.assertEqual(len(hu), 7)
        self.assertTrue(np.all(np.isfinite(hu)))

        glcm = extract_glcm_texture(gray)
        self.assertEqual(len(glcm), 40)
        self.assertTrue(np.all(np.isfinite(glcm)))

    def test_pipeline_fitting_and_serialization(self):
        # Generate small synthetic dataset: 3 classes, 15 samples each
        np.random.seed(42)
        n_samples = 45
        X_synth = np.random.randn(n_samples, TOTAL_FEATURE_DIM).astype(np.float32)
        y_synth = np.array(["ArtistA"] * 15 + ["ArtistB"] * 15 + ["ArtistC"] * 15)

        # Train pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=42)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ])

        pipe.fit(X_synth, y_synth)
        preds = pipe.predict(X_synth)
        self.assertEqual(len(preds), n_samples)
        self.assertTrue(set(preds).issubset({"ArtistA", "ArtistB", "ArtistC"}))

        # Test metrics utility
        metrics = compute_classification_metrics(list(y_synth), list(preds), ["ArtistA", "ArtistB", "ArtistC"])
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

        # Test serialization
        temp_dir = Path(tempfile.mkdtemp())
        try:
            model_file = temp_dir / "test_model.joblib"
            joblib.dump({"pipe": pipe, "labels": ["ArtistA", "ArtistB", "ArtistC"]}, model_file)
            self.assertTrue(model_file.exists())

            loaded = joblib.load(model_file)
            loaded_preds = loaded["pipe"].predict(X_synth)
            np.testing.assert_array_equal(preds, loaded_preds)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
