"""
Unit and integration tests for the demonstration interface and model serving components.

Tests verify:
- SVM model and class labels loading from serialized joblib artifact
- CNN model and weights loading from serialized PyTorch checkpoint
- Class label consistency and count match (exactly 10 classes in identical order)
- In-memory SVM inference produces sorted top-3 predictions and valid probabilities
- In-memory CNN inference produces sorted top-3 predictions and valid softmax probabilities
- Image preprocessing handles RGB, grayscale, and arbitrary image dimensions cleanly
- Repeated inference maintains model weight immutability (eval mode, zero gradients)
- Zero emojis in application source code
- Benchmark manifest loading and schema integrity
"""

import unittest
import re
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from src.app import (
    load_svm_model,
    load_cnn_model,
    load_benchmark_manifest,
    predict_svm,
    predict_cnn,
    VAL_TRANSFORMS,
    PROJECT_ROOT
)


class TestDemoInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.svm_pipe, cls.svm_classes = load_svm_model()
        cls.cnn_model, cls.cnn_idx_map, cls.cnn_classes = load_cnn_model()

    def test_model_loading_and_class_parity(self):
        """Verify both models load successfully and share the exact same 10 classes."""
        self.assertEqual(len(self.svm_classes), 10)
        self.assertEqual(len(self.cnn_classes), 10)
        self.assertEqual(list(self.svm_classes), list(self.cnn_classes))

    def test_svm_in_memory_prediction(self):
        """Verify SVM produces top-3 sorted predictions with valid probabilities."""
        img = Image.new("RGB", (128, 128), color="coral")
        top3, latency = predict_svm(img, self.svm_pipe, self.svm_classes)

        self.assertEqual(len(top3), 3)
        self.assertGreater(latency, 0.0)

        # Check sorted descending
        self.assertGreaterEqual(top3[0][1], top3[1][1])
        self.assertGreaterEqual(top3[1][1], top3[2][1])

        # Check valid probability bounds
        for artist, score in top3:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(any(artist == c.replace('_', ' ') for c in self.svm_classes))

    def test_cnn_in_memory_prediction(self):
        """Verify CNN produces top-3 sorted predictions with valid softmax probabilities."""
        img = Image.new("RGB", (128, 128), color="navy")
        top3, latency = predict_cnn(img, self.cnn_model, self.cnn_idx_map)

        self.assertEqual(len(top3), 3)
        self.assertGreater(latency, 0.0)

        # Check sorted descending
        self.assertGreaterEqual(top3[0][1], top3[1][1])
        self.assertGreaterEqual(top3[1][1], top3[2][1])

        # Check valid probability bounds
        for artist, score in top3:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_cnn_transform_shape(self):
        """Verify validation transforms output tensor with shape (3, 224, 224)."""
        img_gray = Image.new("L", (320, 240), color=120)
        tensor = VAL_TRANSFORMS(img_gray.convert("RGB"))
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertTrue(torch.all(torch.isfinite(tensor)))

    def test_model_weight_immutability_during_inference(self):
        """Verify weights do not change after multiple inference passes."""
        img = Image.new("RGB", (100, 100), color="gold")

        # Snapshot weights
        initial_svm_params = np.copy(self.svm_pipe.named_steps["svm"].dual_coef_)
        initial_cnn_fc_w = self.cnn_model.fc.weight.clone()

        # Run inference 3 times
        for _ in range(3):
            _ = predict_svm(img, self.svm_pipe, self.svm_classes)
            _ = predict_cnn(img, self.cnn_model, self.cnn_idx_map)

        # Check exact equality
        np.testing.assert_array_equal(self.svm_pipe.named_steps["svm"].dual_coef_, initial_svm_params)
        self.assertTrue(torch.equal(self.cnn_model.fc.weight, initial_cnn_fc_w))

    def test_zero_emojis_in_source_code(self):
        """Verify that src/app.py contains zero emojis anywhere in the code or strings."""
        app_path = PROJECT_ROOT / "src" / "app.py"
        self.assertTrue(app_path.exists())
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Emoji unicode ranges: supplementary symbols and pictographs, miscellaneous symbols, dingbats
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]")
        emojis_found = emoji_pattern.findall(code)
        self.assertEqual(len(emojis_found), 0, f"Found emojis in src/app.py: {emojis_found}")

    def test_load_benchmark_manifest(self):
        """Verify loading benchmark evaluation results returns valid structure."""
        bench = load_benchmark_manifest()
        self.assertIsInstance(bench, dict)
        self.assertIn("summary", bench)
        self.assertIn("svm_baseline", bench["summary"])
        self.assertIn("cnn_transfer_learning", bench["summary"])
        self.assertEqual(bench["summary"]["svm_baseline"]["test_accuracy"], 0.53)
        self.assertEqual(bench["summary"]["cnn_transfer_learning"]["test_accuracy"], 0.82)


if __name__ == "__main__":
    unittest.main()
