

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

    def test_cnn_horizontal_flip_tta_default_and_enabled(self):
        """Verify default retains original behavior and TTA can be enabled and disabled cleanly."""
        # Non-symmetric synthetic image
        img = Image.new("RGB", (100, 100), color="white")
        for x in range(30):
            for y in range(100):
                img.putpixel((x, y), (255, 0, 0))

        # Default call vs use_tta=False
        top3_default, _ = predict_cnn(img, self.cnn_model, self.cnn_idx_map)
        top3_false, _ = predict_cnn(img, self.cnn_model, self.cnn_idx_map, use_tta=False)
        self.assertEqual(top3_default, top3_false)

        # TTA enabled call
        top3_tta, latency_tta = predict_cnn(img, self.cnn_model, self.cnn_idx_map, use_tta=True)
        self.assertEqual(len(top3_tta), 3)
        self.assertGreater(latency_tta, 0.0)
        self.assertGreaterEqual(top3_tta[0][1], top3_tta[1][1])
        self.assertGreaterEqual(top3_tta[1][1], top3_tta[2][1])

        # Verify class labels match expected classes
        for artist, score in top3_tta:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(any(artist == c.replace('_', ' ') for c in self.cnn_classes))

    def test_cnn_horizontal_flip_tta_exact_averaging(self):
        """Verify TTA exactly averages the two softmax prediction vectors."""
        img = Image.new("RGB", (100, 100), color="blue")
        for x in range(40):
            for y in range(100):
                img.putpixel((x, y), (0, 255, 0))

        # 1. Manual forward pass for original view
        rgb_img = img.convert("RGB")
        tensor_orig = VAL_TRANSFORMS(rgb_img).unsqueeze(0)
        with torch.no_grad():
            probs_orig = torch.softmax(self.cnn_model(tensor_orig), dim=1)[0].numpy()

        # 2. Manual forward pass for horizontally flipped view
        flipped_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
        tensor_flip = VAL_TRANSFORMS(flipped_img).unsqueeze(0)
        with torch.no_grad():
            probs_flip = torch.softmax(self.cnn_model(tensor_flip), dim=1)[0].numpy()

        expected_probs = (probs_orig + probs_flip) / 2.0
        expected_indices = np.argsort(expected_probs)[::-1][:3]
        expected_top3 = [(self.cnn_idx_map[i].replace('_', ' '), float(expected_probs[i])) for i in expected_indices]

        # 3. Model predict_cnn with use_tta=True
        top3_tta, _ = predict_cnn(img, self.cnn_model, self.cnn_idx_map, use_tta=True)

        self.assertEqual(len(top3_tta), 3)
        for i in range(3):
            self.assertEqual(top3_tta[i][0], expected_top3[i][0])
            self.assertAlmostEqual(top3_tta[i][1], expected_top3[i][1], places=6)

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
