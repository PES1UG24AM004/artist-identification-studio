"""
Unit tests for the PyTorch CNN transfer learning pipeline.

Tests verify:
- ResNet-18 model construction and custom 10-class linear head
- Output tensor dimensions (batch_size, 10) on synthetic batch
- Gradient freeze enforcement (backbone requires_grad=False, fc requires_grad=True)
- Deterministic validation transforms vs. stochastic training augmentations
- Checkpoint saving and reloading mechanics
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import torch
import torch.nn as nn
from PIL import Image

from src.models.train_cnn import (
    build_artist_resnet18,
    get_transforms,
    NUM_CLASSES,
    DEFAULT_SEED
)


class TestCNNPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_artist_resnet18(num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
        self.model.to(self.device)

    def test_model_head_and_class_count(self):
        """Verify the classification head outputs exactly NUM_CLASSES (10)."""
        self.assertIsInstance(self.model.fc, nn.Linear)
        self.assertEqual(self.model.fc.out_features, NUM_CLASSES)
        self.assertEqual(self.model.fc.in_features, 512)

    def test_forward_pass_output_shape(self):
        """Verify forward pass on synthetic input tensor (batch_size=4, 3, 224, 224)."""
        x = torch.randn(4, 3, 224, 224, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))
        self.assertTrue(torch.all(torch.isfinite(out)), "Forward pass output contains NaN/Inf!")

    def test_backbone_gradient_freeze(self):
        """Verify backbone parameters do NOT require gradients, while fc head does."""
        for name, param in self.model.named_parameters():
            if name.startswith("fc."):
                self.assertTrue(param.requires_grad, f"Head parameter {name} should require grad!")
            else:
                self.assertFalse(param.requires_grad, f"Backbone parameter {name} should be frozen!")

    def test_transforms_reproducibility(self):
        """Verify validation transforms produce deterministic output on the same image."""
        train_tf, val_tf = get_transforms()
        img = Image.new("RGB", (300, 300), color="blue")

        t1 = val_tf(img)
        t2 = val_tf(img)
        self.assertEqual(t1.shape, (3, 224, 224))
        self.assertTrue(torch.allclose(t1, t2), "Validation transforms must be deterministic!")

    def test_checkpoint_saving_and_reloading(self):
        """Verify state_dict can be saved and reloaded with identical predictions."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            ckpt_path = temp_dir / "test_cnn_ckpt.pth"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "class_labels": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            }, ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Reload into new model
            new_model = build_artist_resnet18(num_classes=NUM_CLASSES, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            new_model.load_state_dict(ckpt["model_state_dict"])
            new_model.eval()

            x = torch.randn(2, 3, 224, 224)
            self.model.eval()
            new_model.eval()
            with torch.no_grad():
                out1 = self.model(x)
                out2 = new_model(x)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
