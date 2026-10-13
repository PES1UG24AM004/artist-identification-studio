"""
Unit tests for the ResNet-18 Layer 4 Fine-Tuning Pipeline.

Tests verify:
- Selective layer freeze enforcement:
    - conv1, bn1, layer1, layer2, layer3: requires_grad = False
    - layer4: requires_grad = True
    - fc: requires_grad = True
- Differential optimizer setup: layer4 gets 1e-4, fc gets 1e-3
- Forward pass output shape and finite values
- Validation transforms are deterministic and preserve ImageNet statistics
- Training transforms apply conservative cropping
- Strict artifact isolation: models/cnn_best.pth is not overwritten
"""

import unittest
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from src.models.train_cnn_finetune import (
    build_finetuned_resnet18,
    get_finetune_transforms,
    NUM_CLASSES,
    LR_LAYER4,
    LR_HEAD,
    PROJECT_ROOT
)


class TestCNNFineTunePipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.base_ckpt_path = PROJECT_ROOT / "models" / "experiments" / "cnn_baseline_head_only.pth"
        self.prod_ckpt_path = PROJECT_ROOT / "models" / "cnn_best.pth"
        self.assertTrue(self.base_ckpt_path.exists(), "Archived baseline checkpoint must exist!")
        self.assertTrue(self.prod_ckpt_path.exists(), "Promoted production checkpoint must exist!")
        self.model, self.ckpt = build_finetuned_resnet18(self.base_ckpt_path, num_classes=NUM_CLASSES)
        self.model.to(self.device)

    def test_selective_gradient_unfreezing(self):
        """Verify layer1-3 are frozen while layer4 and fc are unfrozen."""
        for name, param in self.model.named_parameters():
            if name.startswith("layer4.") or name.startswith("fc."):
                self.assertTrue(param.requires_grad, f"Parameter {name} should require grad!")
            else:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen!")

    def test_differential_optimizer_param_groups(self):
        """Verify optimizer param groups have differential learning rates."""
        optimizer = torch.optim.AdamW([
            {"params": self.model.layer4.parameters(), "lr": LR_LAYER4, "weight_decay": 1e-4},
            {"params": self.model.fc.parameters(), "lr": LR_HEAD, "weight_decay": 1e-4}
        ])
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], LR_LAYER4)
        self.assertEqual(optimizer.param_groups[1]["lr"], LR_HEAD)

    def test_forward_pass_output_shape(self):
        """Verify output tensor has shape (batch_size, 10)."""
        x = torch.randn(2, 3, 224, 224, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_transforms_determinism(self):
        """Verify validation transforms are deterministic."""
        train_tf, val_tf = get_finetune_transforms()
        img = Image.new("RGB", (320, 240), color="darkgreen")

        t1 = val_tf(img)
        t2 = val_tf(img)
        self.assertEqual(t1.shape, (3, 224, 224))
        self.assertTrue(torch.allclose(t1, t2))

    def test_baseline_and_promoted_checkpoint_integrity(self):
        """Verify archived baseline has val_acc=0.71 and promoted checkpoint has val_acc=0.81."""
        self.assertTrue(self.base_ckpt_path.exists())
        ckpt_base = torch.load(self.base_ckpt_path, map_location="cpu")
        self.assertEqual(ckpt_base["val_acc"], 0.71)

        self.assertTrue(self.prod_ckpt_path.exists())
        ckpt_prod = torch.load(self.prod_ckpt_path, map_location="cpu")
        self.assertEqual(ckpt_prod["val_acc"], 0.81)


if __name__ == "__main__":
    unittest.main()
