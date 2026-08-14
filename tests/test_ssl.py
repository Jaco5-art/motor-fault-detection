from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from motor_fault.models.transformer import build_cnn_transformer
from motor_fault.ssl.contrastive import (
    ContrastiveEncoder,
    augment_batch,
    encoder_only_state_dict,
    overlap_aware_nt_xent,
)


@unittest.skipIf(torch is None, "PyTorch not installed")
class SSLTests(unittest.TestCase):
    def test_augmentations_preserve_shape_and_mask_time(self) -> None:
        inputs = torch.ones(4, 18, 200)
        generator = torch.Generator().manual_seed(7)
        augmented = augment_batch(inputs, 0.02, 0.10, 0.10, generator)
        self.assertEqual(tuple(augmented.shape), (4, 18, 200))
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue((augmented == 0).any())

    def test_overlap_aware_loss_excludes_overlapping_windows(self) -> None:
        first = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        second = first.clone()
        run_codes = torch.tensor([0, 0, 1])
        starts = torch.tensor([0, 10, 0])
        ends = torch.tensor([199, 209, 199])
        loss, diagnostics = overlap_aware_nt_xent(
            first, second, run_codes, starts, ends, temperature=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diagnostics["excluded_overlap_pairs"], 0)
        self.assertGreater(diagnostics["excluded_overlap_fraction"], 0)

    def test_encoder_export_omits_random_classifier_head(self) -> None:
        encoder = build_cnn_transformer(
            {
                "input_channels": 18,
                "output_motors": 6,
                "cnn_channels": [16, 32],
                "cnn_kernels": [7, 5],
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "pooling": "avg_max",
            }
        )
        model = ContrastiveEncoder(encoder, encoder.representation_dim, 32, 16)
        state = encoder_only_state_dict(model)
        self.assertTrue(state)
        self.assertFalse(any(key.startswith("head.") for key in state))
        incompatible = encoder.load_state_dict(state, strict=False)
        self.assertTrue(all(key.startswith("head.") for key in incompatible.missing_keys))
        self.assertEqual(list(incompatible.unexpected_keys), [])


if __name__ == "__main__":
    unittest.main()
