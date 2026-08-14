from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from motor_fault.models.cnn import build_cnn, count_trainable_parameters
from motor_fault.models.transformer import build_cnn_transformer
from motor_fault.training.cnn_trainer import train_only_pos_weight


@unittest.skipIf(torch is None, "PyTorch not installed")
class CNNTests(unittest.TestCase):
    def test_cnn_shape_and_parameter_count(self) -> None:
        model = build_cnn(
            {
                "input_channels": 18,
                "output_motors": 6,
                "channels": [8, 16],
                "kernels": [7, 5],
                "dropout": 0.2,
                "pooling": "avg_max",
            }
        )
        output = model(torch.zeros(4, 18, 200))
        self.assertEqual(tuple(output.shape), (4, 6))
        self.assertGreater(count_trainable_parameters(model), 0)

    def test_class_weights_use_train_label_counts(self) -> None:
        labels = np.zeros((10, 6), dtype=np.float32)
        labels[:2, :] = 1
        weights = train_only_pos_weight(labels)
        np.testing.assert_allclose(weights, np.full(6, 4.0))

    def test_class_weight_cap_limits_sparse_motor_gradient(self) -> None:
        labels = np.zeros((100, 6), dtype=np.float32)
        labels[0, :] = 1
        weights = train_only_pos_weight(labels, cap=30.0)
        np.testing.assert_allclose(weights, np.full(6, 30.0))

    def test_cnn_transformer_shape_and_temporal_tokens(self) -> None:
        model = build_cnn_transformer(
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
        output = model(torch.zeros(3, 18, 200))
        self.assertEqual(tuple(output.shape), (3, 6))
        self.assertGreater(count_trainable_parameters(model), 0)


if __name__ == "__main__":
    unittest.main()
