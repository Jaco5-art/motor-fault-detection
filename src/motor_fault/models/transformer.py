from __future__ import annotations

import math
from typing import Any

from motor_fault.models.cnn import ConvBlock, require_torch

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class SinusoidalPositionEncoding(nn.Module):
        def __init__(self, d_model: int, max_length: int = 256):
            super().__init__()
            positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / d_model)
            )
            encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(positions * divisor)
            encoding[:, 1::2] = torch.cos(positions * divisor)
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            if tokens.size(1) > self.encoding.size(1):
                raise ValueError(
                    f"Token length {tokens.size(1)} exceeds positional limit "
                    f"{self.encoding.size(1)}"
                )
            return tokens + self.encoding[:, : tokens.size(1)]


    class CNNTransformerClassifier(nn.Module):
        """Local Conv1D tokenizer followed by a lightweight temporal Transformer."""

        def __init__(
            self,
            input_channels: int = 18,
            output_motors: int = 6,
            cnn_channels: tuple[int, ...] | list[int] = (32, 64),
            cnn_kernels: tuple[int, ...] | list[int] = (7, 5),
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 1,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
            pooling: str = "avg_max",
            activation: str = "gelu",
            max_tokens: int = 256,
        ):
            super().__init__()
            if len(cnn_channels) != len(cnn_kernels) or not cnn_channels:
                raise ValueError("cnn_channels and cnn_kernels must be equally sized")
            if int(cnn_channels[-1]) != int(d_model):
                raise ValueError("The final CNN channel count must equal d_model")
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            if pooling not in {"avg", "avg_max"}:
                raise ValueError(f"Unsupported pooling: {pooling}")
            blocks = []
            current = int(input_channels)
            for output, kernel in zip(cnn_channels, cnn_kernels):
                blocks.append(ConvBlock(current, int(output), int(kernel)))
                current = int(output)
            self.cnn = nn.Sequential(*blocks)
            self.position = SinusoidalPositionEncoding(int(d_model), int(max_tokens))
            layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation=str(activation),
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=int(n_layers), enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(int(d_model))
            self.pooling = pooling
            head_dimension = int(d_model) if pooling == "avg" else int(d_model) * 2
            self.head = nn.Sequential(
                nn.Dropout(float(dropout)), nn.Linear(head_dimension, int(output_motors))
            )

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            # Conv stem returns [batch, d_model, reduced_time].
            tokens = self.cnn(inputs).transpose(1, 2)
            tokens = self.norm(self.transformer(self.position(tokens)))
            pooled = tokens.mean(dim=1)
            if self.pooling == "avg_max":
                pooled = torch.cat([pooled, tokens.amax(dim=1)], dim=1)
            return pooled

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(inputs))

        @property
        def representation_dim(self) -> int:
            return int(self.head[-1].in_features)


def build_cnn_transformer(config: dict[str, Any]) -> "CNNTransformerClassifier":
    require_torch()
    return CNNTransformerClassifier(
        input_channels=int(config["input_channels"]),
        output_motors=int(config["output_motors"]),
        cnn_channels=tuple(map(int, config["cnn_channels"])),
        cnn_kernels=tuple(map(int, config["cnn_kernels"])),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        pooling=str(config.get("pooling", "avg_max")),
        activation=str(config.get("activation", "gelu")),
        max_tokens=int(config.get("max_tokens", 256)),
    )
