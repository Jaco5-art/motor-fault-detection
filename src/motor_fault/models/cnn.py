from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # Keep Stage 0–2 utilities importable without PyTorch.
    torch = None
    nn = None


if nn is not None:

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
            super().__init__()
            padding = kernel_size // 2
            self.block = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.block(inputs)


    class CNN1DClassifier(nn.Module):
        """Compact raw-sequence multi-label baseline."""

        def __init__(
            self,
            input_channels: int = 18,
            output_motors: int = 6,
            channels: tuple[int, ...] | list[int] = (32, 64),
            kernels: tuple[int, ...] | list[int] = (7, 5),
            dropout: float = 0.2,
            pooling: str = "avg_max",
        ):
            super().__init__()
            if len(channels) != len(kernels) or not channels:
                raise ValueError("channels and kernels must be non-empty and equally sized")
            blocks = []
            current = int(input_channels)
            for output, kernel in zip(channels, kernels):
                blocks.append(ConvBlock(current, int(output), int(kernel)))
                current = int(output)
            self.encoder = nn.Sequential(*blocks)
            if pooling not in {"avg", "avg_max"}:
                raise ValueError(f"Unsupported pooling: {pooling}")
            self.pooling = pooling
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            head_channels = current if pooling == "avg" else current * 2
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_channels, int(output_motors)),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(inputs)
            pooled = self.avg_pool(encoded)
            if self.pooling == "avg_max":
                pooled = torch.cat([pooled, self.max_pool(encoded)], dim=1)
            return self.classifier(pooled)


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required; install requirements-deep.txt")


def build_cnn(config: dict[str, Any]) -> "CNN1DClassifier":
    require_torch()
    return CNN1DClassifier(
        input_channels=int(config["input_channels"]),
        output_motors=int(config["output_motors"]),
        channels=tuple(map(int, config["channels"])),
        kernels=tuple(map(int, config["kernels"])),
        dropout=float(config["dropout"]),
        pooling=str(config.get("pooling", "avg_max")),
    )


def count_trainable_parameters(model: "nn.Module") -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
