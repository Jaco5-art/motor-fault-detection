from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from motor_fault.models.cnn import require_torch

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:

    class ContrastiveEncoder(nn.Module):
        def __init__(
            self,
            encoder: nn.Module,
            representation_dim: int,
            projection_hidden_dim: int = 128,
            projection_dim: int = 64,
        ):
            super().__init__()
            self.encoder = encoder
            self.projection = nn.Sequential(
                nn.Linear(representation_dim, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, projection_dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            representation = self.encoder.encode(inputs)
            return F.normalize(self.projection(representation), dim=1)


def augment_batch(
    inputs: "torch.Tensor",
    jitter_sigma: float,
    scaling_sigma: float,
    time_mask_ratio: float,
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    require_torch()
    values = inputs.clone()
    batch, channels, length = values.shape
    scale = 1.0 + torch.randn(
        batch, channels, 1, generator=generator, device=values.device
    ) * float(scaling_sigma)
    values = values * scale
    values = values + torch.randn(
        values.shape, generator=generator, device=values.device, dtype=values.dtype
    ) * float(jitter_sigma)
    mask_length = max(1, int(round(length * float(time_mask_ratio))))
    if mask_length >= length:
        raise ValueError("time_mask_ratio masks the complete window")
    starts = torch.randint(
        0, length - mask_length + 1, (batch,), generator=generator, device=values.device
    )
    for row, start in enumerate(starts.tolist()):
        values[row, :, start : start + mask_length] = 0.0
    return values


def overlap_aware_nt_xent(
    first: "torch.Tensor",
    second: "torch.Tensor",
    run_codes: "torch.Tensor",
    starts: "torch.Tensor",
    ends: "torch.Tensor",
    temperature: float = 0.2,
) -> tuple["torch.Tensor", dict[str, float]]:
    """NT-Xent excluding overlapping windows from the negative denominator."""

    require_torch()
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Contrastive views must have matching [batch, dim] shapes")
    batch = first.shape[0]
    if batch < 2:
        raise ValueError("NT-Xent requires at least two base windows")
    representations = torch.cat([first, second], dim=0)
    similarity = representations @ representations.T / float(temperature)
    base = torch.arange(batch, device=representations.device).repeat(2)
    run = run_codes.to(representations.device).repeat(2)
    start = starts.to(representations.device).repeat(2)
    end = ends.to(representations.device).repeat(2)

    identity = torch.eye(2 * batch, dtype=torch.bool, device=representations.device)
    same_base = base[:, None] == base[None, :]
    positive_mask = same_base & ~identity
    overlap = (
        (run[:, None] == run[None, :])
        & (start[:, None] <= end[None, :])
        & (start[None, :] <= end[:, None])
        & ~same_base
    )
    denominator_mask = ~identity & ~overlap
    masked_similarity = similarity.masked_fill(~denominator_mask, -torch.inf)
    positive_logits = similarity[positive_mask].reshape(2 * batch)
    loss = -(positive_logits - torch.logsumexp(masked_similarity, dim=1)).mean()
    excluded = overlap.sum().item()
    possible = (2 * batch) * (2 * batch - 1)
    diagnostics = {
        "excluded_overlap_pairs": float(excluded),
        "excluded_overlap_fraction": float(excluded / possible),
    }
    return loss, diagnostics


def encoder_only_state_dict(model: "nn.Module") -> dict[str, "torch.Tensor"]:
    """Return classifier-compatible encoder weights without a random task head."""

    require_torch()
    return {
        key.removeprefix("encoder."): value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("encoder.") and not key.startswith("encoder.head.")
    }
