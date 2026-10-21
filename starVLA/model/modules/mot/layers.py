# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add asymmetric vision-action MoT blocks and full-rank adaptive RMSNorm primitives.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"false", "0", "no", "off", ""}
    return bool(value)


class RMSNorm(nn.Module):
    """RMSNorm with an optional learned affine scale."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight.to(input_dtype)
        return hidden_states


class AdaLNRMSNormZero(nn.Module):
    """Adaptive RMSNorm with a zero-initialized residual gate."""

    def __init__(self, hidden_size: int, rank: int | None = None, eps: float = 1.0e-6) -> None:
        super().__init__()
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive when provided, got {rank}")
        self.norm = RMSNorm(hidden_size, eps=eps, affine=False)
        self.rank = rank
        if rank is None:
            self.modulation: nn.Module = nn.Linear(hidden_size, hidden_size * 3)
        else:
            self.modulation = nn.Sequential(
                nn.Linear(hidden_size, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, hidden_size * 3),
            )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def output_projection(self) -> nn.Linear:
        """Return the zero-initialized projection that emits shift, scale, and gate."""

        if isinstance(self.modulation, nn.Linear):
            return self.modulation
        output_projection = self.modulation[-1]
        if not isinstance(output_projection, nn.Linear):
            raise TypeError("AdaLN-Zero modulation must end with nn.Linear")
        return output_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return modulated states and the channel-wise residual gate."""

        # Chunk order is explicit and shared by every MoT2 residual branch:
        # [shift, scale, gate].
        modulation = self.modulation(F.silu(condition)) if self.rank is None else self.modulation(condition)
        shift, scale, gate = modulation.chunk(3, dim=-1)
        modulated = self.norm(hidden_states) * (1.0 + scale) + shift
        return modulated, gate


class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm for a non-residual output head."""

    def __init__(self, hidden_size: int, rank: int | None = None, eps: float = 1.0e-6) -> None:
        super().__init__()
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive when provided, got {rank}")
        self.norm = RMSNorm(hidden_size, eps=eps, affine=False)
        self.rank = rank
        if rank is None:
            self.modulation: nn.Module = nn.Linear(hidden_size, hidden_size * 2)
        else:
            self.modulation = nn.Sequential(
                nn.Linear(hidden_size, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, hidden_size * 2),
            )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def output_projection(self) -> nn.Linear:
        """Return the zero-initialized projection that emits shift and scale."""

        if isinstance(self.modulation, nn.Linear):
            return self.modulation
        output_projection = self.modulation[-1]
        if not isinstance(output_projection, nn.Linear):
            raise TypeError("Adaptive RMSNorm modulation must end with nn.Linear")
        return output_projection

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply zero-initialized shift/scale modulation before an output head."""

        # Final adaptive norms have no residual branch and therefore no gate.
        modulation = self.modulation(F.silu(condition)) if self.rank is None else self.modulation(condition)
        shift, scale = modulation.chunk(2, dim=-1)
        return self.norm(hidden_states) * (1.0 + scale) + shift


class SwiGLU(nn.Module):
    """Three-matrix SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _split_heads(hidden_states: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads
    return hidden_states.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def _merge_heads(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
    return hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, num_heads * head_dim)


@dataclass
class MoTOutput:
    """Velocity predictions and optional intermediate expert features."""

    vision_velocity: torch.Tensor
    action_velocity: torch.Tensor
    vision_features: dict[int, torch.Tensor]
    action_features: dict[int, torch.Tensor]
