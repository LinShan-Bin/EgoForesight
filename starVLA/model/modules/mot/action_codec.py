# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Identity codec for the released canonical continuous-action model."""
from typing import Any, Mapping

import torch
from torch import nn


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


class ActionCodec(nn.Module):
    """Preserve normalized canonical actions and elementwise validity masks."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.encoding = str(_cfg_get(config, "encoding", "raw"))
        if self.encoding != "raw":
            raise ValueError("EgoForesight supports action_model.encoding=raw only")
        self.action_dim = int(_cfg_get(config, "action_dim", 130))
        self.chunk_size = int(_cfg_get(config, "chunk_size", 48))
        self.max_training_chunk_size = int(_cfg_get(config, "max_training_chunk_size", self.chunk_size))
        if self.max_training_chunk_size < self.chunk_size:
            raise ValueError("max_training_chunk_size must be at least chunk_size")
        self.action_encoder = None
        self.flow_dim = self.action_dim
        self.num_flow_tokens = self.chunk_size
        self.register_buffer("latent_mean", torch.zeros(1), persistent=False)
        self.register_buffer("latent_std", torch.ones(1), persistent=False)

    def encode(
        self, actions: torch.Tensor, meta: dict[str, Any] | None = None, action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the canonical flow target unchanged."""
        return actions

    def decode(self, flow_actions: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Return generated canonical actions unchanged."""
        return flow_actions

    def flow_loss_mask(self, action_mask: torch.Tensor | None, flow_actions: torch.Tensor) -> torch.Tensor | None:
        """Move an elementwise mask to the flow tensor's device and dtype."""
        return None if action_mask is None else action_mask.to(device=flow_actions.device, dtype=flow_actions.dtype)

    def token_mask(self, action_mask: torch.Tensor | None, flow_actions: torch.Tensor) -> torch.Tensor:
        """A token is valid if at least one canonical slot is valid."""
        if action_mask is None:
            return torch.ones(flow_actions.shape[:2], device=flow_actions.device, dtype=torch.bool)
        return action_mask.bool().any(dim=-1)
