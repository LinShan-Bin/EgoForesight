# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Complete MoT2 AdaLN-Zero conditioning with QK RMSNorm and clean/noisy embeddings.

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.mot.flow_matching import FlowMatchSample


def _assert_async(condition: torch.Tensor, message: str) -> None:
    """Assert a scalar device predicate without a CUDA-to-Python value read."""

    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
        return
    if condition.device.type != "cpu":
        raise RuntimeError("MoT2 requires torch._assert_async for non-blocking CUDA validation")
    if not bool(condition):
        raise ValueError(message)


def _validate_binary_mask(mask: torch.Tensor, name: str) -> None:
    """Validate a binary mask without reading a CUDA predicate on the host."""

    if mask.dtype == torch.bool:
        return
    _assert_async(torch.isfinite(mask).all(), f"{name} must contain only finite values")
    _assert_async(
        ((mask == 0) | (mask == 1)).all(),
        f"{name} must contain only boolean or exact 0/1 values",
    )


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)!r}")
    if mask.ndim > reference.ndim:
        raise ValueError(f"mask rank {mask.ndim} cannot broadcast to reference rank {reference.ndim}")
    shape = tuple(mask.shape) + (1,) * (reference.ndim - mask.ndim)
    reshaped = mask.reshape(shape)
    try:
        torch.broadcast_shapes(reshaped.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to reference shape {tuple(reference.shape)}"
        ) from error
    return reshaped.expand_as(reference)


def _expand_timesteps(timesteps: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    time = torch.as_tensor(timesteps, device=reference.device, dtype=reference.dtype)
    if time.ndim > reference.ndim:
        raise ValueError(f"timesteps rank {time.ndim} cannot broadcast to reference rank {reference.ndim}")
    if time.ndim > 0 and time.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"timestep batch {time.shape[0]} does not match reference batch {reference.shape[0]}")
    shape = tuple(time.shape) + (1,) * (reference.ndim - time.ndim)
    time = time.reshape(shape)
    try:
        torch.broadcast_shapes(time.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"timesteps shape {tuple(time.shape)} cannot broadcast to reference shape {tuple(reference.shape)}"
        ) from error
    _assert_async(torch.isfinite(time).all(), "timesteps must contain only finite values")
    _assert_async((time >= 0.0).all(), "timesteps must lie in [0, 1]")
    _assert_async((time <= 1.0).all(), "timesteps must lie in [0, 1]")
    return time


def masked_action_flow_sample(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    action_schema_mask: torch.Tensor,
    noise: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchSample:
    """Build an RF sample while keeping every inactive action element exactly zero."""

    if not isinstance(clean, torch.Tensor) or not clean.is_floating_point() or clean.ndim == 0:
        raise TypeError("clean must be a floating-point tensor with a batch dimension")
    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    if noise.shape != clean.shape or noise.device != clean.device:
        raise ValueError(
            f"noise must match clean shape/device, got {tuple(noise.shape)} on {noise.device} "
            f"for {tuple(clean.shape)} on {clean.device}"
        )

    action_schema_mask = action_schema_mask.to(device=clean.device)
    _validate_binary_mask(action_schema_mask, "action_schema_mask")
    active = _broadcast_mask(action_schema_mask.bool(), clean)
    clean = torch.where(active, clean, torch.zeros_like(clean))
    noise = torch.where(active, noise, torch.zeros_like(noise))
    time = _expand_timesteps(timesteps, clean)
    noisy = clean * (1.0 - time) + noise * time
    return FlowMatchSample(
        noisy=noisy,
        velocity=noise - clean,
        timesteps=timesteps,
        noise=noise,
    )


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a masked mean without converting a device predicate to Python bool."""

    if mask is None:
        return values.mean()
    weights = _broadcast_mask(mask.to(device=values.device, dtype=values.dtype), values)
    _assert_async((weights.detach() >= 0.0).all(), "mask weights must be non-negative")
    weighted_values = torch.where(
        weights > 0.0,
        values * weights,
        torch.zeros_like(values),
    )
    return weighted_values.sum() / weights.sum().clamp_min(1.0)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable masked MSE without a GPU-to-Python mask check."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}"
        )
    return masked_mean((prediction.float() - target.float()).square(), mask)


def masked_cosine_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student and teacher features must match, got {tuple(student_features.shape)} and "
            f"{tuple(teacher_features.shape)}"
        )
    similarity = F.cosine_similarity(
        student_features.float(),
        teacher_features.detach().clone().float(),
        dim=-1,
        eps=eps,
    )
    return masked_mean(1.0 - similarity, mask)
