# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Keep sampled timesteps above the clean boundary while retaining clean edge vision.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from starVLA.model.modules.mot.flow_matching import masked_mean

FORESIGHT_MIN_TIMESTEP = 0.005


@dataclass(frozen=True)
class ForesightTimestepSchedule:
    """Per-sample vision/action timesteps for Foresight Forcing."""

    vision_timesteps: torch.Tensor
    action_timesteps: torch.Tensor
    alpha: torch.Tensor
    teacher_edge_mask: torch.Tensor

    @property
    def t_v(self) -> torch.Tensor:
        """Return the vision timestep shorthand used in the paper."""

        return self.vision_timesteps

    @property
    def t_a(self) -> torch.Tensor:
        """Return the action timestep shorthand used in the paper."""

        return self.action_timesteps


def _validate_reliability(reliability: torch.Tensor) -> torch.Tensor:
    if not isinstance(reliability, torch.Tensor):
        raise TypeError(f"reliability must be a torch.Tensor, got {type(reliability)!r}")
    if reliability.ndim != 1 or reliability.numel() == 0:
        raise ValueError(f"reliability must be a non-empty [B] tensor, got {tuple(reliability.shape)}")
    reliability_fp32 = reliability.float()
    if not torch.isfinite(reliability_fp32).all():
        raise ValueError("reliability must contain only finite values")
    if torch.any(reliability_fp32 < 0.0) or torch.any(reliability_fp32 > 1.0):
        raise ValueError("reliability values must lie in [0, 1]")
    return reliability_fp32


def scale_teacher_action_timesteps(
    action_timesteps: torch.Tensor,
    alpha_teacher: float = 1.0,
) -> torch.Tensor:
    """Scale teacher action time before the affine map, preserving its floor."""

    if not isinstance(action_timesteps, torch.Tensor) or not action_timesteps.is_floating_point():
        raise TypeError("action_timesteps must be a floating-point torch.Tensor")
    if action_timesteps.numel() == 0:
        raise ValueError("action_timesteps must not be empty")
    alpha_teacher = float(alpha_teacher)
    if not math.isfinite(alpha_teacher) or not 0.0 < alpha_teacher <= 1.0:
        raise ValueError(f"alpha_teacher must lie in (0, 1], got {alpha_teacher}")
    detached = action_timesteps.detach()
    if not torch.isfinite(detached).all() or torch.any(detached < FORESIGHT_MIN_TIMESTEP) or torch.any(detached > 1.0):
        raise ValueError(f"action_timesteps must contain finite values in [{FORESIGHT_MIN_TIMESTEP}, 1]")
    if alpha_teacher == 1.0:
        return action_timesteps
    return FORESIGHT_MIN_TIMESTEP + (action_timesteps - FORESIGHT_MIN_TIMESTEP) * alpha_teacher


def sample_uniform_teacher_action_timesteps(
    action_timesteps: torch.Tensor,
    *,
    ratio_min: float = 0.0,
    ratio_max: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ratios in the pre-map time domain so teacher actions stay noisy."""

    if not isinstance(action_timesteps, torch.Tensor) or not action_timesteps.is_floating_point():
        raise TypeError("action_timesteps must be a floating-point torch.Tensor")
    if action_timesteps.numel() == 0:
        raise ValueError("action_timesteps must not be empty")
    ratio_min = float(ratio_min)
    ratio_max = float(ratio_max)
    if not math.isfinite(ratio_min) or not math.isfinite(ratio_max) or not 0.0 <= ratio_min <= ratio_max <= 1.0:
        raise ValueError(
            "teacher timestep ratio bounds must satisfy 0 <= min <= max <= 1, " f"got min={ratio_min}, max={ratio_max}"
        )
    detached = action_timesteps.detach()
    if not torch.isfinite(detached).all() or torch.any(detached < FORESIGHT_MIN_TIMESTEP) or torch.any(detached > 1.0):
        raise ValueError(f"action_timesteps must contain finite values in [{FORESIGHT_MIN_TIMESTEP}, 1]")

    ratios = torch.rand(
        action_timesteps.shape,
        device=action_timesteps.device,
        dtype=torch.float32,
        generator=generator,
    )
    ratios = ratio_min + (ratio_max - ratio_min) * ratios
    teacher_timesteps = FORESIGHT_MIN_TIMESTEP + (action_timesteps - FORESIGHT_MIN_TIMESTEP) * ratios.to(
        dtype=action_timesteps.dtype
    )
    return teacher_timesteps, ratios


def _mapped_foresight_schedule(
    vision_timesteps: torch.Tensor,
    action_timesteps: torch.Tensor,
    alpha: torch.Tensor,
    teacher_edge_mask: torch.Tensor,
) -> ForesightTimestepSchedule:
    """Apply ``t = 0.005 + 0.995 * t`` once, except to fixed clean edge vision."""
    vision_timesteps = FORESIGHT_MIN_TIMESTEP + (1.0 - FORESIGHT_MIN_TIMESTEP) * vision_timesteps
    action_timesteps = FORESIGHT_MIN_TIMESTEP + (1.0 - FORESIGHT_MIN_TIMESTEP) * action_timesteps
    vision_timesteps = vision_timesteps.masked_fill(teacher_edge_mask, 0.0)
    # The affine map can round a tiny positive gap to zero. Preserve ordering.
    action_timesteps = torch.maximum(
        action_timesteps, torch.nextafter(vision_timesteps, torch.ones_like(vision_timesteps))
    )
    return ForesightTimestepSchedule(vision_timesteps, action_timesteps, alpha, teacher_edge_mask)


def _sample_action_gap_ratio(
    alpha: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-item ``Beta(alpha, 2)`` gap ratios in float32."""

    alpha_fp32 = alpha.float()
    alpha_gamma = torch._standard_gamma(alpha_fp32, generator=generator)
    beta_gamma = torch._standard_gamma(torch.full_like(alpha_fp32, 2.0), generator=generator)
    denominator = (alpha_gamma + beta_gamma).clamp_min(torch.finfo(torch.float32).tiny)
    return alpha_gamma / denominator


def sample_foresight_timesteps(
    reliability: torch.Tensor,
    *,
    alpha_min: float = 0.5,
    alpha_max: float = 3.0,
    teacher_edge_prob: float = 0.02,
    vision_timestep_power: float | None = None,
    eps: float = 1.0e-7,
    generator: torch.Generator | None = None,
) -> ForesightTimestepSchedule:
    """
    Sample modality-specific Foresight Forcing timesteps in float32.

    ``t_v`` is uniform. For ordinary samples, ``g ~ Beta(alpha, 2)`` and
    ``t_a = t_v + (1 - t_v) * g``. This makes ``t_a > t_v`` while preserving
    the previous ``Beta(alpha + 1, 1)`` action-timestep marginal. Less reliable
    datasets receive larger ``alpha`` and therefore larger gaps.

    Boundary-calibration samples use ``t_v = 0`` and
    ``t_a ~ Beta(alpha, 1)``.

    When ``vision_timestep_power`` is set, ordinary samples instead draw
    ``t_a ~ Beta(alpha + 1, 1)`` first, then set ``t_v = t_a ** power``.
    Finally, map every sampled time with ``t = 0.005 + 0.995 * t``. Only the
    fixed clean boundary-calibration vision time remains zero. All marginal
    distribution descriptions above refer to the times before this mapping.
    """

    reliability_fp32 = _validate_reliability(reliability)
    if not math.isfinite(alpha_min) or alpha_min <= 0.0:
        raise ValueError(f"alpha_min must be finite and positive, got {alpha_min}")
    if not math.isfinite(alpha_max) or alpha_max < alpha_min:
        raise ValueError(f"alpha_max must be finite and >= alpha_min, got {alpha_max}")
    if not math.isfinite(teacher_edge_prob) or not 0.0 <= teacher_edge_prob <= 1.0:
        raise ValueError(f"teacher_edge_prob must lie in [0, 1], got {teacher_edge_prob}")
    if not math.isfinite(eps) or not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps}")

    batch_size = reliability_fp32.shape[0]
    device = reliability_fp32.device
    if vision_timestep_power is not None:
        power = float(vision_timestep_power)
        if not math.isfinite(power) or power <= 1.0:
            raise ValueError(f"vision_timestep_power must be finite and > 1, got {power}")
        alpha = alpha_min + (1.0 - reliability_fp32) * (alpha_max - alpha_min)
        uniform_action = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator).clamp_min(eps)
        if teacher_edge_prob == 0.0:
            teacher_edge_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        elif teacher_edge_prob == 1.0:
            teacher_edge_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        else:
            teacher_edge_mask = (
                torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator) < teacher_edge_prob
            )
        concentration = torch.where(teacher_edge_mask, alpha, alpha + 1.0)
        action_timesteps = uniform_action.pow(concentration.reciprocal())
        vision_timesteps = action_timesteps.pow(power).masked_fill(teacher_edge_mask, 0.0)
        return _mapped_foresight_schedule(
            vision_timesteps=vision_timesteps,
            action_timesteps=action_timesteps,
            alpha=alpha,
            teacher_edge_mask=teacher_edge_mask,
        )

    vision_timesteps = torch.rand(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    uniform_action = torch.rand(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(eps)
    if teacher_edge_prob == 0.0:
        teacher_edge_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    elif teacher_edge_prob == 1.0:
        teacher_edge_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        teacher_edge_mask = (
            torch.rand(
                batch_size,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            < teacher_edge_prob
        )

    vision_timesteps = vision_timesteps.masked_fill(teacher_edge_mask, 0.0)
    alpha = alpha_min + (1.0 - reliability_fp32) * (alpha_max - alpha_min)
    # Keep all sampling and transforms explicitly FP32 even under autocast.
    gap_ratio = _sample_action_gap_ratio(alpha, generator=generator)
    regular_action_timesteps = vision_timesteps + (1.0 - vision_timesteps) * gap_ratio
    # Very small gaps can round back to t_v in float32. Advance by one ULP in
    # those cases so the intended strict ordering is also reflected in metrics.
    next_vision_timestep = torch.nextafter(
        vision_timesteps,
        torch.ones_like(vision_timesteps),
    )
    regular_action_timesteps = torch.maximum(regular_action_timesteps, next_vision_timestep)
    edge_action_timesteps = uniform_action.float().pow(alpha.float().reciprocal())
    action_timesteps = torch.where(
        teacher_edge_mask,
        edge_action_timesteps,
        regular_action_timesteps,
    )
    return _mapped_foresight_schedule(
        vision_timesteps=vision_timesteps,
        action_timesteps=action_timesteps,
        alpha=alpha,
        teacher_edge_mask=teacher_edge_mask,
    )


def reliability_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """
    Weight unreduced MSE by reliability while normalizing by valid actions.

    The denominator deliberately excludes reliability. Dividing by the sum of
    reliability weights would cancel ``q_d`` for a single-dataset batch.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}"
        )
    reliability_fp32 = _validate_reliability(reliability).to(device=prediction.device)
    if reliability_fp32.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"reliability batch {reliability_fp32.shape[0]} does not match prediction batch " f"{prediction.shape[0]}"
        )
    reliability_shape = (reliability_fp32.shape[0],) + (1,) * (prediction.ndim - 1)
    squared_error = (prediction.float() - target.float()).square()
    weighted_error = squared_error * reliability_fp32.reshape(reliability_shape)
    return masked_mean(weighted_error, valid_mask)
