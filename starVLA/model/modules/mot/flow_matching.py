# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add pure-PyTorch rectified-flow training and Euler sampling utilities for MoT.

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowMatchSample:
    """
    Inputs and targets for rectified-flow training.

    StarVLA uses the convention ``clean == t=0`` and ``noise == t=1``.
    Consequently, the constant target velocity is ``noise - clean``.
    """

    noisy: torch.Tensor
    velocity: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor


def _validate_reference(reference: torch.Tensor) -> None:
    if not isinstance(reference, torch.Tensor):
        raise TypeError(f"reference must be a torch.Tensor, got {type(reference)!r}")
    if reference.ndim == 0:
        raise ValueError("reference must include a batch dimension")
    if not reference.is_floating_point():
        raise TypeError(f"reference must have a floating dtype, got {reference.dtype}")


def _validate_unit_interval(timesteps: torch.Tensor, name: str = "timesteps") -> None:
    if not timesteps.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype, got {timesteps.dtype}")
    if timesteps.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    detached = timesteps.detach()
    if torch.any(detached < 0) or torch.any(detached > 1):
        minimum = detached.min().item()
        maximum = detached.max().item()
        raise ValueError(f"{name} must lie in [0, 1], got range [{minimum}, {maximum}]")


def expand_timesteps(timesteps: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """
    Convert scalar, batch, or per-token timesteps to a reference-broadcastable tensor.

    Supported examples for a reference shaped ``[B, N, D]`` are a scalar,
    ``[B]``, ``[B, N]``, and ``[B, N, 1]``. Missing dimensions are appended
    on the right, so ``[B]`` is interpreted as one timestep per batch item.
    """

    _validate_reference(reference)
    dtype = reference.dtype
    time = torch.as_tensor(timesteps, device=reference.device, dtype=dtype)
    _validate_unit_interval(time)
    if time.ndim > reference.ndim:
        raise ValueError(f"timesteps rank {time.ndim} cannot broadcast to reference rank {reference.ndim}")
    if time.ndim > 0 and time.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"timestep batch {time.shape[0]} does not match reference batch {reference.shape[0]}")

    expanded_shape = tuple(time.shape) + (1,) * (reference.ndim - time.ndim)
    time = time.reshape(expanded_shape)
    try:
        torch.broadcast_shapes(time.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"timesteps shape {tuple(time.shape)} cannot broadcast to reference shape " f"{tuple(reference.shape)}"
        ) from error
    return time


def sample_timesteps(
    batch_size: int,
    *,
    num_tokens: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniform batch-level ``[B]`` or per-token ``[B, N]`` timesteps."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_tokens is not None and num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating, got {dtype}")
    shape = (batch_size,) if num_tokens is None else (batch_size, num_tokens)
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def sample_beta_timesteps(
    batch_size: int,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    num_tokens: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample batch-level ``[B]`` or per-token ``[B, N]`` Beta timesteps."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_tokens is not None and num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating, got {dtype}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and positive, got {beta}")

    shape = (batch_size,) if num_tokens is None else (batch_size, num_tokens)
    sampling_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    alpha_concentration = torch.full(shape, alpha, device=device, dtype=sampling_dtype)
    beta_concentration = torch.full(shape, beta, device=device, dtype=sampling_dtype)
    alpha_gamma = torch._standard_gamma(alpha_concentration, generator=generator)
    beta_gamma = torch._standard_gamma(beta_concentration, generator=generator)
    denominator = (alpha_gamma + beta_gamma).clamp_min(torch.finfo(sampling_dtype).tiny)
    return (alpha_gamma / denominator).to(dtype=dtype)


def sample_logit_normal_timesteps(
    batch_size: int,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    num_tokens: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample timesteps as ``sigmoid(N(mean, std))``."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_tokens is not None and num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating, got {dtype}")
    if not math.isfinite(mean):
        raise ValueError(f"mean must be finite, got {mean}")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be finite and positive, got {std}")

    shape = (batch_size,) if num_tokens is None else (batch_size, num_tokens)
    sampling_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    logits = torch.randn(shape, device=device, dtype=sampling_dtype, generator=generator)
    return torch.sigmoid(logits.mul(std).add(mean)).to(dtype=dtype)


def interpolate(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor | float,
) -> torch.Tensor:
    """
    Interpolate along the rectified-flow path.

    ``x_t = (1 - t) * clean + t * noise`` with clean at zero and noise at one.
    Timesteps may be shared per batch or specified independently per token.
    """

    _validate_reference(clean)
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean and noise must have identical shapes, got {tuple(clean.shape)} and " f"{tuple(noise.shape)}"
        )
    if clean.device != noise.device:
        raise ValueError(f"clean and noise must share a device, got {clean.device} and {noise.device}")
    time = expand_timesteps(timesteps, clean)
    return clean * (1.0 - time) + noise * time


def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the clean-to-noise rectified-flow velocity ``noise - clean``."""

    _validate_reference(clean)
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean and noise must have identical shapes, got {tuple(clean.shape)} and " f"{tuple(noise.shape)}"
        )
    return noise - clean


def dual_timesteps(
    t: torch.Tensor | float,
    s: torch.Tensor | float,
    mask: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Select between two shared timesteps for every token.

    A true mask entry selects ``s`` and a false entry selects ``t``. Sampling
    ``t`` and ``s`` independently from the same distribution preserves the
    ordinary flow-matching marginal timestep distribution for every token.
    """

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)!r}")
    if mask.ndim < 2:
        raise ValueError(f"mask must be at least [B, N], got shape {tuple(mask.shape)}")
    if mask.shape[0] <= 0:
        raise ValueError("mask must include at least one batch item")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating, got {dtype}")
    reference = torch.empty(mask.shape, device=mask.device, dtype=dtype)
    t_expanded = expand_timesteps(t, reference)
    s_expanded = expand_timesteps(s, reference)
    return torch.where(mask.bool(), s_expanded, t_expanded).expand_as(reference)


def make_flow_match_sample(
    clean: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    per_token: bool = False,
    generator: torch.Generator | None = None,
) -> FlowMatchSample:
    """Build an ordinary batch-timestep or per-token rectified-flow training sample."""

    _validate_reference(clean)
    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    if timesteps is None:
        num_tokens = clean.shape[1] if per_token else None
        timesteps = sample_timesteps(
            clean.shape[0],
            num_tokens=num_tokens,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    noisy = interpolate(clean, noise, timesteps)
    return FlowMatchSample(
        noisy=noisy,
        velocity=velocity_target(clean, noise),
        timesteps=timesteps,
        noise=noise,
    )


def make_dual_flow_match_sample(
    clean: torch.Tensor,
    *,
    t: torch.Tensor | float,
    s: torch.Tensor | float,
    mask: torch.Tensor,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowMatchSample:
    """Build a per-token flow sample whose token times are selected from shared ``t`` and ``s``."""

    _validate_reference(clean)
    if clean.ndim < 2:
        raise ValueError(f"dual flow matching expects tokenized input [B, N, ...], got {tuple(clean.shape)}")
    if mask.shape != clean.shape[: mask.ndim]:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match a prefix of clean shape {tuple(clean.shape)}")
    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    times = dual_timesteps(t, s, mask, dtype=clean.dtype)
    return FlowMatchSample(
        noisy=interpolate(clean, noise, times),
        velocity=velocity_target(clean, noise),
        timesteps=times,
        noise=noise,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a weighted mean; an all-zero mask returns a differentiable scalar zero."""

    if mask is None:
        return values.mean()
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)!r}")
    weights = mask.to(device=values.device, dtype=values.dtype)
    if weights.ndim > values.ndim:
        raise ValueError(f"mask rank {weights.ndim} cannot broadcast to value rank {values.ndim}")
    weights = weights.reshape(tuple(weights.shape) + (1,) * (values.ndim - weights.ndim))
    try:
        weights = weights.expand_as(values)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to values shape {tuple(values.shape)}"
        ) from error
    if torch.any(weights.detach() < 0):
        raise ValueError("mask weights must be non-negative")
    numerator = (values * weights).sum()
    denominator = weights.sum().clamp_min(1.0)
    return numerator / denominator


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable float32 MSE with an optional batch, token, or element mask."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}"
        )
    squared_error = (prediction.float() - target.float()).square()
    return masked_mean(squared_error, mask)


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable float32 L1 loss with an optional batch, token, or element mask."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}"
        )
    absolute_error = (prediction.float() - target.float()).abs()
    return masked_mean(absolute_error, mask)


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    *,
    num_steps: int = 4,
    start_time: float = 1.0,
    end_time: float = 0.0,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Integrate a velocity model from noise to clean with explicit Euler steps.

    The default is the requested four-step sampler over ``1 -> 0``. The model
    receives the current state and a float32 batch timestep tensor shaped
    ``[B]``. Returning a trajectory includes the initial noise and every update.
    """

    _validate_reference(noise)
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0.0 <= end_time < start_time <= 1.0:
        raise ValueError(f"expected 0 <= end_time < start_time <= 1, got {end_time} and {start_time}")

    state = noise
    time_grid = torch.linspace(
        start_time,
        end_time,
        num_steps + 1,
        device=noise.device,
        dtype=torch.float32,
    )
    trajectory = [state] if return_trajectory else None
    for step in range(num_steps):
        batch_time = time_grid[step].expand(noise.shape[0])
        velocity = velocity_fn(state, batch_time)
        if not isinstance(velocity, torch.Tensor):
            raise TypeError(f"velocity_fn must return a torch.Tensor, got {type(velocity)!r}")
        if velocity.shape != state.shape:
            raise ValueError(f"velocity shape {tuple(velocity.shape)} does not match state shape {tuple(state.shape)}")
        delta_time = (time_grid[step + 1] - time_grid[step]).to(dtype=state.dtype)
        state = state + delta_time * velocity.to(dtype=state.dtype)
        if trajectory is not None:
            trajectory.append(state)

    if trajectory is not None:
        return state, tuple(trajectory)
    return state
