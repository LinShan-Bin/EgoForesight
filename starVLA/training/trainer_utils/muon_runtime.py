# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add auxiliary-AdamW clipping and remove ZeRO norm-mask scalar synchronization.

"""Runtime helpers for DeepSpeed's hybrid Muon/auxiliary-AdamW optimizer.

DeepSpeed 0.19.3 applies one global clipping coefficient to every optimizer
group. That is unsuitable for a hybrid optimizer: Muon gradients have already
been converted to Newton-Schulz updates when ZeRO builds
``averaged_gradients``, while auxiliary AdamW groups still contain ordinary
gradients. This module disables DeepSpeed's global policy and installs a small
pre-step hook that clips only the combined auxiliary-AdamW norm.

The implementation deliberately patches only the constructed ZeRO optimizer
instance. It does not modify the installed DeepSpeed package.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from functools import update_wrapper
from types import FunctionType, MethodType
from typing import Any, Callable

import torch
from packaging.version import InvalidVersion, Version

_STATE_ATTRIBUTE = "_egofore_muon_aux_adamw_clipper"
_NORM_MASK_ORIGINAL_ATTRIBUTE = "_egofore_original_get_grad_norm_direct"
_METRIC_PREFIX = "aux_adamw"
_SUPPORTED_DEEPSPEED_VERSION = Version("0.19.3")


def _validated_nonnegative_norm(max_norm: float | None) -> float:
    if max_norm is None:
        return 0.0
    if isinstance(max_norm, bool):
        raise ValueError("Gradient clipping max_norm must be a number, not a boolean")
    value = float(max_norm)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Gradient clipping max_norm must be finite and non-negative, got {max_norm!r}")
    return value


def configure_deepspeed_global_gradient_clipping(
    deepspeed_plugin: Any,
    max_norm: float | None,
) -> float:
    """Set the DeepSpeed global clipping value before ``Accelerator`` is built.

    Accelerate keeps the same DeepSpeed configuration in
    ``plugin.deepspeed_config`` and ``plugin.hf_ds_config.config``. Some
    releases or test doubles may expose distinct mutable mappings, so both are
    updated to prevent split-brain configuration. The update is rolled back if
    any mapping rejects the assignment.

    Hybrid Muon callers should pass ``0.0`` and install
    :func:`install_deepspeed_muon_aux_adamw_clipper` after ``prepare``.
    Conventional AdamW callers can pass their ordinary global max norm.
    """

    clipping_value = _validated_nonnegative_norm(max_norm)
    configs: list[MutableMapping[str, Any]] = []

    direct_config = getattr(deepspeed_plugin, "deepspeed_config", None)
    if direct_config is not None:
        if not isinstance(direct_config, MutableMapping):
            raise TypeError(
                f"deepspeed_plugin.deepspeed_config must be a mutable mapping, got {type(direct_config).__name__}"
            )
        configs.append(direct_config)

    hf_ds_config = getattr(deepspeed_plugin, "hf_ds_config", None)
    wrapped_config = getattr(hf_ds_config, "config", None)
    if wrapped_config is not None:
        if not isinstance(wrapped_config, MutableMapping):
            raise TypeError(
                f"deepspeed_plugin.hf_ds_config.config must be a mutable mapping, got {type(wrapped_config).__name__}"
            )
        if all(wrapped_config is not config for config in configs):
            configs.append(wrapped_config)

    if not configs:
        raise RuntimeError(
            "DeepSpeed plugin does not expose a mutable deepspeed_config; "
            "configure clipping before constructing Accelerator"
        )

    missing = object()
    previous_values = [config.get("gradient_clipping", missing) for config in configs]
    try:
        for config in configs:
            config["gradient_clipping"] = clipping_value
    except Exception:
        for config, previous_value in zip(configs, previous_values, strict=True):
            if previous_value is missing:
                config.pop("gradient_clipping", None)
            else:
                config["gradient_clipping"] = previous_value
        raise

    return clipping_value


def _installed_deepspeed_version() -> Version:
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError("Hybrid Muon runtime requires DeepSpeed 0.19.3") from error
    try:
        return Version(str(deepspeed.__version__))
    except (AttributeError, InvalidVersion) as error:
        raise RuntimeError(f"Could not determine the installed DeepSpeed version: {error}") from error


def validate_deepspeed_muon_runtime_version() -> str:
    """Require the DeepSpeed release whose ZeRO reduce-scatter Muon path is fixed."""

    installed_version = _installed_deepspeed_version()
    if installed_version != _SUPPORTED_DEEPSPEED_VERSION:
        raise RuntimeError(
            "EgoForesight's hybrid Muon ZeRO hook is pinned to DeepSpeed "
            f"{_SUPPORTED_DEEPSPEED_VERSION}; detected {installed_version}"
        )
    return str(installed_version)


@dataclass
class _AuxAdamWClipState:
    max_norm: float
    zero_stage: int
    aux_group_indices: tuple[int, ...]
    muon_group_indices: tuple[int, ...]
    group_metric_names: dict[int, str]
    original_step: Callable[..., Any]
    last_metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)


def _zero_stage(engine: Any) -> int:
    stage_method = getattr(engine, "zero_optimization_stage", None)
    if not callable(stage_method):
        raise RuntimeError("Expected a DeepSpeedEngine exposing zero_optimization_stage()")
    stage_value = stage_method()
    return int(getattr(stage_value, "value", stage_value))


def _zero_optimizer(engine: Any) -> Any:
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("DeepSpeedEngine does not expose its ZeRO optimizer")
    return optimizer


def _mask_invalid_norm_without_scalar_sync(tensor: torch.Tensor, device: Any = None) -> None:
    # ZeRO's norm is FP32. A Python scalar selects masked_fill_'s Scalar overload;
    # a CUDA scalar tensor would be copied back to the host via .item().
    norm_is_inf = tensor.isinf()
    norm_is_nan = tensor.isnan()
    tensor.masked_fill_(norm_is_nan.logical_or(norm_is_inf), -1.0)


def deepspeed_async_norm_mask_enabled(engine: Any) -> bool:
    """Report whether this engine uses the instance-local asynchronous norm mask."""
    return hasattr(_zero_optimizer(engine), _NORM_MASK_ORIGINAL_ATTRIBUTE)


def configure_deepspeed_async_norm_mask(engine: Any, enabled: bool = True) -> dict[str, bool]:
    """Remove two scalar-copy synchronizations per ZeRO-1/2 gradient norm.

    Keep DeepSpeed 0.19.3's norm method bytecode, FP64 accumulation, collective
    order, and DP/MP/replicated-parameter handling. A private globals dictionary
    changes only the final invalid-norm mask for this optimizer instance.
    Other engines and installed DeepSpeed modules remain untouched. Disabling
    restores the original bound method, allowing an exact baseline for A/B.
    """
    if not isinstance(enabled, bool):
        raise TypeError("trainer.optimizer.async_norm_mask must be a boolean")
    optimizer = _zero_optimizer(engine)
    original = getattr(optimizer, _NORM_MASK_ORIGINAL_ATTRIBUTE, None)
    if not enabled:
        if original is not None:
            optimizer.get_grad_norm_direct = original
            delattr(optimizer, _NORM_MASK_ORIGINAL_ATTRIBUTE)
        return {"enabled": False, "installed": False}
    if original is not None:
        return {"enabled": True, "installed": False}

    validate_deepspeed_muon_runtime_version()
    if _zero_stage(engine) not in (1, 2) or bool(getattr(optimizer, "cpu_offload", False)):
        raise RuntimeError("Asynchronous norm masking requires DeepSpeed ZeRO-1/2 without optimizer offload")

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    original = optimizer.get_grad_norm_direct
    if not isinstance(original, MethodType) or original.__func__ is not DeepSpeedZeroOptimizer.get_grad_norm_direct:
        raise RuntimeError("Asynchronous norm masking requires the unmodified DeepSpeed get_grad_norm_direct method")
    function = original.__func__
    mask_name = "mask_nan_or_inf_with_val_inplace"
    if mask_name not in function.__code__.co_names or mask_name not in function.__globals__:
        raise RuntimeError("DeepSpeed get_grad_norm_direct no longer calls the supported invalid-norm mask")
    private_globals = dict(function.__globals__)
    private_globals[mask_name] = _mask_invalid_norm_without_scalar_sync
    patched = FunctionType(
        function.__code__, private_globals, function.__name__, function.__defaults__, function.__closure__
    )
    patched.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(patched, function)
    optimizer.get_grad_norm_direct = MethodType(patched, optimizer)
    setattr(optimizer, _NORM_MASK_ORIGINAL_ATTRIBUTE, original)
    return {"enabled": True, "installed": True}


def _indexed(container: Any, index: int, label: str) -> Any:
    try:
        return container[index]
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError(f"DeepSpeed ZeRO {label} does not contain optimizer group {index}") from error


def _group_metric_names(param_groups: list[dict[str, Any]], aux_indices: tuple[int, ...]) -> dict[int, str]:
    result: dict[int, str] = {}
    used_names: set[str] = set()
    for index in aux_indices:
        raw_name = str(param_groups[index].get("name", f"group_{index}")).strip("/")
        base_name = raw_name or f"group_{index}"
        metric_name = base_name
        if metric_name in used_names:
            metric_name = f"{base_name}_{index}"
        used_names.add(metric_name)
        result[index] = metric_name
    return result


def _validate_and_build_state(engine: Any, max_norm: float) -> tuple[Any, _AuxAdamWClipState]:
    validate_deepspeed_muon_runtime_version()
    zero_stage = _zero_stage(engine)
    if zero_stage not in (1, 2):
        raise RuntimeError(f"Auxiliary-AdamW-only clipping currently supports DeepSpeed ZeRO-1/2, got ZeRO-{zero_stage}")

    zero_optimizer = _zero_optimizer(engine)
    if bool(getattr(zero_optimizer, "cpu_offload", False)):
        raise RuntimeError("Auxiliary-AdamW-only clipping does not support ZeRO optimizer offload")

    try:
        global_clip = float(zero_optimizer.clip_grad)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose a numeric clip_grad value") from error
    if not math.isfinite(global_clip) or global_clip != 0.0:
        raise RuntimeError(
            "DeepSpeed global gradient clipping must be exactly 0.0 before installing "
            f"auxiliary-AdamW-only clipping; detected {global_clip!r}"
        )

    inner_optimizer = getattr(zero_optimizer, "optimizer", None)
    if type(inner_optimizer).__name__ != "MuonWithAuxAdam":
        detected = "None" if inner_optimizer is None else type(inner_optimizer).__name__
        raise RuntimeError(f"Expected DeepSpeed MuonWithAuxAdam, detected {detected}")

    param_groups = getattr(inner_optimizer, "param_groups", None)
    if not isinstance(param_groups, list) or not param_groups:
        raise RuntimeError("MuonWithAuxAdam does not expose a non-empty param_groups list")

    aux_indices: list[int] = []
    muon_indices: list[int] = []
    for index, group in enumerate(param_groups):
        if not isinstance(group, dict) or "use_muon" not in group:
            raise RuntimeError(f"Hybrid optimizer group {index} is missing the required use_muon marker")
        (muon_indices if bool(group["use_muon"]) else aux_indices).append(index)
    if not aux_indices:
        raise RuntimeError("Hybrid Muon optimizer has no auxiliary AdamW parameter groups")
    if not muon_indices:
        raise RuntimeError("Hybrid Muon optimizer has no Muon parameter groups")

    params_in_partition = getattr(zero_optimizer, "params_in_partition", None)
    if params_in_partition is None:
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose params_in_partition")
    try:
        partition_group_count = len(params_in_partition)
    except TypeError as error:
        raise RuntimeError("DeepSpeed params_in_partition is not an indexed group collection") from error
    if partition_group_count != len(param_groups):
        raise RuntimeError(
            "DeepSpeed/inner-optimizer group mismatch: "
            f"params_in_partition={partition_group_count}, param_groups={len(param_groups)}"
        )

    if getattr(zero_optimizer, "averaged_gradients", None) is None:
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose averaged_gradients")
    if not callable(getattr(zero_optimizer, "get_grad_norm_direct", None)):
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose get_grad_norm_direct()")
    original_step = getattr(zero_optimizer, "step", None)
    if not callable(original_step):
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose step()")

    aux_group_indices = tuple(aux_indices)
    state = _AuxAdamWClipState(
        max_norm=max_norm,
        zero_stage=zero_stage,
        aux_group_indices=aux_group_indices,
        muon_group_indices=tuple(muon_indices),
        group_metric_names=_group_metric_names(param_groups, aux_group_indices),
        original_step=original_step,
    )
    return zero_optimizer, state


def _collect_aux_group_norms(
    zero_optimizer: Any,
    state: _AuxAdamWClipState,
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    averaged_gradients = zero_optimizer.averaged_gradients
    params_in_partition = zero_optimizer.params_in_partition
    group_norms: dict[int, torch.Tensor] = {}
    aux_gradients: list[Any] = []

    for index in state.aux_group_indices:
        gradients = _indexed(averaged_gradients, index, "averaged_gradients")
        parameters = _indexed(params_in_partition, index, "params_in_partition")
        if gradients is None:
            raise RuntimeError(f"DeepSpeed averaged gradients for auxiliary AdamW group {index} are unavailable")
        norm = zero_optimizer.get_grad_norm_direct(gradients, parameters)
        if not torch.is_tensor(norm) or norm.numel() != 1:
            raise RuntimeError(f"DeepSpeed returned a non-scalar norm for auxiliary AdamW group {index}: {norm!r}")
        group_norms[index] = norm.detach().float()
        aux_gradients.append(gradients)

    if bool(getattr(zero_optimizer, "has_moe_layers", False)):
        average_expert_norms = getattr(zero_optimizer, "_average_expert_grad_norms", None)
        if not callable(average_expert_norms):
            raise RuntimeError("DeepSpeed MoE runtime does not expose _average_expert_grad_norms()")
        zero = next(iter(group_norms.values())).new_zeros(())
        all_group_norms = [
            group_norms.get(index, zero.clone()) for index in range(len(zero_optimizer.optimizer.param_groups))
        ]
        average_expert_norms(all_group_norms)
        for index in state.aux_group_indices:
            group_norms[index] = all_group_norms[index]

    return group_norms, aux_gradients


@torch.no_grad()
def _clip_aux_adamw_gradients(zero_optimizer: Any, state: _AuxAdamWClipState) -> None:
    group_norms, aux_gradient_groups = _collect_aux_group_norms(zero_optimizer, state)
    stacked_norms = torch.stack([group_norms[index] for index in state.aux_group_indices])
    group_norms_valid = torch.isfinite(stacked_norms) & stacked_norms.ge(0.0)
    all_norms_valid = group_norms_valid.all()
    safe_norms = torch.where(group_norms_valid, stacked_norms, torch.zeros_like(stacked_norms))
    scaled_global_norm = torch.linalg.vector_norm(safe_norms, ord=2)

    try:
        loss_scale = float(zero_optimizer.loss_scale)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("DeepSpeed ZeRO optimizer does not expose a numeric loss_scale") from error
    if not math.isfinite(loss_scale) or loss_scale <= 0.0:
        raise RuntimeError(f"DeepSpeed loss_scale must be finite and positive, got {loss_scale!r}")

    unscaled_global_norm = scaled_global_norm / loss_scale
    max_norm = unscaled_global_norm.new_tensor(state.max_norm)
    clip_ratio = (unscaled_global_norm + 1.0e-6) / max_norm
    clip_coefficient = torch.reciprocal(torch.clamp(clip_ratio, min=1.0))
    clip_coefficient = torch.where(
        all_norms_valid,
        clip_coefficient,
        torch.ones_like(clip_coefficient),
    )

    for gradients in aux_gradient_groups:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)

    nan = unscaled_global_norm.new_tensor(float("nan"))
    reported_global_norm = torch.where(all_norms_valid, unscaled_global_norm, nan)
    metrics: dict[str, torch.Tensor | float] = {
        f"{_METRIC_PREFIX}_clip_input_norm/pre_clip_global": reported_global_norm.detach(),
        f"{_METRIC_PREFIX}_grad_clip/max_norm": state.max_norm,
        f"{_METRIC_PREFIX}_grad_clip/coefficient": clip_coefficient.detach(),
        f"{_METRIC_PREFIX}_grad_clip/was_clipped": (all_norms_valid & clip_coefficient.lt(1.0)).detach().float(),
        f"{_METRIC_PREFIX}_grad_clip/invalid": (~all_norms_valid).detach().float(),
    }
    for index in state.aux_group_indices:
        group_norm = group_norms[index] / loss_scale
        group_valid = torch.isfinite(group_norms[index]) & group_norms[index].ge(0.0)
        metrics[f"{_METRIC_PREFIX}_clip_input_norm/group/{state.group_metric_names[index]}"] = torch.where(
            group_valid, group_norm, nan
        ).detach()
    state.last_metrics = metrics


def _zero_step_with_aux_adamw_clipping(zero_optimizer: Any, *args: Any, **kwargs: Any) -> Any:
    state = getattr(zero_optimizer, _STATE_ATTRIBUTE, None)
    if not isinstance(state, _AuxAdamWClipState):
        raise RuntimeError("Muon auxiliary-AdamW clipping state is missing from the ZeRO optimizer")
    _clip_aux_adamw_gradients(zero_optimizer, state)
    return state.original_step(*args, **kwargs)


def install_deepspeed_muon_aux_adamw_clipper(
    engine: Any,
    max_norm: float,
) -> dict[str, Any]:
    """Install an aux-AdamW-only global-norm clipper on a ZeRO-1/2 engine.

    The function is idempotent for the same ``max_norm``. Reinstalling with a
    different threshold raises instead of silently changing an active training
    policy.
    """

    clipping_value = _validated_nonnegative_norm(max_norm)
    if clipping_value <= 0.0:
        raise ValueError(f"Auxiliary AdamW max_norm must be positive, got {max_norm!r}")

    zero_optimizer = _zero_optimizer(engine)
    existing_state = getattr(zero_optimizer, _STATE_ATTRIBUTE, None)
    if existing_state is not None:
        if not isinstance(existing_state, _AuxAdamWClipState):
            raise RuntimeError(f"ZeRO optimizer attribute {_STATE_ATTRIBUTE} is already occupied")
        if existing_state.max_norm != clipping_value:
            raise RuntimeError(
                "Auxiliary AdamW clipper is already installed with "
                f"max_norm={existing_state.max_norm}, requested {clipping_value}"
            )
        return {
            "installed": False,
            "max_norm": existing_state.max_norm,
            "zero_stage": existing_state.zero_stage,
            "aux_group_indices": existing_state.aux_group_indices,
            "muon_group_indices": existing_state.muon_group_indices,
        }

    zero_optimizer, state = _validate_and_build_state(engine, clipping_value)
    setattr(zero_optimizer, _STATE_ATTRIBUTE, state)
    try:
        zero_optimizer.step = MethodType(_zero_step_with_aux_adamw_clipping, zero_optimizer)
    except Exception:
        delattr(zero_optimizer, _STATE_ATTRIBUTE)
        raise

    return {
        "installed": True,
        "max_norm": state.max_norm,
        "zero_stage": state.zero_stage,
        "aux_group_indices": state.aux_group_indices,
        "muon_group_indices": state.muon_group_indices,
    }


def get_deepspeed_muon_aux_adamw_clip_metrics(
    engine: Any,
    *,
    clear: bool = False,
) -> dict[str, float]:
    """Return the most recent hook metrics as host floats.

    Conversion synchronizes scalar device tensors, so callers should invoke
    this accessor only on logging steps. Non-Muon engines and hooks that have
    not completed a step return an empty dictionary.
    """

    zero_optimizer = getattr(engine, "optimizer", None)
    state = getattr(zero_optimizer, _STATE_ATTRIBUTE, None)
    if not isinstance(state, _AuxAdamWClipState) or not state.last_metrics:
        return {}

    result: dict[str, float] = {}
    for name, value in state.last_metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(f"Expected scalar Muon clipping metric {name}, got shape={tuple(value.shape)}")
            result[name] = float(value.detach().float().cpu())
        else:
            result[name] = float(value)
    if clear:
        state.last_metrics = {}
    return result
