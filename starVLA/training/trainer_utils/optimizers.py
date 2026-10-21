# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Build hybrid Muon/AdamW groups with LR policies and fused-projection layouts.

"""Optimizer construction helpers shared by StarVLA training entrypoints."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from packaging.version import Version

from starVLA.training.trainer_utils.muon_fused import fused_muon_layout, install_single_device_fused_context
from starVLA.training.trainer_utils.muon_ns import configure_deepspeed_muon_ns_kernel
from starVLA.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    resolve_optimizer_policy_group,
)

_OUTPUT_MODULE_NAMES = {
    "action_output",
    "classifier",
    "lm_head",
    "output_head",
    "output_layer",
    "output_projection",
    "score",
    "vision_output",
}
_EMBEDDING_NAME_PARTS = {
    "embed",
    "embed_tokens",
    "embedding",
    "embeddings",
    "word_embeddings",
    "wte",
}
_MUON_ADJUST_LR_FUNCTIONS = {"original", "match_rms_adamw", "spectral_unclamped"}
_MOT_MUON_DIRECT_MODULE_NAMES = {
    "action_cross_kv",
    "action_cross_out",
    "action_cross_q",
    "action_joint_out",
    "action_joint_qkv",
    "context_projection",
    "vision_cross_kv",
    "vision_cross_out",
    "vision_cross_q",
    "vision_input",
    "vision_joint_out",
    "vision_joint_qkv",
}
_MOT_MUON_FFN_MODULE_NAMES = {"down_proj", "gate_proj", "up_proj"}
_MOT_MUON_FFN_PARENT_NAMES = {"action_ffn", "vision_ffn"}
_QWEN_LANGUAGE_ATTENTION_MODULE_NAMES = {
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "v_proj",
}
_QWEN_LANGUAGE_ATTENTION_PARENT_NAMES = {"linear_attn", "self_attn"}
_QWEN_LANGUAGE_MLP_MODULE_NAMES = {"down_proj", "gate_proj", "up_proj"}
_QWEN_VISUAL_ATTENTION_MODULE_NAMES = {"proj", "qkv"}
_QWEN_VISUAL_MLP_MODULE_NAMES = {"linear_fc1", "linear_fc2"}


def _parameter_aliases(model: nn.Module) -> dict[int, tuple[str, ...]]:
    """Return every qualified name for each parameter, including tied-weight aliases."""
    aliases: dict[int, list[str]] = defaultdict(list)
    try:
        named_parameters: Iterable[tuple[str, nn.Parameter]] = model.named_parameters(remove_duplicate=False)
    except TypeError:  # PyTorch releases before ``remove_duplicate`` was added.
        named_parameters = model.named_parameters()
    for name, parameter in named_parameters:
        aliases[id(parameter)].append(name)
    return {parameter_id: tuple(names) for parameter_id, names in aliases.items()}


def _embedding_parameter_ids(model: nn.Module) -> set[int]:
    parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            parameter_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    return parameter_ids


def _name_parts(name: str) -> set[str]:
    return {part.lower() for part in name.split(".")}


def _is_named_embedding(names: tuple[str, ...]) -> bool:
    return any(_name_parts(name) & _EMBEDDING_NAME_PARTS for name in names)


def _is_named_output(names: tuple[str, ...]) -> bool:
    return any(_name_parts(name) & _OUTPUT_MODULE_NAMES for name in names)


def _is_index(value: str) -> bool:
    return value.isdigit()


def _is_whitelisted_mot_matrix(name: str) -> bool:
    parts = name.lower().split(".")
    if "mot_model" not in parts or len(parts) < 2 or parts[-1] != "weight":
        return False
    module_name = parts[-2]
    if module_name in _MOT_MUON_DIRECT_MODULE_NAMES:
        return True
    parent_name = parts[-3] if len(parts) >= 3 else ""
    if module_name in _MOT_MUON_FFN_MODULE_NAMES and parent_name in _MOT_MUON_FFN_PARENT_NAMES:
        return True
    if "full_experts" in "_".join(parts):
        if parent_name == "mlp" and module_name in _MOT_MUON_FFN_MODULE_NAMES:
            return True
        if parent_name == "self_attn" and module_name in _QWEN_LANGUAGE_ATTENTION_MODULE_NAMES:
            return True
    if "adapters" in "_".join(parts) and module_name in {"gate_up_proj", "down_proj"}:
        return True
    return False


def _is_whitelisted_qwen_matrix(name: str) -> bool:
    parts = name.lower().split(".")
    if "qwen_vl_interface" not in parts or len(parts) < 7 or parts[-1] != "weight":
        return False

    module_name = parts[-2]
    parent_name = parts[-3]
    for index in range(len(parts) - 4):
        stack_name, collection_name, layer_index = parts[index : index + 3]
        if stack_name == "language_model" and collection_name == "layers" and _is_index(layer_index):
            if parent_name in _QWEN_LANGUAGE_ATTENTION_PARENT_NAMES:
                return module_name in _QWEN_LANGUAGE_ATTENTION_MODULE_NAMES
            if parent_name == "mlp":
                return module_name in _QWEN_LANGUAGE_MLP_MODULE_NAMES
        if stack_name == "visual" and collection_name == "blocks" and _is_index(layer_index):
            if parent_name == "attn":
                return module_name in _QWEN_VISUAL_ATTENTION_MODULE_NAMES
            if parent_name == "mlp":
                return module_name in _QWEN_VISUAL_MLP_MODULE_NAMES
    return False


def _is_whitelisted_muon_matrix(names: tuple[str, ...]) -> bool:
    return any(_is_whitelisted_mot_matrix(name) or _is_whitelisted_qwen_matrix(name) for name in names)


def is_muon_parameter(
    parameter: nn.Parameter,
    names: tuple[str, ...],
    *,
    embedding_parameter_ids: set[int],
) -> bool:
    """Select explicitly whitelisted dense hidden-layer matrices for Muon.

    All parameters default to auxiliary AdamW. Muon is limited to standard MoT
    attention/FFN matrices, dense MoT context/vision projections, and standard
    Qwen language/vision Transformer attention/MLP matrices. In particular,
    adaptive-norm modulation, timestep conditioning, sparse action/schema paths,
    representation projectors, embeddings, output heads, biases, normalization
    gains, and non-matrix parameters remain on the auxiliary AdamW path.
    """
    if parameter.ndim != 2:
        return False
    if id(parameter) in embedding_parameter_ids or _is_named_embedding(names):
        return False
    if _is_named_output(names):
        return False
    return _is_whitelisted_muon_matrix(names)


def _deepspeed_muon_lr_scale(shape: torch.Size, adjust_lr_fn: str) -> float:
    """Scale the group LR so DeepSpeed's original Muon update matches the requested policy."""
    rows, columns = (int(shape[0]), int(shape[1]))
    original_scale = math.sqrt(max(1.0, rows / columns))
    if adjust_lr_fn == "original":
        target_scale = original_scale
    elif adjust_lr_fn == "match_rms_adamw":
        target_scale = 0.2 * math.sqrt(max(rows, columns))
    elif adjust_lr_fn == "spectral_unclamped":
        target_scale = math.sqrt(rows / columns)
    else:
        raise ValueError(
            f"Unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; expected one of " f"{sorted(_MUON_ADJUST_LR_FUNCTIONS)}"
        )
    # DeepSpeed's muon_update already applies ``original_scale``. Adjusting the
    # optimizer-group LR by this ratio produces the selected effective update.
    return target_scale / original_scale


def _group_weight_decay(group_name: str, optimizer_config: Any) -> float:
    """Resolve an optional MoT-specific decay, including aliased action modules."""
    default_weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    mot_weight_decay = optimizer_config.get("mot_weight_decay", None)
    if mot_weight_decay is None or resolve_optimizer_policy_group(group_name) != "mot_model":
        return default_weight_decay

    resolved_weight_decay = float(mot_weight_decay)
    if not math.isfinite(resolved_weight_decay) or resolved_weight_decay < 0.0:
        raise ValueError(
            "trainer.optimizer.mot_weight_decay must be a finite non-negative number, " f"got {mot_weight_decay!r}"
        )
    return resolved_weight_decay


def _group_muon_lr_multiplier(group_name: str, optimizer_config: Any) -> float:
    """Resolve a Muon-only LR multiplier without changing auxiliary AdamW groups."""
    configured_multipliers = optimizer_config.get("muon_lr_multiplier", {})
    policy_group = resolve_optimizer_policy_group(group_name)
    multiplier = float(configured_multipliers.get(policy_group, 1.0))
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "trainer.optimizer.muon_lr_multiplier." f"{policy_group} must be finite and positive, got {multiplier}"
        )
    return multiplier


def build_muon_param_groups(model: nn.Module, cfg: Any) -> list[dict[str, Any]]:
    """Split module-specific LR groups into Muon matrices and auxiliary AdamW parameters."""
    optimizer_config = cfg.trainer.optimizer
    aliases = _parameter_aliases(model)
    embedding_parameter_ids = _embedding_parameter_ids(model)
    base_groups = build_param_lr_groups(model=model, cfg=cfg)

    betas = tuple(optimizer_config.get("betas", (0.9, 0.95)))
    eps = float(optimizer_config.get("eps", 1.0e-8))
    momentum = float(optimizer_config.get("momentum", 0.95))
    ns_method = str(optimizer_config.get("ns_method", "gram"))
    adjust_lr_fn = str(optimizer_config.get("adjust_lr_fn", "match_rms_adamw"))
    if adjust_lr_fn not in _MUON_ADJUST_LR_FUNCTIONS:
        raise ValueError(
            f"Unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; expected one of " f"{sorted(_MUON_ADJUST_LR_FUNCTIONS)}"
        )

    result: list[dict[str, Any]] = []
    for base_group in base_groups:
        group_name = str(base_group.get("name", "base"))
        base_lr = float(base_group["lr"])
        group_weight_decay = _group_weight_decay(group_name, optimizer_config)
        adamw_parameters: list[nn.Parameter] = []
        muon_by_shape: dict[tuple[int, int], list[nn.Parameter]] = defaultdict(list)

        for parameter in base_group["params"]:
            names = aliases.get(id(parameter), ("<unnamed>",))
            use_muon = is_muon_parameter(
                parameter,
                names,
                embedding_parameter_ids=embedding_parameter_ids,
            )
            # DeepSpeed ZeRO inspects this marker on the original, unflattened
            # parameter before it builds optimizer partitions.
            parameter.use_muon = use_muon
            parameter._egofore_muon_layout = None
            if use_muon:
                layouts = [fused_muon_layout(model, name, parameter, adjust_lr_fn) for name in names]
                known_layouts = {layout for layout in layouts if layout is not None}
                if len(known_layouts) > 1:
                    raise ValueError(f"Conflicting fused Muon layouts for tied parameter: {names}")
                parameter._egofore_muon_layout = next(iter(known_layouts), None)
                muon_by_shape[(int(parameter.shape[0]), int(parameter.shape[1]))].append(parameter)
            else:
                adamw_parameters.append(parameter)

        if adamw_parameters:
            result.append(
                {
                    "params": adamw_parameters,
                    "lr": base_lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": group_weight_decay,
                    "use_muon": False,
                    "optimizer_type": "adamw",
                    "name": f"{group_name}/adamw",
                }
            )

        for shape, parameters in sorted(muon_by_shape.items()):
            shape_lr_scale = _deepspeed_muon_lr_scale(torch.Size(shape), adjust_lr_fn)
            muon_lr_multiplier = _group_muon_lr_multiplier(group_name, optimizer_config)
            lr_scale = shape_lr_scale * muon_lr_multiplier
            result.append(
                {
                    "params": parameters,
                    "lr": base_lr * lr_scale,
                    "momentum": momentum,
                    # Compensate only for the shape adjustment. A deliberate
                    # Muon LR multiplier scales both the parameter update and
                    # decoupled weight decay, while auxiliary AdamW is unchanged.
                    "weight_decay": group_weight_decay / shape_lr_scale,
                    "use_muon": True,
                    "ns_method": ns_method,
                    "optimizer_type": "muon",
                    "muon_adjust_lr_fn": adjust_lr_fn,
                    "muon_lr_scale": lr_scale,
                    "muon_shape_lr_scale": shape_lr_scale,
                    "muon_lr_multiplier": muon_lr_multiplier,
                    "muon_shape": shape,
                    "name": f"{group_name}/muon_{shape[0]}x{shape[1]}",
                }
            )

    expected = Counter(id(parameter) for group in base_groups for parameter in group["params"])
    grouped = Counter(id(parameter) for group in result for parameter in group["params"])
    if grouped != expected:
        missing = sum((expected - grouped).values())
        duplicated_or_extra = sum((grouped - expected).values())
        raise RuntimeError(
            "Hybrid Muon parameter grouping did not cover trainable parameters exactly once: "
            f"missing={missing}, extra={duplicated_or_extra}"
        )
    return result


def _load_deepspeed_muon(backend: str) -> type[torch.optim.Optimizer]:
    if backend == "deepspeed":
        try:
            import deepspeed
            from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
        except ImportError as error:
            raise ImportError(
                "trainer.optimizer.name=Muon with backend=deepspeed requires DeepSpeed >=0.19.2, "
                "whose ZeRO implementation orthogonalizes original matrix gradients before flattening."
            ) from error
        if Version(deepspeed.__version__) < Version("0.19.2"):
            raise RuntimeError(
                f"DeepSpeed Muon requires version >=0.19.2, got {deepspeed.__version__}. "
                "Older ZeRO implementations flatten matrix gradients before the optimizer can orthogonalize them."
            )
        return MuonWithAuxAdam
    if backend == "single_device":
        try:
            from deepspeed.runtime.zero.muon.original_muon import SingleDeviceMuonWithAuxAdam
        except ImportError as error:
            raise ImportError(
                "trainer.optimizer.name=Muon with backend=single_device requires DeepSpeed >=0.19.2."
            ) from error
        return SingleDeviceMuonWithAuxAdam
    raise ValueError("trainer.optimizer.backend must be 'deepspeed' or 'single_device'")


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build AdamW or a DeepSpeed-compatible hybrid Muon/AdamW optimizer."""
    optimizer_config = cfg.trainer.optimizer
    optimizer_name = str(optimizer_config.get("name", "AdamW")).lower()
    if optimizer_name == "adamw":
        param_groups = build_param_lr_groups(model=model, cfg=cfg)
        if optimizer_config.get("mot_weight_decay", None) is not None:
            for group in param_groups:
                group_name = str(group.get("name", "base"))
                group["weight_decay"] = _group_weight_decay(group_name, optimizer_config)
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.trainer.learning_rate.base,
            betas=tuple(optimizer_config.betas),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.eps,
        )
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported trainer.optimizer.name={optimizer_config.name!r}; expected AdamW or Muon")

    backend = str(optimizer_config.get("backend", "deepspeed")).lower()
    optimizer_class = _load_deepspeed_muon(backend)
    ns_kernel = str(optimizer_config.get("ns_kernel", "deepspeed")).lower()
    if backend != "deepspeed" and ns_kernel != "deepspeed":
        raise ValueError("A custom Muon ns_kernel is only supported with backend=deepspeed")
    ns_backend = configure_deepspeed_muon_ns_kernel(ns_kernel, require_zero=backend == "deepspeed")
    optimizer = optimizer_class(build_muon_param_groups(model, cfg))
    if backend == "single_device":
        install_single_device_fused_context(optimizer)
    optimizer._egofore_muon_ns_backend = ns_backend
    # Transformers' cosine_with_min_lr scheduler derives min_lr_rate from the
    # optimizer default LR. DeepSpeed's MuonWithAuxAdam intentionally starts
    # with empty defaults because every hybrid group is fully specified.
    optimizer.defaults["lr"] = float(cfg.trainer.learning_rate.base)
    return optimizer


def validate_muon_runtime(model: nn.Module, cfg: Any) -> dict[str, Any] | None:
    """Fail closed if the ZeRO-aware Muon optimizer is not actually running under ZeRO."""
    optimizer_config = cfg.trainer.optimizer
    if str(optimizer_config.get("name", "AdamW")).lower() != "muon":
        return None
    backend = str(optimizer_config.get("backend", "deepspeed")).lower()
    if backend == "single_device":
        if type(model).__name__ == "DeepSpeedEngine":
            raise RuntimeError(
                "Muon backend=single_device cannot be wrapped by DeepSpeed. ZeRO would pre-orthogonalize gradients "
                "and the single-device optimizer would then try to apply Muon again to flattened parameters."
            )
        return {"backend": backend, "zero_stage": 0, "optimizer_offload": False}

    zero_stage_method = getattr(model, "zero_optimization_stage", None)
    if not callable(zero_stage_method) or type(model).__name__ != "DeepSpeedEngine":
        raise RuntimeError(
            "Muon backend=deepspeed is only valid after Accelerator wraps the model in a DeepSpeedEngine. "
            "Without ZeRO, MuonWithAuxAdam would apply raw gradients without Newton-Schulz orthogonalization."
        )
    zero_stage_value = zero_stage_method()
    zero_stage = int(getattr(zero_stage_value, "value", zero_stage_value))
    if zero_stage == 3 and any(getattr(parameter, "_egofore_muon_layout", None) for parameter in model.parameters()):
        raise RuntimeError("Fused Muon projections currently require DeepSpeed ZeRO-1/2; ZeRO-3 is not supported")
    if zero_stage < 1:
        raise RuntimeError(f"Muon backend=deepspeed requires ZeRO stage 1, 2, or 3; detected stage {zero_stage}.")

    offload_config = model.zero_offload_optimizer()
    offload_device = getattr(offload_config, "device", "none") if offload_config is not None else "none"
    optimizer_offload = str(offload_device).lower() not in {"none", "offloaddeviceenum.none"}
    if optimizer_offload:
        raise RuntimeError(
            f"DeepSpeed Muon does not support optimizer offload in this training path; got device={offload_device}."
        )

    zero_optimizer = getattr(model, "optimizer", None)
    inner_optimizer = getattr(zero_optimizer, "optimizer", zero_optimizer)
    if type(inner_optimizer).__name__ != "MuonWithAuxAdam":
        raise RuntimeError(
            "DeepSpeedEngine did not retain the ZeRO-aware MuonWithAuxAdam optimizer; "
            f"detected {type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}."
        )
    return {
        "backend": backend,
        "ns_kernel": getattr(inner_optimizer, "_egofore_muon_ns_backend", {"name": "deepspeed"}),
        "zero_stage": zero_stage,
        "optimizer_offload": optimizer_offload,
        "zero_optimizer_class": f"{type(zero_optimizer).__module__}.{type(zero_optimizer).__name__}",
        "inner_optimizer_class": f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}",
    }
