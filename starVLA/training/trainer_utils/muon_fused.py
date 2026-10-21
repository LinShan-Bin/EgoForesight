# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Orthogonalize independent linear maps in fused Muon parameters.

"""Fused-row layouts and scoped metadata for DeepSpeed's pre-flatten updates."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FusedMuonLayout:
    """Rows stored as ``[groups, sum(sizes), input_dim]``.

    Each slice across all groups is one independent linear map. Groups are
    needed for Qwen's head-interleaved query/gate projection.
    """

    sizes: tuple[int, ...]
    groups: int = 1
    adjust_lr_fn: str = "match_rms_adamw"


_ACTIVE_LAYOUTS: ContextVar[dict[int, FusedMuonLayout] | None] = ContextVar("muon_fused_layouts", default=None)


def fused_muon_layout(model: nn.Module, name: str, parameter: Tensor, policy: str) -> FusedMuonLayout | None:
    """Resolve known fused projections using their actual module dimensions."""
    parts = name.split(".")
    if len(parts) < 2 or parts[-1] != "weight":
        return None
    module_name = parts[-2]
    rows = int(parameter.shape[0])
    parent = model.get_submodule(".".join(parts[:-2]))
    groups = 1
    if module_name == "in_proj_qkv":
        key_dim = int(parent.key_dim)
        sizes = (key_dim, key_dim, int(parent.value_dim))
    elif module_name in {"action_joint_qkv", "vision_joint_qkv", "qkv"}:
        assert rows % 3 == 0, f"Invalid fused QKV rows for {name}: {rows}"
        sizes = (rows // 3,) * 3
    elif module_name in {"action_cross_kv", "vision_cross_kv", "gate_up_proj"}:
        assert rows % 2 == 0, f"Invalid fused projection rows for {name}: {rows}"
        sizes = (rows // 2,) * 2
    elif module_name == "q_proj" and hasattr(parent, "head_dim") and hasattr(parent, "o_proj"):
        # Qwen3.5 stores [query, output gate] inside each head; earlier Qwen
        # versions have an ordinary q_proj whose rows equal o_proj input size.
        query_rows = int(parent.o_proj.weight.shape[1])
        if rows != 2 * query_rows:
            return None
        head_dim = int(parent.head_dim)
        assert query_rows % head_dim == 0, f"Invalid gated query heads for {name}"
        groups = query_rows // head_dim
        sizes = (head_dim, head_dim)
    else:
        return None
    assert (
        all(size > 0 for size in sizes) and groups * sum(sizes) == rows
    ), f"Fused Muon layout does not match {name}: shape={tuple(parameter.shape)}, groups={groups}, sizes={sizes}"
    return FusedMuonLayout(sizes, groups, policy)


def _target_scale(rows: int, columns: int, policy: str) -> float:
    if policy == "original":
        return math.sqrt(max(1.0, rows / columns))
    if policy == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    if policy == "spectral_unclamped":
        return math.sqrt(rows / columns)
    raise ValueError(f"Unsupported Muon LR policy: {policy!r}")


def orthogonalize_fused_update(
    gradient: Tensor,
    update: Tensor,
    orthogonalize: Callable[[Tensor], Tensor],
) -> Tensor:
    """Apply NS and shape scaling to each map, then restore physical row order.

    Momentum is already updated once in the full buffer. The optimizer group
    retains its full-shape LR and decay compensation, so undo only that LR
    compensation here before applying each map's requested effective scale.
    """
    layout = (_ACTIVE_LAYOUTS.get() or {}).get(id(gradient))
    rows, columns = update.shape
    if layout is None:
        return orthogonalize(update) * math.sqrt(max(1.0, rows / columns))
    assert rows == layout.groups * sum(layout.sizes), "Fused Muon gradient shape changed"
    group_lr_scale = _target_scale(rows, columns, layout.adjust_lr_fn) / math.sqrt(max(1.0, rows / columns))
    grouped = update.reshape(layout.groups, sum(layout.sizes), columns)
    blocks = []
    for block in grouped.split(layout.sizes, dim=1):
        matrix = block.reshape(-1, columns).contiguous()
        result = orthogonalize(matrix)
        result = result * (_target_scale(matrix.shape[0], columns, layout.adjust_lr_fn) / group_lr_scale)
        blocks.append(result.reshape(layout.groups, block.shape[1], columns))
    return torch.cat(blocks, dim=1).reshape(rows, columns)


def install_zero_fused_context(zero_class: type) -> None:
    """Scope layouts to the exact original gradients used by ZeRO-1/2."""
    original = zero_class.get_flat_partition
    if getattr(original, "_egofore_fused_context", False):
        return

    @wraps(original)
    def get_flat_partition(
        self: Any,
        tensor_list: list[Tensor],
        first_offset: int,
        partition_size: int,
        dtype: torch.dtype,
        device: torch.device,
        param_group_idx: int,
        return_tensor_list: bool = False,
    ) -> Any:
        layouts = {}
        for index, parameter in enumerate(tensor_list):
            layout = getattr(parameter, "_egofore_muon_layout", None)
            if layout is not None and getattr(parameter, "use_muon", False):
                gradient = self.all_grad_tensors[param_group_idx][index]
                layouts[id(gradient)] = layout
        token = _ACTIVE_LAYOUTS.set(layouts)
        try:
            return original(
                self, tensor_list, first_offset, partition_size, dtype, device, param_group_idx, return_tensor_list
            )
        finally:
            _ACTIVE_LAYOUTS.reset(token)

    get_flat_partition._egofore_fused_context = True
    zero_class.get_flat_partition = get_flat_partition


def install_single_device_fused_context(optimizer: torch.optim.Optimizer) -> None:
    """Expose each original gradient's layout for the single-device fallback."""
    original = optimizer.step

    @wraps(original)
    def step(self: torch.optim.Optimizer, closure: Callable | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # DeepSpeed also materializes absent gradients. Do it before collecting
        # metadata so a fused parameter with momentum still uses its layout.
        for group in self.param_groups:
            if group.get("use_muon", False):
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
        layouts = {
            id(parameter.grad): parameter._egofore_muon_layout
            for group in self.param_groups
            if group.get("use_muon", False)
            for parameter in group["params"]
            if parameter.grad is not None and getattr(parameter, "_egofore_muon_layout", None) is not None
        }
        token = _ACTIVE_LAYOUTS.set(layouts)
        try:
            original()
            return loss
        finally:
            _ACTIVE_LAYOUTS.reset(token)

    optimizer.step = MethodType(step, optimizer)
