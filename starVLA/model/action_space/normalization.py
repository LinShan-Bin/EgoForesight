# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Resolve mixed q99/min-max canonical normalization anchors.

"""Checkpoint-first per-dimension normalization mode resolution."""

from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any, Literal

import numpy as np

NormalizationMode = Literal["min_max", "q99"]
_VALID_NORMALIZATION_MODES = frozenset({"min_max", "q99"})
_DATA_CONFIG_RESOLUTION_LOCK = RLock()


@lru_cache(maxsize=None)
def normalization_modes_from_data_config(
    unnorm_key: str,
    dimension: int,
    *,
    modality: Literal["action", "state"] = "action",
) -> tuple[str, ...] | None:
    """Reconstruct canonical modes for a legacy checkpoint.

    New checkpoints persist ``norm_modes`` beside their canonical statistics.
    This registry path exists only for older checkpoints and deliberately uses
    the same built transform pipeline as training instead of duplicating a
    second robot-specific dimension table.
    """

    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    if modality not in {"action", "state"}:
        raise ValueError(f"modality must be 'action' or 'state', got {modality!r}")

    from starVLA.dataloader.gr00t_lerobot.canonical_action import (
        CanonicalSlot,
        is_action_alignment_enabled,
        set_action_alignment_enabled,
    )
    from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP

    config = ROBOT_TYPE_CONFIG_MAP.get(str(unnorm_key))
    if config is None:
        return None
    with _DATA_CONFIG_RESOLUTION_LOCK:
        alignment_was_enabled = is_action_alignment_enabled()
        try:
            set_action_alignment_enabled(True)
            composed = config.transform()
        finally:
            set_action_alignment_enabled(alignment_was_enabled)

    normalization_modes: dict[str, str] = {}
    concat_transform = None
    slot_attribute = f"{modality}_canonical_slots"
    for transform in getattr(composed, "transforms", []) or []:
        for key, mode in (getattr(transform, "normalization_modes", None) or {}).items():
            existing = normalization_modes.get(key)
            if existing is not None and existing != mode:
                raise ValueError(
                    f"DataConfig {unnorm_key!r} assigns conflicting normalization modes "
                    f"to {key}: {existing!r} vs {mode!r}"
                )
            normalization_modes[key] = str(mode)
        if getattr(transform, slot_attribute, None):
            if concat_transform is not None:
                raise ValueError(f"DataConfig {unnorm_key!r} has multiple canonical {modality} scatter transforms")
            concat_transform = transform
    if concat_transform is None:
        return None

    slot_map = getattr(concat_transform, slot_attribute) or {}
    slot_indices_map = getattr(concat_transform, f"{modality}_canonical_slot_indices", None) or {}
    resolved = [""] * dimension
    for key, slot_name in slot_map.items():
        mode = normalization_modes.get(key)
        if mode not in _VALID_NORMALIZATION_MODES:
            raise ValueError(f"DataConfig {unnorm_key!r} has no valid normalization mode for {key!r}: {mode!r}")
        slot = CanonicalSlot[slot_name] if isinstance(slot_name, str) else slot_name
        configured_indices = slot_indices_map.get(key)
        if configured_indices is None:
            destinations = range(slot.start, min(slot.end, dimension))
        else:
            slot_indices = tuple(int(index) for index in configured_indices)
            if len(set(slot_indices)) != len(slot_indices):
                raise ValueError(
                    f"DataConfig {unnorm_key!r} has duplicate canonical indices for {key}: " f"{list(slot_indices)}"
                )
            if any(index < 0 or index >= slot.size for index in slot_indices):
                raise ValueError(
                    f"DataConfig {unnorm_key!r} has out-of-range canonical indices for {key} "
                    f"in slot {slot.spec.name} (size={slot.size}): {list(slot_indices)}"
                )
            destinations = (slot.start + index for index in slot_indices)
        for index in destinations:
            if not 0 <= index < dimension:
                continue
            existing = resolved[index]
            if existing and existing != mode:
                raise ValueError(
                    f"DataConfig {unnorm_key!r} assigns conflicting modes to canonical "
                    f"{modality} dim {index}: {existing!r} vs {mode!r}"
                )
            resolved[index] = mode
    return tuple(resolved)


def q99_dimensions_from_data_config(
    unnorm_key: str,
    dimension: int,
    *,
    modality: Literal["action", "state"] = "action",
) -> frozenset[int]:
    """Reconstruct q99 action dimensions for checkpoints without ``norm_modes``."""

    modes = normalization_modes_from_data_config(
        unnorm_key,
        dimension,
        modality=modality,
    )
    if modes is None:
        return frozenset()
    return frozenset(index for index, mode in enumerate(modes) if mode == "q99")


def resolve_normalization_modes(
    stats: dict[str, Any],
    dimension: int,
    *,
    modality: Literal["action", "state"] = "action",
    unnorm_key: str | None = None,
    force_mode: NormalizationMode | None = None,
    default_mode: NormalizationMode = "min_max",
) -> tuple[tuple[str, ...], str]:
    """Resolve canonical normalization modes in checkpoint-first order."""

    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, got {dimension}")
    if force_mode is not None and force_mode not in _VALID_NORMALIZATION_MODES:
        raise ValueError(f"force_mode must be None, 'min_max', or 'q99', got {force_mode!r}")
    if default_mode not in _VALID_NORMALIZATION_MODES:
        raise ValueError(f"default_mode must be 'min_max' or 'q99', got {default_mode!r}")

    if force_mode is not None:
        return (force_mode,) * dimension, f"force={force_mode}"

    exported_modes = stats.get("norm_modes")
    if exported_modes is not None:
        statistic_widths = {key: len(stats[key]) for key in ("min", "max", "q01", "q99") if key in stats}
        if statistic_widths and len(set(statistic_widths.values())) != 1:
            raise ValueError(f"Normalization statistics have inconsistent widths: {statistic_widths}")
        schema_width = next(iter(statistic_widths.values()), len(exported_modes))
        if len(exported_modes) != schema_width:
            raise ValueError(f"norm_modes has length {len(exported_modes)}; statistics schema width is {schema_width}")
        if len(exported_modes) < dimension:
            raise ValueError(f"norm_modes has length {len(exported_modes)}; expected at least {dimension}")
        schema_modes = tuple("" if mode is None else str(mode) for mode in exported_modes)
        mask = stats.get("mask")
        if mask is not None:
            if len(mask) != schema_width:
                raise ValueError(f"mask has length {len(mask)}; schema width is {schema_width}")
            schema_active = np.asarray(mask, dtype=bool)
        else:
            slot_names = stats.get("slot_names")
            if slot_names is not None:
                if len(slot_names) != schema_width:
                    raise ValueError(f"slot_names has length {len(slot_names)}; schema width is {schema_width}")
                schema_active = np.asarray([bool(name) for name in slot_names], dtype=bool)
            else:
                if any(not mode for mode in schema_modes):
                    raise ValueError("Blank norm_modes require stats.mask or stats.slot_names to identify placeholders")
                schema_active = np.asarray([bool(mode) for mode in schema_modes], dtype=bool)
        for index, mode in enumerate(schema_modes):
            if schema_active[index] and mode in _VALID_NORMALIZATION_MODES:
                continue
            if not mode and not schema_active[index]:
                continue
            raise ValueError(f"Unsupported normalization mode at dim {index}: {mode!r}")
        return schema_modes[:dimension], "stats.norm_modes"

    if unnorm_key is not None:
        registry_modes = normalization_modes_from_data_config(
            unnorm_key,
            dimension,
            modality=modality,
        )
        if registry_modes is not None:
            active = None
            mask = stats.get("mask")
            slot_names = stats.get("slot_names")
            if mask is not None:
                if len(mask) < dimension:
                    raise ValueError(f"mask has length {len(mask)}; expected at least {dimension}")
                active = np.asarray(mask[:dimension], dtype=bool)
            elif slot_names is not None:
                if len(slot_names) < dimension:
                    raise ValueError(f"slot_names has length {len(slot_names)}; expected at least {dimension}")
                active = np.asarray([bool(name) for name in slot_names[:dimension]], dtype=bool)
            if active is not None:
                registry_modes = tuple(mode if active[index] else "" for index, mode in enumerate(registry_modes))
                missing = [index for index, mode in enumerate(registry_modes) if active[index] and not mode]
                if missing:
                    raise ValueError(
                        f"DataConfig {unnorm_key!r} has no canonical {modality} mode for active dims {missing}"
                    )
            return registry_modes, f"_NORMALIZATION_MODE_MAP[{unnorm_key}].{modality}"

    return (default_mode,) * dimension, f"{default_mode}_fallback"


def resolve_normalization_bounds(
    stats: dict[str, Any],
    dimension: int,
    *,
    modality: Literal["action", "state"] = "action",
    unnorm_key: str | None = None,
    force_mode: NormalizationMode | None = None,
    default_mode: NormalizationMode = "min_max",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Select min/max or q01/q99 anchors for every canonical dimension."""

    modes, source = resolve_normalization_modes(
        stats,
        dimension,
        modality=modality,
        unnorm_key=unnorm_key,
        force_mode=force_mode,
        default_mode=default_mode,
    )

    expected_shape = (dimension,)

    def read_statistic(name: str) -> np.ndarray:
        if name not in stats:
            raise KeyError(f"Missing {name!r} normalization statistics")
        values = np.asarray(stats[name][:dimension], dtype=np.float32)
        if values.shape != expected_shape:
            raise ValueError(f"{name} statistics have shape {values.shape}; expected {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} statistics contain non-finite values")
        return values

    q99_mask = np.asarray([mode == "q99" for mode in modes], dtype=bool)
    if q99_mask.all():
        lower = read_statistic("q01")
        upper = read_statistic("q99")
        if np.any(lower > upper):
            indices = np.flatnonzero(lower > upper).tolist()
            raise ValueError(f"q01 exceeds q99 at dimensions {indices}")
        return lower, upper, source

    lower = read_statistic("min").copy()
    upper = read_statistic("max").copy()
    if q99_mask.any():
        q01 = read_statistic("q01")
        q99 = read_statistic("q99")
        lower[q99_mask] = q01[q99_mask]
        upper[q99_mask] = q99[q99_mask]

    if np.any(lower > upper):
        indices = np.flatnonzero(lower > upper).tolist()
        raise ValueError(f"Normalization lower bound exceeds upper bound at dimensions {indices}")

    return lower, upper, source


def normalize_with_statistics(
    values: np.ndarray,
    stats: dict[str, Any],
    *,
    modality: Literal["action", "state"] = "state",
    unnorm_key: str | None = None,
    force_mode: NormalizationMode | None = None,
    default_mode: NormalizationMode = "min_max",
) -> tuple[np.ndarray, str]:
    """Apply the training normalizer's mixed q99/min-max forward semantics."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("values must have at least one dimension")
    dimension = array.shape[-1]
    modes, source = resolve_normalization_modes(
        stats,
        dimension,
        modality=modality,
        unnorm_key=unnorm_key,
        force_mode=force_mode,
        default_mode=default_mode,
    )
    lower, upper, _ = resolve_normalization_bounds(
        stats,
        dimension,
        modality=modality,
        unnorm_key=unnorm_key,
        force_mode=force_mode,
        default_mode=default_mode,
    )
    scale = upper - lower
    nonconstant = scale != 0.0
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[..., nonconstant] = 2.0 * (array[..., nonconstant] - lower[nonconstant]) / scale[nonconstant] - 1.0
    q99_mask = np.asarray([mode == "q99" for mode in modes], dtype=bool)
    q99_constant = q99_mask & ~nonconstant
    normalized[..., q99_constant] = array[..., q99_constant]
    normalized[..., q99_mask] = np.clip(normalized[..., q99_mask], -1.0, 1.0)
    return normalized, source


def unnormalize_with_statistics(
    normalized_values: np.ndarray,
    stats: dict[str, Any],
    *,
    modality: Literal["action", "state"] = "action",
    unnorm_key: str | None = None,
    force_mode: NormalizationMode | None = None,
    default_mode: NormalizationMode = "min_max",
    clip: bool = False,
) -> tuple[np.ndarray, str]:
    """Invert mixed q99/min-max canonical values with checkpoint anchors."""

    array = np.asarray(normalized_values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("normalized_values must have at least one dimension")
    lower, upper, source = resolve_normalization_bounds(
        stats,
        array.shape[-1],
        modality=modality,
        unnorm_key=unnorm_key,
        force_mode=force_mode,
        default_mode=default_mode,
    )
    if clip:
        array = np.clip(array, -1.0, 1.0)
    values = 0.5 * (array + 1.0) * (upper - lower) + lower
    return values.astype(np.float32, copy=False), source


def resolve_statistics_key(norm_stats: dict[str, Any], requested_key: str) -> str:
    """Resolve old/q99 tag aliases while always preferring an exact key."""

    if requested_key in norm_stats:
        return requested_key
    candidate = requested_key[: -len("_q99")] if requested_key.endswith("_q99") else f"{requested_key}_q99"
    if candidate in norm_stats:
        return candidate
    raise KeyError(f"Normalization statistics key {requested_key!r} is unavailable; " f"available={sorted(norm_stats)}")


__all__ = [
    "normalization_modes_from_data_config",
    "normalize_with_statistics",
    "q99_dimensions_from_data_config",
    "resolve_normalization_bounds",
    "resolve_normalization_modes",
    "resolve_statistics_key",
    "unnormalize_with_statistics",
]
