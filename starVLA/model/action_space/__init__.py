# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Chubin Zhang in 2026.
# Modification: Add model-owned RoboCasa joint/EEF/aligned action-space adapter discovery.
"""Resolve model action-space configuration and build inference adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

ActionSpaceBuilder = Callable[[Any, dict[str, Any]], Any]

_ACTION_SPACE_BUILDERS: dict[str, ActionSpaceBuilder] = {}
_BUILTIN_ACTION_SPACE_MODULES = {
    "robocasa_gr1_joint": "starVLA.model.action_space.robocasa",
    "robocasa_gr1_eef": "starVLA.model.action_space.robocasa",
    "robocasa_gr1_aug_aligned": "starVLA.model.action_space.robocasa",
    "robotwin_aug_aligned": "starVLA.model.action_space.robotwin",
    "robotwin_ee": "starVLA.model.action_space.robotwin",
}


def register_action_space(name: str) -> Callable[[ActionSpaceBuilder], ActionSpaceBuilder]:
    """Register a model-side inference adapter for an action-space name."""

    def decorator(builder: ActionSpaceBuilder) -> ActionSpaceBuilder:
        if name in _ACTION_SPACE_BUILDERS:
            raise ValueError(f"Action-space adapter {name!r} is already registered")
        _ACTION_SPACE_BUILDERS[name] = builder
        return builder

    return decorator


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _as_plain_spec(action_space: Any) -> dict[str, Any]:
    if isinstance(action_space, str):
        return {"name": action_space}
    if isinstance(action_space, Mapping):
        return dict(action_space)
    items = getattr(action_space, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError(
        "framework.action_model.action_space must be a string or mapping, " f"got {type(action_space).__name__}"
    )


def resolve_action_space_spec(policy: Any) -> dict[str, Any] | None:
    """Read action-space ownership from the loaded model configuration.

    New checkpoints declare ``framework.action_model.action_space``. For old
    RoboTwin/RoboCasa checkpoints, the existing dataset action type and normalization
    statistics provide an unambiguous compatibility path.
    """
    config = getattr(policy, "config", None)
    framework_config = _get(config, "framework")
    action_model_config = _get(framework_config, "action_model")
    configured_action_space = _get(action_model_config, "action_space")

    datasets_config = _get(config, "datasets")
    data_config = _get(datasets_config, "vla_data")
    data_mix = str(_get(data_config, "data_mix", ""))
    norm_stats = getattr(policy, "norm_stats", {}) or {}

    inferred_spec = None
    if (
        data_mix in {"robotwin_c2r_aug_aligned", "robotwin_c2r_aug_aligned_no_mirror", "robotwin_full_aug_aligned"}
        and "robotwin_aug_aligned" in norm_stats
    ):
        if _get(data_config, "relative_eef_frame") != "camera_spatial":
            raise ValueError("Augmented RoboTwin checkpoints require camera_spatial training metadata")
        inferred_spec = {
            "name": "robotwin_aug_aligned",
            "representation": "relative" if _get(data_config, "enable_relative_action") else "absolute",
            "joint_representation": "absolute",
            "statistics_key": "robotwin_aug_aligned",
            "relative_eef_frame": "camera_spatial",
            "static_camera_frame": True,
            "frame_convention": "robotwin_head_opencv_v1",
            "source": "legacy_dataset_config",
        }
    elif data_mix == "robocasa_1000_joint_absolute" and "gr1" in norm_stats:
        inferred_spec = {
            "name": "robocasa_gr1_joint",
            "representation": "absolute",
            "statistics_key": "gr1",
            "source": "legacy_dataset_config",
        }
    elif data_mix == "robocasa_1000_ee" and "robocasa_gr1_eef" in norm_stats:
        inferred_spec = {
            "name": "robocasa_gr1_eef",
            "representation": "relative",
            "statistics_key": "robocasa_gr1_eef",
            "relative_eef_frame": "camera_spatial",
            "source": "legacy_dataset_config",
        }
    elif (
        data_mix in {"robocasa_1000_aug_aligned", "robocasa_1000_aug_aligned_egodex"}
        and "robocasa_gr1_aug_aligned" in norm_stats
    ):
        inferred_spec = {
            "name": "robocasa_gr1_aug_aligned",
            "representation": "relative",
            "joint_representation": "absolute",
            "fingertip_representation": "absolute",
            "statistics_key": "robocasa_gr1_aug_aligned",
            "relative_eef_frame": "camera_spatial",
            "static_camera_frame": True,
            "frame_convention": "robocasa_base_origin_right_down_forward_v1",
            "source": "legacy_dataset_config",
        }

    if configured_action_space is not None:
        spec = _as_plain_spec(configured_action_space)
        configured_name = str(spec.get("name", ""))
        if inferred_spec is not None and configured_name in {"", "canonical"}:
            merged_spec = dict(inferred_spec)
            merged_spec.update(spec)
            merged_spec["name"] = inferred_spec["name"]
            merged_spec["source"] = "model_config+legacy_dataset_config"
            return merged_spec
        if not configured_name:
            raise ValueError("framework.action_model.action_space.name must be non-empty")
        spec.setdefault("source", "model_config")
        return spec

    if inferred_spec is not None:
        return inferred_spec

    # Backward compatibility for checkpoints produced before action-space
    # metadata moved to the model. Requiring RoboTwin statistics prevents an
    # unrelated dataset's generic ``absolute_ee`` label from selecting this
    # embodiment-specific adapter.
    if "robotwin" not in norm_stats:
        return None
    action_type = _get(data_config, "action_type")
    relative_enabled = _get(data_config, "enable_relative_action")
    state_relative_enabled = _get(data_config, "enable_state_relative_eef")
    representation_by_type = {
        "delta_ee": "relative",
        "relative_ee": "relative",
        "absolute_ee": "absolute",
    }
    representation = representation_by_type.get(action_type)
    if representation is None and (relative_enabled is True or state_relative_enabled is True):
        representation = "relative"
    elif representation is None and (relative_enabled is False or state_relative_enabled is False):
        representation = "absolute"
    if representation is None:
        return None
    return {
        "name": "robotwin_ee",
        "representation": representation,
        "statistics_key": "robotwin",
        "source": "legacy_dataset_config",
    }


def build_action_space_policy(policy: Any) -> Any:
    """Wrap a loaded model according to its own action-space configuration."""
    spec = resolve_action_space_spec(policy)
    if spec is None:
        return policy

    name = str(spec["name"])
    if name not in _ACTION_SPACE_BUILDERS and name in _BUILTIN_ACTION_SPACE_MODULES:
        importlib.import_module(_BUILTIN_ACTION_SPACE_MODULES[name])
    try:
        builder = _ACTION_SPACE_BUILDERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ACTION_SPACE_BUILDERS)) or "none"
        raise ValueError(f"No inference adapter registered for action space {name!r}; available: {available}") from error
    return builder(policy, spec)


__all__ = [
    "build_action_space_policy",
    "register_action_space",
    "resolve_action_space_spec",
]
