# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-License-Identifier: Apache-2.0
"""
EefPoseAlignTransform: merge split (position, orientation) EE-pose keys and
unify every rotation representation into axis-angle (rotvec).

Some datasets keep position and orientation in two separate keys (e.g.
GenieSim3: ``action.left_ee_position`` + ``action.left_ee_orientation``).
Other datasets store the full pose under one key in a non-rotvec rotation
format (quaternion / 6D / euler / matrix). The canonical layout downstream
assumes a single ``xyz(3) + rotvec(3) = 6`` representation per EE-pose slot.

This transform normalizes both cases so ConcatTransform's canonical scatter
sees one well-formed key per slot.

Example transform-list usage inside a DataConfig:

    EefPoseAlignTransform(
        merges=[
            ("action.left_ee_pose",  "action.left_ee_position",  "action.left_ee_orientation"),
            ("action.right_ee_pose", "action.right_ee_position", "action.right_ee_orientation"),
        ],
        src_rotation={
            "action.left_ee_orientation":  "quaternion_xyzw",
            "action.right_ee_orientation": "quaternion_xyzw",
        },
    )

Or, for an already-merged key whose rotation isn't rotvec yet:

    EefPoseAlignTransform(
        merges=[("action.left_ee_pose", None, "action.left_ee_pose")],
        src_rotation={"action.left_ee_pose": "quaternion_xyzw"},  # last 4 dims are quat
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import Field

from .base import ModalityTransform
from .rotation_align import RotationFormat, to_rotvec


class EefPoseAlignTransform(ModalityTransform):
    """Merge + axis-angle-align EE-pose keys.

    ``merges`` is a list of ``(out_key, position_key | None, rotation_key)`` tuples.

    * If ``position_key`` is given, ``out_key`` is the concat of position[..., :3]
      and rotvec(rotation[..., :])`` along the last axis.
    * If ``position_key`` is None, the rotation key is expected to already be a
      full pose (xyz + rotation). The first 3 dims pass through as xyz, the
      remaining dims are converted from ``src_rotation[rotation_key]`` to rotvec.

    Input keys are removed from the data dict; ``out_key`` is added.
    Tensors are kept in numpy / torch matching the source type.
    """

    # apply_to is unused but required by the base class.
    apply_to: list[str] = Field(default_factory=list)

    merges: list[tuple[str, str | None, str]] = Field(
        default_factory=list,
        description="List of (out_key, position_key|None, rotation_key) merges.",
    )
    src_rotation: dict[str, str] = Field(
        default_factory=dict,
        description="Map of rotation_key -> source rotation format name (see rotation_align.RotationFormat).",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for out_key, pos_key, rot_key in self.merges:
            if rot_key not in data:
                # Allow the transform to silently no-op for missing keys so
                # the same DataConfig can be reused across embodiment variants
                # that don't carry every key.
                continue

            fmt: RotationFormat = self.src_rotation.get(rot_key, "rotvec")  # type: ignore[assignment]

            if pos_key is None:
                # rot_key holds the full pose; first 3 dims are xyz, rest is rotation.
                full = data.pop(rot_key)
                xyz_np, rot_np, was_torch, ref = self._split_xyz_rot(full)
                rotvec_np = to_rotvec(rot_np, fmt)
                merged_np = np.concatenate([xyz_np, rotvec_np], axis=-1).astype(np.float32, copy=False)
                data[out_key] = self._maybe_to_torch(merged_np, was_torch, ref)
            else:
                if pos_key not in data:
                    # Position key missing but rotation present: log nothing,
                    # just skip the merge (downstream will fail loudly if it
                    # actually needed this key).
                    continue
                pos = data.pop(pos_key)
                rot = data.pop(rot_key)
                pos_np, was_torch_p, ref_p = self._as_numpy(pos)
                rot_np, was_torch_r, ref_r = self._as_numpy(rot)
                if pos_np.shape[:-1] != rot_np.shape[:-1]:
                    raise ValueError(
                        f"position/rotation shape mismatch for {pos_key=} {rot_key=}: "
                        f"{pos_np.shape=} vs {rot_np.shape=}"
                    )
                if pos_np.shape[-1] != 3:
                    raise ValueError(f"{pos_key} last-dim expected 3 (xyz), got {pos_np.shape[-1]}")
                rotvec_np = to_rotvec(rot_np, fmt)
                merged_np = np.concatenate([pos_np, rotvec_np], axis=-1).astype(np.float32, copy=False)
                was_torch = was_torch_p or was_torch_r
                ref = ref_p if was_torch_p else ref_r
                data[out_key] = self._maybe_to_torch(merged_np, was_torch, ref)
        return data

    # ----- small helpers -----------------------------------------------------
    @staticmethod
    def _as_numpy(arr) -> tuple[np.ndarray, bool, Any]:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy(), True, arr
        return np.asarray(arr), False, arr

    @staticmethod
    def _maybe_to_torch(arr_np: np.ndarray, was_torch: bool, ref) -> Any:
        if not was_torch:
            return arr_np
        return torch.from_numpy(arr_np).to(device=ref.device, dtype=ref.dtype)

    def _split_xyz_rot(self, full):
        arr, was_torch, ref = self._as_numpy(full)
        if arr.shape[-1] < 4:
            raise ValueError(f"full pose key last dim must be >=4 (xyz + rotation), got {arr.shape[-1]}")
        xyz = arr[..., :3]
        rot = arr[..., 3:]
        return xyz, rot, was_torch, ref


class EefPoseNormalizeTransform(ModalityTransform):
    """Normalize 6D xyz+rotvec keys to [-1, 1] after EefPoseAlignTransform.

    xyz is normalized against per-key min/max bounds passed at construction
    time (typically the xyz slice of the parquet key's stats.json entry).
    rotvec is normalized against fixed [-pi, +pi] bounds since axis-angle
    representation has that as a hard physical upper bound.

    This exists because our EefPoseAlignTransform performs an in-place
    quat(7D) -> rotvec(6D) merge. The dataset's stats.json entry for the
    key is still 7D-shaped and cannot be reused by StateActionTransform's
    min_max mode directly (shape mismatch). Rather than teach that
    normalizer to handle the conversion, we run this dedicated shape-
    aware normalizer AFTER align.

    Args:
        xyz_bounds: {key: {"min": [3], "max": [3]}} per-key xyz bounds.
                    Keys must match the merged 6D key names, e.g.
                    "action.left_ee_pose".
        rotvec_bound: single float, the ±bound for rotvec (default pi).
                      All 3 rotvec dims share the same symmetric bound.
    """

    apply_to: list[str] = Field(default_factory=list)
    xyz_bounds: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    rotvec_bound: float = Field(default=float(np.pi))

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            if key not in self.xyz_bounds:
                # No bounds provided for this key; skip. Downstream will get
                # unnormalized values -- caller should ensure xyz_bounds
                # covers every eef key that reaches ConcatTransform.
                continue
            val = data[key]
            arr, was_torch, ref = self._as_numpy(val)
            if arr.shape[-1] != 6:
                raise ValueError(f"{key} last dim expected 6 (xyz+rotvec), got {arr.shape[-1]}")

            bounds = self.xyz_bounds[key]
            xyz_min = np.asarray(bounds["min"], dtype=np.float64)
            xyz_max = np.asarray(bounds["max"], dtype=np.float64)
            assert xyz_min.shape == (3,) and xyz_max.shape == (
                3,
            ), f"xyz_bounds[{key}] must be 3D vectors, got {xyz_min.shape} / {xyz_max.shape}"

            xyz = arr[..., :3].astype(np.float64)
            rot = arr[..., 3:].astype(np.float64)

            # Normalize xyz: 2*(x - min)/(max - min) - 1, clipped to [-1, 1]
            span = xyz_max - xyz_min
            span = np.where(span == 0, 1.0, span)  # avoid div-by-zero
            xyz_norm = 2.0 * (xyz - xyz_min) / span - 1.0
            xyz_norm = np.clip(xyz_norm, -1.0, 1.0)

            # Normalize rotvec: x / rotvec_bound
            b = float(self.rotvec_bound)
            rot_norm = np.clip(rot / b, -1.0, 1.0)

            out = np.concatenate([xyz_norm, rot_norm], axis=-1).astype(np.float32, copy=False)
            data[key] = self._maybe_to_torch(out, was_torch, ref)
        return data

    # Reuse the array helpers from EefPoseAlignTransform.
    @staticmethod
    def _as_numpy(arr) -> tuple[np.ndarray, bool, Any]:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy(), True, arr
        return np.asarray(arr), False, arr

    @staticmethod
    def _maybe_to_torch(arr_np: np.ndarray, was_torch: bool, ref) -> Any:
        if not was_torch:
            return arr_np
        return torch.from_numpy(arr_np).to(device=ref.device, dtype=ref.dtype)
