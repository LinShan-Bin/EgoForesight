# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add camera-motion tracks for camera-frame relative EEF actions.

"""Camera-pose adapters shared by runtime transforms and relative-action stats.

``extract_camera_world_transforms`` normalizes every supported source to one
convention::

    T_common_from_camera

Relative ``*_camera_pos`` columns encode ``T_Ck_from_C0`` and therefore
must be inverted to map a pose from the current camera into the shared C0
frame. Absolute head extrinsics already map camera coordinates into their shared
frame and are kept as-is. Quaternion columns use ``xyzw`` order; 12-D
extrinsics store a row-major 3x3 rotation followed by translation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .batch_ops import invert_homogeneous

CAMERA_MOTION_CONVENTION_VERSION = 2
CAMERA_STATE_TRANSFORM_KEY = "__camera_motion.state_world_from_camera__"
CAMERA_ACTION_TRANSFORM_KEY = "__camera_motion.action_world_from_camera__"
CAMERA_REFERENCE_TRANSFORM_KEY = "__camera_motion.reference_world_from_camera__"
CAMERA_SKIP_COMPENSATION_KEY = "__camera_motion.skip_compensation__"
CAMERA_TO_ROBOT_COLUMNS = ("head_camera_to_robot_extrinsics", "camera_params.head_extrinsic")
CAMERA_WORLD_TO_OPTICAL_COLUMNS = ("camera_params.head_w2c",)

CAMERA_RELATIVE_POSE_COLUMNS = frozenset(
    {
        "state_camera_pos",
        "action_camera_pos",
        "camera_pos",
    }
)
CAMERA_POSE_COLUMNS = (
    "state_camera_pos",
    "action_camera_pos",
    "camera_pos",
    *CAMERA_TO_ROBOT_COLUMNS,
    *CAMERA_WORLD_TO_OPTICAL_COLUMNS,
)


def _camera_column(columns: Sequence[str], role: str) -> str | None:
    """Resolve the camera-pose column for ``state`` or ``action``."""
    if role not in {"state", "action"}:
        raise ValueError(f"role must be 'state' or 'action', got {role!r}")
    available = set(columns)
    role_column = f"{role}_camera_pos"
    for candidate in (role_column, "camera_pos", "camera_params.head_extrinsic"):
        if candidate in available:
            return candidate
    return None


def camera_pose_vectors_to_homogeneous(values: np.ndarray) -> np.ndarray:
    """Decode camera pose vectors into their encoded ``(..., 4, 4)`` matrices.

    Supported layouts:
      - 7D: ``xyz + quaternion_xyzw``
      - 12D: row-major ``rotation_3x3 + xyz``
      - 16D: row-major homogeneous matrix
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 2:
        raise ValueError(f"camera poses must be [T,D] or [D], got {values.shape}")

    dimension = values.shape[-1]
    transforms = np.zeros((values.shape[0], 4, 4), dtype=np.float64)
    transforms[:, 3, 3] = 1.0
    if dimension == 7:
        quaternions = values[:, 3:7]
        norms = np.linalg.norm(quaternions, axis=-1)
        if np.any(norms < 1e-8):
            bad = np.flatnonzero(norms < 1e-8)[:5].tolist()
            raise ValueError(f"camera quaternion norm is zero at rows {bad}")
        transforms[:, :3, :3] = Rotation.from_quat(quaternions / norms[:, None]).as_matrix()
        transforms[:, :3, 3] = values[:, :3]
    elif dimension == 12:
        transforms[:, :3, :3] = values[:, :9].reshape(-1, 3, 3)
        transforms[:, :3, 3] = values[:, 9:12]
    elif dimension == 16:
        transforms = values.reshape(-1, 4, 4).copy()
    else:
        raise ValueError(f"unsupported camera pose dimension {dimension}; expected 7, 12, or 16")

    rotations = transforms[:, :3, :3]
    orthogonality_error = np.max(
        np.abs(np.matmul(np.transpose(rotations, (0, 2, 1)), rotations) - np.eye(3)),
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)
    invalid = (orthogonality_error > 5e-3) | (np.abs(determinants - 1.0) > 5e-3)
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:5].tolist()
        raise ValueError(
            "camera rotation is not a valid SO(3) matrix at rows "
            f"{bad}; max_orthogonality_error={orthogonality_error[invalid].max():.3g}, "
            f"det_range=({determinants[invalid].min():.3g}, {determinants[invalid].max():.3g})"
        )
    return transforms[0] if squeeze else transforms


def robot_from_camera_vectors_to_homogeneous(values: np.ndarray) -> np.ndarray:
    """Decode ``xyz + quaternion_wxyz`` camera extrinsics."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"robot-from-camera poses must be [T,7] or [7], got {values.shape}")
    xyzw_values = values.copy()
    xyzw_values[:, 3:7] = values[:, [4, 5, 6, 3]]
    transforms = camera_pose_vectors_to_homogeneous(xyzw_values)
    return transforms[0] if squeeze else transforms


def _stack_rows(column: Any, indices: np.ndarray) -> np.ndarray:
    """Gather object-array rows from a pandas Series without densifying it all."""
    rows = []
    for index in indices:
        value = column.iloc[int(index)] if hasattr(column, "iloc") else column[int(index)]
        rows.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return np.stack(rows, axis=0)


def extract_camera_world_transforms(
    frame: Any,
    indices: Sequence[int] | np.ndarray,
    role: str,
) -> tuple[np.ndarray, str, bool] | None:
    """Read clamped camera transforms for the requested state/action indices.

    Args:
        frame: A pandas-like dataframe.
        indices: Source-frame indices, before first/last padding.
        role: ``state`` or ``action``. Separate tracks may exist for each role.

    Returns:
        ``(T_common_from_camera, source_column, skip_compensation)`` or
        ``None`` when no supported camera trajectory exists. Relative
        relative camera poses are inverted here. All-zero 12D
        sentinel sets ``skip_compensation`` for the whole sampled chunk.
    """
    column_name = _camera_column(frame.columns, role)
    if column_name is None:
        return None
    if len(frame) == 0:
        raise ValueError("cannot read camera poses from an empty trajectory")
    indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    indices_array = np.clip(indices_array, 0, len(frame) - 1)
    vectors = _stack_rows(frame[column_name], indices_array)
    contains_zero_extrinsic = (
        column_name == "camera_params.head_extrinsic"
        and vectors.shape[-1] == 12
        and np.any(np.all(vectors == 0.0, axis=1))
    )
    if contains_zero_extrinsic:
        placeholder = np.broadcast_to(np.eye(4), (len(indices_array), 4, 4)).copy()
        return placeholder, column_name, True
    transforms = camera_pose_vectors_to_homogeneous(vectors)
    if column_name in CAMERA_RELATIVE_POSE_COLUMNS:
        transforms = invert_homogeneous(transforms)
    return transforms, column_name, False


def extract_robot_base_reference_camera_transforms(
    frame: Any,
    indices: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, str] | None:
    """Return ``T_robot_base_from_head_camera`` for robot-base EEF sources."""
    column_name = next((name for name in CAMERA_TO_ROBOT_COLUMNS if name in frame.columns), None)
    if len(frame) == 0:
        raise ValueError("cannot read camera extrinsics from an empty trajectory")
    indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    indices_array = np.clip(indices_array, 0, len(frame) - 1)
    if column_name is not None:
        vectors = _stack_rows(frame[column_name], indices_array)
        if column_name == "head_camera_to_robot_extrinsics":
            transforms = robot_from_camera_vectors_to_homogeneous(vectors)
        else:
            transforms = camera_pose_vectors_to_homogeneous(vectors)
        return transforms, column_name

    w2c_column = next((name for name in CAMERA_WORLD_TO_OPTICAL_COLUMNS if name in frame.columns), None)
    if w2c_column is None:
        return None
    world_to_camera = camera_pose_vectors_to_homogeneous(_stack_rows(frame[w2c_column], indices_array))
    return invert_homogeneous(world_to_camera), w2c_column
