# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Define fixed RoboCasa pseudo-camera coordinates and column-packed rotations.
"""Physical geometry shared by RoboCasa conversion, validation, and deployment."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

PSEUDO_FROM_BASE_ROTATION = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0])
FRAME_CONVENTION = "robocasa_base_origin_right_down_forward_v1"


def unpack_rotation_columns(values: np.ndarray) -> np.ndarray:
    """Decode [R[:,0], R[:,1]], preserving the AugPosRot source convention."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite column-packed rot6d, got {values.shape}")
    first, second = values[..., :3], values[..., 3:]
    rotations = np.stack((first, second, np.cross(first, second)), axis=-1)
    error = np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))
    if error > 1e-4:
        raise ValueError(f"Source rot6d is not orthonormal: max error={error}")
    return rotations


def pack_rotation_columns(rotations: np.ndarray) -> np.ndarray:
    """Encode the first two columns consecutively, matching AugPosRot."""
    return np.concatenate((rotations[..., :, 0], rotations[..., :, 1]), axis=-1)


def transform_pose6(poses: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Change the parent frame of xyz+rotvec poses without changing the EE axes."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-1] != 6:
        raise ValueError(f"Expected xyz+rotvec poses, got {poses.shape}")
    rotation = np.asarray(target_from_source, dtype=np.float64)[:3, :3]
    xyz = poses[..., :3] @ rotation.T + target_from_source[:3, 3]
    source_rotation = Rotation.from_rotvec(poses[..., 3:].reshape(-1, 3)).as_matrix()
    rotvec = Rotation.from_matrix(rotation @ source_rotation).as_rotvec().reshape(poses[..., 3:].shape)
    return np.concatenate((xyz, rotvec), axis=-1).astype(np.float32)
