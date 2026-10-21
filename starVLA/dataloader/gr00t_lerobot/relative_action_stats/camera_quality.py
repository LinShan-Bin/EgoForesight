# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Validate 7-D relative camera trajectories for sample-quality indexes.

"""Shared camera trajectory validation for camera-relative action data.

``camera_pos[t]`` is expected to be ``T_{Ct<-C0}`` in xyz+quat_xyzw form.
Absolute translation values are not thresholded: a long, valid trajectory may
move far from its anchor. Instead, this module rejects invalid poses and
physically impossible changes between consecutive camera frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_FILTER_VERSION = 2
CAMERA_WINDOW_POLICY_VERSION = 3
DUAL_TRACK_ACTION_FRAME_OFFSET = 1
CAMERA_RUNTIME_MIN_QUATERNION_NORM = 1.0e-8
CAMERA_RELATIVE_POSE_COLUMNS = frozenset({"state_camera_pos", "action_camera_pos", "camera_pos"})


@dataclass(frozen=True)
class CameraFilterConfig:
    """Conservative hard limits for indisputably broken camera trajectories."""

    quaternion_norm_tolerance: float = 0.05
    minimum_quaternion_norm: float = 1.0e-6
    maximum_translation_per_step_m: float = 1.0
    maximum_rotation_per_step_deg: float = 90.0
    maximum_linear_speed_mps: float = 3.0
    maximum_angular_speed_deg_s: float = 1080.0
    maximum_window_translation_m: float = 5.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    def to_dict(self) -> dict[str, Any]:
        return {"version": CAMERA_FILTER_VERSION, **asdict(self)}


@dataclass(frozen=True)
class CameraTrajectoryQuality:
    frame_bad: np.ndarray
    edge_bad: np.ndarray
    centers_in_anchor: np.ndarray
    rotations_from_anchor: np.ndarray
    timestamps: np.ndarray | None
    timestamp_available: np.ndarray | None
    timestamp_frame_offset: int
    frame_nonfinite_pose: np.ndarray
    frame_undecodable_pose: np.ndarray
    frame_bad_quaternion: np.ndarray
    frame_nonfinite_timestamp: np.ndarray
    edge_nonpositive_dt: np.ndarray
    edge_translation_jump: np.ndarray
    edge_rotation_jump: np.ndarray
    edge_linear_speed: np.ndarray
    edge_angular_speed: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "camera_bad_frames": int(self.frame_bad.sum()),
            "camera_bad_edges": int(self.edge_bad.sum()),
            "camera_nonfinite_pose_frames": int(self.frame_nonfinite_pose.sum()),
            "camera_undecodable_pose_frames": int(self.frame_undecodable_pose.sum()),
            "camera_bad_quaternion_frames": int(self.frame_bad_quaternion.sum()),
            "camera_nonfinite_timestamp_frames": int(self.frame_nonfinite_timestamp.sum()),
            "camera_nonpositive_dt_edges": int(self.edge_nonpositive_dt.sum()),
            "camera_translation_jump_edges": int(self.edge_translation_jump.sum()),
            "camera_rotation_jump_edges": int(self.edge_rotation_jump.sum()),
            "camera_linear_speed_edges": int(self.edge_linear_speed.sum()),
            "camera_angular_speed_edges": int(self.edge_angular_speed.sum()),
        }


def resolve_relative_camera_columns(columns: Sequence[str]) -> tuple[str, str] | None:
    """Resolve state/action 7-D tracks using the same role precedence as runtime."""

    available = set(columns)
    state_column = next((name for name in ("state_camera_pos", "camera_pos") if name in available), None)
    action_column = next((name for name in ("action_camera_pos", "camera_pos") if name in available), None)
    if state_column is None and action_column is None:
        return None
    if state_column is None or action_column is None:
        raise ValueError(
            "Camera quality filtering needs both state and action 7-D relative tracks; "
            f"resolved state={state_column!r}, action={action_column!r}, available={sorted(available)}"
        )
    return state_column, action_column


def evaluate_camera_trajectory(
    camera_pos: np.ndarray,
    timestamps: np.ndarray | None,
    config: CameraFilterConfig,
    *,
    timestamp_frame_offset: int = 0,
) -> CameraTrajectoryQuality:
    """Validate one episode and return frame/transition masks.

    Quaternion signs are intentionally ignored by converting normalized
    quaternions to rotation matrices. A harmless q/-q sign flip is therefore
    never classified as camera motion.
    """

    if timestamp_frame_offset not in {0, DUAL_TRACK_ACTION_FRAME_OFFSET}:
        raise ValueError(
            "timestamp_frame_offset must be 0 or " f"{DUAL_TRACK_ACTION_FRAME_OFFSET}, got {timestamp_frame_offset}"
        )
    camera = np.asarray(camera_pos, dtype=np.float64)
    if camera.ndim != 2 or camera.shape[1] != 7:
        raise ValueError(f"camera_pos must have shape (T, 7), got {camera.shape}")
    length = camera.shape[0]

    frame_nonfinite_pose = ~np.isfinite(camera).all(axis=1)
    quaternion = camera[:, 3:7]
    quaternion_norm = np.linalg.norm(np.where(np.isfinite(quaternion), quaternion, 0.0), axis=1)
    frame_undecodable_pose = (
        frame_nonfinite_pose | ~np.isfinite(quaternion_norm) | (quaternion_norm < CAMERA_RUNTIME_MIN_QUATERNION_NORM)
    )
    quaternion_norm_error = np.abs(quaternion_norm - 1.0)
    exceeds_norm_tolerance = quaternion_norm_error > config.quaternion_norm_tolerance
    # Treat the configured boundary as inclusive despite binary rounding of
    # decimal values such as 0.95 and 1.05 in float64.
    exceeds_norm_tolerance &= ~np.isclose(
        quaternion_norm_error,
        config.quaternion_norm_tolerance,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    frame_bad_quaternion = (quaternion_norm < config.minimum_quaternion_norm) | exceeds_norm_tolerance

    if timestamps is None:
        timestamp = None
        timestamp_available = None
        frame_nonfinite_timestamp = np.zeros(length, dtype=bool)
    else:
        source_timestamp = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if source_timestamp.shape != (length,):
            raise ValueError(f"timestamps must have shape ({length},), got {source_timestamp.shape}")
        timestamp = source_timestamp.copy()
        timestamp_available = np.ones(length, dtype=bool)
        if timestamp_frame_offset:
            timestamp[:] = np.nan
            timestamp_available[:] = False
            if length > 1:
                timestamp[:-1] = source_timestamp[1:]
                timestamp_available[:-1] = True
        frame_nonfinite_timestamp = timestamp_available & ~np.isfinite(timestamp)

    frame_bad = frame_nonfinite_pose | frame_bad_quaternion | frame_nonfinite_timestamp

    safe_quaternion = quaternion.copy()
    safe_quaternion[~np.isfinite(safe_quaternion)] = 0.0
    safe_norm = np.linalg.norm(safe_quaternion, axis=1)
    usable = ~frame_bad & (safe_norm >= config.minimum_quaternion_norm)
    safe_quaternion[usable] /= safe_norm[usable, None]
    safe_quaternion[~usable] = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = Rotation.from_quat(safe_quaternion).as_matrix()

    safe_translation = camera[:, :3].copy()
    safe_translation[~np.isfinite(safe_translation)] = 0.0
    # P_t = T_{Ct<-C0} = [R_t, p_t], so the camera center in C0 is -R_t^T p_t.
    centers = -np.einsum("tji,tj->ti", rotation, safe_translation)

    edge_shape = (max(0, length - 1),)
    edge_nonpositive_dt = np.zeros(edge_shape, dtype=bool)
    edge_translation_jump = np.zeros(edge_shape, dtype=bool)
    edge_rotation_jump = np.zeros(edge_shape, dtype=bool)
    edge_linear_speed = np.zeros(edge_shape, dtype=bool)
    edge_angular_speed = np.zeros(edge_shape, dtype=bool)

    if length > 1:
        translation_delta = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        relative_rotation = np.einsum("tij,tkj->tik", rotation[:-1], rotation[1:])
        rotation_delta = Rotation.from_matrix(relative_rotation).magnitude()
        edge_translation_jump = translation_delta > config.maximum_translation_per_step_m
        edge_rotation_jump = rotation_delta > np.deg2rad(config.maximum_rotation_per_step_deg)

        if timestamp is not None:
            dt = np.diff(timestamp)
            assert timestamp_available is not None
            dt_available = timestamp_available[:-1] & timestamp_available[1:]
            edge_nonpositive_dt = dt_available & (~np.isfinite(dt) | (dt <= 0.0))
            safe_dt = np.where(edge_nonpositive_dt, np.inf, dt)
            edge_linear_speed = dt_available & (translation_delta / safe_dt > config.maximum_linear_speed_mps)
            edge_angular_speed = dt_available & (
                rotation_delta / safe_dt > np.deg2rad(config.maximum_angular_speed_deg_s)
            )

    edge_bad = (
        frame_bad[:-1]
        | frame_bad[1:]
        | edge_nonpositive_dt
        | edge_translation_jump
        | edge_rotation_jump
        | edge_linear_speed
        | edge_angular_speed
    )
    return CameraTrajectoryQuality(
        frame_bad=frame_bad,
        edge_bad=edge_bad,
        centers_in_anchor=centers,
        rotations_from_anchor=rotation,
        timestamps=timestamp,
        timestamp_available=timestamp_available,
        timestamp_frame_offset=int(timestamp_frame_offset),
        frame_nonfinite_pose=frame_nonfinite_pose,
        frame_undecodable_pose=frame_undecodable_pose,
        frame_bad_quaternion=frame_bad_quaternion,
        frame_nonfinite_timestamp=frame_nonfinite_timestamp,
        edge_nonpositive_dt=edge_nonpositive_dt,
        edge_translation_jump=edge_translation_jump,
        edge_rotation_jump=edge_rotation_jump,
        edge_linear_speed=edge_linear_speed,
        edge_angular_speed=edge_angular_speed,
    )


def _range_has_true(mask: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return whether inclusive ranges [lower, upper] contain a true value."""

    result = np.zeros(lower.shape, dtype=bool)
    valid = upper >= lower
    if not valid.any() or len(mask) == 0:
        return result
    prefix = np.empty(len(mask) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(mask, dtype=np.int64, out=prefix[1:])
    result[valid] = (prefix[upper[valid] + 1] - prefix[lower[valid]]) > 0
    return result


def _invalid_window_components(
    quality: CameraTrajectoryQuality,
    starts: np.ndarray,
    reference_offset: int,
    action_offsets: np.ndarray,
    config: CameraFilterConfig,
    *,
    clip_to_episode: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    action_offsets = np.asarray(action_offsets, dtype=np.int64).reshape(-1)
    offsets = np.unique(np.r_[int(reference_offset), action_offsets])
    if offsets.size == 0 or offsets[0] < 0:
        raise ValueError(f"camera offsets must be non-negative, got {offsets}")
    length = len(quality.frame_bad)
    if length == 0:
        return np.ones(len(starts), dtype=bool), np.zeros(len(starts), dtype=bool), np.zeros(len(starts), dtype=bool)

    lower = starts + int(offsets[0])
    upper = starts + int(offsets[-1])
    if clip_to_episode:
        np.clip(lower, 0, length - 1, out=lower)
        np.clip(upper, 0, length - 1, out=upper)
    elif (lower < 0).any() or (upper >= length).any():
        raise IndexError("complete-window camera indices are outside the episode")

    frame_window_bad = _range_has_true(quality.frame_bad, lower, upper)
    edge_window_bad = _range_has_true(quality.edge_bad, lower, np.maximum(lower - 1, upper - 1))

    reference_indices = starts + int(reference_offset)
    if clip_to_episode:
        np.clip(reference_indices, 0, length - 1, out=reference_indices)
    motion_window_bad = np.zeros(len(starts), dtype=bool)
    reference_centers = quality.centers_in_anchor[reference_indices]
    for offset in action_offsets:
        target = starts + int(offset)
        if clip_to_episode:
            np.clip(target, 0, length - 1, out=target)
        displacement = np.linalg.norm(quality.centers_in_anchor[target] - reference_centers, axis=1)
        motion_window_bad |= displacement > config.maximum_window_translation_m
    return frame_window_bad, edge_window_bad, motion_window_bad


def _window_counts(
    frame_window_bad: np.ndarray,
    edge_window_bad: np.ndarray,
    motion_window_bad: np.ndarray,
) -> dict[str, int]:
    bad = frame_window_bad | edge_window_bad | motion_window_bad
    return {
        "camera_frame_window_rejects": int(frame_window_bad.sum()),
        "camera_edge_window_rejects": int(edge_window_bad.sum()),
        "camera_motion_window_rejects": int(motion_window_bad.sum()),
        "camera_window_rejects": int(bad.sum()),
    }


def invalid_camera_windows(
    quality: CameraTrajectoryQuality,
    starts: np.ndarray,
    reference_offset: int,
    action_offsets: np.ndarray,
    config: CameraFilterConfig,
    *,
    clip_to_episode: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reject windows touching invalid camera rows or transitions."""

    components = _invalid_window_components(
        quality,
        starts,
        reference_offset,
        action_offsets,
        config,
        clip_to_episode=clip_to_episode,
    )
    return components[0] | components[1] | components[2], _window_counts(*components)


def invalid_camera_role_windows(
    state_quality: CameraTrajectoryQuality,
    action_quality: CameraTrajectoryQuality,
    starts: np.ndarray,
    reference_offset: int,
    action_offsets: np.ndarray,
    config: CameraFilterConfig,
    *,
    clip_to_episode: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    """Validate a current state reference against same- or dual-track actions.

    Some schemas store distinct state/action camera tracks. For
    that schema, the state reference frame and action trajectory are validated
    independently, then every action camera center is compared with the state
    reference using the original 5 m whole-window criterion.
    """

    action_offsets = np.asarray(action_offsets, dtype=np.int64).reshape(-1)
    if action_offsets.size == 0:
        raise ValueError("camera action_offsets must not be empty")
    if state_quality is action_quality:
        return invalid_camera_windows(
            state_quality,
            starts,
            reference_offset,
            action_offsets,
            config,
            clip_to_episode=clip_to_episode,
        )

    if state_quality.timestamp_frame_offset != 0:
        raise ValueError(
            "Dual-track state camera timestamps must use frame offset 0, got " f"{state_quality.timestamp_frame_offset}"
        )
    if action_quality.timestamps is not None and action_quality.timestamp_frame_offset != DUAL_TRACK_ACTION_FRAME_OFFSET:
        raise ValueError(
            "Dual-track action camera timestamps must use frame offset "
            f"{DUAL_TRACK_ACTION_FRAME_OFFSET}, got {action_quality.timestamp_frame_offset}"
        )

    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    state_length = len(state_quality.frame_bad)
    action_length = len(action_quality.frame_bad)
    if state_length == 0 or action_length == 0:
        bad = np.ones(len(starts), dtype=bool)
        return bad, {
            **_window_counts(bad, np.zeros_like(bad), np.zeros_like(bad)),
            "camera_cross_track_motion_window_rejects": 0,
        }

    state_indices = starts + int(reference_offset)
    if clip_to_episode:
        np.clip(state_indices, 0, state_length - 1, out=state_indices)
    elif (state_indices < 0).any() or (state_indices >= state_length).any():
        raise IndexError("complete-window state camera indices are outside the episode")
    state_frame_bad = state_quality.frame_bad[state_indices]

    action_frame_bad, action_edge_bad, _ = _invalid_window_components(
        action_quality,
        starts,
        reference_offset,
        action_offsets,
        config,
        clip_to_episode=clip_to_episode,
    )

    action_reference_indices = starts + int(reference_offset)
    if clip_to_episode:
        np.clip(action_reference_indices, 0, action_length - 1, out=action_reference_indices)
    elif (action_reference_indices < 0).any() or (action_reference_indices >= action_length).any():
        raise IndexError("complete-window action camera reference indices are outside the episode")

    boundary_translation = np.linalg.norm(
        action_quality.centers_in_anchor[action_reference_indices] - state_quality.centers_in_anchor[state_indices],
        axis=1,
    )
    boundary_relative_rotation = np.einsum(
        "tij,tkj->tik",
        state_quality.rotations_from_anchor[state_indices],
        action_quality.rotations_from_anchor[action_reference_indices],
    )
    boundary_rotation = Rotation.from_matrix(boundary_relative_rotation).magnitude()
    boundary_translation_jump = boundary_translation > config.maximum_translation_per_step_m
    boundary_rotation_jump = boundary_rotation > np.deg2rad(config.maximum_rotation_per_step_deg)
    boundary_nonpositive_dt = np.zeros(len(starts), dtype=bool)
    boundary_linear_speed = np.zeros(len(starts), dtype=bool)
    boundary_angular_speed = np.zeros(len(starts), dtype=bool)
    if state_quality.timestamps is not None and action_quality.timestamps is not None:
        assert state_quality.timestamp_available is not None
        assert action_quality.timestamp_available is not None
        boundary_time_available = (
            state_quality.timestamp_available[state_indices]
            & action_quality.timestamp_available[action_reference_indices]
        )
        boundary_dt = action_quality.timestamps[action_reference_indices] - state_quality.timestamps[state_indices]
        boundary_nonpositive_dt = boundary_time_available & (~np.isfinite(boundary_dt) | (boundary_dt <= 0.0))
        safe_dt = np.where(boundary_nonpositive_dt, np.inf, boundary_dt)
        boundary_linear_speed = boundary_time_available & (
            boundary_translation / safe_dt > config.maximum_linear_speed_mps
        )
        boundary_angular_speed = boundary_time_available & (
            boundary_rotation / safe_dt > np.deg2rad(config.maximum_angular_speed_deg_s)
        )
    boundary_bad = (
        boundary_translation_jump
        | boundary_rotation_jump
        | boundary_nonpositive_dt
        | boundary_linear_speed
        | boundary_angular_speed
    )

    cross_track_motion_bad = np.zeros(len(starts), dtype=bool)
    reference_centers = state_quality.centers_in_anchor[state_indices]
    for offset in action_offsets:
        target_indices = starts + int(offset)
        if clip_to_episode:
            np.clip(target_indices, 0, action_length - 1, out=target_indices)
        elif (target_indices < 0).any() or (target_indices >= action_length).any():
            raise IndexError("complete-window action camera indices are outside the episode")
        displacement = np.linalg.norm(
            action_quality.centers_in_anchor[target_indices] - reference_centers,
            axis=1,
        )
        cross_track_motion_bad |= displacement > config.maximum_window_translation_m

    frame_bad = state_frame_bad | action_frame_bad
    motion_bad = cross_track_motion_bad
    bad = frame_bad | action_edge_bad | boundary_bad | motion_bad
    counts = {
        **_window_counts(frame_bad, action_edge_bad, motion_bad),
        "camera_state_frame_window_rejects": int(state_frame_bad.sum()),
        "camera_boundary_window_rejects": int(boundary_bad.sum()),
        "camera_boundary_translation_jump_rejects": int(boundary_translation_jump.sum()),
        "camera_boundary_rotation_jump_rejects": int(boundary_rotation_jump.sum()),
        "camera_boundary_nonpositive_dt_rejects": int(boundary_nonpositive_dt.sum()),
        "camera_boundary_linear_speed_rejects": int(boundary_linear_speed.sum()),
        "camera_boundary_angular_speed_rejects": int(boundary_angular_speed.sum()),
        "camera_cross_track_motion_window_rejects": int(cross_track_motion_bad.sum()),
    }
    counts["camera_window_rejects"] = int(bad.sum())
    return bad, counts
