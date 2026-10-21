# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-License-Identifier: Apache-2.0
"""
Vectorised SE(3) relative-chunking helpers.

The single-frame ``EndEffectorPose`` / ``EndEffectorActionChunk`` path in
``pose.py`` and ``action_chunking.py`` builds one scipy Rotation object per
frame and loops in pure Python, which is fine for occasional operations but
becomes the hot spot when generating relative-action statistics (millions
of poses per dataset).

The functions below produce **exactly** the same output as the single-frame
code path but in a fully batched numpy + scipy form:

  - ``rotvec_to_matrix_batch(rv)``  : (N, 3)  -> (N, 3, 3)
  - ``matrix_to_rotvec_batch(R)``   : (N, 3, 3) -> (N, 3)
  - ``xyz_rotvec_to_homogeneous(x)``: (..., 6) -> (..., 4, 4)
  - ``invert_homogeneous(T)``       : (..., 4, 4) -> (..., 4, 4)
  - ``compute_relative_xyz_rotvec_chunks(actions, refs, delta_indices)``
        : (T, 6), (M, 6), (H,) -> (M, H, 6)
        where actions[i] holds the absolute pose at step i,
              refs[j] is the reference (last state) for the j-th window,
              and delta_indices are the H offsets into actions per window.
  - ``recover_camera_spatial_xyz_rotvec_actions(relative_actions, reference_pose, ...)``
        : (H, 6), (6,) -> (H, 6)
        recovers absolute EEF poses from camera-spatial translation and rotation deltas.

Semantics of ``compute_relative_xyz_rotvec_chunks`` mirror the original loop exactly:

    reference = EndEffectorPose(refs[j, :3], refs[j, 3:], rot="rotvec")
    curr      = EndEffectorPose(actions[delta_indices[k] + j, :3],
                                actions[delta_indices[k] + j, 3:], rot="rotvec")
    T_rel     = invert(ref.homogeneous) @ curr.homogeneous
    out[j, k] = concat( T_rel[:3, 3], as_rotvec(T_rel[:3, :3]) )
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def rotvec_to_matrix_batch(rv: NDArray) -> NDArray:
    """Batched axis-angle -> rotation matrix.

    Args:
        rv: (N, 3) rotation vectors.
    Returns:
        (N, 3, 3) rotation matrices, float64.
    """
    # Some Arrow/pandas slices expose a read-only buffer, while older scipy
    # Rotation builds request a writable contiguous memoryview.
    rv = np.array(rv, dtype=np.float64, copy=True, order="C")
    return Rotation.from_rotvec(rv).as_matrix()


def matrix_to_rotvec_batch(R: NDArray) -> NDArray:
    """Batched rotation matrix -> axis-angle. Output ``||v|| in [0, pi]``
    (matches scipy default and single-frame code).

    Args:
        R: (N, 3, 3) rotation matrices.
    Returns:
        (N, 3) rotation vectors, float64.
    """
    R = np.array(R, dtype=np.float64, copy=True, order="C")
    return Rotation.from_matrix(R).as_rotvec()


def xyz_rotvec_to_homogeneous(x: NDArray) -> NDArray:
    """(..., 6) xyz+rotvec -> (..., 4, 4) homogeneous transforms.

    Preserves any leading batch dims. Matches the single-frame
    ``EndEffectorPose(...).homogeneous`` construction.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 6:
        raise ValueError(f"expected last dim 6 (xyz+rotvec), got shape {x.shape}")
    lead = x.shape[:-1]
    flat = x.reshape(-1, 6)
    R = rotvec_to_matrix_batch(flat[:, 3:6])  # (N, 3, 3)
    T = np.zeros((flat.shape[0], 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = flat[:, :3]
    T[:, 3, 3] = 1.0
    return T.reshape(*lead, 4, 4)


def invert_homogeneous(T: NDArray) -> NDArray:
    """Analytic inverse of rigid-body 4x4 transforms.

    Uses the closed-form ``T^{-1} = [R^T | -R^T t; 0 0 0 1]`` which is
    both exact and much faster than ``np.linalg.inv``. Matches
    ``pose.invert_transformation`` from the single-frame path.

    Args:
        T: (..., 4, 4) homogeneous transforms whose top-left 3x3 is a
           pure rotation.
    Returns:
        Same shape.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"expected trailing shape (4, 4), got {T.shape}")
    lead = T.shape[:-2]
    flat = T.reshape(-1, 4, 4)
    R = flat[:, :3, :3]  # (N, 3, 3)
    t = flat[:, :3, 3:4]  # (N, 3, 1)
    R_T = np.transpose(R, (0, 2, 1))  # (N, 3, 3)
    t_inv = -np.matmul(R_T, t)  # (N, 3, 1)
    out = np.zeros_like(flat)
    out[:, :3, :3] = R_T
    out[:, :3, 3:4] = t_inv
    out[:, 3, 3] = 1.0
    return out.reshape(*lead, 4, 4)


def compute_relative_xyz_rotvec_chunks(
    actions: NDArray,
    refs: NDArray,
    delta_indices: NDArray,
    world_from_ref_frames: NDArray | None = None,
    world_from_action_frames: NDArray | None = None,
) -> NDArray:
    """Vectorised drop-in for the ``for i in range(usable_length)`` +
    ``EndEffectorActionChunk(...).relative_chunking(reference_frame=...)``
    pattern used in ``RelativeActionLoader.load_relative_actions``.

    Args:
        actions: (T, 6) absolute EE poses in xyz+rotvec form across the
                 whole episode.
        refs:    (M, 6) reference poses -- one per window. Typically
                 ``state_data[state_ind : state_ind + M]``, so
                 ``refs[j] = state at chunk j``.
        delta_indices: (H,) integer offsets into ``actions`` for each
                 step within a chunk. Absolute action for chunk j, step
                 k is ``actions[delta_indices[k] + j]``.
        world_from_ref_frames: Optional ``(M, 4, 4)`` transforms from the
                 reference pose's storage frame to a shared world frame.
        world_from_action_frames: Optional ``(T, 4, 4)`` transforms from
                 every action pose's storage frame to the same world frame.
                 Supplying both frame tracks compensates moving-camera EEF
                 poses before the relative transform is computed.

    Returns:
        (M, H, 6) array of relative xyz+rotvec poses (float32), matching
        ``np.stack([p.xyz_rotvec for p in traj.poses], dtype=np.float32)``
        of the single-frame path.
    """
    actions = np.asarray(actions, dtype=np.float64)
    refs = np.asarray(refs, dtype=np.float64)
    delta_indices = np.asarray(delta_indices, dtype=np.int64)

    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"actions must be (T, 6), got {actions.shape}")
    if refs.ndim != 2 or refs.shape[1] != 6:
        raise ValueError(f"refs must be (M, 6), got {refs.shape}")
    if delta_indices.ndim != 1:
        raise ValueError(f"delta_indices must be 1-D, got shape {delta_indices.shape}")
    if (world_from_ref_frames is None) != (world_from_action_frames is None):
        raise ValueError("world_from_ref_frames and world_from_action_frames must be supplied together")

    M = refs.shape[0]
    H = delta_indices.shape[0]

    # 1. Build (M, 4, 4) inverse references.
    T_ref = xyz_rotvec_to_homogeneous(refs)  # (M, 4, 4)
    if world_from_ref_frames is not None:
        world_from_ref_frames = np.asarray(world_from_ref_frames, dtype=np.float64)
        world_from_action_frames = np.asarray(world_from_action_frames, dtype=np.float64)
        if world_from_ref_frames.shape != (M, 4, 4):
            raise ValueError(f"world_from_ref_frames must be {(M, 4, 4)}, got {world_from_ref_frames.shape}")
        if world_from_action_frames.shape != (actions.shape[0], 4, 4):
            raise ValueError(
                "world_from_action_frames must match the full action track: "
                f"expected {(actions.shape[0], 4, 4)}, got {world_from_action_frames.shape}"
            )
        T_ref = np.matmul(world_from_ref_frames, T_ref)
    T_ref_inv = invert_homogeneous(T_ref)  # (M, 4, 4)

    # 2. Build (M, H) index grid into `actions` and gather absolute chunk
    #    poses in one shot.
    #    chunk_idx[j, k] = delta_indices[k] + j
    j = np.arange(M, dtype=np.int64)[:, None]  # (M, 1)
    k = delta_indices[None, :]  # (1, H)
    chunk_idx = k + j  # (M, H)
    if chunk_idx.max() >= actions.shape[0]:
        raise IndexError(
            f"chunk_idx.max()={chunk_idx.max()} exceeds actions.shape[0]={actions.shape[0]}. "
            f"Caller must ensure M <= T - delta_indices.max()."
        )
    abs_actions = actions[chunk_idx]  # (M, H, 6)
    T_abs = xyz_rotvec_to_homogeneous(abs_actions)  # (M, H, 4, 4)
    if world_from_action_frames is not None:
        T_abs = np.matmul(world_from_action_frames[chunk_idx], T_abs)

    # 3. Batched T_ref_inv @ T_abs   -->  (M, H, 4, 4)
    T_rel = np.matmul(T_ref_inv[:, None, :, :], T_abs)  # (M, H, 4, 4)

    # 4. Extract translation + rotvec in one batched call.
    R_rel = T_rel[..., :3, :3].reshape(-1, 3, 3)  # (M*H, 3, 3)
    rv_rel = matrix_to_rotvec_batch(R_rel).reshape(M, H, 3)  # (M, H, 3)
    xyz_rel = T_rel[..., :3, 3]  # (M, H, 3)

    out = np.concatenate([xyz_rel, rv_rel], axis=-1).astype(np.float32, copy=False)
    return out


def compute_camera_spatial_xyz_rotvec_chunks(
    actions: NDArray,
    refs: NDArray,
    delta_indices: NDArray,
    world_from_reference_cameras: NDArray,
    world_from_ref_frames: NDArray | None = None,
    world_from_action_frames: NDArray | None = None,
) -> NDArray:
    """Express spatial EEF deltas in the current observation-camera frame.

    Translation is ``R_Ct_from_W @ (p_W_Ek - p_W_Et)`` and rotation is the
    spatial delta ``R_Ct_from_W @ R_W_Ek @ R_W_Et.T @ R_W_from_Ct``. Pose-frame
    tracks are optional for robot-base poses and required as a pair for moving-camera poses.
    """
    actions = np.asarray(actions, dtype=np.float64)
    refs = np.asarray(refs, dtype=np.float64)
    delta_indices = np.asarray(delta_indices, dtype=np.int64)
    world_from_reference_cameras = np.asarray(world_from_reference_cameras, dtype=np.float64)

    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"actions must be (T, 6), got {actions.shape}")
    if refs.ndim != 2 or refs.shape[1] != 6:
        raise ValueError(f"refs must be (M, 6), got {refs.shape}")
    if delta_indices.ndim != 1:
        raise ValueError(f"delta_indices must be 1-D, got shape {delta_indices.shape}")
    if (world_from_ref_frames is None) != (world_from_action_frames is None):
        raise ValueError("world_from_ref_frames and world_from_action_frames must be supplied together")

    window_count = refs.shape[0]
    horizon = delta_indices.shape[0]
    if world_from_reference_cameras.shape != (window_count, 4, 4):
        raise ValueError(
            "world_from_reference_cameras must match refs: "
            f"expected {(window_count, 4, 4)}, got {world_from_reference_cameras.shape}"
        )

    reference_poses = xyz_rotvec_to_homogeneous(refs)
    if world_from_ref_frames is not None:
        world_from_ref_frames = np.asarray(world_from_ref_frames, dtype=np.float64)
        world_from_action_frames = np.asarray(world_from_action_frames, dtype=np.float64)
        if world_from_ref_frames.shape != (window_count, 4, 4):
            raise ValueError(f"world_from_ref_frames must be {(window_count, 4, 4)}, got {world_from_ref_frames.shape}")
        if world_from_action_frames.shape != (actions.shape[0], 4, 4):
            raise ValueError(
                "world_from_action_frames must match the full action track: "
                f"expected {(actions.shape[0], 4, 4)}, got {world_from_action_frames.shape}"
            )
        reference_poses = np.matmul(world_from_ref_frames, reference_poses)

    window_indices = np.arange(window_count, dtype=np.int64)[:, None] + delta_indices[None, :]
    if window_indices.size and window_indices.max() >= actions.shape[0]:
        raise IndexError(f"chunk_idx.max()={window_indices.max()} exceeds actions.shape[0]={actions.shape[0]}")
    action_poses = xyz_rotvec_to_homogeneous(actions[window_indices])
    if world_from_action_frames is not None:
        action_poses = np.matmul(world_from_action_frames[window_indices], action_poses)

    camera_from_world_rotations = np.transpose(
        world_from_reference_cameras[:, :3, :3],
        (0, 2, 1),
    )
    translation_world = action_poses[..., :3, 3] - reference_poses[:, None, :3, 3]
    translation_camera = np.einsum("mij,mhj->mhi", camera_from_world_rotations, translation_world)

    reference_rotations = reference_poses[:, :3, :3]
    action_rotations = action_poses[..., :3, :3]
    spatial_rotation_world = np.matmul(
        action_rotations,
        np.transpose(reference_rotations, (0, 2, 1))[:, None, :, :],
    )
    spatial_rotation_camera = np.matmul(
        np.matmul(camera_from_world_rotations[:, None, :, :], spatial_rotation_world),
        world_from_reference_cameras[:, None, :3, :3],
    )
    rotation_camera = matrix_to_rotvec_batch(spatial_rotation_camera.reshape(-1, 3, 3)).reshape(
        window_count,
        horizon,
        3,
    )
    return np.concatenate([translation_camera, rotation_camera], axis=-1).astype(np.float32, copy=False)


def recover_camera_spatial_xyz_rotvec_actions(
    relative_actions: NDArray,
    reference_pose: NDArray,
    world_from_reference_camera: NDArray,
    world_from_reference_frame: NDArray | None = None,
    world_from_output_frames: NDArray | None = None,
) -> NDArray:
    """Recover absolute EEF poses from current-camera spatial deltas.

    This is the inverse of :func:`compute_camera_spatial_xyz_rotvec_chunks` for
    one action chunk. The relative label convention is::

        dp_C = R_C_from_W @ (p_W_Ek - p_W_Et)
        dR_C = R_C_from_W @ R_W_Ek @ R_W_Et.T @ R_W_from_C

    By default, recovered poses are expressed in the same frame as
    ``reference_pose``. Passing ``world_from_output_frames`` instead recovers
    every pose in its corresponding output frame, which is useful for exact
    round trips when the source action poses were stored in moving cameras.

    Args:
        relative_actions: ``(H, 6)`` camera-spatial xyz+rotvec deltas.
        reference_pose: ``(6,)`` absolute reference EEF pose in the reference
            storage frame.
        world_from_reference_camera: ``(4, 4)`` current observation-camera
            transform into the shared world frame.
        world_from_reference_frame: Optional ``(4, 4)`` transform from the
            reference pose's storage frame into the shared world frame. When
            omitted, the reference pose is assumed to already be in world.
        world_from_output_frames: Optional ``(H, 4, 4)`` transforms from each
            requested output storage frame into world. When omitted, all
            recovered poses use the reference storage frame.

    Returns:
        ``(H, 6)`` recovered absolute xyz+rotvec EEF poses.
    """
    relative_actions = np.asarray(relative_actions, dtype=np.float64)
    reference_pose = np.asarray(reference_pose, dtype=np.float64)
    world_from_reference_camera = np.asarray(world_from_reference_camera, dtype=np.float64)

    if relative_actions.ndim != 2 or relative_actions.shape[1] != 6:
        raise ValueError(f"relative_actions must be (H, 6), got {relative_actions.shape}")
    if reference_pose.shape != (6,):
        raise ValueError(f"reference_pose must be (6,), got {reference_pose.shape}")
    if world_from_reference_camera.shape != (4, 4):
        raise ValueError("world_from_reference_camera must be (4, 4), got " f"{world_from_reference_camera.shape}")

    action_count = relative_actions.shape[0]
    if action_count == 0:
        return np.empty((0, 6), dtype=np.float32)

    if world_from_reference_frame is None:
        world_from_reference_frame = np.eye(4, dtype=np.float64)
    else:
        world_from_reference_frame = np.asarray(world_from_reference_frame, dtype=np.float64)
        if world_from_reference_frame.shape != (4, 4):
            raise ValueError("world_from_reference_frame must be (4, 4), got " f"{world_from_reference_frame.shape}")

    if world_from_output_frames is None:
        world_from_output_frames = np.broadcast_to(
            world_from_reference_frame,
            (action_count, 4, 4),
        )
    else:
        world_from_output_frames = np.asarray(world_from_output_frames, dtype=np.float64)
        if world_from_output_frames.shape != (action_count, 4, 4):
            raise ValueError(
                "world_from_output_frames must match the action chunk: "
                f"expected {(action_count, 4, 4)}, got {world_from_output_frames.shape}"
            )

    reference_pose_world = world_from_reference_frame @ xyz_rotvec_to_homogeneous(reference_pose)
    world_from_camera_rotation = world_from_reference_camera[:3, :3]
    camera_from_world_rotation = world_from_camera_rotation.T

    delta_positions_camera = relative_actions[:, :3]
    delta_rotations_camera = rotvec_to_matrix_batch(relative_actions[:, 3:])

    positions_world = reference_pose_world[:3, 3] + np.einsum(
        "ij,hj->hi",
        world_from_camera_rotation,
        delta_positions_camera,
    )
    rotations_world = np.matmul(
        np.matmul(
            np.matmul(world_from_camera_rotation[None, ...], delta_rotations_camera),
            camera_from_world_rotation[None, ...],
        ),
        reference_pose_world[None, :3, :3],
    )

    poses_world = np.zeros((action_count, 4, 4), dtype=np.float64)
    poses_world[:, :3, :3] = rotations_world
    poses_world[:, :3, 3] = positions_world
    poses_world[:, 3, 3] = 1.0

    poses_output = np.matmul(invert_homogeneous(world_from_output_frames), poses_world)
    rotations_output = matrix_to_rotvec_batch(poses_output[:, :3, :3])
    return np.concatenate([poses_output[:, :3, 3], rotations_output], axis=-1).astype(
        np.float32,
        copy=False,
    )
