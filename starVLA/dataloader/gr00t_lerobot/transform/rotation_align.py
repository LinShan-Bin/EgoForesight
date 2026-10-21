# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-License-Identifier: Apache-2.0
"""
Vectorized rotation-format conversions used by the canonical action pipeline.

Different datasets store EE rotations in different formats (axis-angle / quat /
6D / euler / matrix). The canonical EE_POSE slot is fixed at xyz(3) + axis-angle
rotvec(3) = 6 dims. This module provides batch-friendly converters that operate
on numpy arrays of shape (..., d) so they can be dropped into a DataLoader
transform pipeline without per-sample Python overhead.

Single-sample/dataclass-style logic already lives in
``starVLA.dataloader.gr00t_lerobot.state_action.pose.EndEffectorPose`` -- this
file is the (..., d) batched counterpart.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

RotationFormat = Literal[
    "axis_angle",
    "rotvec",  # alias for axis_angle
    "quaternion_xyzw",
    "quaternion_wxyz",
    "rotation_6d",
    "euler_xyz",
    "euler_zyx",
    "matrix",
]


_ROTVEC_ALIASES = {"axis_angle", "rotvec"}


def _flatten(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    leading = arr.shape[:-1]
    return arr.reshape(-1, arr.shape[-1]), leading


def _unflatten(flat: np.ndarray, leading: tuple[int, ...]) -> np.ndarray:
    return flat.reshape(*leading, flat.shape[-1])


def _rot6d_to_matrix_batch(rot6d_flat: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6D->3x3 (Zhou et al. 2019). Input (N, 6), output (N, 3, 3)."""
    a1 = rot6d_flat[:, :3]
    a2 = rot6d_flat[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    dot = (b1 * a2).sum(axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)


# --------------------------------------------------------------------------
# Rotvec canonicalisation and continuity guards
#
# Axis-angle (rotvec) v represents a rotation of angle ||v|| radians about
# axis v/||v||. Two failure modes bite training if left unguarded:
#
#   (A) Sign ambiguity on the shell ||v||=pi:  +pi*n and -pi*n are the SAME
#       rotation, so tiny sensor jitter on a pose near this shell can flip
#       the whole vector by ~2*pi in Euclidean norm. Also, scipy's
#       Rotation.as_rotvec() canonicalises the angle to [0, pi] and does
#       not fix the hemisphere -- consecutive frames representing the same
#       physical orientation can come out with opposite signs.
#
#   (B) Per-frame regression loss (L2 on rotvec) treats these ~6.28 jumps
#       as huge errors even though the physical delta is ~0.
#
# We provide two guards:
#
#   canonicalize_rotvec(v)       : (frame-local) forces a deterministic
#     representation for every v by (i) folding ||v||>pi via a 2*pi cycle
#     to keep norm<=pi and (ii) picking a canonical hemisphere.
#
#   unwrap_rotvec_sequence(x)    : (chunk-local) walks along the time axis
#     and, whenever ||v_t - v_{t-1}|| >> ||v_t + v_{t-1}||, flips v_t's
#     sign. The output stays inside ||.||<=pi (so downstream q99 statistics
#     do not shift), but consecutive frames no longer see 2*pi jumps.
#
# Both are pure numpy, batch-friendly, and side-effect free.
# --------------------------------------------------------------------------


def _fold_to_pi_shell(flat: np.ndarray) -> np.ndarray:
    """Given (N, 3) rotvec, wrap norms into [0, pi] by shifting angle mod 2*pi.

    For each row v: let a = ||v||, axis = v/a. Reduce a mod 2*pi into
    (-pi, pi]; if the reduced angle is negative, flip its sign and the
    axis together (equivalent rotation). The result has ||v_out|| <= pi.
    """
    a = np.linalg.norm(flat, axis=1, keepdims=True)  # (N, 1)
    zero = (a < 1e-12).reshape(-1)
    if zero.all():
        return flat.copy()
    axis = np.where(a > 1e-12, flat / np.maximum(a, 1e-12), flat)  # (N, 3)
    # reduce angle to (-pi, pi]
    a_flat = a[:, 0]
    a_wrapped = np.mod(a_flat + np.pi, 2.0 * np.pi) - np.pi  # in (-pi, pi]
    # keep sign on axis so vector = axis * angle; enforce angle in [0, pi]
    neg = a_wrapped < 0
    a_wrapped = np.where(neg, -a_wrapped, a_wrapped)
    axis = np.where(neg[:, None], -axis, axis)
    out = axis * a_wrapped[:, None]
    # rows with essentially zero norm stay at zero
    out[zero] = 0.0
    return out.astype(flat.dtype, copy=False)


def _hemisphere_flip(flat: np.ndarray, shell_eps: float = 1e-3, dead_eps: float = 1e-4) -> np.ndarray:
    """Pick a deterministic hemisphere for rotvecs ON THE pi SHELL.

    SAFETY: rotvec v and -v represent the SAME rotation ONLY when ||v||=pi
    (because a rotation of angle pi about axis n equals a rotation of angle
    pi about axis -n). For sub-pi rotvecs, v and -v are OPPOSITE rotations
    and must not be conflated.

    Therefore this function only flips rows whose norm is within
    ``shell_eps`` of pi. Rows with ||v|| < pi - shell_eps are passed
    through untouched.

    Rule for shell rows: flip the whole vector so that its first
    "significant" component (checked in order x, y, z) is positive.
    """
    if flat.shape[0] == 0:
        return flat
    norms = np.linalg.norm(flat, axis=1)  # (N,)
    on_shell = norms > (np.pi - shell_eps)
    if not on_shell.any():
        return flat
    # First significant component per row
    ax_sig = np.abs(flat[:, 0]) > dead_eps
    ay_sig = np.abs(flat[:, 1]) > dead_eps
    sign = np.where(
        ax_sig,
        np.sign(flat[:, 0]),
        np.where(ay_sig, np.sign(flat[:, 1]), np.sign(flat[:, 2])),
    )
    sign = np.where(sign == 0, 1.0, sign)
    # Only apply flip for shell rows
    apply = on_shell & (sign < 0)
    out = flat.copy()
    out[apply] = -flat[apply]
    return out


def canonicalize_rotvec(rotvec: np.ndarray, hemisphere: bool = True) -> np.ndarray:
    """Return a canonicalised rotvec of the SAME shape.

    Steps:
      1. Wrap the angle into [0, pi] so ||v||<=pi (redundant if the input
         already came from scipy.as_rotvec, but safe and cheap).
      2. If ``hemisphere`` (default), pick a canonical sign for the vector
         so that (v, -v) pairs collapse to one representative. This kills
         the pi-shell sign ambiguity for FRAME-INDEPENDENT lookup, but see
         ``unwrap_rotvec_sequence`` for the time-connected fix.
    """
    if rotvec.size == 0:
        return rotvec
    flat, leading = _flatten(rotvec)
    flat = _fold_to_pi_shell(flat)
    if hemisphere:
        flat = _hemisphere_flip(flat)
    return _unflatten(flat.astype(rotvec.dtype, copy=False), leading)


def _quat_to_rotvec_no_fold(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (N, 4) in xyzw order to rotvec (N, 3),
    WITHOUT folding the angle into [0, pi].

    Preserves the sign of w: if w<0 the resulting rotvec has ||v||>pi,
    which is still a valid input to scipy.Rotation.from_rotvec. This is
    exactly what we need for continuity: hemisphere-aligned quaternions
    map to continuous rotvecs (no 2*pi jumps at the shell).
    """
    xyz = q[:, :3]
    w = q[:, 3]
    axis_norm = np.linalg.norm(xyz, axis=1)  # (N,)
    # angle = 2 * atan2(||xyz||, w) in [0, 2*pi). Handles w<0 correctly.
    angle = 2.0 * np.arctan2(axis_norm, w)  # (N,)
    # Where axis_norm ~ 0, rotation is (near-)identity: return zero.
    safe = axis_norm > 1e-8
    axis = np.where(safe[:, None], xyz / np.maximum(axis_norm[:, None], 1e-12), np.zeros_like(xyz))
    return axis * angle[:, None]


def unwrap_rotvec_sequence(
    rotvec: np.ndarray,
    time_axis: int = -2,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Unwrap a time series of rotvecs so consecutive frames are continuous.

    Uses the quaternion double-cover trick: a rotation has TWO unit-quaternion
    representatives, q and -q. Along a smooth trajectory they only differ
    at the ``w=0`` shell, which corresponds exactly to the rotvec
    ``||v||=pi`` shell. So we:

        1. Convert every frame to a unit quaternion via ``from_rotvec``.
        2. Walk forward: whenever q_t . q_{t-1} < 0, negate q_t (same
           rotation, opposite hemisphere).
        3. Convert back to rotvec WITHOUT folding the angle into [0, pi]
           (see ``_quat_to_rotvec_no_fold``). This is the crucial step:
           scipy's ``as_rotvec`` would fold, undoing the alignment.

    The result is rotation-preserving to floating-point precision and
    completely kills the 2*pi discontinuity that plain ``as_rotvec``
    introduces around ``||v||=pi``.

    Args:
        rotvec: (..., T, 3) array of consecutive rotvec frames.
        time_axis: which axis is time. Default -2.
        anchor: optional (..., 3) reference used as the pre-t=0 frame.
                When provided, the first frame is aligned to this anchor
                (useful when the state at t=-1 exists and the chunk
                should be continuous with it).

    Returns:
        Array of the same shape and dtype. Note: after unwrapping,
        ``||v_t||`` may exceed pi. That is fine: ``Rotation.from_rotvec``
        accepts any magnitude (it internally reduces mod 2*pi), and the
        physical rotation is preserved.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError(f"rotvec last dim must be 3, got shape {rotvec.shape}")
    if rotvec.ndim < 2:
        raise ValueError(f"rotvec must have at least 2 dims (time, 3); got {rotvec.shape}")

    # Move time axis to position -2 for uniform processing, then move back.
    arr = np.moveaxis(rotvec, time_axis, -2).copy()  # (..., T, 3)
    T = arr.shape[-2]
    if T < 2 and anchor is None:
        return rotvec.copy()

    # Flatten the batch dims to a single leading dim for vectorised ops.
    leading = arr.shape[:-2]
    N = int(np.prod(leading)) if leading else 1
    flat = arr.reshape(N, T, 3).astype(np.float64)  # use f64 for numerics

    # 1. Convert whole (N*T, 3) block to quaternions in one call.
    q_all = Rotation.from_rotvec(flat.reshape(-1, 3)).as_quat()  # (N*T, 4) xyzw
    q = q_all.reshape(N, T, 4)

    # 2. Anchor alignment (optional) and hemisphere alignment forward
    if anchor is not None:
        anchor_arr = np.asarray(anchor, dtype=np.float64).reshape(N, 3)
        anchor_q = Rotation.from_rotvec(anchor_arr).as_quat()  # (N, 4)
        dot0 = (q[:, 0] * anchor_q).sum(axis=1)
        q[:, 0] = np.where(dot0[:, None] < 0.0, -q[:, 0], q[:, 0])
    for t in range(1, T):
        dot = (q[:, t] * q[:, t - 1]).sum(axis=1)  # (N,)
        q[:, t] = np.where(dot[:, None] < 0.0, -q[:, t], q[:, t])

    # 3. Convert back to rotvec, NO angle fold.
    out_flat = _quat_to_rotvec_no_fold(q.reshape(-1, 4))  # (N*T, 3)
    out = out_flat.reshape(N, T, 3)

    out = out.reshape(*leading, T, 3) if leading else out.reshape(T, 3)
    out = np.moveaxis(out, -2, time_axis)
    return out.astype(rotvec.dtype, copy=False)


def to_rotvec(
    rotation: np.ndarray,
    src_format: RotationFormat,
    canonicalize: bool = True,
) -> np.ndarray:
    """Convert any-format rotation to axis-angle rotvec.

    Args:
        rotation: (..., d) where d depends on src_format
                  axis_angle/rotvec: d=3
                  quaternion_xyzw / quaternion_wxyz: d=4
                  rotation_6d: d=6
                  euler_xyz / euler_zyx: d=3
                  matrix: d=9 (flattened row-major)
        src_format: source representation name.
        canonicalize: if True (default), apply frame-local
                      ``canonicalize_rotvec`` on the result so callers get
                      a hemisphere-normalised, |v|<=pi representation. Set
                      False only if you have an upstream reason to keep
                      scipy's raw as_rotvec output.

    Returns:
        (..., 3) axis-angle rotvec.
    """
    fmt = src_format.lower()
    flat, leading = _flatten(rotation)

    if fmt in _ROTVEC_ALIASES:
        if flat.shape[1] != 3:
            raise ValueError(f"rotvec expects last dim 3, got {flat.shape[1]}")
        out = flat.astype(np.float32, copy=True)
        if canonicalize:
            out = _fold_to_pi_shell(out)
            out = _hemisphere_flip(out)
        return _unflatten(out, leading)

    if fmt == "quaternion_xyzw":
        if flat.shape[1] != 4:
            raise ValueError(f"quaternion_xyzw expects last dim 4, got {flat.shape[1]}")
        rot = Rotation.from_quat(flat)
    elif fmt == "quaternion_wxyz":
        if flat.shape[1] != 4:
            raise ValueError(f"quaternion_wxyz expects last dim 4, got {flat.shape[1]}")
        # scipy uses (x,y,z,w) order, so reorder
        rot = Rotation.from_quat(flat[:, [1, 2, 3, 0]])
    elif fmt == "rotation_6d":
        if flat.shape[1] != 6:
            raise ValueError(f"rotation_6d expects last dim 6, got {flat.shape[1]}")
        mats = _rot6d_to_matrix_batch(flat)
        rot = Rotation.from_matrix(mats)
    elif fmt == "euler_xyz":
        if flat.shape[1] != 3:
            raise ValueError(f"euler_xyz expects last dim 3, got {flat.shape[1]}")
        rot = Rotation.from_euler("xyz", flat, degrees=False)
    elif fmt == "euler_zyx":
        if flat.shape[1] != 3:
            raise ValueError(f"euler_zyx expects last dim 3, got {flat.shape[1]}")
        rot = Rotation.from_euler("zyx", flat, degrees=False)
    elif fmt == "matrix":
        if flat.shape[1] != 9:
            raise ValueError(f"matrix expects last dim 9 (3x3 flattened), got {flat.shape[1]}")
        rot = Rotation.from_matrix(flat.reshape(-1, 3, 3))
    else:
        raise NotImplementedError(f"unsupported source rotation format: {src_format}")

    out = rot.as_rotvec().astype(np.float32, copy=False)
    if canonicalize:
        out = _fold_to_pi_shell(out)
        out = _hemisphere_flip(out)
    return _unflatten(out, leading)


def from_rotvec(rotvec: np.ndarray, dst_format: RotationFormat) -> np.ndarray:
    """Inverse of ``to_rotvec`` -- for canonical -> embodiment unmapping at
    inference time.
    """
    fmt = dst_format.lower()
    flat, leading = _flatten(rotvec)
    if flat.shape[1] != 3:
        raise ValueError(f"rotvec source expects last dim 3, got {flat.shape[1]}")

    if fmt in _ROTVEC_ALIASES:
        return _unflatten(flat.astype(np.float32, copy=False), leading)

    rot = Rotation.from_rotvec(flat)

    if fmt == "quaternion_xyzw":
        out = rot.as_quat()
    elif fmt == "quaternion_wxyz":
        q = rot.as_quat()
        out = q[:, [3, 0, 1, 2]]
    elif fmt == "rotation_6d":
        mats = rot.as_matrix()
        out = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)
    elif fmt == "euler_xyz":
        out = rot.as_euler("xyz", degrees=False)
    elif fmt == "euler_zyx":
        out = rot.as_euler("zyx", degrees=False)
    elif fmt == "matrix":
        out = rot.as_matrix().reshape(-1, 9)
    else:
        raise NotImplementedError(f"unsupported destination rotation format: {dst_format}")

    return _unflatten(out.astype(np.float32, copy=False), leading)


def xyz_plus_rotation_to_xyz_rotvec(eef: np.ndarray, src_rotation_format: RotationFormat) -> np.ndarray:
    """Convert (..., 3 + rot_d) end-effector pose to canonical xyz+rotvec (..., 6).

    The translation channels (first 3) pass through unchanged.
    """
    xyz = eef[..., :3]
    rot = eef[..., 3:]
    rotvec = to_rotvec(rot, src_rotation_format)
    return np.concatenate([xyz, rotvec], axis=-1).astype(np.float32, copy=False)


def xyz_rotvec_to_xyz_plus_rotation(eef_canonical: np.ndarray, dst_rotation_format: RotationFormat) -> np.ndarray:
    """Inverse of ``xyz_plus_rotation_to_xyz_rotvec``. Returns (..., 3 + rot_d)."""
    xyz = eef_canonical[..., :3]
    rotvec = eef_canonical[..., 3:6]
    rot = from_rotvec(rotvec, dst_rotation_format)
    return np.concatenate([xyz, rot], axis=-1).astype(np.float32, copy=False)
