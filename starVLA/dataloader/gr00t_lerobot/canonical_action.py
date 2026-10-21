# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-License-Identifier: Apache-2.0
"""
Canonical (physically-aligned) action / state space.

Every embodiment's action and state vectors are scattered into one fixed 130-D
layout where each dimension has a globally unique physical meaning. This
mirrors the Qwen-RobotManip "unified state-action representation" design
(two 29-D per-arm joint/EE/gripper/dex blocks + 22 shared/reserved dims +
two 25-D per-hand finger-joint blocks).

A per-dim binary mask travels with the canonical vector so that loss /
gradients only flow through slots that this particular dataset actually fills.

Layout (indices are [start, end), half-open):

  LEFT  ARM block  ([ 0:29])
      0:7    LEFT_ARM_JOINT      up to 7-DoF arm joint angles
      7:13   LEFT_EEF_POSE       xyz(3) + axis-angle rotvec(3) = 6
      13:14  LEFT_GRIPPER        continuous gripper value
      14:29  LEFT_DEX_HAND       up to 15 dexterous-hand FINGERTIP positions
                                 (5 fingers x xyz = 15)

  RIGHT ARM block  ([29:58])
      29:36  RIGHT_ARM_JOINT
      36:42  RIGHT_EEF_POSE
      42:43  RIGHT_GRIPPER
      43:58  RIGHT_DEX_HAND      fingertip positions (mirror of LEFT_DEX_HAND)

  SHARED / RESERVED  ([58:80])
      58:64  BASE_VELOCITY       mobile-base (vx,vy,vz,wx,wy,wz)
      64:68  HEAD_POSE
      68:72  WAIST_POSE
      72:80  RESERVED_FUTURE

  FINGER-JOINT block  ([80:130])
      80:105 LEFT_DEX_JOINT      up to 25 dexterous-hand finger JOINT angles
                                 (5 fingers x 5 joints/finger = 25;
                                  finger order: thumb, index, middle, ring,
                                  pinky; joint order proximal -> distal:
                                  abduction, MCP, PIP, DIP, reserved)
      105:130 RIGHT_DEX_JOINT    mirror of LEFT_DEX_JOINT

The split between DEX_HAND (Cartesian fingertip xyz) and DEX_JOINT (joint
angles) is intentional: they are different physical quantities and should
not share dimensions. Most embodiments populate one OR the other, not both.

Embodiments with fewer joints than 25 per hand (e.g. unitree_g1 has 6 or 7
per hand; Inspire-Hand has 6) are left-aligned into the slot: source dims
fill ``[slot.start, slot.start + d_src)`` and trailing dims stay zero with
mask=False, unless their DataConfig supplies explicit canonical_slot_indices.
In particular, RoboCasa's six hand controls use sparse offsets matching the
canonical Sharpa 22-D source order (not the conceptual 5x5 order above).
See hand_joint_layout.py; do not reorder pretrained Sharpa slots.

The R1Pro Sharpa convention preserves the retargeter's packed 22-D
order (index4, middle4, pinky5, ring4, thumb5) in [80:102] / [105:127].
These hands do not use the illustrative five-entries-per-finger layout
above; their final three entries per hand remain masked out.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

CANONICAL_ACTION_DIM = 130
CANONICAL_STATE_DIM = 130

# Effective dims. Default to the static CANONICAL_*_DIM. Can be overridden
# at run start via ``set_canonical_dims`` so the head and dataloader stay
# in sync when launching from a hyperparam like
# ``--framework.action_model.action_dim``.
_EFFECTIVE_ACTION_DIM: int = CANONICAL_ACTION_DIM
_EFFECTIVE_STATE_DIM: int = CANONICAL_STATE_DIM


def set_canonical_dims(action_dim: int | None = None, state_dim: int | None = None) -> None:
    """Override the effective canonical action / state dims used by the
    dataloader pipeline. Pass None to keep the current value (defaults
    derive from the static module constants).
    """
    global _EFFECTIVE_ACTION_DIM, _EFFECTIVE_STATE_DIM
    if action_dim is not None:
        _EFFECTIVE_ACTION_DIM = int(action_dim)
    if state_dim is not None:
        _EFFECTIVE_STATE_DIM = int(state_dim)


def get_canonical_action_dim() -> int:
    """Return the effective canonical action dim (hyperparam override or static)."""
    return _EFFECTIVE_ACTION_DIM


def get_canonical_state_dim() -> int:
    """Return the effective canonical state dim (hyperparam override or static)."""
    return _EFFECTIVE_STATE_DIM


# ---------------------------------------------------------------------------
# Global enable flag.
#
# Whether the dataloader pipeline should route through the canonical 130-D
# layout. Off by default for backwards compatibility: DataConfigs that have
# been migrated will detect this flag and either install canonical_slots /
# EefPoseAlignTransform (when True) or fall back to their legacy concat
# concat path (when False).
#
# ``build_dataloader`` toggles this at run start from
# ``cfg.datasets.vla_data.enable_action_alignment``.
# ---------------------------------------------------------------------------

_ACTION_ALIGNMENT_ENABLED: bool = False


def set_action_alignment_enabled(flag: bool) -> None:
    """Toggle the canonical 130-D pipeline on or off process-wide."""
    global _ACTION_ALIGNMENT_ENABLED
    _ACTION_ALIGNMENT_ENABLED = bool(flag)


def is_action_alignment_enabled() -> bool:
    return _ACTION_ALIGNMENT_ENABLED


# ---------------------------------------------------------------------------
# State-relative EEF action flag.
#
# When ON, DataConfigs that have EEF pose slots (LEFT_EEF_POSE / RIGHT_EEF_POSE)
# will inject a RelativeActionTransform in their transform pipeline. Every
# action in a chunk is expressed as an SE(3) transform relative to the current
# state's EEF pose (state chunk of length 1, reference_index=0). Gripper /
# joint / dex / head / waist slots stay absolute.
#
# build_dataloader toggles this at run start from
# cfg.datasets.vla_data.enable_state_relative_eef.
# ---------------------------------------------------------------------------

_STATE_RELATIVE_EEF_ENABLED: bool = False
_RELATIVE_EEF_FRAME: str = "eef"
VALID_RELATIVE_EEF_FRAMES = frozenset({"eef", "camera_spatial"})


def set_state_relative_eef_enabled(flag: bool) -> None:
    global _STATE_RELATIVE_EEF_ENABLED
    _STATE_RELATIVE_EEF_ENABLED = bool(flag)


def is_state_relative_eef_enabled() -> bool:
    return _STATE_RELATIVE_EEF_ENABLED


def set_relative_eef_frame(frame: str) -> None:
    """Select the frame/convention used for relative EEF labels."""
    normalized = str(frame).lower().replace("-", "_")
    if normalized not in VALID_RELATIVE_EEF_FRAMES:
        raise ValueError(f"relative_eef_frame must be one of {sorted(VALID_RELATIVE_EEF_FRAMES)}, got {frame!r}")
    global _RELATIVE_EEF_FRAME
    _RELATIVE_EEF_FRAME = normalized


def get_relative_eef_frame() -> str:
    return _RELATIVE_EEF_FRAME


class SlotKind(str, Enum):
    """Physical-meaning category of a canonical slot."""

    JOINT = "joint"
    EE_POSE = "ee_pose"
    GRIPPER = "gripper"
    DEX_HAND = "dex_hand"  # fingertip xyz positions (5*3=15)
    DEX_JOINT = "dex_joint"  # finger joint angles
    BASE = "base"
    HEAD = "head"
    WAIST = "waist"
    RESERVED = "reserved"


@dataclass(frozen=True)
class SlotSpec:
    name: str
    start: int
    end: int
    kind: SlotKind
    side: str | None  # "left" | "right" | None for shared blocks

    @property
    def size(self) -> int:
        return self.end - self.start


class CanonicalSlot(Enum):
    """Globally unique action/state slot identifiers."""

    LEFT_ARM_JOINT = SlotSpec("left_arm_joint", 0, 7, SlotKind.JOINT, "left")
    LEFT_EEF_POSE = SlotSpec("left_eef_pose", 7, 13, SlotKind.EE_POSE, "left")
    LEFT_GRIPPER = SlotSpec("left_gripper", 13, 14, SlotKind.GRIPPER, "left")
    LEFT_DEX_HAND = SlotSpec("left_dex_hand", 14, 29, SlotKind.DEX_HAND, "left")

    RIGHT_ARM_JOINT = SlotSpec("right_arm_joint", 29, 36, SlotKind.JOINT, "right")
    RIGHT_EEF_POSE = SlotSpec("right_eef_pose", 36, 42, SlotKind.EE_POSE, "right")
    RIGHT_GRIPPER = SlotSpec("right_gripper", 42, 43, SlotKind.GRIPPER, "right")
    RIGHT_DEX_HAND = SlotSpec("right_dex_hand", 43, 58, SlotKind.DEX_HAND, "right")

    BASE_VELOCITY = SlotSpec("base_velocity", 58, 64, SlotKind.BASE, None)
    HEAD_POSE = SlotSpec("head_pose", 64, 68, SlotKind.HEAD, None)
    WAIST_POSE = SlotSpec("waist_pose", 68, 72, SlotKind.WAIST, None)
    RESERVED_FUTURE = SlotSpec("reserved_future", 72, 80, SlotKind.RESERVED, None)

    LEFT_DEX_JOINT = SlotSpec("left_dex_joint", 80, 105, SlotKind.DEX_JOINT, "left")
    RIGHT_DEX_JOINT = SlotSpec("right_dex_joint", 105, 130, SlotKind.DEX_JOINT, "right")

    @property
    def spec(self) -> SlotSpec:
        return self.value

    @property
    def start(self) -> int:
        return self.value.start

    @property
    def end(self) -> int:
        return self.value.end

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def kind(self) -> SlotKind:
        return self.value.kind


_ALL_SLOTS: tuple[CanonicalSlot, ...] = tuple(CanonicalSlot)


def _validate_layout() -> None:
    covered = np.zeros(CANONICAL_ACTION_DIM, dtype=bool)
    for slot in _ALL_SLOTS:
        s, e = slot.start, slot.end
        assert 0 <= s < e <= CANONICAL_ACTION_DIM, f"bad slot {slot}: [{s},{e})"
        assert not covered[s:e].any(), f"slot {slot} overlaps existing layout"
        covered[s:e] = True
    assert covered.all(), f"layout has holes: {(~covered).nonzero()[0].tolist()}"


_validate_layout()


def scatter_to_canonical(
    sources: dict[str, np.ndarray],
    slot_map: dict[str, CanonicalSlot],
    horizon: int,
    *,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-key arrays into a single static canonical [T, 130] tensor.

    Args:
        sources: data["action.left_arm"] -> ndarray [T, d_src], etc. Keys not in
                 ``slot_map`` are ignored (caller drops or handles them).
        slot_map: which canonical slot each source key writes into. A source
                  with d_src < slot.size is left-aligned and the trailing dims
                  stay zero; the mask follows the actual filled width.
        horizon: T, time dimension.
        dtype: output dtype.

    Returns:
        (canonical, mask) where canonical.shape == mask.shape == (T, 130) and
        mask is bool with True on dims actually written by some source.
    """
    canonical = np.zeros((horizon, CANONICAL_ACTION_DIM), dtype=dtype)
    mask = np.zeros((horizon, CANONICAL_ACTION_DIM), dtype=bool)
    for key, slot in slot_map.items():
        if key not in sources:
            continue
        arr = np.asarray(sources[key])
        if arr.ndim == 1:
            arr = arr[None, :]
        assert arr.shape[0] == horizon, f"source {key} has horizon {arr.shape[0]}, expected {horizon}"
        d_src = arr.shape[1]
        if d_src > slot.size:
            raise ValueError(f"source {key} dim {d_src} exceeds slot {slot.name} capacity {slot.size}")
        canonical[:, slot.start : slot.start + d_src] = arr.astype(dtype, copy=False)
        mask[:, slot.start : slot.start + d_src] = True
    return canonical, mask


def canonical_to_embodiment(
    canonical: np.ndarray,
    slot_map: dict[str, CanonicalSlot],
    source_dims: dict[str, int],
) -> dict[str, np.ndarray]:
    """Inverse of ``scatter_to_canonical`` for inference / deployment.

    Args:
        canonical: [..., 130] canonical action tensor.
        slot_map: same map used during scatter.
        source_dims: how many leading dims of each slot to keep for each key.

    Returns:
        dict[key, ndarray of shape [..., source_dims[key]]] preserving the
        embodiment's original concatenation order is the caller's job.
    """
    out: dict[str, np.ndarray] = {}
    for key, slot in slot_map.items():
        d = source_dims[key]
        if d > slot.size:
            raise ValueError(f"requested {d} dims of slot {slot.name} (capacity {slot.size})")
        out[key] = canonical[..., slot.start : slot.start + d]
    return out


def slots_by_kind(kind: SlotKind) -> tuple[CanonicalSlot, ...]:
    return tuple(s for s in _ALL_SLOTS if s.kind == kind)


def all_slots() -> Iterable[CanonicalSlot]:
    return _ALL_SLOTS
