# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-License-Identifier: Apache-2.0
"""
RotvecContinuityTransform: chunk-local rotvec unwrap.

Motivation
----------
The canonical EE-pose slot in the 130-D layout is ``xyz(3) + axis-angle
rotvec(3) = 6``. Axis-angle has two well-known failure modes:

  1. **pi-shell sign ambiguity.** A rotation of angle ``pi`` about axis
     ``n`` equals a rotation of ``pi`` about ``-n``: (v, -v) map to the
     same rotation on ``||v||=pi``. Tiny sensor jitter at this shell can
     flip the whole 3-vector by ~2*pi in Euclidean norm frame-to-frame.

  2. **Frame-to-frame representation flips.** ``scipy.Rotation.as_rotvec``
     folds ``||v||`` into ``[0, pi]``. A trajectory that smoothly crosses
     the pi shell in one direction gets its rotvec representation flipped
     to the opposite hemisphere on the far side. The regression loss sees
     a ~6.28 jump although the physical rotation moved by ~0 radians.

The head that consumes this signal is a flow-matching L2 regressor over
the canonical vector; unaddressed pi-shell jumps corrupt the training
signal on data crossing that shell.

This transform walks each ``xyz+rotvec`` key along its time axis using
quaternion double-cover alignment (see ``rotation_align.
unwrap_rotvec_sequence``) so the rotation half of every EE-pose key is
frame-to-frame continuous. Rotation matrices are preserved to floating-
point precision; only the vector representation changes.

Where to install
----------------
Two placement modes:

**Per-key mode** — right after ``EefPoseAlignTransform`` (which produces
canonical xyz+rotvec keys) and BEFORE any normalization or scatter step.
Only useful for configs whose source EE-pose is quat/rot6d/euler (i.e.
non-rotvec) AND uses ``EefPoseAlignTransform`` to merge into xyz+rotvec::

    RotvecContinuityTransform(
        keys=[
            "action.left_ee_pose",  "action.right_ee_pose",
            "state.left_ee_pose",   "state.right_ee_pose",
        ],
    )

**Canonical-slot mode** — AFTER ``ConcatTransform`` (which scatters into
the canonical 130-D layout) and BEFORE ``GR00TTransform``. Works for
EVERY dataset uniformly, including configs where the EE-pose source is
raw rotvec, euler, or anything else -- because by the time
ConcatTransform is done, all rotations have been converted into the
canonical rotvec slots regardless of upstream representation::

    RotvecContinuityTransform(
        canonical_slots=["LEFT_EEF_POSE", "RIGHT_EEF_POSE"],
        state_key="state",
        action_key="action",
    )

Prefer canonical-slot mode when possible: it is one line per DataConfig
and covers all sources through a single unified path.

Keys / slots missing from ``data`` are silently skipped so the same
DataConfig can be reused across embodiment variants.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import Field

from ..canonical_action import CanonicalSlot, SlotKind
from .base import ModalityTransform
from .rotation_align import unwrap_rotvec_sequence


class RotvecContinuityTransform(ModalityTransform):
    """Frame-to-frame rotvec unwrap on xyz+rotvec segments.

    Two mutually compatible modes:

    * **Per-key mode**: set ``keys`` to a list of data-dict keys whose
      value has last dim >= 6 and is laid out as xyz(3) + rotvec(3) +
      (optional trailing dims). The rotvec segment ``[..., 3:6]`` is
      unwrapped in-place; xyz and any trailing dims pass through.

    * **Canonical-slot mode**: set ``canonical_slots`` to a list of
      canonical slot names (e.g. ``["LEFT_EEF_POSE", "RIGHT_EEF_POSE"]``)
      and provide ``state_key`` / ``action_key`` (default "state" /
      "action") — the ``[start+3:end]`` sub-range of each named slot in
      the corresponding scatter output tensor is unwrapped.

    Both modes may be enabled at once; the per-key pass runs first.
    """

    apply_to: list[str] = Field(default_factory=list)

    # --- per-key mode ---
    keys: list[str] = Field(
        default_factory=list,
        description="Keys holding xyz+rotvec (or xyz+rotvec+more) arrays.",
    )
    # --- canonical-slot mode ---
    canonical_slots: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical slot names (e.g. 'LEFT_EEF_POSE') whose rotvec sub-range "
            "should be unwrapped in the state/action scatter tensors."
        ),
    )
    state_key: str = Field(
        default="state",
        description="Name of the state scatter tensor in the data dict (canonical-slot mode).",
    )
    action_key: str = Field(
        default="action",
        description="Name of the action scatter tensor in the data dict (canonical-slot mode).",
    )

    time_axis: int = Field(
        default=-2,
        description="Axis along which time flows. Default -2 (i.e. shape (..., T, D)).",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        # ----- per-key mode -----
        for key in self.keys:
            if key not in data:
                continue
            val = data[key]
            arr, was_torch, ref = self._as_numpy(val)
            if arr.shape[-1] < 6:
                raise ValueError(f"{key} last dim must be >= 6 (xyz+rotvec), got {arr.shape[-1]}")
            if arr.ndim < 2:
                continue

            ta = self.time_axis
            if ta == -1 or ta == arr.ndim - 1:
                raise ValueError(
                    f"time_axis {self.time_axis} points at the feature dim of {key} "
                    f"(shape={arr.shape}); pick the time axis instead."
                )
            rot = arr[..., 3:6]
            unwrapped = unwrap_rotvec_sequence(rot, time_axis=ta)
            new_arr = arr.copy()
            new_arr[..., 3:6] = unwrapped
            data[key] = self._maybe_to_torch(new_arr, was_torch, ref)

        # ----- canonical-slot mode -----
        if self.canonical_slots:
            slot_ranges = self._resolve_slots(self.canonical_slots)
            for scatter_key in (self.state_key, self.action_key):
                if scatter_key not in data:
                    continue
                val = data[scatter_key]
                arr, was_torch, ref = self._as_numpy(val)
                if arr.ndim < 2:
                    continue

                ta = self.time_axis
                if ta == -1 or ta == arr.ndim - 1:
                    raise ValueError(
                        f"time_axis {self.time_axis} points at the feature dim of {scatter_key} "
                        f"(shape={arr.shape}); pick the time axis instead."
                    )

                new_arr = arr.copy()
                touched = False
                for slot_name, (start, end) in slot_ranges:
                    if end > arr.shape[-1]:
                        # Effective canonical dim can be smaller than
                        # CANONICAL_*_DIM at runtime (see set_canonical_dims).
                        # Skip slots that lie outside the current tensor.
                        continue
                    if end - start != 6:
                        raise ValueError(
                            f"slot {slot_name} spans {end-start} dims, expected 6 "
                            f"(xyz+rotvec). Only EE_POSE slots are supported."
                        )
                    rot = new_arr[..., start + 3 : end]
                    unwrapped = unwrap_rotvec_sequence(rot, time_axis=ta)
                    new_arr[..., start + 3 : end] = unwrapped
                    touched = True
                if touched:
                    data[scatter_key] = self._maybe_to_torch(new_arr, was_torch, ref)

        return data

    # ----- small helpers ----------------------------------------------------
    @staticmethod
    def _resolve_slots(names: list[str]) -> list[tuple[str, tuple[int, int]]]:
        out = []
        for n in names:
            try:
                spec = CanonicalSlot[n].value
            except KeyError as e:
                raise ValueError(
                    f"unknown canonical slot {n!r}. Available: " f"{[m.name for m in CanonicalSlot]}"
                ) from e
            if spec.kind != SlotKind.EE_POSE:
                raise ValueError(
                    f"slot {n!r} is not an EE_POSE slot (kind={spec.kind}); " f"only EE_POSE slots hold xyz+rotvec."
                )
            out.append((n, (spec.start, spec.end)))
        return out

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
