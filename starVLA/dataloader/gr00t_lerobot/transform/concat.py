# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import torch
from pydantic import Field

from ..canonical_action import (
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    CanonicalSlot,
    get_canonical_action_dim,
    get_canonical_state_dim,
)
from ..schema import DatasetMetadata, StateActionMetadata
from .base import InvertibleModalityTransform
from .rotation_align import (
    xyz_plus_rotation_to_xyz_rotvec,
    xyz_rotvec_to_xyz_plus_rotation,
)


def _resolve_slot(name_or_slot) -> CanonicalSlot:
    """Accept either a CanonicalSlot enum or its name (configs serialize to JSON)."""
    if isinstance(name_or_slot, CanonicalSlot):
        return name_or_slot
    if isinstance(name_or_slot, str):
        return CanonicalSlot[name_or_slot]
    raise TypeError(f"cannot resolve canonical slot: {name_or_slot!r}")


class ConcatTransform(InvertibleModalityTransform):
    """
    Concatenate the keys according to specified order.

    Two output modes are supported per modality:

    1. **Legacy concat** (default): outputs ``data["action"]`` of shape ``[T, D]``
       where ``D = sum(action_dims[k] for k in action_concat_order)``.
       Downstream ``GR00TTransform`` then zero-pads up to ``max_action_dim``.

    2. **Canonical scatter** (opt-in via ``action_canonical_slots``):
       writes each source key into a globally fixed canonical slot layout
       (130-D by default, or the effective dim configured for the model head).
       Yields ``[T, D]`` action + a binary ``action_mask [T, D]`` indicating
       which dims were actually filled. Same scheme applies to state when
       ``state_canonical_slots`` is set.

    Mixing modes within a single ConcatTransform is allowed: e.g. canonical
    action + legacy state.
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(default_factory=list, description="Not used in this transform, kept for compatibility.")

    video_concat_order: list[str] = Field(
        ...,
        description="Concatenation order for each video modality. " "Format: ['video.ego_view_pad_res224_freq20', ...]",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. " "Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    # --- Canonical (physically-aligned) scatter mode --------------------------
    # Mapping {modality_key -> canonical slot name}. When set for a modality,
    # the canonical scatter path is taken and the legacy concat path skipped.
    action_canonical_slots: Optional[dict[str, str]] = Field(
        default=None,
        description="Map of action_key -> CanonicalSlot name; enables canonical scatter for actions.",
    )
    state_canonical_slots: Optional[dict[str, str]] = Field(
        default=None,
        description="Map of state_key -> CanonicalSlot name; enables canonical scatter for state.",
    )
    action_canonical_slot_indices: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Optional source-dim -> slot-relative index mapping for non-contiguous action fields.",
    )
    state_canonical_slot_indices: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Optional source-dim -> slot-relative index mapping for non-contiguous state fields.",
    )
    # Per-key source rotation format for EE pose keys (e.g. 'quaternion_xyzw',
    # 'rotation_6d', 'euler_xyz', ...). Used to convert each EE key to the
    # canonical xyz(3) + rotvec(3) representation before scattering. Keys not
    # listed here are assumed to already be in xyz+rotvec.
    action_eef_rotation_source: dict[str, str] = Field(
        default_factory=dict,
        description="Source rotation format for action EE-pose keys.",
    )
    state_eef_rotation_source: dict[str, str] = Field(
        default_factory=dict,
        description="Source rotation format for state EE-pose keys.",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
                "action_canonical_slots",
                "state_canonical_slots",
                "action_canonical_slot_indices",
                "state_canonical_slot_indices",
                "action_eef_rotation_source",
                "state_eef_rotation_source",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            # Check if keys in video_concat_order, state_concat_order, action_concat_order are
            # ineed contained in the data. If not, then the keys are misspecified
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None, f"{self.video_concat_order=}, {video_keys=}"
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_video = np.expand_dims(video_data, axis=-4)  # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

            # Video
            data["video"] = unsqueezed_video

        # "state"
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            if self.state_canonical_slots is not None:
                # Canonical scatter path: write each key into its fixed slot.
                data = self._scatter_canonical(
                    data=data,
                    keys=state_keys,
                    canonical_slots=self.state_canonical_slots,
                    canonical_slot_indices=self.state_canonical_slot_indices,
                    eef_rotation_source=self.state_eef_rotation_source,
                    out_dim=get_canonical_state_dim(),
                    out_key="state",
                    mask_key="state_mask",
                )
            else:
                assert self.state_concat_order is not None, f"{self.state_concat_order=}"
                assert all(
                    item in state_keys for item in self.state_concat_order
                ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
                # Check the state dims
                for key in self.state_concat_order:
                    target_shapes = [self.state_dims[key]]
                    if self.is_rotation_key(key):
                        target_shapes.append(6)  # Allow for rotation_6d
                    # if key in ["state.right_arm", "state.right_hand"]:
                    target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                    assert (
                        data[key].shape[-1] in target_shapes
                    ), f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
                # Concatenate the state keys
                # We'll have StateActionToTensor before this transform, so here we use torch.cat
                data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)  # [T, D_state]

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            if self.action_canonical_slots is not None:
                # Canonical scatter path.
                data = self._scatter_canonical(
                    data=data,
                    keys=action_keys,
                    canonical_slots=self.action_canonical_slots,
                    canonical_slot_indices=self.action_canonical_slot_indices,
                    eef_rotation_source=self.action_eef_rotation_source,
                    out_dim=get_canonical_action_dim(),
                    out_key="action",
                    mask_key="action_mask",
                )
            else:
                assert self.action_concat_order is not None, f"{self.action_concat_order=}"
                # Check if all keys in concat_order are present
                assert set(self.action_concat_order) == set(
                    action_keys
                ), f"{set(self.action_concat_order)=}, {set(action_keys)=}"
                # Record the action dims
                for key in self.action_concat_order:
                    target_shapes = [self.action_dims[key]]
                    if self.is_rotation_key(key):
                        target_shapes.append(3)  # Allow for axis angle
                    assert (
                        self.action_dims[key] == data[key].shape[-1]
                    ), f"Action dim mismatch for {key=}, {self.action_dims[key]=}, {data[key].shape[-1]=}"
                # Concatenate the action keys
                # We'll have StateActionToTensor before this transform, so here we use torch.cat
                data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)  # [T, D_action]

        return data

    # ------------------------------------------------------------------ helpers
    def _scatter_canonical(
        self,
        *,
        data: dict,
        keys: list[str],
        canonical_slots: dict[str, str],
        canonical_slot_indices: dict[str, list[int]],
        eef_rotation_source: dict[str, str],
        out_dim: int,
        out_key: str,
        mask_key: str,
    ) -> dict:
        """Scatter per-key tensors into a fixed-size canonical vector.

        Steps for each key in ``canonical_slots``:
          1. EE-pose keys whose source rotation format is registered in
             ``eef_rotation_source`` are converted to xyz(3) + rotvec(3) first.
          2. The resulting tensor is written into its canonical slot range,
             left-aligned. Trailing slot dims stay zero, mask stays False.

        Keys that appear in ``data`` under this modality but are NOT in
        ``canonical_slots`` are dropped from the output (so the model never
        sees ambiguous extra dims).
        """
        first_key = next(iter(canonical_slots.keys()))
        ref = data[first_key]
        horizon = ref.shape[0]
        if isinstance(ref, torch.Tensor):
            device = ref.device
            canonical = torch.zeros((horizon, out_dim), dtype=ref.dtype, device=device)
            mask = torch.zeros((horizon, out_dim), dtype=torch.bool, device=device)
            to_torch = True
        else:
            canonical = np.zeros((horizon, out_dim), dtype=np.float32)
            mask = np.zeros((horizon, out_dim), dtype=bool)
            to_torch = False

        for key, slot_name in canonical_slots.items():
            assert key in data, f"canonical key {key!r} missing from data (have {keys})"
            slot = _resolve_slot(slot_name)
            arr = data.pop(key)

            # Rotation alignment for EE pose keys (works on numpy; torch tensor
            # is detached/cpu'd then restored).
            if key in eef_rotation_source:
                np_arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
                np_arr = xyz_plus_rotation_to_xyz_rotvec(np_arr, eef_rotation_source[key])
                if to_torch:
                    arr = torch.from_numpy(np_arr).to(device=device, dtype=ref.dtype)
                else:
                    arr = np_arr

            d_src = arr.shape[-1]
            slot_indices = canonical_slot_indices.get(key)
            if slot_indices is not None:
                if len(slot_indices) != d_src:
                    raise ValueError(
                        f"key {key!r} has {d_src} dims but canonical_slot_indices has " f"{len(slot_indices)} entries"
                    )
                if len(set(slot_indices)) != len(slot_indices):
                    raise ValueError(f"key {key!r} canonical_slot_indices contains duplicates: {slot_indices}")
                if any(index < 0 or index >= slot.size for index in slot_indices):
                    raise ValueError(
                        f"key {key!r} canonical_slot_indices {slot_indices} exceed "
                        f"slot {slot.spec.name} capacity {slot.size}"
                    )
                destination = [slot.start + index for index in slot_indices]
                canonical[..., destination] = arr
                mask[..., destination] = True
                continue
            if d_src > slot.size:
                raise ValueError(f"key {key!r} dim {d_src} exceeds canonical slot {slot.spec.name} capacity {slot.size}")
            canonical[..., slot.start : slot.start + d_src] = arr
            mask[..., slot.start : slot.start + d_src] = True

        # Drop any keys for this modality that were not mapped to canonical
        # slots, to avoid leaking unexpected fields downstream.
        for stray in list(keys):
            if stray in data and stray not in canonical_slots:
                # Only drop fields whose dotted prefix matches the modality
                # we are scattering (the original ``keys`` list).
                data.pop(stray)

        data[out_key] = canonical
        data[mask_key] = mask
        return data

    def unapply(self, data: dict) -> dict:
        assert "action" in data, f"{data.keys()=}"
        # ----- Action -----
        action_tensor = data.pop("action")
        if self.action_canonical_slots is not None:
            # Canonical mode: pull each key out of its slot, optionally undo
            # the EE-rotation alignment so the embodiment-native format comes
            # back (useful for inference / sanity-check).
            data.pop("action_mask", None)
            for key, slot_name in self.action_canonical_slots.items():
                slot = _resolve_slot(slot_name)
                d = self.action_dims.get(key, slot.size)
                slot_indices = self.action_canonical_slot_indices.get(key)
                if slot_indices is not None:
                    if len(slot_indices) != d:
                        raise ValueError(
                            f"action_dims[{key}]={d} does not match canonical_slot_indices "
                            f"length {len(slot_indices)}"
                        )
                    segment = action_tensor[..., [slot.start + index for index in slot_indices]]
                elif d > slot.size:
                    raise ValueError(f"action_dims[{key}]={d} exceeds slot {slot.spec.name} ({slot.size})")
                else:
                    segment = action_tensor[..., slot.start : slot.start + d]
                if key in self.action_eef_rotation_source:
                    np_seg = segment.detach().cpu().numpy() if isinstance(segment, torch.Tensor) else np.asarray(segment)
                    np_seg = xyz_rotvec_to_xyz_plus_rotation(np_seg, self.action_eef_rotation_source[key])
                    if isinstance(segment, torch.Tensor):
                        segment = torch.from_numpy(np_seg).to(device=segment.device, dtype=segment.dtype)
                    else:
                        segment = np_seg
                data[key] = segment
        else:
            # Legacy concat mode.
            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            start_dim = 0
            for key in self.action_concat_order:
                if key not in self.action_dims:
                    raise ValueError(f"Action dim {key} not found in action_dims.")
                end_dim = start_dim + self.action_dims[key]
                data[key] = action_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        # ----- State -----
        if "state" in data:
            state_tensor = data.pop("state")
            if self.state_canonical_slots is not None:
                data.pop("state_mask", None)
                for key, slot_name in self.state_canonical_slots.items():
                    slot = _resolve_slot(slot_name)
                    d = self.state_dims.get(key, slot.size)
                    slot_indices = self.state_canonical_slot_indices.get(key)
                    if slot_indices is not None:
                        if len(slot_indices) != d:
                            raise ValueError(
                                f"state_dims[{key}]={d} does not match canonical_slot_indices "
                                f"length {len(slot_indices)}"
                            )
                        segment = state_tensor[..., [slot.start + index for index in slot_indices]]
                    elif d > slot.size:
                        raise ValueError(f"state_dims[{key}]={d} exceeds slot {slot.spec.name} ({slot.size})")
                    else:
                        segment = state_tensor[..., slot.start : slot.start + d]
                    if key in self.state_eef_rotation_source:
                        np_seg = (
                            segment.detach().cpu().numpy() if isinstance(segment, torch.Tensor) else np.asarray(segment)
                        )
                        np_seg = xyz_rotvec_to_xyz_plus_rotation(np_seg, self.state_eef_rotation_source[key])
                        if isinstance(segment, torch.Tensor):
                            segment = torch.from_numpy(np_seg).to(device=segment.device, dtype=segment.dtype)
                        else:
                            segment = np_seg
                    data[key] = segment
            else:
                assert self.state_concat_order is not None, f"{self.state_concat_order=}"
                start_dim = 0
                for key in self.state_concat_order:
                    end_dim = start_dim + self.state_dims[key]
                    data[key] = state_tensor[..., start_dim:end_dim]
                    start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys.
        # In legacy concat mode this drives ``action_concat_order`` shapes; in
        # canonical mode it is used by ``unapply`` to recover the per-key tail
        # of each slot.
        #
        # For canonical keys we additionally clamp to slot.size, because the
        # actual tensor that reaches scatter may be smaller than what the raw
        # dataset metadata says (e.g. EefPoseAlignTransform shrinks an
        # xyz+quaternion 7-D key to canonical xyz+rotvec 6-D). The clamp
        # prevents unapply() from trying to slice past the slot capacity.
        #
        # Some canonical keys are SYNTHETIC -- created by an upstream transform
        # such as EefPoseAlignTransform (e.g. ``action.left_wrist_pose``) or
        # ``_PackFingertipPositionsTransform`` (e.g. ``action.left_finger_positions``)
        # and therefore do NOT exist in the dataset's modality.json. For those
        # we cannot look up dims via dataset_metadata; fall back to slot.size.

        def _safe_lookup(key: str, slot_size: int) -> int:
            try:
                return self.get_state_action_dims(key)
            except (AssertionError, AttributeError, KeyError):
                return slot_size

        keys_action: list[str] = []
        if self.action_concat_order is not None:
            keys_action.extend(self.action_concat_order)
        if self.action_canonical_slots is not None:
            keys_action.extend(self.action_canonical_slots.keys())
        for key in keys_action:
            if key not in self.action_dims:
                slot_size_default = 0
                if self.action_canonical_slots and key in self.action_canonical_slots:
                    slot_size_default = _resolve_slot(self.action_canonical_slots[key]).size
                self.action_dims[key] = _safe_lookup(key, slot_size_default)
        if self.action_canonical_slots is not None:
            for key, slot_name in self.action_canonical_slots.items():
                slot = _resolve_slot(slot_name)
                if self.action_dims.get(key, slot.size) > slot.size:
                    self.action_dims[key] = slot.size

        keys_state: list[str] = []
        if self.state_concat_order is not None:
            keys_state.extend(self.state_concat_order)
        if self.state_canonical_slots is not None:
            keys_state.extend(self.state_canonical_slots.keys())
        for key in keys_state:
            if key not in self.state_dims:
                slot_size_default = 0
                if self.state_canonical_slots and key in self.state_canonical_slots:
                    slot_size_default = _resolve_slot(self.state_canonical_slots[key]).size
                self.state_dims[key] = _safe_lookup(key, slot_size_default)
        if self.state_canonical_slots is not None:
            for key, slot_name in self.state_canonical_slots.items():
                slot = _resolve_slot(slot_name)
                if self.state_dims.get(key, slot.size) > slot.size:
                    self.state_dims[key] = slot.size
