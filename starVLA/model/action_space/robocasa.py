# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Chubin Zhang in 2026.
# Modification: Add train-consistent RoboCasa GR1 joint and relative-EEF deployment.
"""RoboCasa GR1 preprocessing and action decoding for model inference."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.dataloader.gr00t_lerobot.canonical_meta import get_dataset_family, get_robot_family
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import embodiment_tag_to_index
from starVLA.dataloader.gr00t_lerobot.hand_joint_layout import ROBOCASA_HAND_SHARPA_INDICES
from starVLA.dataloader.gr00t_lerobot.state_action.batch_ops import recover_camera_spatial_xyz_rotvec_actions
from starVLA.model.action_space import register_action_space
from starVLA.model.action_space.normalization import (
    normalize_with_statistics,
    unnormalize_with_statistics,
)


def _configure_hand_slot_layout(adapter: Any) -> None:
    """Use sparse Sharpa slots, or recover the old layout from both exported masks.

    Existing post-training checkpoints learned contiguous six-value hand blocks.
    Do not reinterpret those weights/statistics as the new finger-aligned layout.
    Malformed or mixed layouts are still rejected by the regular validators.
    """
    legacy_left = np.arange(80, 86)
    legacy_right = np.arange(105, 111)
    non_hand_slots = adapter._ACTIVE_SLOTS[adapter._ACTIVE_SLOTS < 80]
    legacy_mask = np.zeros(130, dtype=bool)
    legacy_mask[np.concatenate((non_hand_slots, legacy_left, legacy_right))] = True
    legacy = all(
        np.array_equal(np.asarray(stats.get("mask"), dtype=bool), legacy_mask)
        for stats in (adapter.action_stats, adapter.state_stats)
    )
    if legacy:
        adapter._LEFT_HAND_SLOTS = legacy_left
        adapter._RIGHT_HAND_SLOTS = legacy_right
        adapter._ACTIVE_SLOTS = np.concatenate((non_hand_slots, legacy_left, legacy_right))
        adapter._EXPECTED_SLOT_NAMES = {
            **{index: name for index, name in adapter._EXPECTED_SLOT_NAMES.items() if index < 80},
            **{int(index): "left_dex_joint" for index in legacy_left},
            **{int(index): "right_dex_joint" for index in legacy_right},
        }
    adapter.action_space_metadata["hand_slot_layout"] = "legacy_contiguous" if legacy else "egodex_sharpa_v1"


def _xyz_quat_to_rotvec(pose: np.ndarray) -> np.ndarray:
    """Convert xyz + quaternion_xyzw poses to xyz + rotation-vector poses."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 7:
        raise ValueError(f"EEF pose must be xyz+quaternion_xyzw with width 7, got {pose.shape}")
    rotations = Rotation.from_quat(pose[..., 3:7].reshape(-1, 4)).as_rotvec().reshape(*pose.shape[:-1], 3)
    return np.concatenate((pose[..., :3], rotations), axis=-1).astype(np.float32, copy=False)


def _xyz_rotvec_to_quat(pose: np.ndarray) -> np.ndarray:
    """Convert xyz + rotation-vector poses to xyz + quaternion_xyzw poses."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 6:
        raise ValueError(f"EEF pose must be xyz+rotvec with width 6, got {pose.shape}")
    quaternions = Rotation.from_rotvec(pose[..., 3:6].reshape(-1, 3)).as_quat().reshape(*pose.shape[:-1], 4)
    return np.concatenate((pose[..., :3], quaternions), axis=-1).astype(np.float32, copy=False)


class RobocasaGr1AbsoluteJointPolicy:
    """Adapt compact simulator joint state/actions to the 130-D canonical model space."""

    _LEFT_ARM_SLOTS: ClassVar[np.ndarray] = np.arange(0, 7)
    _RIGHT_ARM_SLOTS: ClassVar[np.ndarray] = np.arange(29, 36)
    _WAIST_SLOTS: ClassVar[np.ndarray] = np.arange(68, 71)
    _LEFT_HAND_SLOTS: ClassVar[np.ndarray] = 80 + np.asarray(ROBOCASA_HAND_SHARPA_INDICES)
    _RIGHT_HAND_SLOTS: ClassVar[np.ndarray] = 105 + np.asarray(ROBOCASA_HAND_SHARPA_INDICES)
    _ACTIVE_SLOTS: ClassVar[np.ndarray] = np.concatenate(
        (_LEFT_ARM_SLOTS, _RIGHT_ARM_SLOTS, _LEFT_HAND_SLOTS, _RIGHT_HAND_SLOTS, _WAIST_SLOTS)
    )
    _EXPECTED_SLOT_NAMES: ClassVar[dict[int, str]] = {
        **{int(index): "left_arm_joint" for index in _LEFT_ARM_SLOTS},
        **{int(index): "right_arm_joint" for index in _RIGHT_ARM_SLOTS},
        **{int(index): "waist_pose" for index in _WAIST_SLOTS},
        **{int(index): "left_dex_joint" for index in _LEFT_HAND_SLOTS},
        **{int(index): "right_dex_joint" for index in _RIGHT_HAND_SLOTS},
    }
    _DEFAULT_META: ClassVar[dict[str, Any]] = {
        "embodiment_tag": embodiment_tag_to_index("gr1"),
        "end_effector_type": "hand",
        "dataset_id": "gr1",
        "dataset_family": "unknown",
        "robot_family": "unknown",
        "robot_type": "fourier_gr1_arms_waist_absolute",
        "control_mode": "joint",
    }

    def __init__(
        self,
        policy: Any,
        *,
        statistics_key: str = "gr1",
        action_space_spec: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.action_space_spec = dict(action_space_spec or {})
        self.action_space_spec.setdefault("name", "robocasa_gr1_joint")
        self.action_space_spec.setdefault("representation", "absolute")
        self.action_space_spec.setdefault("statistics_key", statistics_key)
        self.action_space_metadata = dict(self.action_space_spec)

        norm_stats = getattr(policy, "norm_stats", {}) or {}
        if statistics_key not in norm_stats:
            raise KeyError(f"Missing {statistics_key!r} model normalization statistics; available={list(norm_stats)}")
        statistics = norm_stats[statistics_key]
        self.action_stats = statistics["action"]
        self.state_stats = statistics["state"]
        _configure_hand_slot_layout(self)
        self._validate_statistics()

        self.action_mask = self._validated_exported_mask(self.action_stats, "action")
        self.state_mask = self._validated_exported_mask(self.state_stats, "state")
        action_codec = getattr(policy, "action_codec", None)
        chunk_size = getattr(action_codec, "chunk_size", None)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"Invalid policy action chunk size: {chunk_size!r}")
        self.action_schema_mask = np.broadcast_to(self.action_mask, (chunk_size, self.action_mask.size)).copy()

        # Resolve modes at construction so malformed checkpoint exports fail
        # before the simulator starts an episode.
        _, self.state_normalization_source = normalize_with_statistics(
            np.zeros(self.state_mask.size, dtype=np.float32),
            self.state_stats,
            modality="state",
            unnorm_key=statistics_key,
        )
        _, self.action_normalization_source = unnormalize_with_statistics(
            np.zeros(self.action_mask.size, dtype=np.float32),
            self.action_stats,
            modality="action",
            unnorm_key=statistics_key,
            clip=True,
        )

    def _validate_statistics(self) -> None:
        expected_width = 130
        for name, stats in (("action", self.action_stats), ("state", self.state_stats)):
            slot_names = list(stats.get("slot_names", []))
            if len(slot_names) != expected_width:
                raise ValueError(
                    f"RoboCasa GR1 {name} statistics have {len(slot_names)} slots; expected {expected_width}"
                )
            for index, expected_name in self._EXPECTED_SLOT_NAMES.items():
                if slot_names[index] != expected_name:
                    raise ValueError(
                        f"Unexpected RoboCasa GR1 {name} slot {index}: {slot_names[index]!r}; "
                        f"expected {expected_name!r}"
                    )

    def _validated_exported_mask(self, stats: dict[str, Any], modality: str) -> np.ndarray:
        exported_mask = np.asarray(stats.get("mask"), dtype=bool)
        if exported_mask.shape != (130,):
            raise ValueError(f"RoboCasa GR1 {modality} mask has shape {exported_mask.shape}; expected (130,)")
        expected_mask = np.zeros(130, dtype=bool)
        expected_mask[self._ACTIVE_SLOTS] = True
        if not np.array_equal(exported_mask, expected_mask):
            differing = np.flatnonzero(exported_mask != expected_mask).tolist()
            raise ValueError(
                f"RoboCasa GR1 {modality} mask disagrees with the physical joint layout at slots {differing}"
            )
        return exported_mask

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        prepared_payload = self._prepare_payload(payload)
        output = self.policy.predict_action(**prepared_payload)
        if not isinstance(output, dict) or "normalized_actions" not in output:
            raise RuntimeError("Wrapped policy did not return normalized_actions")
        output = dict(output)
        output["absolute_actions"] = self._decode_compact_actions(
            np.asarray(output["normalized_actions"], dtype=np.float32)
        )
        return output

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("RoboCasa action-space adapter expects a non-empty examples list")
        prepared_examples = []
        for example in examples:
            prepared_example = copy.copy(example)
            if "robocasa_joint_state" not in prepared_example:
                raise ValueError("Missing robocasa_joint_state in RoboCasa inference example")
            compact_state = np.asarray(prepared_example.pop("robocasa_joint_state"), dtype=np.float32).reshape(-1)
            if compact_state.shape != (29,):
                raise ValueError(f"robocasa_joint_state must contain 29 values, got shape {compact_state.shape}")
            prepared_example["state"] = self._normalize_canonical_state(compact_state)[None, :]
            prepared_example["state_mask"] = self.state_mask[None, :].copy()
            prepared_example["action_schema_mask"] = self.action_schema_mask.copy()
            meta = dict(self._DEFAULT_META)
            meta.update(prepared_example.get("meta", {}) or {})
            prepared_example["meta"] = meta
            prepared_examples.append(prepared_example)
        prepared_payload = dict(payload)
        prepared_payload["examples"] = prepared_examples
        return prepared_payload

    def _scatter_compact(self, compact: np.ndarray) -> np.ndarray:
        canonical = np.zeros((*compact.shape[:-1], 130), dtype=np.float32)
        canonical[..., self._LEFT_ARM_SLOTS] = compact[..., 0:7]
        canonical[..., self._RIGHT_ARM_SLOTS] = compact[..., 7:14]
        canonical[..., self._LEFT_HAND_SLOTS] = compact[..., 14:20]
        canonical[..., self._RIGHT_HAND_SLOTS] = compact[..., 20:26]
        canonical[..., self._WAIST_SLOTS] = compact[..., 26:29]
        return canonical

    def _normalize_canonical_state(self, compact_state: np.ndarray) -> np.ndarray:
        canonical_state = self._scatter_compact(compact_state)
        normalized_state, _ = normalize_with_statistics(
            canonical_state,
            self.state_stats,
            modality="state",
            unnorm_key=str(self.action_space_spec["statistics_key"]),
        )
        normalized_state[~self.state_mask] = 0.0
        return normalized_state

    def _decode_compact_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :, :]
        if normalized_actions.ndim != 3 or normalized_actions.shape[-1] != 130:
            raise ValueError(f"normalized_actions must be [B,T,130], got {normalized_actions.shape}")
        canonical_actions, _ = unnormalize_with_statistics(
            normalized_actions,
            self.action_stats,
            modality="action",
            unnorm_key=str(self.action_space_spec["statistics_key"]),
            clip=True,
        )
        return np.concatenate(
            (
                canonical_actions[..., self._LEFT_ARM_SLOTS],
                canonical_actions[..., self._RIGHT_ARM_SLOTS],
                canonical_actions[..., self._LEFT_HAND_SLOTS],
                canonical_actions[..., self._RIGHT_HAND_SLOTS],
                canonical_actions[..., self._WAIST_SLOTS],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)


class RobocasaGr1AugAlignedPolicy(RobocasaGr1AbsoluteJointPolicy):
    """Consume joint-only state and decode either joint or mixed EEF/joint control.

    Auxiliary spatial EEF deltas and absolute fingertips remain available in
    normalized_actions. The default ``absolute_actions`` output preserves the
    joint-only deployment path. When the request also supplies fixed-pseudo-frame
    EEF references, ``eef_joint_actions`` contains two absolute EEF poses
    followed by joint hands and joint waist. This dataset was trained with a
    static camera-spatial basis, so the deltas use the same pseudo-frame axes.
    """

    _AUXILIARY_SLOTS: ClassVar[np.ndarray] = np.r_[7:13, 14:29, 36:42, 43:58]
    _AUXILIARY_SLOT_NAMES: ClassVar[dict[int, str]] = {
        **dict.fromkeys(range(7, 13), "left_eef_pose"),
        **dict.fromkeys(range(14, 29), "left_dex_hand"),
        **dict.fromkeys(range(36, 42), "right_eef_pose"),
        **dict.fromkeys(range(43, 58), "right_dex_hand"),
    }
    _DEFAULT_META: ClassVar[dict[str, Any]] = {
        "embodiment_tag": embodiment_tag_to_index("robocasa_gr1_aug_aligned"),
        "end_effector_type": "hand",
        "dataset_id": "robocasa_gr1_aug_aligned",
        "dataset_family": get_dataset_family("robocasa_gr1_aug_aligned"),
        "robot_family": get_robot_family("robocasa_gr1_aug_aligned"),
        "robot_type": "robocasa_gr1_aug_aligned",
        "control_mode": "ee",
        "action_representation": "relative",
    }

    def __init__(
        self,
        policy: Any,
        *,
        statistics_key: str = "robocasa_gr1_aug_aligned",
        action_space_spec: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(policy, statistics_key=statistics_key, action_space_spec=action_space_spec)
        self.action_space_metadata["execution_modes"] = ["absolute_joint", "aligned_eef_joint"]

    def _validate_statistics(self) -> None:
        super()._validate_statistics()
        for index, expected_name in self._AUXILIARY_SLOT_NAMES.items():
            if self.action_stats["slot_names"][index] != expected_name:
                raise ValueError(f"Unexpected aligned RoboCasa action slot {index}; expected {expected_name!r}")

    def _validated_exported_mask(self, stats: dict[str, Any], modality: str) -> np.ndarray:
        expected = np.zeros(130, dtype=bool)
        expected[self._ACTIVE_SLOTS] = True
        if modality == "action":
            expected[self._AUXILIARY_SLOTS] = True
        exported = np.asarray(stats.get("mask"), dtype=bool)
        if not np.array_equal(exported, expected):
            raise ValueError(f"Aligned RoboCasa {modality} mask must expose exactly {int(expected.sum())} slots")
        return exported

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        """Run the joint-conditioned model and optionally recover absolute EEF targets."""
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("Aligned RoboCasa action-space adapter expects a non-empty examples list")

        cleaned_examples: list[dict[str, Any]] = []
        reference_poses: list[np.ndarray] = []
        mixed_requests: list[bool] = []
        for example in examples:
            cleaned = copy.copy(example)
            has_pose = "robocasa_eef_reference" in cleaned
            mixed_requests.append(has_pose)
            if has_pose:
                reference = np.asarray(cleaned.pop("robocasa_eef_reference"), dtype=np.float32).reshape(-1)
                if reference.shape != (14,) or not np.all(np.isfinite(reference)):
                    raise ValueError(
                        "robocasa_eef_reference must contain two finite xyz+quaternion poses "
                        f"(14 values), got shape {reference.shape}"
                    )
                reference_poses.append(reference)
            cleaned_examples.append(cleaned)
        if any(mixed_requests) and not all(mixed_requests):
            raise ValueError("A batched RoboCasa request cannot mix joint-only and aligned EEF execution")

        prepared_payload = dict(payload)
        prepared_payload["examples"] = cleaned_examples
        output = self.policy.predict_action(**self._prepare_payload(prepared_payload))
        if not isinstance(output, dict) or "normalized_actions" not in output:
            raise RuntimeError("Wrapped policy did not return normalized_actions")
        output = dict(output)
        normalized_actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        output["absolute_actions"] = self._decode_compact_actions(normalized_actions)
        if all(mixed_requests):
            output["eef_joint_actions"] = self._decode_eef_joint_actions(
                normalized_actions,
                np.stack(reference_poses, axis=0),
            )
        return output

    def _decode_eef_joint_actions(
        self,
        normalized_actions: np.ndarray,
        reference_poses: np.ndarray,
    ) -> np.ndarray:
        """Recover pseudo-frame EEF arms while retaining absolute joint hand/waist slots."""
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :, :]
        if normalized_actions.ndim != 3 or normalized_actions.shape[-1] != 130:
            raise ValueError(f"normalized_actions must be [B,T,130], got {normalized_actions.shape}")
        batch_size = normalized_actions.shape[0]
        if reference_poses.shape != (batch_size, 14):
            raise ValueError(f"EEF reference poses must be [B,14] for B={batch_size}, got {reference_poses.shape}")
        canonical_actions, _ = unnormalize_with_statistics(
            normalized_actions,
            self.action_stats,
            modality="action",
            unnorm_key=str(self.action_space_spec["statistics_key"]),
            clip=True,
        )
        identity = np.eye(4, dtype=np.float64)
        recovered_arms: list[tuple[np.ndarray, np.ndarray]] = []
        for batch_index in range(batch_size):
            left_absolute = recover_camera_spatial_xyz_rotvec_actions(
                canonical_actions[batch_index][:, 7:13],
                _xyz_quat_to_rotvec(reference_poses[batch_index, 0:7]),
                world_from_reference_camera=identity,
                world_from_reference_frame=identity,
            )
            right_absolute = recover_camera_spatial_xyz_rotvec_actions(
                canonical_actions[batch_index][:, 36:42],
                _xyz_quat_to_rotvec(reference_poses[batch_index, 7:14]),
                world_from_reference_camera=identity,
                world_from_reference_frame=identity,
            )
            recovered_arms.append((left_absolute, right_absolute))

        left = np.stack([_xyz_rotvec_to_quat(item[0]) for item in recovered_arms], axis=0)
        right = np.stack([_xyz_rotvec_to_quat(item[1]) for item in recovered_arms], axis=0)
        return np.concatenate(
            (
                left,
                right,
                canonical_actions[..., self._LEFT_HAND_SLOTS],
                canonical_actions[..., self._RIGHT_HAND_SLOTS],
                canonical_actions[..., self._WAIST_SLOTS],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)


class RobocasaGr1RelativeEefPolicy:
    """Adapt RoboCasa 7-D EEF poses to canonical camera-spatial relative actions.

    The simulator stores each arm as ``xyz + quaternion_xyzw`` while training
    uses canonical ``xyz + rotvec`` slots. Relative rotation labels are spatial
    deltas in the current (static) head-camera frame, so absolute recovery is
    ``R_abs = R_delta @ R_ref`` rather than rotvec addition or body-frame
    ``R_ref @ R_delta`` composition.
    """

    _LEFT_EEF_SLOTS: ClassVar[np.ndarray] = np.arange(7, 13)
    _RIGHT_EEF_SLOTS: ClassVar[np.ndarray] = np.arange(36, 42)
    _WAIST_SLOTS: ClassVar[np.ndarray] = np.arange(68, 71)
    _LEFT_HAND_SLOTS: ClassVar[np.ndarray] = 80 + np.asarray(ROBOCASA_HAND_SHARPA_INDICES)
    _RIGHT_HAND_SLOTS: ClassVar[np.ndarray] = 105 + np.asarray(ROBOCASA_HAND_SHARPA_INDICES)
    _ACTIVE_SLOTS: ClassVar[np.ndarray] = np.concatenate(
        (_LEFT_EEF_SLOTS, _RIGHT_EEF_SLOTS, _LEFT_HAND_SLOTS, _RIGHT_HAND_SLOTS, _WAIST_SLOTS)
    )
    _EXPECTED_SLOT_NAMES: ClassVar[dict[int, str]] = {
        **{int(index): "left_eef_pose" for index in _LEFT_EEF_SLOTS},
        **{int(index): "right_eef_pose" for index in _RIGHT_EEF_SLOTS},
        **{int(index): "waist_pose" for index in _WAIST_SLOTS},
        **{int(index): "left_dex_joint" for index in _LEFT_HAND_SLOTS},
        **{int(index): "right_dex_joint" for index in _RIGHT_HAND_SLOTS},
    }
    _DEFAULT_META: ClassVar[dict[str, Any]] = {
        "embodiment_tag": embodiment_tag_to_index("robocasa_gr1_eef"),
        "end_effector_type": "unknown",
        "dataset_id": "robocasa_gr1_eef",
        "dataset_family": "robocasa",
        "robot_family": "fourier_gr1",
        "robot_type": "robocasa_gr1_eef",
        "control_mode": "ee",
        "action_representation": "relative",
    }

    def __init__(
        self,
        policy: Any,
        *,
        statistics_key: str = "robocasa_gr1_eef",
        action_space_spec: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.action_space_spec = dict(action_space_spec or {})
        self.action_space_spec.setdefault("name", "robocasa_gr1_eef")
        self.action_space_spec.setdefault("representation", "relative")
        self.action_space_spec.setdefault("statistics_key", statistics_key)
        self.action_space_spec.setdefault("relative_eef_frame", "camera_spatial")
        self.action_space_metadata = dict(self.action_space_spec)

        if self.action_space_spec["relative_eef_frame"] != "camera_spatial":
            raise ValueError(
                "RoboCasa GR1 EEF deployment requires relative_eef_frame='camera_spatial', got "
                f"{self.action_space_spec['relative_eef_frame']!r}"
            )
        norm_stats = getattr(policy, "norm_stats", {}) or {}
        if statistics_key not in norm_stats:
            raise KeyError(f"Missing {statistics_key!r} model normalization statistics; available={list(norm_stats)}")
        statistics = norm_stats[statistics_key]
        self.action_stats = statistics["action"]
        self.state_stats = statistics["state"]
        _configure_hand_slot_layout(self)
        self._validate_statistics()
        self.action_mask = self._validated_exported_mask(self.action_stats, "action")
        self.state_mask = self._validated_exported_mask(self.state_stats, "state")

        action_codec = getattr(policy, "action_codec", None)
        chunk_size = getattr(action_codec, "chunk_size", None)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"Invalid policy action chunk size: {chunk_size!r}")
        self.action_schema_mask = np.broadcast_to(self.action_mask, (chunk_size, 130)).copy()

        _, self.state_normalization_source = normalize_with_statistics(
            np.zeros(130, dtype=np.float32),
            self.state_stats,
            modality="state",
            unnorm_key=statistics_key,
        )
        _, self.action_normalization_source = unnormalize_with_statistics(
            np.zeros(130, dtype=np.float32),
            self.action_stats,
            modality="action",
            unnorm_key=statistics_key,
            clip=True,
        )

    def _validate_statistics(self) -> None:
        for name, stats in (("action", self.action_stats), ("state", self.state_stats)):
            slot_names = list(stats.get("slot_names", []))
            if len(slot_names) != 130:
                raise ValueError(f"RoboCasa GR1 EEF {name} statistics have {len(slot_names)} slots; expected 130")
            for index, expected_name in self._EXPECTED_SLOT_NAMES.items():
                if slot_names[index] != expected_name:
                    raise ValueError(
                        f"Unexpected RoboCasa GR1 EEF {name} slot {index}: {slot_names[index]!r}; "
                        f"expected {expected_name!r}"
                    )

    def _validated_exported_mask(self, stats: dict[str, Any], modality: str) -> np.ndarray:
        exported_mask = np.asarray(stats.get("mask"), dtype=bool)
        expected_mask = np.zeros(130, dtype=bool)
        expected_mask[self._ACTIVE_SLOTS] = True
        if exported_mask.shape != (130,):
            raise ValueError(f"RoboCasa GR1 EEF {modality} mask has shape {exported_mask.shape}; expected (130,)")
        if not np.array_equal(exported_mask, expected_mask):
            differing = np.flatnonzero(exported_mask != expected_mask).tolist()
            raise ValueError(f"RoboCasa GR1 EEF {modality} mask disagrees with canonical slots {differing}")
        return exported_mask

    @staticmethod
    def _xyz_quat_to_rotvec(pose: np.ndarray) -> np.ndarray:
        return _xyz_quat_to_rotvec(pose)

    @staticmethod
    def _xyz_rotvec_to_quat(pose: np.ndarray) -> np.ndarray:
        return _xyz_rotvec_to_quat(pose)

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        prepared_payload, reference_poses = self._prepare_payload(payload)
        output = self.policy.predict_action(**prepared_payload)
        if not isinstance(output, dict) or "normalized_actions" not in output:
            raise RuntimeError("Wrapped policy did not return normalized_actions")
        output = dict(output)
        output["absolute_actions"] = self._decode_absolute_actions(
            np.asarray(output["normalized_actions"], dtype=np.float32), reference_poses
        )
        return output

    def _prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("RoboCasa GR1 EEF adapter expects a non-empty examples list")
        prepared_examples: list[dict[str, Any]] = []
        reference_poses: list[np.ndarray] = []
        for example in examples:
            prepared = copy.copy(example)
            if "robocasa_eef_state" not in prepared:
                raise ValueError("Missing robocasa_eef_state in RoboCasa EEF inference example")
            compact = np.asarray(prepared.pop("robocasa_eef_state"), dtype=np.float32).reshape(-1)
            if compact.shape != (29,):
                raise ValueError(f"robocasa_eef_state must contain 29 values, got shape {compact.shape}")
            canonical = self._scatter_compact_state(compact)
            normalized, _ = normalize_with_statistics(
                canonical,
                self.state_stats,
                modality="state",
                unnorm_key=str(self.action_space_spec["statistics_key"]),
            )
            normalized[~self.state_mask] = 0.0
            prepared["state"] = normalized[None, :]
            prepared["state_mask"] = self.state_mask[None, :].copy()
            prepared["action_schema_mask"] = self.action_schema_mask.copy()
            meta = dict(self._DEFAULT_META)
            meta.update(prepared.get("meta", {}) or {})
            prepared["meta"] = meta
            prepared_examples.append(prepared)
            reference_poses.append(compact[:14])
        prepared_payload = dict(payload)
        prepared_payload["examples"] = prepared_examples
        return prepared_payload, np.stack(reference_poses, axis=0)

    def _scatter_compact_state(self, compact: np.ndarray) -> np.ndarray:
        canonical = np.zeros(130, dtype=np.float32)
        canonical[self._LEFT_EEF_SLOTS] = self._xyz_quat_to_rotvec(compact[0:7])
        canonical[self._RIGHT_EEF_SLOTS] = self._xyz_quat_to_rotvec(compact[7:14])
        canonical[self._LEFT_HAND_SLOTS] = compact[14:20]
        canonical[self._RIGHT_HAND_SLOTS] = compact[20:26]
        canonical[self._WAIST_SLOTS] = compact[26:29]
        return canonical

    def _decode_absolute_actions(self, normalized_actions: np.ndarray, reference_poses: np.ndarray) -> np.ndarray:
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :, :]
        if normalized_actions.ndim != 3 or normalized_actions.shape[-1] != 130:
            raise ValueError(f"normalized_actions must be [B,T,130], got {normalized_actions.shape}")
        if normalized_actions.shape[0] != reference_poses.shape[0]:
            raise ValueError("Model batch size does not match RoboCasa EEF reference-pose batch size")
        relative_actions, _ = unnormalize_with_statistics(
            normalized_actions,
            self.action_stats,
            modality="action",
            unnorm_key=str(self.action_space_spec["statistics_key"]),
            clip=True,
        )
        identity = np.eye(4, dtype=np.float64)
        absolute_arms: list[tuple[np.ndarray, np.ndarray]] = []
        for batch_index, compact_reference in enumerate(reference_poses):
            left_reference = self._xyz_quat_to_rotvec(compact_reference[0:7])
            right_reference = self._xyz_quat_to_rotvec(compact_reference[7:14])
            left_absolute = recover_camera_spatial_xyz_rotvec_actions(
                relative_actions[batch_index][:, self._LEFT_EEF_SLOTS],
                left_reference,
                world_from_reference_camera=identity,
                world_from_reference_frame=identity,
            )
            right_absolute = recover_camera_spatial_xyz_rotvec_actions(
                relative_actions[batch_index][:, self._RIGHT_EEF_SLOTS],
                right_reference,
                world_from_reference_camera=identity,
                world_from_reference_frame=identity,
            )
            absolute_arms.append((left_absolute, right_absolute))

        left = np.stack([self._xyz_rotvec_to_quat(item[0]) for item in absolute_arms], axis=0)
        right = np.stack([self._xyz_rotvec_to_quat(item[1]) for item in absolute_arms], axis=0)
        return np.concatenate(
            (
                left,
                right,
                relative_actions[..., self._LEFT_HAND_SLOTS],
                relative_actions[..., self._RIGHT_HAND_SLOTS],
                relative_actions[..., self._WAIST_SLOTS],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)


@register_action_space("robocasa_gr1_joint")
def build_robocasa_gr1_joint_policy(policy: Any, spec: dict[str, Any]) -> RobocasaGr1AbsoluteJointPolicy:
    representation = str(spec.get("representation", "")).lower()
    if representation != "absolute":
        raise ValueError(f"RoboCasa GR1 joint action representation must be 'absolute', got {representation!r}")
    statistics_key = str(spec.get("statistics_key", "gr1"))
    return RobocasaGr1AbsoluteJointPolicy(
        policy,
        statistics_key=statistics_key,
        action_space_spec=spec,
    )


@register_action_space("robocasa_gr1_eef")
def build_robocasa_gr1_eef_policy(policy: Any, spec: dict[str, Any]) -> RobocasaGr1RelativeEefPolicy:
    representation = str(spec.get("representation", "")).lower()
    if representation != "relative":
        raise ValueError(f"RoboCasa GR1 EEF action representation must be 'relative', got {representation!r}")
    statistics_key = str(spec.get("statistics_key", "robocasa_gr1_eef"))
    return RobocasaGr1RelativeEefPolicy(policy, statistics_key=statistics_key, action_space_spec=spec)


@register_action_space("robocasa_gr1_aug_aligned")
def build_robocasa_gr1_aug_aligned_policy(policy: Any, spec: dict[str, Any]) -> RobocasaGr1AugAlignedPolicy:
    """Build an aligned adapter supporting joint-only and mixed EEF/joint execution."""
    from starVLA.dataloader.gr00t_lerobot.state_action.robocasa_frames import FRAME_CONVENTION

    required = {
        "representation": "relative",
        "joint_representation": "absolute",
        "fingertip_representation": "absolute",
        "relative_eef_frame": "camera_spatial",
        "static_camera_frame": True,
        "frame_convention": FRAME_CONVENTION,
    }
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise ValueError(f"Aligned RoboCasa requires {key}={expected!r}, got {spec.get(key)!r}")
    return RobocasaGr1AugAlignedPolicy(
        policy, statistics_key=str(spec.get("statistics_key", "robocasa_gr1_aug_aligned")), action_space_spec=spec
    )


__all__ = [
    "RobocasaGr1AbsoluteJointPolicy",
    "RobocasaGr1AugAlignedPolicy",
    "RobocasaGr1RelativeEefPolicy",
    "build_robocasa_gr1_eef_policy",
    "build_robocasa_gr1_joint_policy",
]
