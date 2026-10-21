# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add model-owned RoboTwin EE action-space inference adapters.
"""RoboTwin action-space preprocessing and decoding for model inference."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.model.action_space import register_action_space
from starVLA.model.action_space.normalization import (
    normalize_with_statistics,
    unnormalize_with_statistics,
)


class RobotwinRelativeActionPolicy:
    """Return compact absolute RoboTwin targets from relative model actions."""

    _LEFT_EE_SLOTS = np.arange(7, 13)
    _LEFT_GRIPPER_SLOT = 13
    _RIGHT_EE_SLOTS = np.arange(36, 42)
    _RIGHT_GRIPPER_SLOT = 42

    def __init__(
        self,
        policy: Any,
        *,
        statistics_key: str = "robotwin",
        action_space_spec: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.action_space_spec = dict(action_space_spec or {})
        self.action_space_spec.setdefault("name", "robotwin_ee")
        self.action_space_spec.setdefault("representation", "relative")
        self.action_space_spec.setdefault("statistics_key", statistics_key)
        self.action_space_metadata = dict(self.action_space_spec)

        norm_stats = getattr(policy, "norm_stats", {}) or {}
        if statistics_key not in norm_stats:
            raise KeyError(f"Missing {statistics_key!r} model normalization statistics; available={list(norm_stats)}")
        statistics = norm_stats[statistics_key]
        self.action_stats = statistics["action"]
        self.state_stats = statistics["state"]
        self._validate_statistics()
        self.action_q01 = np.asarray(self.action_stats["q01"], dtype=np.float32)
        self.action_q99 = np.asarray(self.action_stats["q99"], dtype=np.float32)
        self.state_q01 = np.asarray(self.state_stats["q01"], dtype=np.float32)
        self.state_q99 = np.asarray(self.state_stats["q99"], dtype=np.float32)
        self.state_mask = self._canonical_state_mask()
        self.action_schema_mask = self._canonical_action_schema_mask()

        # Resolve both modalities at construction time so malformed exported
        # schemas fail before the first request instead of midway through a rollout.
        _, self.state_normalization_source = normalize_with_statistics(
            np.zeros_like(self.state_q01),
            self.state_stats,
            modality="state",
            default_mode="q99",
        )
        _, self.action_normalization_source = unnormalize_with_statistics(
            np.zeros_like(self.action_q01),
            self.action_stats,
            modality="action",
            default_mode="q99",
            clip=True,
        )

    def _validate_statistics(self) -> None:
        expected = {7: "left_eef_pose", 13: "left_gripper", 36: "right_eef_pose", 42: "right_gripper"}
        for name, stats in (("action", self.action_stats), ("state", self.state_stats)):
            slot_names = stats.get("slot_names", [])
            if len(slot_names) <= self._RIGHT_GRIPPER_SLOT:
                raise ValueError(f"RoboTwin {name} statistics do not have canonical action slots")
            for index, expected_name in expected.items():
                if slot_names[index] != expected_name:
                    raise ValueError(
                        f"Unexpected {name} slot {index}: {slot_names[index]!r}; expected {expected_name!r}"
                    )
            for quantile in ("q01", "q99"):
                if quantile not in stats:
                    raise KeyError(f"Missing RoboTwin {name} {quantile} statistics")
                values = np.asarray(stats[quantile])
                if values.shape != (len(slot_names),):
                    raise ValueError(
                        f"RoboTwin {name} {quantile} statistics have shape {values.shape}; "
                        f"expected {(len(slot_names),)}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"RoboTwin {name} {quantile} statistics contain non-finite values")

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        prepared_payload, reference_poses = self._prepare_payload(payload)
        output = self.policy.predict_action(**prepared_payload)
        if not isinstance(output, dict) or "normalized_actions" not in output:
            raise RuntimeError("Wrapped policy did not return normalized_actions")
        output = dict(output)
        output["absolute_actions"] = self._recover_absolute_actions(
            np.asarray(output["normalized_actions"], dtype=np.float32), reference_poses
        )
        return output

    def _prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("RoboTwin action-space adapter expects a non-empty examples list")
        prepared_examples: list[dict[str, Any]] = []
        poses: list[np.ndarray] = []
        for example in examples:
            prepared_example = copy.copy(example)
            if "robotwin_ee_pose" not in prepared_example:
                raise ValueError("Missing robotwin_ee_pose; enable server_pose_recovery in the client")
            pose = np.asarray(prepared_example.pop("robotwin_ee_pose"), dtype=np.float32)
            if pose.size != 14:
                raise ValueError(f"robotwin_ee_pose must contain 14 values, got shape {pose.shape}")
            pose = pose.reshape(14)
            prepared_example["state"] = self._build_canonical_normalized_state(pose)[None, :]
            prepared_example["state_mask"] = self.state_mask[None, :]
            if self.action_schema_mask is not None:
                # MoT2 conditions inference on the embodiment's canonical action
                # schema. Keep this server-owned alongside state normalization and
                # relative-pose recovery rather than duplicating 130-D slot layout
                # knowledge in every simulator client.
                prepared_example["action_schema_mask"] = self.action_schema_mask.copy()
            prepared_examples.append(prepared_example)
            poses.append(pose)
        prepared_payload = dict(payload)
        prepared_payload["examples"] = prepared_examples
        return prepared_payload, np.stack(poses, axis=0)

    def _canonical_state_mask(self) -> np.ndarray:
        """Reproduce the mask emitted by RoboTwin's canonical scatter."""
        mask = np.zeros_like(self.state_q01, dtype=bool)
        mask[self._LEFT_EE_SLOTS] = True
        mask[self._LEFT_GRIPPER_SLOT] = True
        mask[self._RIGHT_EE_SLOTS] = True
        mask[self._RIGHT_GRIPPER_SLOT] = True
        exported_mask = self.state_stats.get("mask")
        if exported_mask is not None:
            exported_mask = np.asarray(exported_mask, dtype=bool)
            if exported_mask.shape != mask.shape:
                raise ValueError(f"RoboTwin state mask has shape {exported_mask.shape}; expected {mask.shape}")
            if not np.array_equal(exported_mask, mask):
                differing = np.flatnonzero(exported_mask != mask).tolist()
                raise ValueError(
                    "RoboTwin state mask disagrees with the canonical EE action space " f"at slots {differing}"
                )
        return mask

    def _canonical_action_schema_mask(self) -> np.ndarray | None:
        """Build the per-timestep canonical schema required by MoT2 inference."""
        action_codec = getattr(self.policy, "action_codec", None)
        chunk_size = getattr(action_codec, "chunk_size", None)
        if chunk_size is None:
            return None
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"Invalid policy action chunk size: {chunk_size!r}")

        mask = np.zeros_like(self.action_q01, dtype=bool)
        mask[self._LEFT_EE_SLOTS] = True
        mask[self._LEFT_GRIPPER_SLOT] = True
        mask[self._RIGHT_EE_SLOTS] = True
        mask[self._RIGHT_GRIPPER_SLOT] = True
        exported_mask = self.action_stats.get("mask")
        if exported_mask is not None:
            exported_mask = np.asarray(exported_mask, dtype=bool)
            if exported_mask.shape != mask.shape:
                raise ValueError(f"RoboTwin action mask has shape {exported_mask.shape}; expected {mask.shape}")
            if not np.array_equal(exported_mask, mask):
                differing = np.flatnonzero(exported_mask != mask).tolist()
                raise ValueError(
                    "RoboTwin action mask disagrees with the canonical EE action space " f"at slots {differing}"
                )
        return np.broadcast_to(mask, (chunk_size, mask.size)).copy()

    def _build_canonical_normalized_state(self, pose: np.ndarray) -> np.ndarray:
        # Training normalizes each real modality before canonical scatter, so
        # unowned canonical slots must stay literal zero padding.
        canonical_state = np.zeros_like(self.state_q01, dtype=np.float32)
        active_slots = np.concatenate(
            [self._LEFT_EE_SLOTS, [self._LEFT_GRIPPER_SLOT], self._RIGHT_EE_SLOTS, [self._RIGHT_GRIPPER_SLOT]]
        )
        active_values = np.concatenate([pose[:6], pose[6:7], pose[7:13], pose[13:14]])
        canonical_state[active_slots] = active_values
        normalized_state, _ = normalize_with_statistics(
            canonical_state,
            self.state_stats,
            modality="state",
            default_mode="q99",
        )
        normalized_state[~self.state_mask] = 0.0
        return normalized_state

    def _recover_absolute_actions(self, normalized_actions: np.ndarray, reference_poses: np.ndarray) -> np.ndarray:
        relative_actions = self._decode_normalized_actions(normalized_actions, reference_poses)
        left_absolute = self._compose_absolute_pose(reference_poses[:, :6], relative_actions[..., self._LEFT_EE_SLOTS])
        right_absolute = self._compose_absolute_pose(
            reference_poses[:, 7:13], relative_actions[..., self._RIGHT_EE_SLOTS]
        )
        left_gripper = relative_actions[..., self._LEFT_GRIPPER_SLOT : self._LEFT_GRIPPER_SLOT + 1]
        right_gripper = relative_actions[..., self._RIGHT_GRIPPER_SLOT : self._RIGHT_GRIPPER_SLOT + 1]
        return np.concatenate([left_absolute, left_gripper, right_absolute, right_gripper], axis=-1).astype(
            np.float32, copy=False
        )

    def _decode_normalized_actions(self, normalized_actions: np.ndarray, reference_poses: np.ndarray) -> np.ndarray:
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :, :]
        if normalized_actions.ndim != 3:
            raise ValueError(f"normalized_actions must be [B, T, D], got {normalized_actions.shape}")
        if normalized_actions.shape[0] != reference_poses.shape[0]:
            raise ValueError("Model batch size does not match reference-pose batch size")
        if normalized_actions.shape[-1] != self.action_q01.size:
            raise ValueError(f"Expected {self.action_q01.size}-D action, got {normalized_actions.shape[-1]}")
        actions, _ = unnormalize_with_statistics(
            normalized_actions,
            self.action_stats,
            modality="action",
            default_mode="q99",
            clip=True,
        )
        return actions

    @staticmethod
    def _compose_absolute_pose(reference_pose: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
        batch_size, horizon, _ = relative_pose.shape
        reference = np.repeat(reference_pose[:, None, :], horizon, axis=1)
        reference_rotation = Rotation.from_rotvec(reference[..., 3:6].reshape(-1, 3))
        relative_rotation = Rotation.from_rotvec(relative_pose[..., 3:6].reshape(-1, 3))
        absolute_xyz = reference[..., :3] + reference_rotation.apply(relative_pose[..., :3].reshape(-1, 3)).reshape(
            batch_size, horizon, 3
        )
        absolute_rotvec = (reference_rotation * relative_rotation).as_rotvec().reshape(batch_size, horizon, 3)
        return np.concatenate([absolute_xyz, absolute_rotvec], axis=-1)


class RobotwinAbsoluteActionPolicy(RobotwinRelativeActionPolicy):
    """Return compact absolute RoboTwin targets from absolute model actions."""

    def _recover_absolute_actions(self, normalized_actions: np.ndarray, reference_poses: np.ndarray) -> np.ndarray:
        absolute_actions = self._decode_normalized_actions(normalized_actions, reference_poses)
        left_absolute = absolute_actions[..., self._LEFT_EE_SLOTS]
        left_gripper = absolute_actions[..., self._LEFT_GRIPPER_SLOT : self._LEFT_GRIPPER_SLOT + 1]
        right_absolute = absolute_actions[..., self._RIGHT_EE_SLOTS]
        right_gripper = absolute_actions[..., self._RIGHT_GRIPPER_SLOT : self._RIGHT_GRIPPER_SLOT + 1]
        return np.concatenate([left_absolute, left_gripper, right_absolute, right_gripper], axis=-1).astype(
            np.float32, copy=False
        )


@register_action_space("robotwin_ee")
def build_robotwin_action_policy(policy: Any, spec: dict[str, Any]) -> RobotwinRelativeActionPolicy:
    """Build the RoboTwin adapter selected by model-owned representation metadata."""
    representation = str(spec.get("representation", "")).lower()
    statistics_key = str(spec.get("statistics_key", "robotwin"))
    policy_class_by_representation = {
        "relative": RobotwinRelativeActionPolicy,
        "absolute": RobotwinAbsoluteActionPolicy,
    }
    try:
        policy_class = policy_class_by_representation[representation]
    except KeyError as error:
        raise ValueError(
            "RoboTwin action-space representation must be 'relative' or 'absolute', " f"got {representation!r}"
        ) from error
    return policy_class(policy, statistics_key=statistics_key, action_space_spec=spec)


__all__ = [
    "RobotwinAbsoluteActionPolicy",
    "RobotwinRelativeActionPolicy",
    "build_robotwin_action_policy",
]


@register_action_space("robotwin_aug_aligned")
class RobotwinAugAlignedJointPolicy:
    """Deploy training-aligned joint state with selectable joint or camera-spatial EEF control."""

    _JOINT_SLOTS = np.r_[0:6, 13, 29:35, 42]
    _EEF_SLOTS = np.r_[7:13, 36:42]

    def __init__(self, policy: Any, spec: dict[str, Any]) -> None:
        self.policy = policy
        self.action_space_spec = dict(spec)
        self.action_space_metadata = dict(spec)
        self.eef_representation = str(spec.get("representation", "")).lower()
        if self.eef_representation not in {"relative", "absolute"} or spec.get("joint_representation") != "absolute":
            raise ValueError("RobotWin augmented actions require relative or absolute EEFs and absolute joints")
        key = str(spec.get("statistics_key", "robotwin_aug_aligned"))
        stats = policy.norm_stats[key]
        self.state_stats, self.action_stats = stats["state"], stats["action"]
        self.state_mask = np.zeros(130, dtype=bool)
        self.state_mask[self._JOINT_SLOTS] = True
        self.action_mask = self.state_mask.copy()
        self.action_mask[self._EEF_SLOTS] = True
        for name, mask in (("state", self.state_mask), ("action", self.action_mask)):
            if not np.array_equal(np.asarray(stats[name].get("mask"), dtype=bool), mask):
                raise ValueError(f"RobotWin augmented {name} statistics disagree with the 14/26-D schema")
        horizon = policy.action_codec.chunk_size
        self.action_schema_mask = np.broadcast_to(self.action_mask, (horizon, 130)).copy()

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        """Accept robotwin_joint_state ordered left six/gripper, right six/gripper."""
        from starVLA.dataloader.gr00t_lerobot.embodiment_tags import embodiment_tag_to_index

        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("Expected non-empty examples with robotwin_joint_state")
        prepared = []
        reference_poses = []
        control_mode = payload.pop("robotwin_control_mode", "joint")
        if control_mode not in {"joint", "eef"}:
            raise ValueError(f"Unsupported RoboTwin control mode: {control_mode}")
        for example in examples:
            item = copy.copy(example)
            pose = item.pop("robotwin_ee_pose", None)
            if control_mode == "eef":
                if (
                    (
                        self.eef_representation == "relative"
                        and self.action_space_spec.get("relative_eef_frame") != "camera_spatial"
                    )
                    or self.action_space_spec.get("static_camera_frame") is not True
                    or self.action_space_spec.get("frame_convention") != "robotwin_head_opencv_v1"
                ):
                    raise ValueError("EEF control requires the training camera-spatial OpenCV frame contract")
                if self.eef_representation == "relative":
                    pose = np.asarray(pose, dtype=np.float32)
                    if pose.shape != (12,) or not np.isfinite(pose).all():
                        raise ValueError("EEF control requires 12 finite current camera-frame EEF pose values")
                    reference_poses.append(pose)
            joints = np.asarray(item.pop("robotwin_joint_state"), dtype=np.float32).reshape(-1)
            if joints.shape != (14,) or not np.isfinite(joints).all():
                raise ValueError("robotwin_joint_state must contain 14 finite joint/gripper values")
            state = np.zeros(130, dtype=np.float32)
            state[self._JOINT_SLOTS] = joints
            state, _ = normalize_with_statistics(state, self.state_stats, modality="state", default_mode="q99")
            state[~self.state_mask] = 0
            item["state"] = state[None]
            item["state_mask"] = self.state_mask[None].copy()
            item["action_schema_mask"] = self.action_schema_mask.copy()
            item["meta"] = {
                **(item.get("meta") or {}),
                "embodiment_tag": embodiment_tag_to_index("robotwin_aug_aligned"),
                "dataset_id": "robotwin_aug_aligned",
                "dataset_family": "robotwin",
                "robot_family": "aloha",
                "robot_type": "my_robotwin",
                "control_mode": "ee",
                "end_effector_type": "gripper",
                "action_representation": self.eef_representation,
            }
            prepared.append(item)
        sample_seed = payload.pop("robotwin_sample_seed", None)
        if sample_seed is None:
            output = dict(self.policy.predict_action(**{**payload, "examples": prepared}))
        else:
            import torch

            if len(prepared) != 1:
                raise ValueError("Seeded RoboTwin evaluation requires max_batch_size=1")
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(sample_seed))
                output = dict(self.policy.predict_action(**{**payload, "examples": prepared}))
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim not in (2, 3) or normalized.shape[-1] != 130:
            raise ValueError(f"Expected canonical [B,T,130] actions, got {normalized.shape}")
        actions, _ = unnormalize_with_statistics(
            normalized,
            self.action_stats,
            modality="action",
            default_mode="q99",
            clip=True,
        )
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned non-finite actions")
        output["absolute_actions"] = actions[..., self._JOINT_SLOTS]
        if control_mode == "eef":
            if actions.ndim == 2:
                actions = actions[None]
            if self.eef_representation == "relative" and len(actions) != len(reference_poses):
                raise ValueError("Action batch does not match EEF reference batch")
            references = np.asarray(reference_poses) if self.eef_representation == "relative" else None
            compact = []
            for slots, offset, gripper in ((np.arange(7, 13), 0, 13), (np.arange(36, 42), 6, 42)):
                eef = actions[..., slots]
                if self.eef_representation == "relative":
                    ref = references[:, None, offset : offset + 6]
                    xyz = ref[..., :3] + eef[..., :3]
                    # Training uses spatial rotation R_future @ R_current.T.
                    rotation = (
                        (
                            Rotation.from_rotvec(eef[..., 3:].reshape(-1, 3))
                            * Rotation.from_rotvec(np.broadcast_to(ref[..., 3:], eef[..., 3:].shape).reshape(-1, 3))
                        )
                        .as_rotvec()
                        .reshape((*eef.shape[:-1], 3))
                    )
                    eef = np.concatenate([xyz, rotation], axis=-1)
                compact.extend([eef, actions[..., gripper : gripper + 1]])
            output["absolute_actions"] = np.concatenate(compact, axis=-1).astype(np.float32)
        output["control_mode"] = control_mode
        return output
