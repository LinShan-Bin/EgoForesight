# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Adapt pick-fruit canonical predictions to physical R1 EEF or joint targets.

"""Training-specific adapter. Wire observations are measurements; outputs are targets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.dataloader.gr00t_lerobot.state_action.sharpa import SHARPA_HARDWARE_JOINT_NAMES
from starVLA.dataloader.pick_fruit_camera_dataset import SUPERVISED_SLOTS
from starVLA.dataloader.pick_fruit_dataset import HAND_NAMES, META, STATE_SLOTS
from starVLA.dataloader.pick_fruit_geometry import absolute_pose, arm_fk
from starVLA.dataloader.r1pro_transforms import denormalize_values, make_rgb_canvas, normalize_values

WIRE_HAND_NAMES = list(SHARPA_HARDWARE_JOINT_NAMES)
HAND_TO_CANONICAL = [WIRE_HAND_NAMES.index(name) for name in HAND_NAMES]
HAND_TO_WIRE = [HAND_NAMES.index(name) for name in WIRE_HAND_NAMES]


def file_sha256(path: Path) -> str:
    """Return a digest of exact asset bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract(horizon: int, control_mode: str) -> dict:
    """Describe physical channels under the core policy.inference 1.0 contract."""
    if control_mode not in {"eef", "joint"} or not 1 <= horizon <= 100:
        raise ValueError("Expected eef/joint control and a horizon of 1..100")
    observations, actions = {}, {}
    for side in ("left", "right"):
        for kind, names in (("arm", [f"arm_joint{i}" for i in range(1, 8)]), ("hand", WIRE_HAND_NAMES)):
            descriptor = {
                "component": f"{kind}_{side}",
                "quantity": "joint_position",
                "representation": "absolute",
                "shape": [len(names)],
                "dtype": "float32",
                "required": True,
                "joint_names": list(names),
                "units": ["rad"] * len(names),
            }
            observations[f"{kind}_{side}"] = {
                **descriptor,
                "history": {"min_samples": 1, "max_samples": 1, "max_age_ns": 200_000_000},
            }
            if kind == "hand" or control_mode == "joint":
                actions[f"{kind}_{side}"] = descriptor
        if control_mode == "eef":
            actions[f"eef_{side}"] = {
                "component": f"arm_{side}",
                "quantity": "pose",
                "representation": "absolute",
                "shape": [7],
                "dtype": "float32",
                "required": True,
                "parent_frame": "torso_link4",
                "child_frame": f"r1_{side}_eef",
                "units": "m_rad",
                "convention": "xyzw",
            }
    for role in ("head", "wrist_left", "wrist_right"):
        observations[role] = {
            "component": role,
            "quantity": "rgb",
            "representation": "absolute",
            "shape": [720, 1280, 3],
            "dtype": "uint8",
            "required": True,
            "color": "RGB",
            "history": {"min_samples": 1, "max_samples": 1, "max_age_ns": 200_000_000},
        }
    return {
        "observations": observations,
        "actions": actions,
        "required_extensions": [],
        "timing": {
            "modes": ["relative_sequence"],
            "offsets_ns": [i * 100_000_000 for i in range(horizon)],
            "prediction_alignment": "observation",
            "allow_retime": True,
        },
    }


class R1ProAdapter:
    """Preserve the 70-slot model schema in both selectable wire control modes."""

    def __init__(
        self, statistics: dict, manifest: dict, urdf: Path, control_mode: str = "eef", statistics_key: str | None = None
    ) -> None:
        self.statistics = statistics
        self.manifest = manifest
        self.urdf = urdf
        self.control_mode = control_mode
        self.statistics_key = statistics_key or META["dataset_id"]
        self.horizon = manifest["horizon"]
        self.transform = np.asarray(manifest["T_C_B"], dtype=np.float64)
        self.contract = build_contract(self.horizon, control_mode)

    def inputs(self, channels: dict[str, np.ndarray], task: str) -> tuple[dict, np.ndarray]:
        """Normalize current measured joints and reproduce the inference canvas/prompt."""
        state = np.zeros((1, 130), dtype=np.float32)
        for side, arm_offset, hand_offset in (("left", 0, 80), ("right", 29, 105)):
            state[0, arm_offset : arm_offset + 7] = channels[f"arm_{side}"]
            state[0, hand_offset : hand_offset + 22] = channels[f"hand_{side}"][HAND_TO_CANONICAL]
        example = {
            "lang": f"Task: {task}. Control Mode: relative ee. Embodiment: galaxea r1pro sharpa.",
            "state": normalize_values(state, self.statistics["state"]),
            "state_mask": np.asarray(self.statistics["state"]["mask"], dtype=bool)[None],
            "image": make_rgb_canvas(channels),
            "action_schema_mask": (
                np.broadcast_to(np.asarray(self.statistics["action"]["mask"], dtype=bool), (self.horizon, 130)).copy()
            ),
            "meta": {
                **META,
                "dataset_id": self.statistics_key,
                "action_representation": "relative",
                "eef_frame": "head_rgb_optical",
                "relative_eef_frame": "camera_spatial",
                "joint_representation": "absolute",
            },
        }
        # Geometry must use the original measured joints, never clipped/normalized state.
        return example, state

    def outputs(self, normalized: np.ndarray, measured_state: np.ndarray) -> dict[str, np.ndarray]:
        """Denormalize once; decode camera spatial deltas or select absolute joint targets."""
        normalized = np.asarray(normalized)
        if normalized.shape != (1, self.horizon, 130) or not np.isfinite(normalized).all():
            raise ValueError(f"Invalid model output: expected finite [1,{self.horizon},130], got {normalized.shape}")
        physical = denormalize_values(normalized[0], self.statistics["action"])
        result = {}
        for side, arm_offset, eef_offset, hand_offset in (("left", 0, 7, 80), ("right", 29, 36, 105)):
            if self.control_mode == "eef":
                reference = arm_fk(
                    self.urdf, measured_state[:, arm_offset : arm_offset + 7], side, f"{side}_gripper_link"
                )
                pose = absolute_pose(physical[:, eef_offset : eef_offset + 6], reference, self.transform)
                result[f"eef_{side}"] = np.concatenate(
                    (pose[:, :3], Rotation.from_rotvec(pose[:, 3:]).as_quat()), axis=-1
                ).astype(np.float32)
            else:
                # These model labels were future encoders, not recorded commanded arm joints.
                result[f"arm_{side}"] = physical[:, arm_offset : arm_offset + 7].copy()
            result[f"hand_{side}"] = physical[:, hand_offset : hand_offset + 22][:, HAND_TO_WIRE].copy()
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite decoded physical prediction")
        return result


def load_adapter(
    run_dir: Path, checkpoint: Path, calibration: Path, urdf: Path, control_mode: str
) -> tuple[R1ProAdapter, Any, dict]:
    """Validate the existing training run's saved assets; never substitute pretraining statistics."""
    from omegaconf import OmegaConf

    run_dir, checkpoint = run_dir.resolve(), checkpoint.resolve()
    is_step = checkpoint.parent == run_dir / "checkpoints" and re.fullmatch(
        r"steps_[0-9]+_pytorch_model.pt", checkpoint.name
    )
    is_final = checkpoint == run_dir / "final_model" / "pytorch_model.pt"
    if not (is_step or is_final):
        raise ValueError("Checkpoint must be a saved step or final_model/pytorch_model.pt inside this run")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = run_dir / "config_full.yaml"
    manifest_path = run_dir / "dataset_manifest.json"
    statistics_path = run_dir / "dataset_statistics.json"
    cfg = OmegaConf.load(config_path)
    manifest = json.loads(manifest_path.read_text())
    statistics_key = OmegaConf.select(cfg, "framework.action_model.action_space.statistics_key")
    if not isinstance(statistics_key, str) or not statistics_key:
        raise ValueError("Training config must explicitly select its dataset statistics key")
    statistics = json.loads(statistics_path.read_text())[statistics_key]
    dataset_py = OmegaConf.select(cfg, "datasets.vla_data.dataset_py")
    if dataset_py not in {"pick_fruit_dataset", "r1pro_merged_dataset"}:
        raise ValueError(f"Unsupported training dataset: {dataset_py!r}")
    arm_source = manifest.get("arm_joint_source", "measured")
    arm_targets = {
        "measured": "measured joints at the same future source rows; not commanded joint targets",
        "emitted_ik": "recorded emitted IK joint commands; approximate-rpc alignment; no causal EEF pairing",
    }
    if arm_source not in arm_targets or (dataset_py == "r1pro_merged_dataset" and arm_source != "emitted_ik"):
        raise ValueError(f"Unsupported arm joint source: {arm_source!r}")
    expected = {
        "framework.name": "Qwen35MoT3",
        "framework.mot.action_dim": 130,
        "framework.mot.robot_state.enabled": True,
        "framework.mot.robot_state.tokenization": "single_token",
        "framework.action_model.action_dim": 130,
        "framework.action_model.state_dim": 130,
        "framework.action_model.encoding": "raw",
        "framework.action_model.chunk_size": manifest["horizon"],
        "framework.action_model.action_space.name": "canonical",
        "framework.action_model.action_space.representation": "relative",
        "framework.action_model.action_space.relative_eef_frame": "camera_spatial",
        "framework.action_model.action_space.frame_convention": "head_rgb_optical_opencv",
        "framework.action_model.action_space.static_camera_frame": True,
        "framework.action_model.action_space.joint_representation": "absolute",
        "datasets.vla_data.dataset_py": dataset_py,
        "datasets.vla_data.action_horizon": manifest["horizon"],
        "datasets.vla_data.image_size": [384, 320],
        "datasets.vla_data.image_pair_head_size": [256, 320],
        "datasets.vla_data.image_pair_wrist_size": [128, 160],
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Unsupported training setting {key}: expected {value!r}")
    expected_manifest = {
        "format": "pick_fruit_camera_spatial_v1",
        "fps": 10,
        "eef_frame": "head_rgb_optical",
        "rotation": "camera_spatial_xyz_rotvec",
        "reference_frame": "torso_link4",
        "fk_tips": ["left_gripper_link", "right_gripper_link"],
        "hand_order": list(HAND_NAMES),
        "state_slots": STATE_SLOTS.tolist(),
        "action_slots": SUPERVISED_SLOTS.tolist(),
        "state": "measured_arm_and_hand_joints_only; FK is label/decoder geometry, never model input",
        "arm_joint_target": arm_targets[arm_source],
        "hand_joint_target": "recorded controller hand targets at source row; no time shift",
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unsupported dataset manifest field {key}: expected {value!r}")
    for key, path in (("calibration_sha256", calibration), ("urdf_sha256", urdf)):
        if manifest[key] != file_sha256(path):
            raise ValueError(f"{key} differs from the asset used to generate training labels")
    cal = json.loads(calibration.read_text())
    transform = np.asarray(manifest["T_C_B"])
    if (
        cal != manifest["calibration"]
        or cal["from_frame"] != "torso_link4"
        or cal["to_frame"] != "head_rgb_optical"
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.array_equal(transform, np.asarray(cal["T_C_B"]))
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
    ):
        raise ValueError("Training camera transform is inconsistent or non-rigid")
    for field, slots in (("state", STATE_SLOTS), ("action", SUPERVISED_SLOTS)):
        stats = statistics[field]
        mask = np.zeros(130, dtype=bool)
        mask[slots] = True
        if len(stats["mask"]) != 130 or any(type(item) is not bool for item in stats["mask"]):
            raise ValueError(f"Invalid {field} normalization mask")
        if not np.array_equal(stats["mask"], mask):
            raise ValueError(f"Wrong {field} supervised slots")
        for key in ("center", "scale", "q01", "q99"):
            array = np.asarray(stats[key])
            if array.shape != (130,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field}.{key} statistics")
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        if (
            np.any(high < low)
            or not np.allclose(stats["center"], (low + high) / 2, rtol=0, atol=1e-10)
            or not np.allclose(stats["scale"], np.maximum((high - low) / 2, 1e-4), rtol=0, atol=1e-10)
        ):
            raise ValueError(f"Inconsistent {field} normalization recipe")
    adapter = R1ProAdapter(statistics, manifest, urdf, control_mode, statistics_key)
    provenance = {
        "adapter": "pick_fruit_camera_spatial_v1",
        "checkpoint": str(checkpoint),
        "control_mode": control_mode,
        "dataset_fps": 10,
        "state_source": "current measured arm and hand joints only",
        "arm_joint_target_source": manifest["arm_joint_target"],
        "eef_target_source": "recorded controller EEF target; decoded against current measured joint FK",
        "hand_target_source": manifest["hand_joint_target"],
        "preprocessing_owner": "model_server",
        "physical_units_on_wire": True,
        "image_preprocessing": "RGB INTER_AREA letterbox; head 256x320 above two 128x160 wrists",
        "normalization": "train q01/q99; clip input state; inverse prediction once without output clipping",
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "statistics_sha256": file_sha256(statistics_path),
        "statistics_key": statistics_key,
        "urdf_sha256": manifest["urdf_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
        "training_calibration": cal,
        "artifact_binding": "existing pick-fruit run assets; no integrated.training_view.v1 artifact",
    }
    return adapter, cfg, provenance


class R1ProPolicy:
    """A stateless model invocation with train-identical adaptation on both sides."""

    def __init__(self, model: Any, adapter: R1ProAdapter, flow_steps: int = 4) -> None:
        self.model = model
        self.adapter = adapter
        self.flow_steps = flow_steps
        self.contract = adapter.contract

    def predict(self, channels: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        """Return only physical actions in the selected wire control mode."""
        example, measured = self.adapter.inputs(channels, task)
        output = self.model.predict_action([example], num_flow_steps=self.flow_steps)
        return self.adapter.outputs(output["normalized_actions"], measured)

    def reset(self) -> None:
        """No cached context: every predict call uses only its complete observation.

        The async transport drains any previous invocation before this barrier.
        This adapter never reuses model KV, images, state or previous actions.
        """

    def warmup(self, count: int = 1) -> None:
        """Exercise model and complete decoder before any ready welcome can be sent."""
        if count < 1:
            raise ValueError("At least one successful warmup is required")
        channels = {
            key: np.zeros(desc["shape"], dtype=desc["dtype"]) for key, desc in self.contract["observations"].items()
        }
        for _ in range(count):
            self.predict(channels, "Pick fruit.")
