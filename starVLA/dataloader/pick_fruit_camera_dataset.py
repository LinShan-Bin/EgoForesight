# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Joint and camera-spatial EEF supervision with measured FK references.

"""Calibrated R1Pro cache with explicit held-out episodes and reversible EEF labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.dataloader.pick_fruit_dataset import ACTION_SLOTS, STATE_SLOTS, PickFruitDataset
from starVLA.dataloader.pick_fruit_geometry import absolute_pose, arm_fk, relative_pose

JOINT_SLOTS = np.array([*range(7), *range(29, 36)])
SUPERVISED_SLOTS = np.sort(np.union1d(ACTION_SLOTS, JOINT_SLOTS))


def fit_statistics(values: np.ndarray, slots: np.ndarray) -> dict:
    """Fit train-only q01/q99 normalization in the actual supervision space."""
    low, high = np.quantile(values, [0.01, 0.99], axis=0)
    mask = np.zeros(130, dtype=bool)
    mask[slots] = True
    return {
        "q01": low.tolist(),
        "q99": high.tolist(),
        "center": ((low + high) / 2).tolist(),
        "scale": np.maximum((high - low) / 2, 1e-4).tolist(),
        "mask": mask.tolist(),
        "count": len(values),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
    }


def prepare_camera_cache(source: Path, destination: Path, calibration: Path, urdf: Path, horizon: int = 16) -> dict:
    """Derive labels without modifying native data or the previous validation cache."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite derived cache: {destination}")
    cal = json.loads(calibration.read_text())
    assert cal["from_frame"] == "torso_link4" and cal["to_frame"] == "head_rgb_optical"
    transform = np.asarray(cal["T_C_B"], dtype=np.float64)
    assert np.allclose(transform @ np.asarray(cal["T_B_C"]), np.eye(4), atol=1e-8)
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)
    manifest = json.loads((source / "manifest.json").read_text())
    held_out = {tuple(item) for item in manifest["held_out"]}
    if not held_out:
        raise ValueError("Provide an explicit nonempty held-out episode split")
    destination.mkdir(parents=True)
    train_actions, train_states, qa = [], [], {}
    roundtrip_position, roundtrip_rotation = 0.0, 0.0
    for subset in manifest["subsets"]:
        src = source / subset["name"]
        dst = destination / subset["name"]
        dst.mkdir()
        (dst / "images.npy").symlink_to((src / "images.npy").resolve())
        actions = np.load(src / "actions.npy").copy()
        measured = np.load(src / "states.npy")
        states = measured.copy()
        references = np.zeros_like(actions)
        if manifest.get("arm_joint_source") == "emitted_ik":
            joint_targets = np.load(src / "arm_joint_targets.npy")
            assert joint_targets.shape == (len(actions), 14) and np.isfinite(joint_targets).all()
            actions[:, JOINT_SLOTS] = joint_targets
        else:
            actions[:, JOINT_SLOTS] = measured[:, JOINT_SLOTS]
        subset_qa = {}
        for side, joint_offset, eef_offset in (("left", 0, 7), ("right", 29, 36)):
            pose = arm_fk(urdf, measured[:, joint_offset : joint_offset + 7], side, f"{side}_gripper_link")
            references[:, eef_offset : eef_offset + 6] = pose
            target = actions[:, eef_offset : eef_offset + 6]
            subset_qa[side] = {
                "fk_vs_controller_position_cm_median_p95": (
                    np.quantile(np.linalg.norm(pose[:, :3] - target[:, :3], axis=-1) * 100, [0.5, 0.95]).tolist()
                ),
                "fk_vs_controller_rotation_deg_median_p95": (
                    np.quantile(
                        np.rad2deg(
                            (Rotation.from_rotvec(pose[:, 3:]).inv() * Rotation.from_rotvec(target[:, 3:])).magnitude()
                        ),
                        [0.5, 0.95],
                    ).tolist()
                ),
            }
        chunks = np.zeros((len(actions), horizon, 130), dtype=np.float32)
        for ep in subset["episodes"]:
            ep["split"] = "validation" if (subset["name"], ep["episode_index"]) in held_out else "train"
            frames = np.arange(ep["start"], ep["end"])
            indices = np.minimum(frames[:, None] + np.arange(horizon), ep["end"] - 1)
            chunk = actions[indices].copy()
            for offset in (7, 36):
                target = chunk[..., offset : offset + 6].reshape(-1, 6)
                ref = np.repeat(references[frames, offset : offset + 6], horizon, axis=0)
                relative = relative_pose(target, ref, transform)
                recovered = absolute_pose(relative, ref, transform)
                roundtrip_position = max(roundtrip_position, float(np.max(np.abs(recovered[:, :3] - target[:, :3]))))
                roundtrip_rotation = max(
                    roundtrip_rotation,
                    float(
                        np.max(
                            (
                                Rotation.from_rotvec(recovered[:, 3:]).inv() * Rotation.from_rotvec(target[:, 3:])
                            ).magnitude()
                        )
                    ),
                )
                chunk[..., offset : offset + 6] = relative.reshape(len(frames), horizon, 6)
            chunks[frames] = chunk
            if ep["split"] == "train":
                valid = frames[:, None] + np.arange(horizon) < ep["end"]
                train_actions.append(chunk[valid])
                train_states.append(states[frames])
        for filename, values in (
            ("actions", actions),
            ("states", states),
            ("references", references),
            ("relative_actions", chunks),
        ):
            np.save(dst / f"{filename}.npy", values)
        qa[subset["name"]] = subset_qa
    assert roundtrip_position < 1e-6 and roundtrip_rotation < 1e-6
    manifest.update(
        {
            "format": "pick_fruit_camera_spatial_v1",
            "horizon": horizon,
            "eef_frame": "head_rgb_optical",
            "rotation": "camera_spatial_xyz_rotvec",
            "state": "measured_arm_and_hand_joints_only; FK is label/decoder geometry, never model input",
            "action": "controller_eef_camera_spatial_relative_to_current_measured_FK_and_absolute_joints",
            "arm_joint_target": manifest.get(
                "arm_joint_target", "measured joints at the same future source rows; not commanded joint targets"
            ),
            "hand_joint_target": "recorded controller hand targets at source row; no time shift",
            "reference_frame": "torso_link4",
            "fk_tips": ["left_gripper_link", "right_gripper_link"],
            "action_slots": SUPERVISED_SLOTS.tolist(),
            "state_slots": STATE_SLOTS.tolist(),
            "action_names": [f"canonical_slot_{int(i)}" for i in SUPERVISED_SLOTS],
            "calibration": cal,
            "T_C_B": transform.tolist(),
            "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "metric_protocol": manifest.get(
                "metric_protocol", "legacy 56 absolute torso EEF/hand slots and frozen previous train-only normalizer"
            ),
            "source_cache": str(source.resolve()),
        }
    )
    statistics = {
        "action": fit_statistics(np.concatenate(train_actions), SUPERVISED_SLOTS),
        "state": fit_statistics(np.concatenate(train_states), STATE_SLOTS),
    }
    # Freeze the previous normalizer for numerical comparability of absolute EEF/hand MSE.
    (destination / "absolute_metric_statistics.json").write_text((source / "statistics.json").read_text())
    (destination / "statistics.json").write_text(json.dumps(statistics, indent=2))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    qa.update(
        {
            "roundtrip_max_position_m": roundtrip_position,
            "roundtrip_max_rotation_rad": roundtrip_rotation,
            "held_out": sorted(held_out),
            "train_episodes": sum(ep["split"] == "train" for s in manifest["subsets"] for ep in s["episodes"]),
            "validation_episodes": sum(ep["split"] == "validation" for s in manifest["subsets"] for ep in s["episodes"]),
        }
    )
    assert qa["train_episodes"] > 0 and qa["validation_episodes"] == len(held_out) > 0
    (destination / "qa.json").write_text(json.dumps(qa, indent=2))
    (destination / "READY").write_text("camera spatial labels and train-only normalization verified\n")
    return qa


class PickFruitCameraDataset(PickFruitDataset):
    """Joint supervision plus camera-spatial EEF, decoded using current measured FK."""

    def __init__(self, cache_dir: str | Path, chunk_size: int = 16, split: str = "train") -> None:
        super().__init__(cache_dir, chunk_size, split)
        assert self.manifest["format"] == "pick_fruit_camera_spatial_v1"
        assert self.manifest["horizon"] == chunk_size
        self.transform = np.asarray(self.manifest["T_C_B"])
        self.color_jitter = None
        self.absolute_metric_statistics = json.loads((self.cache_dir / "absolute_metric_statistics.json").read_text())
        for subset, data in zip(self.manifest["subsets"], self.subsets, strict=True):
            for key in ("references", "relative_actions"):
                data[key] = np.load(self.cache_dir / subset["name"] / f"{key}.npy", mmap_mode="r")

    def make_example(self, subset_index: int, frame: int, end: int, task: str, training: bool = True) -> dict:
        """Expose only current RGB and measured state at inference."""
        result = super().make_example(subset_index, frame, end, task, training)
        result["lang"] = f"Task: {task}. Control Mode: relative ee. Embodiment: galaxea r1pro sharpa."
        result["meta"].update(
            {
                "action_representation": "relative",
                "eef_frame": "head_rgb_optical",
                "relative_eef_frame": "camera_spatial",
                "joint_representation": "absolute",
            }
        )
        if training:
            result["action"] = self.normalize(self.subsets[subset_index]["relative_actions"][frame], "action")
            if self.color_jitter is not None:
                import torch

                pair = torch.from_numpy(result["image_pair"][:, 0].copy()).permute(0, 3, 1, 2)
                result["image_pair"] = self.color_jitter(pair).permute(0, 2, 3, 1).numpy()[:, None]
        return result

    def recover_absolute(self, prediction: np.ndarray, subset_index: int, frame: int) -> np.ndarray:
        """Decode a denormalized chunk into the original torso EEF/absolute joint space."""
        result = prediction.copy()
        for offset in (7, 36):
            ref = self.subsets[subset_index]["references"][frame, offset : offset + 6]
            result[:, offset : offset + 6] = absolute_pose(prediction[:, offset : offset + 6], ref, self.transform)
        return result
