# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight additions and modifications by Chubin Zhang, 2026.
# Modification: Generalize native R1Pro preparation to explicit recordings and splits.
"""Prepare R1Pro LeRobot v3 recordings; preserve emitted-command versus measured-joint semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from starVLA.dataloader.pick_fruit_camera_dataset import prepare_camera_cache
from starVLA.dataloader.pick_fruit_dataset import (
    ACTION_NAMES,
    ACTION_SLOTS,
    HAND_NAMES,
    STATE_SLOTS,
    calculate_statistics,
    prepare_subset,
)

ARM_TARGETS = {
    "emitted_ik": "recorded emitted IK joint commands; approximate-rpc alignment; no causal EEF pairing",
    "measured": "measured joints at the same future source rows; not commanded joint targets",
}


def validate_emitted_commands(source: Path, table: pa.Table, calibration: Path) -> tuple[np.ndarray, dict]:
    """Read native emitted IK channels, checking their controller boundary and calibration."""
    features_path = source / "training_features.json"
    features = json.loads(features_path.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    calibration_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
    transform = np.asarray(json.loads(calibration.read_text())["T_C_B"])
    for key, value in features["calibrations"].items():
        extrinsics = value["cameras"]["head"]["extrinsics"]
        if extrinsics["result_sha256"] != calibration_hash:
            raise ValueError(f"Calibration mismatch for source {key}")
        np.testing.assert_allclose(extrinsics["T_C_B"], transform, atol=1e-10)
    raw_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    raw_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    targets, geometry = [], {}
    for side, offset in (("left", 0), ("right", 7)):
        key = f"action.arm_{side}.joint.emitted_ik"
        if (
            info["features"][key]["names"] != [f"arm_joint{i}" for i in range(1, 8)]
            or features["channels"][key]["units"] != ["rad"] * 7
            or features["channel_sources"][key]["boundary"] != "verified_IK_output_at_jointTracker_input"
        ):
            raise ValueError(f"Unsupported emitted-command ordering/units/controller boundary: {key}")
        joints = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if joints.shape != (len(table), 7) or not np.isfinite(joints).all():
            raise ValueError(f"Expected finite [N,7] emitted joints: {key}")
        targets.append(joints)
        np.testing.assert_allclose(
            raw_state[:, offset : offset + 7], table[f"observation.arm_{side}.joint"].to_pylist(), atol=1e-7
        )
        for kind, vector in (("action", raw_action), ("observation", raw_state)):
            torso = np.asarray(table[f"{kind}.arm_{side}.eef.torso_link4"].to_pylist())
            camera = np.asarray(table[f"{kind}.arm_{side}.eef.head_rgb_optical"].to_pylist())
            if not np.isfinite(torso).all() or not np.isfinite(camera).all():
                raise ValueError(f"Nonfinite {kind} {side} EEF poses")
            np.testing.assert_allclose(np.linalg.norm(torso[:, 3:], axis=1), 1, atol=1e-4)
            np.testing.assert_allclose(np.linalg.norm(camera[:, 3:], axis=1), 1, atol=1e-4)
            position_error = float(np.max(np.abs(torso[:, :3] @ transform[:3, :3].T + transform[:3, 3] - camera[:, :3])))
            rotation_error = float(
                (
                    Rotation.from_quat(camera[:, 3:]).inv()
                    * Rotation.from_matrix(transform[:3, :3])
                    * Rotation.from_quat(torso[:, 3:])
                )
                .magnitude()
                .max()
            )
            if position_error >= 1e-6 or rotation_error >= 1e-6:
                raise ValueError(f"Inconsistent torso/camera poses: {kind}.{side}")
            geometry[f"{kind}.{side}"] = {"position_m": position_error, "rotation_rad": rotation_error}
            if kind == "action":
                np.testing.assert_allclose(raw_action[:, offset : offset + 7], torso, atol=1e-7)
            names = info["features"]["action" if kind == "action" else "observation.state"]["names"]
            hand_key = f"{kind}.hand_{side}.joint"
            prefix = f"{side}_" if kind == "action" else ""
            indices = [names.index(f"sharpa_{side}.{prefix}{name}") for name in info["features"][hand_key]["names"]]
            np.testing.assert_allclose(vector[:, indices], table[hand_key].to_pylist(), atol=1e-7)
    return np.concatenate(targets, axis=1), {
        "training_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "source_geometry_checks": geometry,
    }


def validate_split(table: pa.Table, train: set[int], validation: set[int]) -> None:
    """Keep every fragment of the same source episode in the same split."""
    ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if "provenance.source" not in table.column_names:
        return  # Older measured-only recordings have native episode boundaries, not fragments.
    groups = np.asarray(table["provenance.source"].to_pylist(), dtype=np.int64)
    assignments = {}
    for episode in np.unique(ids):
        source = np.unique(groups[ids == episode], axis=0)
        if len(source) != 1:
            raise ValueError(f"Episode {episode} crosses source recording groups")
        group = tuple(source[0])
        split = "train" if episode in train else "validation"
        if group in assignments and assignments[group] != split:
            raise ValueError(f"Source episode {group} leaks across train/validation; keep its fragments together")
        assignments[group] = split


def main() -> None:
    """Cache native three-camera recordings, then derive calibrated camera-spatial supervision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True, help="Explicit dataset/split JSON; see docs/r1pro.md")
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    if args.raw_cache.exists() or args.output.exists():
        raise FileExistsError(
            "Use new cache directories; source recordings and previous statistics are never overwritten"
        )
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    spec = json.loads(args.recordings.read_text())
    arm_source = spec.get("arm_joint_source", "emitted_ik")
    if arm_source not in ARM_TARGETS or not str(spec.get("dataset_id", "")).strip() or not spec.get("subsets"):
        raise ValueError("Specify dataset_id, subsets, and arm_joint_source (emitted_ik or measured)")
    names = [row["name"] for row in spec["subsets"]]
    if len(names) != len(set(names)) or any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Subset names must be unique single path components")
    verified = []
    sources = set()
    for row in spec["subsets"]:
        source = Path(row["path"]).expanduser().resolve()
        if source in sources:
            raise ValueError(f"List each recording source once: {source}")
        sources.add(source)
        meta = pq.read_table(sorted((source / "meta/episodes").glob("chunk-*/*.parquet")))
        available = set(map(int, meta["episode_index"].to_pylist()))
        train, validation = set(row["train"]), set(row["validation"])
        if (
            train & validation
            or train | validation != available
            or len(train) != len(row["train"])
            or len(validation) != len(row["validation"])
        ):
            raise ValueError(f"Split must cover every episode exactly once: {row['name']}")
        table = pq.read_table(sorted((source / "data").glob("chunk-*/*.parquet"))).sort_by("index")
        if arm_source == "emitted_ik" and "provenance.source" not in table.column_names:
            raise ValueError("Emitted-command recordings require provenance.source to group episode fragments")
        validate_split(table, train, validation)
        values, provenance = (
            validate_emitted_commands(source, table, args.calibration) if arm_source == "emitted_ik" else (None, {})
        )
        verified.append((row, source, values, provenance))
    if not any(row["train"] for row in spec["subsets"]) or not any(row["validation"] for row in spec["subsets"]):
        raise ValueError("Need both training and held-out episodes; statistics fit training only")
    args.raw_cache.mkdir(parents=True)
    subsets = []
    for row, source, values, provenance in verified:
        destination = args.raw_cache / row["name"]
        result = prepare_subset(str(source), str(destination), list(row["validation"]))
        result.update(name=row["name"], **provenance)
        (destination / "manifest.json").write_text(json.dumps(result, indent=2))
        if values is not None:
            np.save(destination / "arm_joint_targets.npy", values)
        subsets.append(result)
    manifest = {
        "format": "pick_fruit_validation_v1",
        "dataset_id": spec["dataset_id"],
        "fps": 10,
        "subsets": subsets,
        "held_out": [[row["name"], index] for row in spec["subsets"] for index in row["validation"]],
        "split_policy": "explicit recording/episode split; source fragments kept together",
        "eef_frame": "torso_link4",
        "rotation": "xyz_rotvec",
        "canvas": "384x320: head 256x320, left/right wrist 128x160 each",
        "normalization": "train-only q01/q99, clip [-1,1], minimum half-range 1e-4",
        "hand_order": HAND_NAMES,
        "action_names": ACTION_NAMES,
        "action_slots": ACTION_SLOTS.tolist(),
        "state_slots": STATE_SLOTS.tolist(),
        "arm_joint_source": arm_source,
        "arm_joint_target": ARM_TARGETS[arm_source],
        "metric_protocol": "held-out episode split; train-only q01/q99 normalization",
        "source_manifest_sha256": hashlib.sha256(args.recordings.read_bytes()).hexdigest(),
    }
    (args.raw_cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.raw_cache / "statistics.json").write_text(json.dumps(calculate_statistics(args.raw_cache, subsets), indent=2))
    qa = prepare_camera_cache(args.raw_cache, args.output, args.calibration, args.urdf, args.horizon)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
