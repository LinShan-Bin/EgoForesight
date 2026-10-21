# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add joints to filtered original and deterministically mirrored episodes.
"""Add absolute joints and OpenCV EEF poses without re-encoding videos or invoking a VLM."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from scripts.robotwin_aligned.common import (
    ROBOTWIN_HEAD_W2C,
    StaticThresholds,
    _statistics,
    compose_state,
    mirror_state_action,
    pose7_wxyz_to_pose6,
)

JOINT_SIGNS = np.array([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)
VERSION = 1
RAW_SETTINGS = ("clean_50", "randomized_500")
VARIANT_SUFFIXES = {"both": ("", "-mirrored"), "original": ("",), "mirrored": ("-mirrored",)}


def infer_raw_setting(source_root: Path, requested: str | None = None) -> str:
    """Resolve the RoboTwin source split without silently selecting the wrong raw episode."""
    if requested is not None:
        return requested
    matches = [setting for setting in RAW_SETTINGS if setting in source_root.name]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer raw setting from {source_root}; pass --raw-setting")
    return matches[0]


def raw_episode_mapping(source: Path) -> dict[int, int]:
    """Map dense aligned episode indices back to raw IDs after audited exclusions."""
    episodes = {int(path.stem.removeprefix("episode_")) for path in (source / "data").glob("chunk-*/*.parquet")}
    source_ids = source / "meta/source_episodes.json"
    if source_ids.is_file():
        mapping = {int(key): int(value) for key, value in json.loads(source_ids.read_text()).items()}
        if set(mapping) != episodes or len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Invalid raw episode mapping: {source_ids}")
        return mapping
    manifest_path = source / "meta/audit_exclusions.json"
    if not manifest_path.is_file():
        expected = set(range(len(episodes)))
        if episodes != expected:
            raise ValueError(f"Non-dense episode indices without exclusion metadata: {source}")
        return {index: index for index in expected}

    manifest = json.loads(manifest_path.read_text())
    original_count = int(manifest["original_count"])
    remaining_count = int(manifest["remaining_count"])
    excluded = {int(index) for index in manifest["excluded_old_indices"]}
    old_to_new = {int(old): int(new) for old, new in manifest["old_to_new"].items()}
    expected_old = set(range(original_count)) - excluded
    expected_new = set(range(remaining_count))
    if (
        len(excluded) != len(manifest["excluded_old_indices"])
        or old_to_new.keys() != expected_old
        or set(old_to_new.values()) != expected_new
        or episodes != expected_new
    ):
        raise ValueError(f"Invalid audit exclusion mapping: {manifest_path}")
    return {new: old for old, new in old_to_new.items()}


def mirror_joints(joints: np.ndarray) -> np.ndarray:
    """Swap arms and reflect axial joints (1, 5, 6); preserve gripper aperture.

    This is the ideal symmetric-arm augmentation convention, not exact IK for
    the slightly asymmetric physical URDF. It accompanies the existing image
    reflection about the robot sagittal plane (world x=0). Camera calibration
    is reflected with it, preserving the already mirrored camera-frame EEFs.
    """
    return np.concatenate((joints[:, 7:] * JOINT_SIGNS, joints[:, :7] * JOINT_SIGNS), axis=1)


def camera_pose(pose: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Convert world xyz/rotvec to the fixed head-camera OpenCV frame."""
    rotation = Rotation.from_matrix(w2c[:3, :3])
    return np.concatenate(
        (pose[:, :3] @ w2c[:3, :3].T + w2c[:3, 3], (rotation * Rotation.from_rotvec(pose[:, 3:])).as_rotvec()),
        axis=1,
    ).astype(np.float32)


def retained_indices(raw: np.ndarray, state: np.ndarray, action: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Recover exact retained rows, rejecting ambiguous poses with different joints."""
    np.testing.assert_array_equal(state[1:], action[:-1])
    targets = np.concatenate((state, action[-1:]))
    lookup: dict[bytes, list[int]] = {}
    for index, row in enumerate(raw):
        lookup.setdefault(row.tobytes(), []).append(index)
    keep = []
    for row in targets:
        candidates = [i for i in lookup.get(row.tobytes(), []) if not keep or i > keep[-1]]
        if not candidates:
            raise ValueError("Filtered pose not found in raw episode")
        if not np.all(joints[candidates] == joints[candidates[0]]):
            raise ValueError("Ambiguous repeated pose with different joint values")
        keep.append(candidates[0])
    return np.asarray(keep, dtype=np.int64)


def augment_vector(eef: np.ndarray, joints: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Pack 14 joints/grippers followed by two 6-D absolute camera EEFs."""
    return np.concatenate((joints, camera_pose(eef[:, :6], w2c), camera_pose(eef[:, 7:13], w2c)), axis=1)


def modality_metadata(original: dict) -> dict:
    """Replace state/action schema while retaining video and language bindings."""
    result = {key: value for key, value in original.items() if key not in ("state", "action")}
    for modality, column in (("state", "observation.state"), ("action", "action")):
        result[modality] = {}
        for key, start, end in (
            ("left_arm_joint", 0, 6),
            ("left_gripper", 6, 7),
            ("right_arm_joint", 7, 13),
            ("right_gripper", 13, 14),
            ("left_eef_pose", 14, 20),
            ("right_eef_pose", 20, 26),
        ):
            entry = {"original_key": column, "start": start, "end": end, "absolute": True, "dtype": "float32"}
            if key.endswith("eef_pose"):
                entry["rotation_type"] = "axis_angle"
            result[modality][key] = entry
    return result


def convert_task(job: tuple[str, str, str, str, str, str]) -> dict:
    """Build selected task variants atomically, preserving original source repositories."""
    source_root, output_root, raw_root, task = map(Path, job[:4])
    raw_setting = job[4]
    requested_suffixes = VARIANT_SUFFIXES[job[5]]
    task_name = str(task)
    source = source_root / task
    reports = []
    suffixes = []
    for suffix in requested_suffixes:
        destination = output_root / (task_name + suffix)
        report_path = destination / "meta/joint_augmentation.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            if not (
                report["version"] == VERSION
                and report["source_root"] == str(source_root)
                and report.get("raw_setting", "clean_50") == raw_setting
            ):
                raise ValueError(f"Incompatible existing output: {destination}")
            reports.append(report)
        elif destination.exists():
            raise FileExistsError(f"Incomplete task output; inspect before removing: {destination}")
        else:
            suffixes.append(suffix)
    if not suffixes:
        return {"task": task_name, "episodes": reports[0]["episodes"], "rows": reports[0]["rows"], "cached": True}
    destinations = [output_root / (task_name + suffix) for suffix in suffixes]
    temporaries = [path.with_name(path.name + ".building") for path in destinations]
    for suffix, temporary in zip(suffixes, temporaries, strict=True):
        original = source_root / (task_name + suffix)
        if not (original / "meta/info.json").is_file():
            raise FileNotFoundError(f"Source variant is incomplete: {original}")
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "meta").mkdir(parents=True)
        for name in (
            "info.json",
            "modality.json",
            "episodes.jsonl",
            "tasks.jsonl",
            "tasks.original.jsonl",
            "audit_exclusions.json",
            "source_episodes.json",
        ):
            if (original / "meta" / name).exists():
                shutil.copy2(original / "meta" / name, temporary / "meta" / name)
        (temporary / "videos").symlink_to((original / "videos").resolve(), target_is_directory=True)
    vectors: list[dict[str, list[np.ndarray]]] = [{"observation.state": [], "action": []} for _ in temporaries]
    paths = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError(f"No parquet episodes: {source}")
    episode_mapping = raw_episode_mapping(source)
    for path in paths:
        table = pq.read_table(path)
        episode = int(table["episode_index"][0].as_py())
        raw_episode = episode_mapping[episode]
        raw_path = raw_root / task / f"aloha-agilex_{raw_setting}" / "data" / f"episode{raw_episode}.hdf5"
        with h5py.File(raw_path) as raw_file:
            raw_eef = compose_state(
                pose7_wxyz_to_pose6(raw_file["endpose/left_endpose"][:]),
                raw_file["endpose/left_gripper"][:],
                pose7_wxyz_to_pose6(raw_file["endpose/right_endpose"][:]),
                raw_file["endpose/right_gripper"][:],
            )
            joints = raw_file["joint_action/vector"][:].astype(np.float32)
            calibration = raw_file["observation/head_camera/extrinsic_cv"][:]
            np.testing.assert_allclose(
                calibration[..., :3, :4],
                np.broadcast_to(ROBOTWIN_HEAD_W2C[:3, :4], calibration[..., :3, :4].shape),
                atol=1e-6,
                rtol=0,
            )
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        try:
            keep = retained_indices(raw_eef, state, action, joints)
        except ValueError:
            # Rare ambiguous rows: replay only deterministic filtering, never the VLM audit.
            from scripts.robotwin_aligned.build_lerobot import RAW_CAMERA_KEYS
            from scripts.robotwin_aligned.common import decode_jpeg, select_non_static_frames

            with h5py.File(raw_path) as raw_file:
                frames = {
                    k: [decode_jpeg(v) for v in raw_file[f"observation/{camera}/rgb"][:]]
                    for k, camera in RAW_CAMERA_KEYS.items()
                }
                keep = np.asarray(
                    select_non_static_frames(
                        raw_file["endpose/left_endpose"][:],
                        raw_file["endpose/right_endpose"][:],
                        raw_file["endpose/left_gripper"][:],
                        raw_file["endpose/right_gripper"][:],
                        frames,
                        StaticThresholds(),
                    ),
                    dtype=np.int64,
                )
        np.testing.assert_array_equal(raw_eef[keep[:-1]], state)
        np.testing.assert_array_equal(raw_eef[keep[1:]], action)
        np.testing.assert_allclose(joints[:, [6, 13]], raw_eef[:, [6, 13]], atol=1e-6, rtol=0)
        for index, (suffix, temporary) in enumerate(zip(suffixes, temporaries, strict=True)):
            mirrored = suffix == "-mirrored"
            current = pq.read_table(source_root / (task_name + suffix) / path.relative_to(source))
            current_state = np.asarray(current["observation.state"].to_pylist(), dtype=np.float32)
            current_action = np.asarray(current["action"].to_pylist(), dtype=np.float32)
            expected_state = mirror_state_action(state, ROBOTWIN_HEAD_W2C) if mirrored else state
            expected_action = mirror_state_action(action, ROBOTWIN_HEAD_W2C) if mirrored else action
            np.testing.assert_allclose(current_state, expected_state, atol=1e-6, rtol=0)
            np.testing.assert_allclose(current_action, expected_action, atol=1e-6, rtol=0)
            # Reflect the robot about world x=0, and also reflect the camera.
            # F restores the optical right-handed basis after world reflection S:
            # C_mirror = F @ C @ S. Existing flipped pixels need no re-encoding.
            reflection = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            w2c = reflection @ ROBOTWIN_HEAD_W2C @ reflection if mirrored else ROBOTWIN_HEAD_W2C
            world_state = mirror_state_action(state, np.eye(4)) if mirrored else state
            world_action = mirror_state_action(action, np.eye(4)) if mirrored else action
            # Changing the reflection plane AND camera leaves optical poses unchanged.
            for old, new in ((current_state, world_state), (current_action, world_action)):
                np.testing.assert_allclose(
                    augment_vector(old, np.zeros((len(old), 14)), ROBOTWIN_HEAD_W2C),
                    augment_vector(new, np.zeros((len(new), 14)), w2c),
                    atol=1e-6,
                    rtol=0,
                )
            current_state, current_action = world_state, world_action
            calibration_array = pa.FixedSizeListArray.from_arrays(
                pa.array(np.broadcast_to(w2c.ravel(), (len(state), 16)).ravel(), type=pa.float32()), 16
            )
            current = current.set_column(
                current.schema.get_field_index("camera_params.head_w2c"), "camera_params.head_w2c", calibration_array
            )
            current_joints = mirror_joints(joints) if mirrored else joints
            for column, eef, indices in (
                ("observation.state", current_state, keep[:-1]),
                ("action", current_action, keep[1:]),
            ):
                values = augment_vector(eef, current_joints[indices], w2c)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite labels: {path}")
                array = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), 26)
                current = current.set_column(current.schema.get_field_index(column), column, array)
                vectors[index][column].append(values)
            current = current.append_column("source_state_frame_index", pa.array(keep[:-1]))
            current = current.append_column("source_action_frame_index", pa.array(keep[1:]))
            current = current.append_column("source_episode_index", pa.array([raw_episode] * len(current)))
            target = temporary / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(current, target, compression="zstd")
    rows = sum(len(v) for v in vectors[0]["action"])
    for index, temporary in enumerate(temporaries):
        meta = temporary / "meta"
        modality = modality_metadata(json.loads((meta / "modality.json").read_text()))
        (meta / "modality.json").write_text(json.dumps(modality, indent=2))
        stats = {key: _statistics(np.concatenate(value)) for key, value in vectors[index].items()}
        (meta / "stats.json").write_text(json.dumps(stats, indent=2))
        info = json.loads((meta / "info.json").read_text())
        for column in vectors[index]:
            info["features"][column] = {
                "dtype": "float32",
                "shape": [26],
                "names": [
                    f"{key}_{i}"
                    for key, entry in modality["state"].items()
                    for i in range(entry["end"] - entry["start"])
                ],
            }
        for column in ("source_state_frame_index", "source_action_frame_index", "source_episode_index"):
            info["features"][column] = {"dtype": "int64", "shape": [1], "names": None}
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        report = {
            "version": VERSION,
            "source_root": str(source_root),
            "raw_root": str(raw_root),
            "raw_setting": raw_setting,
            "task": task_name,
            "episodes": len(paths),
            "rows": rows,
            "mirrored": suffixes[index] == "-mirrored",
            "frame_convention": "robotwin_head_opencv_v1",
            "action_shift": "next_retained_frame",
            "joint_source": "joint_action/vector at retained state/action frame indices",
            "mirror_plane": "robot sagittal plane world x=0",
            "mirrored_camera_w2c": (
                (np.diag([-1.0, 1.0, 1.0, 1.0]) @ ROBOTWIN_HEAD_W2C @ np.diag([-1.0, 1.0, 1.0, 1.0])).tolist()
            ),
            "mirror_joint_signs": JOINT_SIGNS.tolist(),
            "mirror_joint_convention": "ideal_symmetric_arm",
            "mirror_limitation": "URDF/base mounts are asymmetric; joint reflection is not exact physical IK",
            "videos": "symlink to aligned source; no video re-encoding",
        }
        (meta / "joint_augmentation.json").write_text(json.dumps(report, indent=2))
    for temporary, destination in zip(temporaries, destinations, strict=True):
        os.rename(temporary, destination)
    return {"task": task_name, "episodes": len(paths), "rows": rows}


def main() -> None:
    """Convert all tasks plus existing mirrors with bounded CPU parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/robotwin/aligned/aloha_agilex-clean_50_single_task"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/robotwin/raw"))
    parser.add_argument("--raw-setting", choices=RAW_SETTINGS)
    parser.add_argument("--variants", choices=VARIANT_SUFFIXES, default="both")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = args.source_root.resolve()
    raw_setting = infer_raw_setting(source, args.raw_setting)
    output = (args.output_root or source.with_name(source.name + "-joint-aug")).resolve()
    if output == source:
        raise ValueError("Output must not overwrite the source dataset")
    tasks = sorted(
        p.name for p in source.iterdir() if (p / "meta/info.json").is_file() and not p.name.endswith("-mirrored")
    )
    selected = args.tasks or tasks
    if not selected or not set(selected) <= set(tasks):
        raise ValueError(f"Unknown or empty task selection: {selected}")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(str(source), str(output), str(args.raw_root), task, raw_setting, args.variants) for task in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = []
        for report in pool.map(convert_task, jobs):
            reports.append(report)
            print(json.dumps(report), flush=True)
    complete_reports = []
    for task in tasks:
        for suffix in VARIANT_SUFFIXES["both"]:
            report_path = output / (task + suffix) / "meta/joint_augmentation.json"
            if report_path.is_file():
                complete_reports.append(json.loads(report_path.read_text()))
    if len(complete_reports) == 2 * len(tasks):
        (output / "READY").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "tasks": len(tasks),
                    "repositories": 2 * len(tasks),
                    "episodes": sum(r["episodes"] for r in complete_reports),
                    "rows": sum(r["rows"] for r in complete_reports),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
