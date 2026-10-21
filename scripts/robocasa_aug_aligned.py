#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Build a fixed-frame RoboCasa joint+EEF derivative with shared high-resolution videos.
"""Convert AugPosRot world poses to a fixed base-origin pseudo-camera frame.

The source's 161-D arrays are preserved in layout: joint controls are unchanged,
and every augmented world position/rotation becomes pseudo-camera-relative.
Separate xyz+rotvec EEF columns use the controller's grip_site axes. Fingertips
are reconstructed using a defined mesh-end landmark on each terminal link. Action pose rows
already represent state[t+1]; no extra time shift is applied.

Geometry extraction runs in an existing HDF5-capable interpreter; conversion
and statistics run in the training environment. This avoids changing either
environment. Re-running with the same recipe resumes completed task shards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Load the numpy-only geometry module without importing the training dataloader
# package (the HDF5 interpreter intentionally does not require PyTorch).
_frames_spec = importlib.util.spec_from_file_location(
    "robocasa_frames",
    Path(__file__).resolve().parents[1] / "starVLA/dataloader/gr00t_lerobot/state_action/robocasa_frames.py",
)
_frames = importlib.util.module_from_spec(_frames_spec)
_frames_spec.loader.exec_module(_frames)
FRAME_CONVENTION = _frames.FRAME_CONVENTION
OPENGL_FROM_OPENCV = _frames.OPENGL_FROM_OPENCV
PSEUDO_FROM_BASE_ROTATION = _frames.PSEUDO_FROM_BASE_ROTATION
pack_rotation_columns = _frames.pack_rotation_columns
unpack_rotation_columns = _frames.unpack_rotation_columns

DEFAULT_SOURCE = "data/robocasa/augmented-world"
DEFAULT_VIDEO = "data/robocasa/videos"
DEFAULT_HDF5 = "data/robocasa/official/HDF5"
DEFAULT_OUTPUT = "data/robocasa/aligned"
DEFAULT_GEOMETRY_PYTHON = sys.executable
VERSION = 1
EEF_KEYS = ("left_eef_pose", "right_eef_pose")
FINGER_KEYS = ("left_finger_positions", "right_finger_positions")
POSE_KEYS = (*EEF_KEYS, *FINGER_KEYS)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def mesh_tip_landmarks(mesh_root: Path) -> dict:
    """Define tips as the centroid of unique vertices in the distal 1 mm cap.

    These terminal meshes extend along local +x. This reproducible geometric
    landmark is near the physical distal surface, not the finger-joint origin.
    It is a geometric fingertip definition, not a simulator-provided contact site.
    """
    result = {}
    for side in ("L", "R"):
        for finger in FINGER_ORDER:
            suffix = "distal" if finger == "thumb" else "intermediate"
            path = mesh_root / f"{side}_{finger}_{suffix}_link.STL"
            raw = path.read_bytes()
            count = struct.unpack_from("<I", raw, 80)[0]
            if len(raw) != 84 + 50 * count:
                raise ValueError(f"Expected a binary STL: {path}")
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            vertices = np.unique(np.frombuffer(raw, dtype, count=count, offset=84)["vertices"].reshape(-1, 3), axis=0)
            cap = vertices[vertices[:, 0] >= vertices[:, 0].max() - 0.001]
            result[path.name] = {
                "offset": cap.mean(axis=0, dtype=np.float64).tolist(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    return result


def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    """Atomically publish metadata after its corresponding data is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def episodes(path: Path) -> list[dict]:
    """Read episode identities in their metadata order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def xml_transform(element: ET.Element) -> np.ndarray:
    """Decode compiled MuJoCo XML's local xyz and wxyz quaternion."""
    transform = np.eye(4)
    transform[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quaternion = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return transform


def extract_geometry(
    source: Path, hdf5_root: Path, output: Path, mesh_root: Path, hdf5_file: Path | None = None
) -> None:
    """Extract each episode's fixed base pose and palm-to-grip rigid transforms."""
    import h5py

    task = source.name.removeprefix("gr1_unified.")
    result = {}
    landmarks = mesh_tip_landmarks(mesh_root)
    with h5py.File(hdf5_file or hdf5_root / f"{task}.hdf5", "r") as handle:
        for episode in episodes(source / "meta/episodes.jsonl"):
            demo = f"demo_{int(episode['trajectory_id'].rsplit('-', 1)[1])}"
            group = handle[f"data/{demo}"]
            if group["states"].shape[0] != episode["length"]:
                raise ValueError(f"HDF5 length mismatch: {task}/{demo}")
            xml = ET.fromstring(group.attrs["model_file"])
            base = xml.find("./worldbody/body[@name='robot0_base']")
            if base is None or any(child.tag in {"joint", "freejoint"} for child in base):
                raise ValueError(f"Expected a fixed world-child robot0_base: {task}/{demo}")
            world_from_base = xml_transform(base)
            pseudo_from_world = np.eye(4)
            pseudo_from_world[:3, :3] = PSEUDO_FROM_BASE_ROTATION @ world_from_base[:3, :3].T
            pseudo_from_world[:3, 3] = -pseudo_from_world[:3, :3] @ world_from_base[:3, 3]
            record = {
                "demo": demo,
                "world_from_base": world_from_base.tolist(),
                "pseudo_from_world": pseudo_from_world.tolist(),
            }
            for side, letter in (("left", "l"), ("right", "r")):
                hand = xml.find(f".//body[@name='gripper0_{side}_{side}_hand']")
                found = {}

                def visit(element: ET.Element, transform: np.ndarray, collected: dict) -> None:
                    if element.tag in {"body", "site"}:
                        transform = transform @ xml_transform(element)
                        collected[element.get("name", "")] = transform
                    for child in element:
                        visit(child, transform, collected)

                if hand is None:
                    raise ValueError(f"Missing {side} hand in {demo}")
                visit(hand, np.eye(4), found)
                palm_from_grip = (
                    np.linalg.inv(found[f"gripper0_{side}_{letter}_palm"]) @ found[f"gripper0_{side}_grip_site"]
                )
                record[f"{side}_palm_from_grip"] = palm_from_grip.tolist()
            camera = xml.find(".//camera[@name='egoview']")
            if camera is None:
                raise ValueError(f"Missing egoview camera in {demo}")
            record["camera_fovy_degrees"] = float(camera.get("fovy", "45"))
            record["fingertip_landmarks"] = {}
            meshes = {element.get("name"): element for element in xml.findall("./asset/mesh")}
            for side, letter in (("left", "L"), ("right", "R")):
                for finger in FINGER_ORDER:
                    suffix = "distal" if finger == "thumb" else "intermediate"
                    body_name = f"gripper0_{side}_{letter}_{finger}_{suffix}_link"
                    body = xml.find(f".//body[@name='{body_name}']")
                    if body is None:
                        raise ValueError(f"Missing terminal finger body: {body_name}")
                    geom = next((item for item in body.findall("geom") if item.get("mesh", "").endswith("_vis")), None)
                    if geom is None:
                        raise ValueError(f"Missing terminal finger mesh: {body_name}")
                    mesh = meshes[geom.get("mesh")]
                    name = Path(mesh.get("file")).name
                    landmark = landmarks[name]
                    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                    local = np.asarray(landmark["offset"]) * scale
                    local = xml_transform(geom)[:3, :3] @ local + xml_transform(geom)[:3, 3]
                    record["fingertip_landmarks"][f"{finger}_{letter.lower()}"] = {
                        "offset": local.tolist(),
                        "mesh": name,
                        "mesh_sha256": landmark["sha256"],
                        "definition": "unique_vertex_centroid_distal_1mm_cap_v1",
                    }
            result[str(episode["episode_index"])] = record
    write_json(output, result)


def transform_augmented(values: np.ndarray, fields: dict, geometry: dict) -> tuple[np.ndarray, dict]:
    """Align all 13 augmented body/camera poses and derive two controller EEFs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 161 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [T,161] AugPosRot arrays, got {values.shape}")
    output = values.copy()
    transform = np.asarray(geometry["pseudo_from_world"])
    frame_rotation = transform[:3, :3]
    extra = {}
    fingertips = {}
    for key, position_meta in fields.items():
        if not key.endswith("_pos"):
            continue
        rotation_meta = fields[key.removesuffix("_pos") + "_rot6d"]
        ps = slice(position_meta["start"], position_meta["end"])
        rs = slice(rotation_meta["start"], rotation_meta["end"])
        position = values[:, ps] @ frame_rotation.T + transform[:3, 3]
        rotation = frame_rotation @ unpack_rotation_columns(values[:, rs])
        output[:, ps] = position
        output[:, rs] = pack_rotation_columns(rotation)
        if key in {"wrist_l_pos", "wrist_r_pos"}:
            side = "left" if key == "wrist_l_pos" else "right"
            offset = np.asarray(geometry[f"{side}_palm_from_grip"])
            xyz = position + np.einsum("tij,j->ti", rotation, offset[:3, 3])
            rotvec = Rotation.from_matrix(rotation @ offset[:3, :3]).as_rotvec()
            extra[f"{side}_eef_pose"] = np.concatenate((xyz, rotvec), axis=1).astype(np.float32)
        elif key == "camera_egoview_pos":
            camera_rotation = rotation @ OPENGL_FROM_OPENCV
            w2c = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
            w2c[:, :3, :3] = np.swapaxes(camera_rotation, -1, -2)
            w2c[:, :3, 3] = -np.einsum("tij,tj->ti", w2c[:, :3, :3], position)
            extra["camera_w2c"] = w2c.reshape(-1, 16).astype(np.float32)
        else:
            name = key.removesuffix("_pos")
            offset = np.asarray(geometry["fingertip_landmarks"][name]["offset"])
            fingertips[name] = position + np.einsum("tij,j->ti", rotation, offset)
    for side, letter in (("left", "l"), ("right", "r")):
        extra[f"{side}_finger_positions"] = np.concatenate(
            [fingertips[f"{finger}_{letter}"] for finger in FINGER_ORDER], axis=1
        ).astype(np.float32)
    if set(extra) != {*POSE_KEYS, "camera_w2c"}:
        raise ValueError(f"Incomplete pose schema: {list(extra)}")
    return output.astype(np.float32), extra


def array_statistics(values: np.ndarray) -> dict:
    """Compute full-dataset marginal statistics in converted physical units."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def convert_task(task: str, options: dict) -> dict:
    """Convert one complete task, preserving episode identity and highres video alignment."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = Path(options["source_root"]) / task
    video_source = Path(options["video_root"]) / f"{task}_ee"
    output = Path(options["output_root"]) / task
    marker = output / "meta/alignment.json"
    recipe = {
        "version": VERSION,
        "frame_convention": FRAME_CONVENTION,
        "source": str(source),
        "video_source": str(video_source),
        "pseudo_from_base_rotation": PSEUDO_FROM_BASE_ROTATION.tolist(),
    }
    if marker.exists():
        saved = read_json(marker)
        if any(saved.get(key) != value for key, value in recipe.items()):
            raise ValueError(f"Existing output uses a different recipe: {output}")
        for relative in saved["parquet_files"]:
            if not (output / relative).is_file():
                raise ValueError(f"Completed output lost {relative}")
        # These describe the consumer configuration; numerical data is unchanged.
        saved.pop("active_state_dimensions", None)
        saved.update(stored_state_dimensions_for_canonical=71, model_state_dimensions=29)
        write_json(marker, saved)
        info = read_json(output / "meta/info.json")
        info["robot_type"] = "robocasa_gr1_aug_aligned"
        write_json(output / "meta/info.json", info)
        return {"task": task, "frames": saved["frames"], "episodes": saved["episodes"], "resumed": True}
    output.mkdir(parents=True, exist_ok=True)
    geometry_path = output / "meta/episode_geometry.json"
    if not geometry_path.exists():
        subprocess.run(
            [
                options["geometry_python"],
                str(Path(__file__).resolve()),
                "--export-geometry",
                str(source),
                "--hdf5-root",
                options["hdf5_root"],
                "--mesh-root",
                options["mesh_root"],
                "--geometry-output",
                str(geometry_path),
            ],
            check=True,
        )
    geometry = read_json(geometry_path)
    source_episodes = episodes(source / "meta/episodes.jsonl")
    video_episodes = {item["episode_index"]: item for item in episodes(video_source / "meta/episodes.jsonl")}
    info = read_json(source / "meta/info.json")
    video_info = read_json(video_source / "meta/info.json")
    modality = read_json(source / "meta/modality.json")
    original_modality = json.loads(json.dumps(modality))
    stats_blocks: dict[str, list] = {key: [] for key in ("observation.state", "action")}
    stats_blocks.update({f"{prefix}.{key}": [] for prefix in ("observation.state", "action") for key in POSE_KEYS})
    parquet_files = []
    max_shift_error = 0.0
    for episode in source_episodes:
        index = episode["episode_index"]
        matching = video_episodes.get(index, {})
        if any(episode.get(key) != matching.get(key) for key in ("trajectory_id", "length")):
            raise ValueError(f"Highres episode identity/length mismatch: {task}/{index}")
        pattern_args = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        relative = info["data_path"].format(**pattern_args)
        video_relative = video_info["video_path"].format(video_key="observation.images.ego_view", **pattern_args)
        if not (video_source / video_relative).is_file():
            raise FileNotFoundError(video_source / video_relative)
        table = pq.read_table(source / relative)
        if len(table) != episode["length"]:
            raise ValueError(f"Parquet length mismatch: {source / relative}")
        originals = {}
        for role, prefix in (("state", "observation.state"), ("action", "action")):
            values = np.stack(table[prefix].to_numpy())
            originals[role] = values
            aligned, extra = transform_augmented(values, original_modality[role], geometry[str(index)])
            table = table.set_column(
                table.schema.get_field_index(prefix), prefix, pa.array(aligned.tolist(), pa.list_(pa.float32()))
            )
            stats_blocks[prefix].append(aligned)
            for key in POSE_KEYS:
                name = f"{prefix}.{key}"
                table = table.append_column(name, pa.array(extra[key].tolist(), pa.list_(pa.float32())))
                stats_blocks[name].append(extra[key])
            if role == "state":
                table = table.append_column(
                    "camera_params.head_w2c", pa.array(extra["camera_w2c"].tolist(), pa.list_(pa.float32()))
                )
        if len(table) > 1:
            shift_error = np.max(np.abs(originals["action"][:-1, 44:] - originals["state"][1:, 44:]))
            max_shift_error = max(max_shift_error, float(shift_error))
            if shift_error > 1e-5:
                raise ValueError(f"Augmented action is not next-state pose: {task}/{index}, error={shift_error}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.tmp")
        pq.write_table(table.replace_schema_metadata(None), temporary, compression="zstd")
        os.replace(temporary, target)
        parquet_files.append(relative)
    stats = {key: array_statistics(np.concatenate(blocks)) for key, blocks in stats_blocks.items()}
    for role, prefix in (("state", "observation.state"), ("action", "action")):
        for key in POSE_KEYS:
            name = f"{prefix}.{key}"
            width = 6 if key in EEF_KEYS else 15
            modality[role][key] = {"original_key": name, "start": 0, "end": width, "absolute": True}
            info["features"][name] = {"dtype": "float32", "shape": [width]}
    info["features"]["camera_params.head_w2c"] = {"dtype": "float32", "shape": [16]}
    info["features"]["observation.images.ego_view"] = video_info["features"]["observation.images.ego_view"]
    info["robot_type"] = "robocasa_gr1_aug_aligned"
    info["splits"] = {"train": f"0:{len(source_episodes)}"}
    for name in ("episodes.jsonl", "tasks.jsonl", "embodiment.json"):
        shutil.copy2(source / "meta" / name, output / "meta" / name)
    video_link = output / "videos"
    if video_link.is_symlink():
        if video_link.resolve() != (video_source / "videos").resolve():
            raise ValueError(f"Unexpected video symlink: {video_link}")
    elif video_link.exists():
        raise ValueError(f"Refusing to replace existing video directory: {video_link}")
    else:
        video_link.symlink_to(video_source / "videos", target_is_directory=True)
    write_json(output / "meta/info.json", info)
    write_json(output / "meta/modality.json", modality)
    write_json(output / "meta/stats.json", stats)
    recipe.update(
        {
            "episodes": len(source_episodes),
            "frames": sum(item["length"] for item in source_episodes),
            "parquet_files": parquet_files,
            "stored_state_dimensions_for_canonical": 71,
            "model_state_dimensions": 29,
            "active_action_dimensions": 71,
            "eef_action_semantics": "next-observed grip-site pose; runtime state-relative spatial delta",
            "joint_action_semantics": "original absolute controller commands",
            "auxiliary_fingers": "terminal finger-link origins and rotations, not fingertip targets",
            "fingertip_targets": (
                "geometric distal 1mm mesh-cap centroids; thumb,index,middle,ring,pinky; absolute pseudo-camera xyz"
            ),
            "action_index_shift": 0,
            "max_next_state_error": max_shift_error,
            "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        }
    )
    write_json(marker, recipe)
    return {"task": task, "frames": recipe["frames"], "episodes": recipe["episodes"], "resumed": False}


def main() -> None:
    """Run geometry extraction or the complete resumable data conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO)
    parser.add_argument("--hdf5-root", default=DEFAULT_HDF5)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-python", default=DEFAULT_GEOMETRY_PYTHON)
    parser.add_argument("--mesh-root", required=True, help="Official robosuite assets/grippers/meshes/fourier_hands")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--export-geometry")
    parser.add_argument("--geometry-output")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.export_geometry:
        extract_geometry(
            Path(args.export_geometry), Path(args.hdf5_root), Path(args.geometry_output), Path(args.mesh_root)
        )
        return
    if Path(args.source_root).resolve() == Path(args.output_root).resolve():
        raise ValueError("Output must differ from the source")
    tasks = args.tasks or sorted(path.name for path in Path(args.source_root).glob("gr1_unified.*") if path.is_dir())
    if not tasks:
        raise ValueError(f"No task directories found in {args.source_root}")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_task, task, vars(args)): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    write_json(
        Path(args.output_root) / "conversion_manifest.json",
        {
            "version": VERSION,
            "frame_convention": FRAME_CONVENTION,
            "tasks": sorted(results, key=lambda item: item["task"]),
            "episodes": sum(item["episodes"] for item in results),
            "frames": sum(item["frames"] for item in results),
        },
    )
    if not args.tasks:
        (Path(args.output_root) / "READY").write_text(f"{len(results)} tasks converted; recipe version {VERSION}\n")


if __name__ == "__main__":
    main()
