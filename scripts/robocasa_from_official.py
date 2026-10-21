# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT
"""Official GR00T Teleop-Sim GR1 LeRobot + HDF5 -> EgoForesight aligned data.

The official LeRobot files provide controller/joint ordering and instructions;
matching official HDF5 states/XML provide exact body and camera kinematics.
No private AugPosRot/highres derivative, action playback, object removal, or
learned relabeling is used. This is the GR1 24-task benchmark, not RoboCasa365.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from scripts.robocasa_aug_aligned import (
    EEF_KEYS,
    FINGER_ORDER,
    FRAME_CONVENTION,
    POSE_KEYS,
    array_statistics,
    episodes,
    extract_geometry,
    pack_rotation_columns,
    read_json,
    transform_augmented,
    write_json,
)

JOINT_FIELDS = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}
VIDEO_KEY = "observation.images.ego_view"


def pose_bodies() -> dict[str, str]:
    """World-frame body origins in the original 161-D augmentation order."""
    result = {}
    for side, lower, upper in (("left", "l", "L"), ("right", "r", "R")):
        result[f"wrist_{lower}"] = f"gripper0_{side}_{lower}_palm"
        for finger in FINGER_ORDER:
            suffix = "distal" if finger == "thumb" else "intermediate"
            result[f"{finger}_{lower}"] = f"gripper0_{side}_{upper}_{finger}_{suffix}_link"
    return result


def augmented_fields(modality: dict) -> dict:
    """Validate the official 44-D layout before adding world pose fields."""
    result = json.loads(json.dumps(modality))
    for role, column in (("state", "observation.state"), ("action", "action")):
        for name, (start, end) in JOINT_FIELDS.items():
            meta = result[role].get(name, {})
            if (meta.get("start"), meta.get("end")) != (start, end):
                raise ValueError(f"Not the official GR1 44-D schema: {role}.{name}={meta}")
        offset = 44
        for name in (*pose_bodies(), "camera_egoview"):
            for suffix, width in (("pos", 3), ("rot6d", 6)):
                result[role][f"{name}_{suffix}"] = {
                    "original_key": column,
                    "start": offset,
                    "end": offset + width,
                    "absolute": True,
                }
                offset += width
    return result


def resolve_xml_assets(xml: str, asset_roots: list[Path], path_map: dict[str, str]) -> str:
    """Resolve recorded absolute asset paths without deleting scene contents or guessing basenames."""
    root = ET.fromstring(xml)
    if root.findall(".//include"):
        raise ValueError("Expected compiled episode XML; unresolved <include> is unsupported")
    for element in root.iter():
        value = element.get("file")
        if not value:
            continue
        # An explicit mapping is authoritative. Do not reintroduce unrelated
        # same-suffix assets (e.g. both projects have textures/ceramic.png).
        mappings = [(old, new) for old, new in path_map.items() if value.startswith(old)]
        if mappings:
            old, new = max(mappings, key=lambda pair: len(pair[0]))
            candidate = Path(new + value[len(old) :]).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Mapped asset does not exist: {value} -> {candidate}")
            candidates = {candidate}
        elif Path(value).is_absolute() and Path(value).is_file():
            candidates = {Path(value).resolve()}
        else:
            candidates = set()
            parts = Path(value).parts
            suffixes = [Path(*parts[i + 1 :]) for i, part in enumerate(parts) if part == "assets"]
            if not Path(value).is_absolute():
                suffixes.append(Path(value))
            for assets in asset_roots:
                for suffix in suffixes:
                    candidate = (assets / suffix).resolve()
                    if candidate.is_relative_to(assets) and candidate.is_file():
                        candidates.add(candidate)
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Asset must resolve uniquely: {value}; candidates={sorted(map(str, candidates))}. "
                "Provide the matching official assets or --path-map; no assets are removed."
            )
        element.set("file", str(candidates.pop()))
    for compiler in root.findall("compiler"):
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    return ET.tostring(root, encoding="unicode")


def replay_poses(xml: str, states: np.ndarray, args: argparse.Namespace, video: Path | None) -> np.ndarray:
    """Restore recorded [time,qpos,qvel] and forward only; never apply controls or advance simulation."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(resolve_xml_assets(xml, args.assets_root, args.path_map))
    if model.na or states.ndim != 2 or states.shape[1] != 1 + model.nq + model.nv:
        raise ValueError(
            f"HDF5 state layout differs from robosuite MjSimState: {states.shape}, "
            f"nq={model.nq}, nv={model.nv}, na={model.na}"
        )
    data = mujoco.MjData(model)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in pose_bodies().values()]
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "egoview")
    if min([*bodies, camera]) < 0:
        raise ValueError("Recorded XML lacks GR1 Fourier hand bodies or egoview camera")
    renderer = None
    container = None
    if video is not None:
        import av

        video.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        container = av.open(str(video), mode="w")
        stream = container.add_stream("libx264", rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
        stream.options = {"crf": str(args.crf)}
    output = np.empty((len(states), 117), dtype=np.float64)
    try:
        for index, state in enumerate(states):
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            values = []
            for body in bodies:
                values.extend((data.xpos[body].copy(), pack_rotation_columns(data.xmat[body].reshape(3, 3))))
            values.extend((data.cam_xpos[camera].copy(), pack_rotation_columns(data.cam_xmat[camera].reshape(3, 3))))
            output[index] = np.concatenate(values)
            if renderer is not None:
                renderer.update_scene(data, camera="egoview")
                for packet in stream.encode(av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")):
                    container.mux(packet)
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        if container is not None:
            container.close()
        if renderer is not None:
            renderer.close()
    if not np.isfinite(output).all():
        raise ValueError("Non-finite poses after recorded-state replay")
    return output


def main() -> None:
    """Convert one official task atomically, retaining every episode and controller command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-lerobot", type=Path, required=True, help="One unmodified official GR1 task repository"
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Matching official <task>.hdf5")
    parser.add_argument("--output", type=Path, required=True, help=".../aligned/gr1_unified.<TaskName>")
    parser.add_argument(
        "--assets-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for official robosuite/models/assets and robocasa/models/assets",
    )
    parser.add_argument("--mesh-root", type=Path, required=True, help="Official fourier_hands mesh directory")
    parser.add_argument(
        "--path-map", type=Path, help="Optional JSON mapping recorded path prefixes to local asset roots"
    )
    parser.add_argument(
        "--render-video", action="store_true", help="Re-render egoview, rather than copy official videos"
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()
    source, output = args.official_lerobot.resolve(), args.output.resolve()
    if source == output or output.exists():
        raise FileExistsError(f"Use a new output directory; source is never modified: {output}")
    args.assets_root = [root.expanduser().resolve() for root in args.assets_root]
    args.path_map = read_json(args.path_map) if args.path_map else {}
    if args.height < 2 or args.width < 2 or args.height % 2 or args.width % 2:
        parser.error("Video dimensions must be positive and even")
    info = read_json(source / "meta/info.json")
    args.fps = int(info["fps"])
    if args.fps != 20:
        raise ValueError(f"Expected the GR1 official 20 Hz corpus, got {args.fps}")
    for column in ("observation.state", "action"):
        if list(info["features"][column]["shape"]) != [44]:
            raise ValueError(f"Expected unaugmented official [44] {column}")
    modality = augmented_fields(read_json(source / "meta/modality.json"))
    records = episodes(source / "meta/episodes.jsonl")
    if not records or [r["episode_index"] for r in records] != list(range(len(records))):
        raise ValueError("Official episodes must be nonempty, dense, and in recorded metadata order")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Inspect previous interrupted conversion before retrying: {temporary}")
    (temporary / "meta").mkdir(parents=True)
    geometry_path = temporary / "meta/episode_geometry.json"
    extract_geometry(source, args.hdf5.parent, geometry_path, args.mesh_root, hdf5_file=args.hdf5)
    geometry = read_json(geometry_path)
    blocks = {key: [] for key in ("observation.state", "action")}
    blocks.update({f"{prefix}.{key}": [] for prefix in ("observation.state", "action") for key in POSE_KEYS})
    data_paths = []
    with h5py.File(args.hdf5, "r") as handle:
        for record in tqdm(records, desc="official GR1 -> aligned"):
            index, length = record["episode_index"], record["length"]
            demo = f"demo_{int(record['trajectory_id'].rsplit('-', 1)[1])}"
            group = handle[f"data/{demo}"]
            states = group["states"][:]
            if len(states) != length or len(group["actions"]) != length:
                raise ValueError(f"Official HDF5/LeRobot identity or length mismatch: {demo}")
            fmt = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
            relative = info["data_path"].format(**fmt)
            video_relative = info["video_path"].format(video_key=VIDEO_KEY, **fmt)
            video_target = temporary / video_relative
            xml = group.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            poses = replay_poses(xml, states, args, video_target if args.render_video else None)
            if not args.render_video:
                video_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / video_relative, video_target)
            table = pq.read_table(source / relative)
            if len(table) != length or length < 1:
                raise ValueError(f"Invalid official parquet length: {relative}")
            for role, column in (("state", "observation.state"), ("action", "action")):
                joints = np.asarray(table[column].to_pylist())
                if joints.shape != (length, 44):
                    raise ValueError(f"Invalid official joint/control array: {relative}/{column} {joints.shape}")
                pose = poses if role == "state" else np.concatenate((poses[1:], poses[-1:]), axis=0)
                aligned, extra = transform_augmented(
                    np.concatenate((joints, pose), axis=1), modality[role], geometry[str(index)]
                )
                # The 44-D prefix stays exactly the original controller target/state in float32.
                np.testing.assert_array_equal(aligned[:, :44], joints.astype(np.float32))
                table = table.set_column(
                    table.schema.get_field_index(column), column, pa.array(aligned.tolist(), type=pa.list_(pa.float32()))
                )
                blocks[column].append(aligned)
                for key in POSE_KEYS:
                    name = f"{column}.{key}"
                    table = table.append_column(name, pa.array(extra[key].tolist(), type=pa.list_(pa.float32())))
                    blocks[name].append(extra[key])
                if role == "state":
                    table = table.append_column(
                        "camera_params.head_w2c", pa.array(extra["camera_w2c"].tolist(), type=pa.list_(pa.float32()))
                    )
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.replace_schema_metadata(None), target, compression="zstd")
            data_paths.append(relative)
    for role, prefix in (("state", "observation.state"), ("action", "action")):
        info["features"][prefix] = {"dtype": "float32", "shape": [161]}
        for key in POSE_KEYS:
            name = f"{prefix}.{key}"
            width = 6 if key in EEF_KEYS else 15
            modality[role][key] = {"original_key": name, "start": 0, "end": width, "absolute": True}
            info["features"][name] = {"dtype": "float32", "shape": [width]}
    info["features"]["camera_params.head_w2c"] = {"dtype": "float32", "shape": [16]}
    # Only egoview is consumed by this corpus; do not advertise nonexistent video copies.
    for key in list(info["features"]):
        if info["features"][key]["dtype"] == "video" and key != VIDEO_KEY:
            del info["features"][key]
    modality["video"] = {
        key: value for key, value in modality.get("video", {}).items() if value.get("original_key") == VIDEO_KEY
    }
    if args.render_video:
        info["features"][VIDEO_KEY] = {
            "dtype": "video",
            "shape": [args.height, args.width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": args.fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    info.update(robot_type="robocasa_gr1_aug_aligned", total_videos=len(records), splits={"train": f"0:{len(records)}"})
    for name in ("episodes.jsonl", "tasks.jsonl", "embodiment.json"):
        if (source / "meta" / name).is_file():
            shutil.copy2(source / "meta" / name, temporary / "meta" / name)
    write_json(temporary / "meta/info.json", info)
    write_json(temporary / "meta/modality.json", modality)
    write_json(
        temporary / "meta/stats.json", {key: array_statistics(np.concatenate(values)) for key, values in blocks.items()}
    )
    write_json(
        temporary / "meta/alignment.json",
        {
            "version": 1,
            "source_kind": "official_lerobot_and_hdf5_recorded_state_replay",
            "frame_convention": FRAME_CONVENTION,
            "episodes": len(records),
            "frames": sum(r["length"] for r in records),
            "action_index_shift": 0,
            "eef_action_semantics": "next-observed grip-site pose; terminal pose repeated",
            "joint_action_semantics": "official absolute controller commands, unchanged",
            "parquet_files": data_paths,
            "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
            "mujoco_version": importlib.metadata.version("mujoco"),
            "videos": "recorded-state egoview rendering" if args.render_video else "byte-copied official egoview",
        },
    )
    os.rename(temporary, output)
    print(f"Converted {len(records)} official episodes: {output}")


if __name__ == "__main__":
    main()
