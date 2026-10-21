# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Build camera-aligned RoboTwin LeRobot datasets and mirrored variants.

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

FPS = 50
CHUNK_SIZE = 1000
ROBOTWIN_ALIGNED_PIPELINE_VERSION = 2
ROBOTWIN_VIDEO_COLOR_SPACE = "rgb"
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

# Verified on 2026-08-26 against observation/head_camera/extrinsic_cv for all
# frames of episode0 in all 50 ALOHA tasks, for both clean_50 and randomized_500.
# Every matrix was bit-identical. Convention: p_camera_optical = W2C @ p_world,
# with +x right in the image, +y down, and +z forward into the scene.
ROBOTWIN_HEAD_W2C = np.array(
    [
        [1.0, 0.0, 0.0, 0.03200000524520874],
        [0.0, -0.800000011920929, -0.6000000238418579, 0.44999998807907104],
        [0.0, 0.6000000238418579, -0.800000011920929, 1.350000023841858],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class StaticThresholds:
    translation_m: float = 1e-4
    rotation_rad: float = 1e-3
    gripper: float = 1e-4
    image_mean_abs: float = 0.5


@dataclass
class Episode:
    state: np.ndarray
    action: np.ndarray
    w2c: np.ndarray
    frames: dict[str, list[np.ndarray]]
    prompt: str


def decode_jpeg(value: bytes | np.bytes_) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode RoboTwin JPEG frame")
    # RoboTwin passes semantic RGB arrays directly to cv2.imencode.  OpenCV's
    # matching decoder therefore restores those semantic channels in the
    # returned ndarray already.  Applying BGR->RGB here would swap red/blue a
    # second time (for example, a red block would become blue in LeRobot).
    return np.ascontiguousarray(frame)


def quaternion_wxyz_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    xyzw = quaternion[..., [1, 2, 3, 0]]
    xyzw /= np.linalg.norm(xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(xyzw).as_rotvec().astype(np.float32)


def pose7_wxyz_to_pose6(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    return np.concatenate([pose[..., :3], quaternion_wxyz_to_rotvec(pose[..., 3:7])], axis=-1).astype(np.float32)


def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0)))


def _image_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_small = cv2.resize(left, (40, 30), interpolation=cv2.INTER_AREA).astype(np.float32)
    right_small = cv2.resize(right, (40, 30), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.abs(left_small - right_small)))


def frames_are_static(
    reference: int,
    candidate: int,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    frames: dict[str, list[np.ndarray]],
    thresholds: StaticThresholds,
) -> bool:
    if np.linalg.norm(left_pose[candidate, :3] - left_pose[reference, :3]) > thresholds.translation_m:
        return False
    if np.linalg.norm(right_pose[candidate, :3] - right_pose[reference, :3]) > thresholds.translation_m:
        return False
    if _quaternion_distance(left_pose[candidate, 3:7], left_pose[reference, 3:7]) > thresholds.rotation_rad:
        return False
    if _quaternion_distance(right_pose[candidate, 3:7], right_pose[reference, 3:7]) > thresholds.rotation_rad:
        return False
    if abs(float(left_gripper[candidate] - left_gripper[reference])) > thresholds.gripper:
        return False
    if abs(float(right_gripper[candidate] - right_gripper[reference])) > thresholds.gripper:
        return False
    return all(
        _image_distance(frames[key][reference], frames[key][candidate]) <= thresholds.image_mean_abs
        for key in VIDEO_KEYS
    )


def select_non_static_frames(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    frames: dict[str, list[np.ndarray]],
    thresholds: StaticThresholds,
) -> list[int]:
    """Drop every interior static frame while preserving the terminal static suffix."""
    frame_count = len(left_pose)
    if frame_count < 2:
        return list(range(frame_count))

    pair_is_static = [False]
    pair_is_static.extend(
        frames_are_static(
            index - 1,
            index,
            left_pose,
            right_pose,
            left_gripper,
            right_gripper,
            frames,
            thresholds,
        )
        for index in range(1, frame_count)
    )
    last_motion = max((index for index in range(1, frame_count) if not pair_is_static[index]), default=0)
    terminal_start = last_motion + 1

    keep = [0]
    for index in range(1, terminal_start):
        if not frames_are_static(
            keep[-1],
            index,
            left_pose,
            right_pose,
            left_gripper,
            right_gripper,
            frames,
            thresholds,
        ):
            keep.append(index)
    keep.extend(range(max(terminal_start, keep[-1] + 1), frame_count))
    return keep


def compose_state(
    left_pose6: np.ndarray,
    left_gripper: np.ndarray,
    right_pose6: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    return np.concatenate([left_pose6, left_gripper[..., None], right_pose6, right_gripper[..., None]], axis=-1).astype(
        np.float32
    )


def mirror_pose6_world(pose: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    output = np.empty_like(pose)
    r_cw = np.asarray(w2c[:3, :3], dtype=np.float64)
    t_cw = np.asarray(w2c[:3, 3], dtype=np.float64)
    r_wc = r_cw.T
    reflection = np.diag([-1.0, 1.0, 1.0])
    for index, value in enumerate(pose):
        p_camera = r_cw @ value[:3] + t_cw
        r_camera_eef = r_cw @ Rotation.from_rotvec(value[3:6]).as_matrix()
        p_camera_mirrored = reflection @ p_camera
        r_camera_eef_mirrored = reflection @ r_camera_eef @ reflection
        output[index, :3] = r_wc @ (p_camera_mirrored - t_cw)
        output[index, 3:6] = Rotation.from_matrix(r_wc @ r_camera_eef_mirrored).as_rotvec()
    return output.astype(np.float32)


def mirror_state_action(values: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    left_pose = mirror_pose6_world(values[:, 0:6], w2c)
    right_pose = mirror_pose6_world(values[:, 7:13], w2c)
    return np.concatenate(
        [right_pose, values[:, 13:14], left_pose, values[:, 6:7]],
        axis=-1,
    ).astype(np.float32)


def encode_video(path: Path, frames: Sequence[np.ndarray], fps: int = FPS) -> None:
    if not frames:
        raise ValueError(f"cannot encode empty video {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for image in frames:
            if image.shape != (height, width, 3):
                raise ValueError(f"video frame shape changed in {path}: {image.shape}")
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decode_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _fixed_float_list(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), size)


class LeRobotWriter:
    def __init__(self, output: Path, *, overwrite: bool = False, fps: int = FPS):
        self.output = output
        self.overwrite = overwrite
        self.fps = fps
        self.temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
        if self.temporary.exists():
            shutil.rmtree(self.temporary)
        self.temporary.mkdir(parents=True)
        self.episodes: list[dict] = []
        self.tasks: dict[str, int] = {}
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.global_index = 0
        self.height: int | None = None
        self.width: int | None = None

    def add_episode(self, episode: Episode) -> None:
        episode_index = len(self.episodes)
        length = len(episode.state)
        if length == 0 or len(episode.action) != length:
            raise ValueError(f"invalid episode length state={length}, action={len(episode.action)}")
        if any(len(episode.frames[key]) != length for key in VIDEO_KEYS):
            raise ValueError("video/state lengths do not match")
        if episode.w2c.shape != (4, 4):
            raise ValueError(f"W2C must be 4x4, got {episode.w2c.shape}")

        task_index = self.tasks.setdefault(episode.prompt, len(self.tasks))
        timestamps = np.arange(length, dtype=np.float32) / self.fps
        frame_indices = np.arange(length, dtype=np.int64)
        global_indices = np.arange(self.global_index, self.global_index + length, dtype=np.int64)
        w2c = np.broadcast_to(episode.w2c.reshape(1, 16), (length, 16)).copy()
        table = pa.Table.from_arrays(
            [
                _fixed_float_list(episode.state, 14),
                _fixed_float_list(episode.action, 14),
                _fixed_float_list(w2c, 16),
                pa.array(timestamps, type=pa.float32()),
                pa.array(frame_indices, type=pa.int64()),
                pa.array(np.full(length, episode_index), type=pa.int64()),
                pa.array(global_indices, type=pa.int64()),
                pa.array(np.full(length, task_index), type=pa.int64()),
            ],
            names=[
                "observation.state",
                "action",
                "camera_params.head_w2c",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        chunk = episode_index // CHUNK_SIZE
        data_path = self.temporary / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path, compression="zstd")

        for key in VIDEO_KEYS:
            video_path = self.temporary / f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
            encode_video(video_path, episode.frames[key], self.fps)
        self.height, self.width = episode.frames[VIDEO_KEYS[0]][0].shape[:2]
        self.states.append(episode.state)
        self.actions.append(episode.action)
        self.episodes.append({"episode_index": episode_index, "tasks": [episode.prompt], "length": length})
        self.global_index += length

    def finalize(self) -> Path:
        if not self.episodes or self.height is None or self.width is None:
            raise ValueError("no episodes were written")
        meta = self.temporary / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        with (meta / "episodes.jsonl").open("w") as handle:
            for episode in self.episodes:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        with (meta / "tasks.jsonl").open("w") as handle:
            for task, task_index in sorted(self.tasks.items(), key=lambda item: item[1]):
                handle.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")
        with (meta / "modality.json").open("w") as handle:
            json.dump(_modality_metadata(), handle, indent=2)
        with (meta / "stats.json").open("w") as handle:
            json.dump(
                {
                    "observation.state": _statistics(np.concatenate(self.states)),
                    "action": _statistics(np.concatenate(self.actions)),
                },
                handle,
                indent=2,
            )
        with (meta / "info.json").open("w") as handle:
            json.dump(self._info(), handle, indent=2)

        if self.output.exists():
            if not self.overwrite:
                raise FileExistsError(f"output already exists: {self.output}; pass --overwrite to replace it")
            shutil.rmtree(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.replace(self.output)
        return self.output

    def _info(self) -> dict:
        video_feature = {
            "dtype": "video",
            "shape": [3, self.height, self.width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": self.height,
                "video.width": self.width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": self.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
        vector_names = [
            "left_x",
            "left_y",
            "left_z",
            "left_rx",
            "left_ry",
            "left_rz",
            "left_gripper",
            "right_x",
            "right_y",
            "right_z",
            "right_rx",
            "right_ry",
            "right_rz",
            "right_gripper",
        ]
        features = {
            "observation.state": {"dtype": "float32", "shape": [14], "names": [vector_names]},
            "action": {"dtype": "float32", "shape": [14], "names": [vector_names]},
            "camera_params.head_w2c": {
                "dtype": "float32",
                "shape": [16],
                "names": [[f"w2c_{row}{column}" for row in range(4) for column in range(4)]],
            },
            **{key: video_feature for key in VIDEO_KEYS},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        return {
            "codebase_version": "v2.1",
            "robotwin_aligned_pipeline_version": ROBOTWIN_ALIGNED_PIPELINE_VERSION,
            "video_color_space": ROBOTWIN_VIDEO_COLOR_SPACE,
            "robot_type": "aloha",
            "total_episodes": len(self.episodes),
            "total_frames": self.global_index,
            "total_tasks": len(self.tasks),
            "total_videos": len(self.episodes) * len(VIDEO_KEYS),
            "total_chunks": math.ceil(len(self.episodes) / CHUNK_SIZE),
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }


def _modality_metadata() -> dict:
    pose = {"rotation_type": "axis_angle", "absolute": True, "dtype": "float32"}
    scalar = {"absolute": True, "dtype": "float32"}
    return {
        "state": {
            "left_arm": {"original_key": "observation.state", "start": 0, "end": 6, **pose},
            "left_gripper": {"original_key": "observation.state", "start": 6, "end": 7, **scalar},
            "right_arm": {"original_key": "observation.state", "start": 7, "end": 13, **pose},
            "right_gripper": {"original_key": "observation.state", "start": 13, "end": 14, **scalar},
        },
        "action": {
            "left_arm": {"original_key": "action", "start": 0, "end": 6, **pose},
            "left_gripper": {"original_key": "action", "start": 6, "end": 7, **scalar},
            "right_arm": {"original_key": "action", "start": 7, "end": 13, **pose},
            "right_gripper": {"original_key": "action", "start": 13, "end": 14, **scalar},
        },
        "video": {
            "image_high": {"original_key": VIDEO_KEYS[0]},
            "image_left_wrist": {"original_key": VIDEO_KEYS[1]},
            "image_right_wrist": {"original_key": VIDEO_KEYS[2]},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
