# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Read native pick-fruit LeRobot v3 data with explicit torso-frame EEF supervision.

"""Small, cached validation dataset with disjoint episode splits and canonical masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.gr00t_lerobot.state_action.sharpa import (
    HARDWARE_TO_CANONICAL_SHARPA_INDICES,
    SHARPA_HARDWARE_JOINT_NAMES,
)
from starVLA.dataloader.r1pro_transforms import denormalize_values, letterbox_rgb, normalize_values

ACTION_SLOTS = np.array([*range(7, 13), *range(36, 42), *range(80, 102), *range(105, 127)])
STATE_SLOTS = np.array([*range(0, 7), *range(29, 36), *range(80, 102), *range(105, 127)])
HAND_NAMES = tuple(SHARPA_HARDWARE_JOINT_NAMES[i] for i in HARDWARE_TO_CANONICAL_SHARPA_INDICES)
ACTION_NAMES = [f"{side}_{name}" for side in ("left", "right") for name in ("x", "y", "z", "rx", "ry", "rz")]
ACTION_NAMES += [f"{side}_{name}" for side in ("left", "right") for name in HAND_NAMES]
META = {
    "embodiment_tag": 31,
    "end_effector_type": "dexterous_hand",
    "dataset_id": "r1pro_pick_fruit",
    # Historical hashed checkpoint identity, not a bundled/private dataset dependency.
    "dataset_family": "dexterous_internal",
    "robot_family": "galaxea",
    "robot_type": "r1pro_sharpa_vive_manus",
    "control_mode": "ee",
    "action_representation": "absolute",
    "eef_frame": "torso_link4",
}


def _hand_indices(names: list[str], side: str) -> list[int]:
    expected = [f"sharpa_{side}.{name}" for name in HAND_NAMES]
    cleaned = [name.replace(f"sharpa_{side}.{side}_", f"sharpa_{side}.") for name in names]
    return [cleaned.index(name) for name in expected]


def prepare_subset(source: str, destination: str, validation_episodes: list[int]) -> dict[str, Any]:
    """Cache correctly ordered raw vectors and the three-view RGB canvas."""
    import av
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text())
    table = pq.read_table(sorted((source / "data").glob("chunk-*/*.parquet"))).sort_by("index")
    episodes = (
        pq.read_table(sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))).sort_by("episode_index").to_pylist()
    )
    raw_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    raw_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_ids = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    count = len(table)
    assert raw_action.shape == raw_state.shape == (count, 58)
    assert count == info["total_frames"] and info["fps"] == 10
    assert np.isfinite(raw_action).all() and np.isfinite(raw_state).all()
    action_names, state_names = (info["features"][key]["names"] for key in ("action", "observation.state"))
    expected_eef = [
        f"r1_{side}_eef.{name}" for side in ("left", "right") for name in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
    ]
    assert action_names[:14] == expected_eef
    actions = np.zeros((count, 130), dtype=np.float32)
    states = np.zeros((count, 130), dtype=np.float32)
    quaternion_errors = []
    for side, source_offset, eef_slot, arm_slot, hand_slot in (("left", 0, 7, 0, 80), ("right", 7, 36, 29, 105)):
        quat = raw_action[:, source_offset + 3 : source_offset + 7]
        quaternion_errors.append(float(np.max(np.abs(np.linalg.norm(quat, axis=1) - 1))))
        assert quaternion_errors[-1] < 1e-4
        actions[:, eef_slot : eef_slot + 3] = raw_action[:, source_offset : source_offset + 3]
        actions[:, eef_slot + 3 : eef_slot + 6] = Rotation.from_quat(quat).as_rotvec()
        actions[:, hand_slot : hand_slot + 22] = raw_action[:, _hand_indices(action_names, side)]
        arm_indices = [state_names.index(f"r1_{side}_arm_joint_{j}") for j in range(7)]
        states[:, arm_slot : arm_slot + 7] = raw_state[:, arm_indices]
        states[:, hand_slot : hand_slot + 22] = raw_state[:, _hand_indices(state_names, side)]
        recovered = Rotation.from_rotvec(actions[:, eef_slot + 3 : eef_slot + 6])
        assert np.max((recovered.inv() * Rotation.from_quat(quat)).magnitude()) < 1e-6
    np.save(destination / "actions.npy", actions)
    np.save(destination / "states.npy", states)
    np.save(destination / "episode_ids.npy", episode_ids)
    np.save(destination / "frame_ids.npy", frame_ids)
    canvas = np.lib.format.open_memmap(destination / "images.npy", mode="w+", dtype=np.uint8, shape=(count, 384, 320, 3))
    canvas[:] = 0
    video_checks = {}
    for name, y, x, height, width in (
        ("head", 0, 0, 256, 320),
        ("wrist_left", 256, 0, 128, 160),
        ("wrist_right", 256, 160, 128, 160),
    ):
        key = f"observation.images.{name}"
        offset = 0
        files = sorted((source / "videos" / key).glob("chunk-*/*.mp4"))
        file_offsets = {}
        for video_path in files:
            chunk_index = int(video_path.parent.name.split("-")[-1])
            file_index = int(video_path.stem.split("-")[-1])
            file_offsets[(chunk_index, file_index)] = offset
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 2
                assert abs(float(stream.average_rate) - 10) < 1e-6
                for frame in container.decode(stream):
                    if offset >= count:
                        raise ValueError(f"Too many frames in {video_path}")
                    rgb = frame.to_ndarray(format="rgb24")
                    canvas[offset, y : y + height, x : x + width] = letterbox_rgb(rgb, height, width)
                    offset += 1
            print(f"[prepare] {source.name}/{name}/{video_path.name}: {offset}/{count}", flush=True)
        assert offset == count, (source.name, name, offset, count)
        for ep in episodes:
            prefix = f"videos/{key}"
            video_offset = file_offsets[(ep[f"{prefix}/chunk_index"], ep[f"{prefix}/file_index"])]
            assert video_offset + round(ep[f"{prefix}/from_timestamp"] * 10) == ep["dataset_from_index"]
            assert round((ep[f"{prefix}/to_timestamp"] - ep[f"{prefix}/from_timestamp"]) * 10) == ep["length"]
        video_checks[name] = {"frames": offset, "files": len(files), "episode_timestamps_verified": True}
    canvas.flush()
    manifest_episodes = []
    held_out = set(validation_episodes)
    if not held_out <= set(episode_ids.tolist()):
        raise ValueError(f"Unknown validation episode IDs in {source}: {sorted(held_out)}")
    for ep in episodes:
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        assert np.all(episode_ids[start:end] == ep["episode_index"])
        assert np.array_equal(frame_ids[start:end], np.arange(end - start))
        manifest_episodes.append(
            {
                "episode_index": int(ep["episode_index"]),
                "start": start,
                "end": end,
                "task": ep["tasks"][0],
                "split": "validation" if ep["episode_index"] in held_out else "train",
            }
        )
    result = {
        "name": source.name,
        "source": str(source),
        "frames": count,
        "fps": 10,
        "episodes": manifest_episodes,
        "video_checks": video_checks,
        "max_quaternion_norm_errors": quaternion_errors,
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def calculate_statistics(cache_dir: Path, subsets: list[dict]) -> dict:
    """Fit normalization exclusively to training episodes, including constant dimensions."""
    result = {}
    for field, filename, slots in (("action", "actions.npy", ACTION_SLOTS), ("state", "states.npy", STATE_SLOTS)):
        samples = []
        for subset in subsets:
            values = np.load(cache_dir / subset["name"] / filename, mmap_mode="r")
            samples.extend(values[ep["start"] : ep["end"]] for ep in subset["episodes"] if ep["split"] == "train")
        values = np.concatenate(samples)
        low, high = np.quantile(values, [0.01, 0.99], axis=0)
        mask = np.zeros(130, dtype=bool)
        mask[slots] = True
        scale = np.maximum((high - low) / 2, 1e-4)
        result[field] = {
            "q01": low.tolist(),
            "q99": high.tolist(),
            "center": ((high + low) / 2).tolist(),
            "scale": scale.tolist(),
            "mask": mask.tolist(),
            "count": len(values),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
        }
    return result


class PickFruitDataset(Dataset):
    """Return causal RGB/state inputs and absolute future controller targets."""

    def __init__(self, cache_dir: str | Path, chunk_size: int = 16, split: str = "train") -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"Unknown pick-fruit split: {split}")
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.statistics = json.loads((self.cache_dir / "statistics.json").read_text())
        self.chunk_size = chunk_size
        self.subsets = []
        self.examples = []
        self.selected_episodes = []
        for subset in self.manifest["subsets"]:
            path = self.cache_dir / subset["name"]
            data = {
                key: np.load(path / filename, mmap_mode="r")
                for key, filename in (("images", "images.npy"), ("actions", "actions.npy"), ("states", "states.npy"))
            }
            subset_index = len(self.subsets)
            self.subsets.append(data)
            for ep in subset["episodes"]:
                if split != "all" and ep["split"] != split:
                    continue
                self.selected_episodes.append({**ep, "subset_index": subset_index, "dataset_name": subset["name"]})
                self.examples.extend(
                    (subset_index, frame, ep["end"], ep["task"]) for frame in range(ep["start"], ep["end"])
                )

    def __len__(self) -> int:
        return len(self.examples)

    def normalize(self, values: np.ndarray, field: str) -> np.ndarray:
        return normalize_values(values, self.statistics[field])

    def denormalize_actions(self, values: np.ndarray) -> np.ndarray:
        return denormalize_values(values, self.statistics["action"])

    def make_example(self, subset_index: int, frame: int, end: int, task: str, training: bool = True) -> dict:
        """Construct one sample; inference does not expose future RGB or action labels."""
        data = self.subsets[subset_index]
        indices = np.minimum(frame + np.arange(self.chunk_size), end - 1)
        result = {
            "lang": f"Task: {task}. Control Mode: absolute ee. Embodiment: galaxea r1pro sharpa.",
            "state": self.normalize(data["states"][frame : frame + 1], "state"),
            "state_mask": np.asarray(self.statistics["state"]["mask"], dtype=bool)[None],
            "meta": {**META, "dataset_id": self.manifest.get("dataset_id", META["dataset_id"])},
        }
        if training:
            result["image_pair"] = np.asarray(data["images"][[frame, min(frame + self.chunk_size, end - 1)]])[:, None]
            result["action"] = self.normalize(data["actions"][indices], "action")
            valid = frame + np.arange(self.chunk_size) < end
            result["action_mask"] = valid[:, None] & np.asarray(self.statistics["action"]["mask"], dtype=bool)[None]
        else:
            result["image"] = np.asarray(data["images"][frame]).copy()
            result["action_schema_mask"] = np.broadcast_to(
                np.asarray(self.statistics["action"]["mask"], dtype=bool), (self.chunk_size, 130)
            ).copy()
        return result

    def __getitem__(self, index: int) -> dict:
        return self.make_example(*self.examples[index])


def collate_examples(examples: list[dict]) -> list[dict]:
    """Preserve the framework's list-of-examples input contract."""
    return examples


def load_pick_fruit_dataset(cache_dir: str | Path, chunk_size: int = 16, split: str = "train") -> PickFruitDataset:
    """Dispatch by explicit cache format, preserving legacy absolute validation."""
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    if manifest["format"] == "pick_fruit_camera_spatial_v1":
        from starVLA.dataloader.pick_fruit_camera_dataset import PickFruitCameraDataset

        return PickFruitCameraDataset(cache_dir, chunk_size, split)
    if manifest["format"] != "pick_fruit_validation_v1":
        raise ValueError(f"Unsupported pick-fruit cache format: {manifest['format']}")
    return PickFruitDataset(cache_dir, chunk_size, split)


def build_pick_fruit_dataloader(cfg: Any) -> DataLoader:
    """Build the validation-run loader with reproducible shuffled training frames."""
    data = cfg.datasets.vla_data
    dataset = load_pick_fruit_dataset(data.cache_dir, int(data.action_horizon), "train")
    jitter = data.get("image_pair_color_jitter", {})
    if hasattr(dataset, "color_jitter") and jitter.get("enabled", False):
        from torchvision.transforms import ColorJitter

        dataset.color_jitter = ColorJitter(
            **{key: float(jitter.get(key, 0)) for key in ("brightness", "contrast", "saturation", "hue")}
        )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "dataset_statistics.json").write_text(
            json.dumps({dataset.manifest.get("dataset_id", "r1pro_pick_fruit"): dataset.statistics}, indent=2)
        )
        (output / "dataset_manifest.json").write_text(json.dumps(dataset.manifest, indent=2))
    workers = int(data.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(data.per_device_batch_size),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_examples,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers else None,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )
