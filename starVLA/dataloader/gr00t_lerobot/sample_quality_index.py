# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add mmap-friendly per-dataset sample quality indexes.

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.canonical_meta import get_canonical_robot_type
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.camera_quality import (
    CAMERA_RELATIVE_POSE_COLUMNS,
    CAMERA_WINDOW_POLICY_VERSION,
    CameraFilterConfig,
)

SAMPLE_QUALITY_INDEX_VERSION = 3
METADATA_FILENAME = "metadata.json"
KEYS_FILENAME = "keys.npy"
DELTA_MASKS_FILENAME = "delta_masks.npy"
PAIRS_FILENAME = "pairs.npy"
_POPCOUNT_TABLE = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def configured_future_deltas(data_cfg) -> list[int]:
    random_range = data_cfg.get("image_pair_random_future_delta_range", None)
    if random_range not in (None, False, "False"):
        low, high = sorted(int(value) for value in random_range)
        if low < 1:
            raise ValueError(f"image_pair_random_future_delta_range min must be >= 1, got {low}")
        deltas = list(range(low, high + 1))
    else:
        fixed_deltas = [int(value) for value in data_cfg.get("image_pair_delta_indices", [0, 1])]
        if len(fixed_deltas) != 2:
            raise ValueError(f"image_pair_delta_indices must contain two values, got {fixed_deltas}")
        deltas = [abs(fixed_deltas[1] - fixed_deltas[0])]
    if len(deltas) > 64:
        raise ValueError(f"Sample quality indexes support at most 64 future deltas, got {len(deltas)}")
    return deltas


def full_delta_mask(num_deltas: int) -> np.uint64:
    if not 1 <= num_deltas <= 64:
        raise ValueError(f"num_deltas must be in [1, 64], got {num_deltas}")
    return np.uint64(np.iinfo(np.uint64).max if num_deltas == 64 else (1 << num_deltas) - 1)


def effective_action_offsets_by_future_delta(dataset, future_deltas: list[int]) -> list[list[int]]:
    """Mirror the runtime action horizon used for each sampled image-pair delta."""

    first_action_key = dataset.modality_keys["action"][0]
    action_deltas = np.asarray(dataset.delta_indices[first_action_key], dtype=np.int64)
    data_cfg = getattr(dataset, "data_cfg", {}) or {}
    random_range = data_cfg.get("image_pair_random_future_delta_range", None)
    truncate_to_pair_delta = random_range not in (None, False, "False", "false", "")
    if not truncate_to_pair_delta:
        full_offsets = [int(value) for value in action_deltas]
        return [list(full_offsets) for _ in future_deltas]

    action_shift = int(data_cfg.get("action_index_shift", 0))
    temporal_stride = int(dataset.temporal_downsample_stride)
    return [
        [int(value) for value in action_deltas[action_deltas - action_shift < int(delta) * temporal_stride]]
        for delta in future_deltas
    ]


def popcount_uint64(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.uint64)
    byte_view = values.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_TABLE[byte_view].sum(axis=1, dtype=np.uint64)


def write_explicit_pairs(
    keys: np.ndarray,
    delta_masks: np.ndarray,
    valid_rows: int,
    future_deltas: list[int],
    output_path: Path,
    chunk_size: int = 100_000,
) -> int:
    """Expand allowed pairs into one mmap table for O(1), single-read sampling."""
    total_pairs = 0
    count_chunk_size = max(chunk_size, 4_000_000)
    for start in range(0, valid_rows, count_chunk_size):
        end = min(valid_rows, start + count_chunk_size)
        total_pairs += int(popcount_uint64(delta_masks[start:end]).sum(dtype=np.uint64))
    if total_pairs <= 0:
        raise ValueError("Cannot write a sample quality index with no valid pairs")

    pairs = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int64, shape=(total_pairs, 3))
    bit_values = np.arange(len(future_deltas), dtype=np.uint64)
    future_delta_array = np.asarray(future_deltas, dtype=np.int64)
    write_offset = 0
    for start in range(0, valid_rows, chunk_size):
        end = min(valid_rows, start + chunk_size)
        masks = np.asarray(delta_masks[start:end], dtype=np.uint64)
        allowed = ((masks[:, None] >> bit_values[None, :]) & np.uint64(1)).astype(bool)
        row_offsets, delta_bits = np.nonzero(allowed)
        pair_count = len(row_offsets)
        pairs[write_offset : write_offset + pair_count, :2] = keys[start + row_offsets]
        pairs[write_offset : write_offset + pair_count, 2] = future_delta_array[delta_bits]
        write_offset += pair_count
    if write_offset != total_pairs:
        raise RuntimeError(f"Pair expansion wrote {write_offset} rows, expected {total_pairs}")
    pairs.flush()
    return total_pairs


def _file_signature(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"path": path.as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _resolved_path_value(value: object) -> str:
    """Resolve symlinked mount aliases before comparing cached index metadata."""
    return Path(str(value)).expanduser().resolve().as_posix()


def _normalized_metadata_signature(signature: dict) -> dict:
    normalized = dict(signature)
    if normalized.get("path"):
        normalized["path"] = _resolved_path_value(normalized["path"])
    return normalized


def _info_semantic_signature(path: Path) -> dict | None:
    """Hash only info.json fields that can change quality-index membership."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    relevant = {
        key: info.get(key)
        for key in (
            "codebase_version",
            "total_episodes",
            "total_frames",
            "total_videos",
            "total_chunks",
            "chunks_size",
            "fps",
            "data_path",
            "video_path",
        )
    }
    relevant["video_features"] = {
        key: value
        for key, value in (info.get("features") or {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    }
    relevant["camera_features"] = {
        key: value
        for key, value in (info.get("features") or {}).items()
        if key in CAMERA_RELATIVE_POSE_COLUMNS or key == "camera_params.head_extrinsic"
    }
    relevant["timestamp_feature"] = (info.get("features") or {}).get("timestamp")
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"path": path.as_posix(), "semantic_sha256": digest}


def _non_info_metadata_signatures(spec: dict, *, ignore_modality: bool = False) -> list[dict]:
    return [
        _normalized_metadata_signature(signature)
        for signature in spec.get("metadata_signatures", [])
        if Path(str(signature.get("path", ""))).name != "info.json"
        and not (ignore_modality and Path(str(signature.get("path", ""))).name == "modality.json")
    ]


def _info_metadata_signature(spec: dict) -> dict | None:
    signature = next(
        (
            signature
            for signature in spec.get("metadata_signatures", [])
            if Path(str(signature.get("path", ""))).name == "info.json"
        ),
        None,
    )
    return None if signature is None else _normalized_metadata_signature(signature)


def _camera_filter_requires_info_signature(spec: dict) -> bool:
    return bool(spec.get("camera_trajectory_filter", {}).get("enabled", False))


def _index_specs_are_compatible(cached: dict, expected: dict) -> bool:
    """Allow legacy indexes when only non-semantic info.json metadata changed."""
    ignored_keys = {"fingerprint", "metadata_signatures", "_schema_alias"}
    for key, expected_value in expected.items():
        if key in ignored_keys:
            continue
        if key == "dataset_path":
            if _resolved_path_value(cached.get(key)) != _resolved_path_value(expected_value):
                return False
            continue
        if cached.get(key) != expected_value:
            return False
    if _camera_filter_requires_info_signature(expected):
        if _info_metadata_signature(cached) != _info_metadata_signature(expected):
            return False
    ignore_modality = bool(expected.get("_schema_alias"))
    return _non_info_metadata_signatures(cached, ignore_modality=ignore_modality) == _non_info_metadata_signatures(
        expected,
        ignore_modality=ignore_modality,
    )


def _index_spec_supports_fixed_delta(cached: dict, expected: dict) -> bool:
    """Return whether a multi-delta cache can serve one configured fixed delta."""
    cached_deltas = [int(value) for value in cached.get("future_deltas", [])]
    expected_deltas = [int(value) for value in expected.get("future_deltas", [])]
    if len(expected_deltas) != 1 or expected_deltas[0] not in cached_deltas:
        return False

    ignored_keys = {"fingerprint", "metadata_signatures", "future_deltas", "_schema_alias"}
    for key, expected_value in expected.items():
        if key in ignored_keys:
            continue
        if key == "dataset_path":
            if _resolved_path_value(cached.get(key)) != _resolved_path_value(expected_value):
                return False
            continue
        if cached.get(key) != expected_value:
            return False
    if _camera_filter_requires_info_signature(expected):
        if _info_metadata_signature(cached) != _info_metadata_signature(expected):
            return False
    ignore_modality = bool(expected.get("_schema_alias"))
    return _non_info_metadata_signatures(cached, ignore_modality=ignore_modality) == _non_info_metadata_signatures(
        expected,
        ignore_modality=ignore_modality,
    )


def build_index_spec(dataset, data_cfg, *, kind: str) -> dict:
    dataset_path = Path(dataset.dataset_path).resolve()
    runtime_robot_type = str(dataset.robot_type)
    quality_index_robot_type = get_canonical_robot_type(runtime_robot_type)
    future_deltas = configured_future_deltas(data_cfg)
    zero_action_enabled = kind == "lerobot" and bool(data_cfg.get("dataset_quality_index_filter_zero_action", True))
    source_robot_types = data_cfg.get("dataset_quality_index_zero_action_source_robot_types", [])
    if isinstance(source_robot_types, str):
        source_robot_types = [source_robot_types]
    zero_action_mode = "source_column" if quality_index_robot_type in source_robot_types else "raw_action"

    quality_filter = getattr(dataset, "quality_filter", None)
    quality_signature = None
    if quality_filter is not None:
        source_path = quality_filter.index.source_path
        quality_signature = {
            "dataset_id": quality_filter.binding.dataset_id,
            "video_key": quality_filter.binding.video_key,
            "tier_weights": list(quality_filter.tier_weights),
            "source": None if source_path is None else _file_signature(Path(source_path)),
        }

    camera_filter_enabled = (
        kind == "lerobot"
        and data_cfg.get("dataset_quality_index_filter_camera_trajectory", True)
        not in (False, "False", "false", None, "", 0)
        and bool(getattr(dataset, "_camera_motion_required", False))
    )
    if camera_filter_enabled:
        configured_camera_filter = data_cfg.get("dataset_quality_index_camera_filter", {}) or {}
        try:
            camera_filter_config = CameraFilterConfig(**dict(configured_camera_filter))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dataset_quality_index_camera_filter must contain only valid CameraFilterConfig fields"
            ) from exc
        camera_trajectory_filter = {
            "enabled": True,
            "layout": "relative_xyz_quaternion_xyzw_7d",
            "columns": sorted(CAMERA_RELATIVE_POSE_COLUMNS),
            "window_policy_version": CAMERA_WINDOW_POLICY_VERSION,
            "dual_track_policy": "state_reference_to_next_action_camera_then_action_track",
            "dual_track_timestamp_policy": "next_frame_timestamp_skip_unavailable_tail",
            "head_extrinsic_policy": "bypass_12d_and_keep_runtime_zero_sentinel",
            "source_revision": data_cfg.get("dataset_quality_index_camera_source_revision", None),
            "config": camera_filter_config.to_dict(),
        }
    else:
        camera_trajectory_filter = {"enabled": False}

    metadata_signatures = []
    info_signature = _info_semantic_signature(dataset_path / "meta/info.json")
    if info_signature is not None:
        metadata_signatures.append(info_signature)
    for relative_path in (
        "meta/modality.json",
        "meta/episodes.jsonl",
        "meta/steps_data_index.pkl",
        f"meta/steps_data_index_{getattr(dataset, 'base_step_stride', 1)}.pkl",
    ):
        signature = _file_signature(dataset_path / relative_path)
        if signature is not None and signature not in metadata_signatures:
            metadata_signatures.append(signature)

    if kind == "lerobot":
        input_count = len(dataset.all_steps)
        video_keys = [key.replace("video.", "", 1) for key in dataset.modality_keys["video"]]
        temporal_stride = int(dataset.temporal_downsample_stride)
        action_index_shift = int(dataset.data_cfg.get("action_index_shift", 0))
        effective_action_offsets = effective_action_offsets_by_future_delta(dataset, future_deltas)
        first_state_key = dataset.modality_keys["state"][0]
        state_deltas = np.asarray(dataset.delta_indices[first_state_key], dtype=np.int64)
        if state_deltas.size == 0:
            raise ValueError("Sample quality indexes need at least one state delta index")
        state_reference_offset = int(state_deltas[0])
    elif kind == "video_only":
        input_count = int(dataset.num_videos)
        video_keys = [str(getattr(dataset, "_video_key", "video"))]
        temporal_stride = 1
        action_index_shift = 0
        effective_action_offsets = []
        state_reference_offset = None
    else:
        raise ValueError(f"Unsupported sample quality index kind: {kind}")

    payload = {
        "version": SAMPLE_QUALITY_INDEX_VERSION,
        "kind": kind,
        "dataset_path": dataset_path.as_posix(),
        "dataset_name": dataset.dataset_name,
        # Complementary-supervision schemas deliberately reuse the Stage-2
        # sample set; schema differences are represented by the action mask.
        "robot_type": quality_index_robot_type,
        "input_count": int(input_count),
        "base_step_stride": int(getattr(dataset, "base_step_stride", 1)),
        "temporal_downsample_stride": temporal_stride,
        "action_index_shift": action_index_shift,
        "state_reference_offset": state_reference_offset,
        "effective_action_offsets": effective_action_offsets,
        "future_deltas": future_deltas,
        "video_keys": video_keys,
        "check_required_videos": bool(data_cfg.get("dataset_quality_index_check_videos", True)),
        "zero_action_enabled": zero_action_enabled,
        "zero_action_mode": zero_action_mode,
        "zero_action_source_column": str(data_cfg.get("dataset_quality_index_zero_action_source_column", "is_outlier")),
        "max_zero_action_frames": int(data_cfg.get("max_zero_action_frames", 8)),
        "camera_trajectory_filter": camera_trajectory_filter,
        "quality_filter": quality_signature,
        "metadata_signatures": metadata_signatures,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :20
    ]
    payload["fingerprint"] = fingerprint
    if quality_index_robot_type != runtime_robot_type:
        # This marker is intentionally outside the fingerprint. Compatibility
        # may ignore modality.json because the alias only adds supervised
        # columns and must preserve Stage-2 quality-filter membership.
        payload["_schema_alias"] = runtime_robot_type
    return payload


def index_directory(data_cfg, spec: dict) -> Path:
    cache_root_value = data_cfg.get("dataset_quality_index_cache_dir", None)
    if not cache_root_value:
        raise ValueError("dataset_quality_index_cache_dir is required when dataset quality indexes are enabled")
    cache_root = Path(str(cache_root_value)).expanduser().resolve()
    try:
        cache_root.relative_to("/mnt")
    except ValueError as exc:
        raise ValueError(f"dataset_quality_index_cache_dir must be under /mnt, got {cache_root}") from exc
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(spec["dataset_name"]))
    return cache_root / f"{safe_name}-{spec['fingerprint']}"


@dataclass
class DatasetSampleQualityIndex:
    directory: Path
    metadata: dict
    keys: np.ndarray
    delta_masks: np.ndarray
    pairs: np.ndarray
    fixed_delta: int | None = None
    fixed_delta_bit: int | None = None
    fixed_sampling_length: int | None = None

    @classmethod
    def load(cls, directory: Path, expected_spec: dict) -> "DatasetSampleQualityIndex":
        metadata_path = directory / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset quality index metadata does not exist: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        fixed_delta = None
        fixed_delta_bit = None
        if metadata.get("fingerprint") != expected_spec["fingerprint"]:
            if _index_spec_supports_fixed_delta(metadata, expected_spec):
                fixed_delta = int(expected_spec["future_deltas"][0])
                fixed_delta_bit = 1 << [int(value) for value in metadata["future_deltas"]].index(fixed_delta)
            elif not _index_specs_are_compatible(metadata, expected_spec):
                raise ValueError(f"Stale dataset quality index fingerprint in {metadata_path}")

        valid_rows = int(metadata["valid_rows"])
        total_pairs = int(metadata["total_pairs"])
        keys = np.load(directory / KEYS_FILENAME, mmap_mode="r", allow_pickle=False)[:valid_rows]
        delta_masks = np.load(directory / DELTA_MASKS_FILENAME, mmap_mode="r", allow_pickle=False)[:valid_rows]
        pairs = np.load(directory / PAIRS_FILENAME, mmap_mode="r", allow_pickle=False)
        if keys.shape != (valid_rows, 2) or delta_masks.shape != (valid_rows,):
            raise ValueError(
                f"Invalid dataset quality index shapes in {directory}: keys={keys.shape}, masks={delta_masks.shape}"
            )
        if pairs.shape != (total_pairs, 3) or pairs.dtype != np.int64:
            raise ValueError(f"Invalid explicit pair table in {directory}: shape={pairs.shape}, dtype={pairs.dtype}")
        if valid_rows <= 0 or total_pairs <= 0:
            raise ValueError(f"Dataset quality index contains no valid samples: {directory}")
        fixed_sampling_length = None
        if fixed_delta_bit is not None:
            fixed_sampling_length = 0
            for start in range(0, valid_rows, 4_000_000):
                masks = np.asarray(delta_masks[start : start + 4_000_000], dtype=np.uint64)
                fixed_sampling_length += int(np.count_nonzero(masks & np.uint64(fixed_delta_bit)))
            if fixed_sampling_length <= 0:
                raise ValueError(f"Dataset quality index {directory} has no valid samples for fixed delta {fixed_delta}")
        return cls(
            directory,
            metadata,
            keys,
            delta_masks,
            pairs,
            fixed_delta=fixed_delta,
            fixed_delta_bit=fixed_delta_bit,
            fixed_sampling_length=fixed_sampling_length,
        )

    @property
    def future_deltas(self) -> tuple[int, ...]:
        if self.fixed_delta is not None:
            return (self.fixed_delta,)
        return tuple(int(value) for value in self.metadata["future_deltas"])

    @property
    def sampling_length(self) -> int:
        if self.fixed_sampling_length is not None:
            return self.fixed_sampling_length
        return max(1, int(round(int(self.metadata["total_pairs"]) / len(self.future_deltas))))

    def sample(self, rng: np.random.Generator) -> tuple[int, int, int]:
        if self.fixed_delta is not None and self.fixed_delta_bit is not None:
            while True:
                row = int(rng.integers(0, len(self.keys), dtype=np.int64))
                if int(self.delta_masks[row]) & self.fixed_delta_bit:
                    return int(self.keys[row, 0]), int(self.keys[row, 1]), self.fixed_delta
        total_pairs = int(self.metadata["total_pairs"])
        pair = self.pairs[int(rng.integers(0, total_pairs, dtype=np.int64))]
        return int(pair[0]), int(pair[1]), int(pair[2])


def load_dataset_quality_index(dataset, data_cfg, *, kind: str) -> DatasetSampleQualityIndex | None:
    if data_cfg.get("dataset_quality_index_enabled", False) in (False, "False", None):
        return None
    spec = build_index_spec(dataset, data_cfg, kind=kind)
    directory = index_directory(data_cfg, spec)
    if not directory.exists():
        prefix = directory.name.rsplit("-", 1)[0]
        candidates = sorted(
            directory.parent.glob(f"{prefix}-*"),
            key=lambda path: (
                (path / METADATA_FILENAME).stat().st_mtime_ns if (path / METADATA_FILENAME).exists() else -1
            ),
            reverse=True,
        )
        for candidate in candidates:
            try:
                compatible_index = DatasetSampleQualityIndex.load(candidate, spec)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
            print(
                f"Reusing compatible per-dataset quality index for {dataset.dataset_name}: "
                f"{candidate} (expected fingerprint {spec['fingerprint']})"
            )
            return compatible_index
        if data_cfg.get("dataset_quality_index_required", True) not in (False, "False", None):
            raise FileNotFoundError(
                f"Required per-dataset quality index does not exist: {directory}. "
                "Run scripts/build_dataset_quality_indexes.py before training."
            )
        print(f"Warning: per-dataset quality index is missing for {dataset.dataset_name}: {directory}")
        return None
    return DatasetSampleQualityIndex.load(directory, spec)
