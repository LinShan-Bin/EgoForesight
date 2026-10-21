#!/usr/bin/env python
# EgoForesight modifications Copyright 2026 Chubin Zhang.
"""
Calculate dataset statistics for LeRobot datasets.
Note: Please update the `gr00t/configs/data/embodiment_configs.py` file with the correct modality configurations for the dataset you are using before running this script.

Usage:
    python gr00t/data/stats.py <dataset_path> <embodiment_tag>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations from `gr00t/configs/data/embodiment_configs.py`.
"""

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.embodiment_configs import MODALITY_CONFIGS
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.lerobot_episode_loader import (
    LeRobotEpisodeLoader,
)
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.types import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.utils import (
    to_json_serializable,
)
from starVLA.dataloader.gr00t_lerobot.state_action.action_chunking import (
    EndEffectorActionChunk,
    JointActionChunk,
)
from starVLA.dataloader.gr00t_lerobot.state_action.batch_ops import (
    compute_camera_spatial_xyz_rotvec_chunks,
    compute_relative_xyz_rotvec_chunks,
)
from starVLA.dataloader.gr00t_lerobot.state_action.camera_motion import (
    CAMERA_MOTION_CONVENTION_VERSION,
    CAMERA_POSE_COLUMNS,
    extract_camera_world_transforms,
    extract_robot_base_reference_camera_transforms,
)
from starVLA.dataloader.gr00t_lerobot.state_action.pose import (
    EndEffectorPose,
    JointPose,
)
from starVLA.dataloader.gr00t_lerobot.transform.rotation_align import (
    unwrap_rotvec_sequence,
)

LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
# Legacy filename kept for backward-compat migration only. New runs write to
# the versioned filename returned by _versioned_stats_filename() so different
# (action_horizon, reference_index, action_format) configurations coexist
# without silently overwriting each other.
LEGACY_REL_STATS_FILENAME = "meta/relative_stats.json"
LE_ROBOT_REL_STATS_FILENAME = LEGACY_REL_STATS_FILENAME  # exported for backwards compatibility
REL_STATS_META_VERSION = 3
REL_STATS_MODE = "flatten_horizon"
REL_STATS_STAT_NAMES = ("max", "min", "q01", "q99", "mean", "std")
REL_STATS_QUANTILE_SAMPLE_SIZE = 2_000_000


@contextmanager
def _exclusive_stats_lock(stats_path: Path):
    """Serialize generation of one shared statistics cache across ranks and nodes."""
    lock_path = stats_path.with_name(f".{stats_path.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _atomic_json_dump(payload: dict, output_path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp.json",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(to_json_serializable(payload), handle, indent=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fmt_slug(action_format: "ActionFormat") -> str:
    """action_format enum -> filename-safe slug."""
    v = action_format.value if hasattr(action_format, "value") else str(action_format)
    return v.replace("+", "-")


def _delta_slug(action_delta_indices: list[int]) -> str:
    if not action_delta_indices:
        return "dempty"
    if len(action_delta_indices) == 1:
        return f"d{action_delta_indices[0]}"
    diffs = np.diff(action_delta_indices)
    if np.all(diffs == diffs[0]):
        return f"d{action_delta_indices[0]}-{action_delta_indices[-1]}s{int(diffs[0])}"
    return "d" + "-".join(str(int(i)) for i in action_delta_indices)


def _versioned_stats_filename(
    action_horizon: int,
    reference_index: int,
    action_format: "ActionFormat",
    action_delta_indices: list[int],
    camera_motion_compensation: bool = False,
    relative_eef_frame: str = "eef",
    static_camera_frame: bool = False,
) -> str:
    """meta/relative_action_stats__H{H}_ref{r}_{fmt}_{deltas}[_camcomp-vN].json"""
    camera_slug = f"_camcomp-v{CAMERA_MOTION_CONVENTION_VERSION}" if camera_motion_compensation else ""
    frame_slug = "_camera-spatial" if relative_eef_frame == "camera_spatial" else ""
    static_camera_slug = "_static-camera" if static_camera_frame else ""
    return (
        f"meta/relative_action_stats__H{action_horizon}"
        f"_ref{reference_index}_{_fmt_slug(action_format)}"
        f"_{_delta_slug(action_delta_indices)}{frame_slug}{static_camera_slug}{camera_slug}.json"
    )


def _build_meta_signature(
    action_horizon,
    reference_index,
    action_format,
    d_per_key,
    action_delta_indices,
    rotvec_continuity,
    camera_motion_compensation,
    relative_eef_frame="eef",
    static_camera_frame=False,
):
    signature = {
        "version": REL_STATS_META_VERSION,
        "stats_mode": REL_STATS_MODE,
        "action_horizon": int(action_horizon),
        "action_delta_indices": [int(i) for i in action_delta_indices],
        "reference_index": int(reference_index),
        "action_format": action_format.value if hasattr(action_format, "value") else str(action_format),
        "d_per_key": {k: int(v) for k, v in d_per_key.items()},
        "rotvec_continuity": bool(rotvec_continuity),
        "camera_motion_compensation": bool(camera_motion_compensation),
        "relative_eef_frame": str(relative_eef_frame),
        "static_camera_frame": bool(static_camera_frame),
    }
    if camera_motion_compensation:
        signature["camera_motion_convention_version"] = CAMERA_MOTION_CONVENTION_VERSION
    return signature


def _stats_meta_matches(existing_meta: dict, expected_meta: dict) -> tuple[bool, str]:
    """Return (matches, reason). reason is empty when matches=True."""
    if not existing_meta:
        return False, "existing file has no _meta signature (legacy format)"
    for k in (
        "version",
        "stats_mode",
        "action_horizon",
        "action_delta_indices",
        "reference_index",
        "action_format",
        "rotvec_continuity",
    ):
        if existing_meta.get(k) != expected_meta.get(k):
            return False, f"{k} mismatch: existing={existing_meta.get(k)!r} expected={expected_meta.get(k)!r}"
    if existing_meta.get("camera_motion_compensation", False) != expected_meta.get("camera_motion_compensation", False):
        return (
            False,
            "camera_motion_compensation mismatch: "
            f"existing={existing_meta.get('camera_motion_compensation', False)!r} "
            f"expected={expected_meta.get('camera_motion_compensation', False)!r}",
        )
    if existing_meta.get("relative_eef_frame", "eef") != expected_meta.get("relative_eef_frame", "eef"):
        return (
            False,
            "relative_eef_frame mismatch: "
            f"existing={existing_meta.get('relative_eef_frame', 'eef')!r} "
            f"expected={expected_meta.get('relative_eef_frame', 'eef')!r}",
        )
    if existing_meta.get("static_camera_frame", False) != expected_meta.get("static_camera_frame", False):
        return (
            False,
            "static_camera_frame mismatch: "
            f"existing={existing_meta.get('static_camera_frame', False)!r} "
            f"expected={expected_meta.get('static_camera_frame', False)!r}",
        )
    if expected_meta.get("camera_motion_compensation") and existing_meta.get(
        "camera_motion_convention_version"
    ) != expected_meta.get("camera_motion_convention_version"):
        return (
            False,
            "camera_motion_convention_version mismatch: "
            f"existing={existing_meta.get('camera_motion_convention_version')!r} "
            f"expected={expected_meta.get('camera_motion_convention_version')!r}",
        )
    exp_dpk = expected_meta.get("d_per_key", {})
    got_dpk = existing_meta.get("d_per_key", {})
    for k, v in exp_dpk.items():
        if got_dpk.get(k) != v:
            return False, f"d_per_key[{k}] mismatch: existing={got_dpk.get(k)!r} expected={v!r}"
    return True, ""


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


def _effective_delta_indices(
    base_delta_indices: list[int],
    temporal_downsample_stride: int = 1,
    action_index_shift: int = 0,
) -> list[int]:
    stride = max(1, int(temporal_downsample_stride))
    shift = max(0, int(action_index_shift))
    return [int(i) * stride + shift for i in base_delta_indices]


def _unwrap_relative_xyz_rotvec_chunks(chunks: np.ndarray) -> np.ndarray:
    out = np.asarray(chunks).copy()
    out[..., 3:6] = unwrap_rotvec_sequence(out[..., 3:6], time_axis=-2)
    return out.astype(np.float32, copy=False)


class RelativeActionLoader:
    def __init__(
        self,
        dataset_path: Path | str,
        embodiment_tag: EmbodimentTag,
        action_key: str,
        action_delta_indices: list[int] | None = None,
        camera_motion_compensation: bool = False,
        relative_eef_frame: str = "eef",
        eef_frame: str | None = None,
        static_camera_frame: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        self.camera_motion_compensation = bool(camera_motion_compensation)
        self.relative_eef_frame = str(relative_eef_frame).lower().replace("-", "_")
        self.eef_frame = eef_frame
        self.static_camera_frame = bool(static_camera_frame)
        if self.relative_eef_frame not in {"eef", "camera_spatial"}:
            raise ValueError("relative_eef_frame must be 'eef' or 'camera_spatial'")
        # Check action config
        assert action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(action_key)
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value]["action"]
        self.action_config = action_configs[idx]
        if action_delta_indices is None:
            action_delta_indices = MODALITY_CONFIGS[embodiment_tag.value]["action"].delta_indices
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=[int(i) for i in action_delta_indices],
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        assert state_key in MODALITY_CONFIGS[embodiment_tag.value]["state"].modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert self.modality_configs["state"].delta_indices[-1] <= self.modality_configs["action"].delta_indices[0], (
            f"State reference delta {self.modality_configs['state'].delta_indices[-1]} must not be after "
            f"the first action delta {self.modality_configs['action'].delta_indices[0]}"
        )
        self.loader = LeRobotEpisodeLoader(
            dataset_path,
            self.modality_configs,
            auxiliary_columns=CAMERA_POSE_COLUMNS if self.camera_motion_compensation else None,
        )

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)

        # Normalise potential object-dtype columns to a proper (T, D) float array.
        # pandas returns dtype=object when the parquet column holds numpy arrays
        # as its elements. Stacking eagerly here means the rest of the function
        # can assume regular 2-D arrays.
        def _densify(x: np.ndarray) -> np.ndarray:
            if x.dtype != object:
                return x
            return np.stack([np.asarray(e) for e in x.flat if isinstance(e, np.ndarray)])

        state_data = _densify(state_data)
        action_data = _densify(action_data)

        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices, dtype=np.int64)
        state_ref_offset = self.modality_configs["state"].delta_indices[-1]

        # --- FAST PATH: EEF + XYZ_ROTVEC. This is by far the hot case
        # for camera-spatial post-training. One
        # vectorised call over the whole episode replaces T*H Python-level
        # scipy Rotation objects. Numerically identical to the slow path
        # for both position and rotation halves.
        if self.action_config.type == ActionType.EEF and self.action_config.format == ActionFormat.XYZ_ROTVEC:
            assert state_data.shape[1] == 6, f"Expected 6D state (xyz+rotvec), got {state_data.shape[1]}D"
            assert action_data.shape[1] == 6, f"Expected 6D action (xyz+rotvec), got {action_data.shape[1]}D"
            refs = state_data[state_ref_offset : state_ref_offset + usable_length]  # (M, 6)
            world_from_ref_frames = None
            world_from_action_frames = None
            world_from_reference_cameras = None
            if self.camera_motion_compensation:
                ref_indices = np.arange(
                    state_ref_offset,
                    state_ref_offset + usable_length,
                    dtype=np.int64,
                )
                action_indices = np.arange(len(action_data), dtype=np.int64)
                if self.static_camera_frame and self.relative_eef_frame == "camera_spatial":
                    world_from_reference_cameras = np.broadcast_to(
                        np.eye(4, dtype=np.float64),
                        (usable_length, 4, 4),
                    ).copy()
                elif self.eef_frame == "robot_base" and self.relative_eef_frame == "camera_spatial":
                    reference_camera = extract_robot_base_reference_camera_transforms(df, ref_indices)
                    if reference_camera is None:
                        raise ValueError(
                            "camera_spatial relative stats require head_camera_to_robot_extrinsics in "
                            f"{self.dataset_path}"
                        )
                    world_from_reference_cameras = reference_camera[0]
                else:
                    ref_camera = extract_camera_world_transforms(df, ref_indices, role="state")
                    action_camera = extract_camera_world_transforms(df, action_indices, role="action")
                    if ref_camera is None or action_camera is None:
                        available = [column for column in CAMERA_POSE_COLUMNS if column in df.columns]
                        raise ValueError(
                            "Camera-motion compensated relative stats requested, but a current/future camera "
                            f"trajectory is missing in {self.dataset_path}; available={available}"
                        )
                    if self.relative_eef_frame == "camera_spatial" and (ref_camera[2] or action_camera[2]):
                        raise ValueError(
                            "camera_spatial relative stats require nonzero camera extrinsics for every sample"
                        )
                    if not (ref_camera[2] or action_camera[2]):
                        world_from_ref_frames = ref_camera[0]
                        world_from_action_frames = action_camera[0]
                        world_from_reference_cameras = ref_camera[0]
            # compute_relative_xyz_rotvec_chunks expects delta_indices to be
            # non-negative offsets into `action_data`. In this loader j-th
            # window uses actions[k + j] for k in delta_indices, matching.
            frame_kwargs = {
                "world_from_ref_frames": world_from_ref_frames,
                "world_from_action_frames": world_from_action_frames,
            }
            if self.relative_eef_frame == "camera_spatial":
                if world_from_reference_cameras is None:
                    raise ValueError("camera_spatial relative stats require a reference camera transform")
                batched = compute_camera_spatial_xyz_rotvec_chunks(
                    actions=action_data,
                    refs=refs,
                    delta_indices=action_delta_indices,
                    world_from_reference_cameras=world_from_reference_cameras,
                    **frame_kwargs,
                )
            else:
                batched = compute_relative_xyz_rotvec_chunks(
                    actions=action_data,
                    refs=refs,
                    delta_indices=action_delta_indices,
                    **frame_kwargs,
                )
            batched = _unwrap_relative_xyz_rotvec_chunks(batched)
            # Preserve return contract: one (H, 6) ndarray per window.
            return list(batched)

        # --- SLOW PATH: kept intact for EEF+XYZ_ROT6D and NON_EEF (joint).
        trajectories: list[np.ndarray] = []
        for i in range(usable_length):
            state_ind = state_ref_offset + i
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]

            if self.action_config.type == ActionType.EEF:
                # Determine rotation type based on action format
                if self.action_config.format == ActionFormat.XYZ_ROT6D:
                    # 9D: xyz (3) + rot6d (6)
                    assert len(last_state) == 9, f"Expected 9D state (xyz+rot6d), got {len(last_state)}D"
                    assert actions.shape[1] == 9, f"Expected 9D action (xyz+rot6d), got {actions.shape[1]}D"
                    rotation_type = "rot6d"
                    translation_dim = 3
                else:
                    raise ValueError(f"Unsupported action format: {self.action_config.format}")

                reference_frame = EndEffectorPose(
                    translation=last_state[:translation_dim],
                    rotation=last_state[translation_dim:],
                    rotation_type=rotation_type,
                )

                traj = EndEffectorActionChunk(
                    [
                        EndEffectorPose(
                            translation=m[:translation_dim], rotation=m[translation_dim:], rotation_type=rotation_type
                        )
                        for m in actions
                    ]
                ).relative_chunking(reference_frame=reference_frame)

                if self.action_config.format == ActionFormat.XYZ_ROT6D:
                    trajectories.append(np.stack([p.xyz_rot6d for p in traj.poses], dtype=np.float32))
                else:
                    raise ValueError(f"Unsupported action format: {self.action_config.format}")
            elif self.action_config.type == ActionType.NON_EEF:
                reference_frame = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    action_delta_indices: list[int] | None = None,
    max_episodes: int = -1,
    episode_indices: list[int] | None = None,
    camera_motion_compensation: bool = False,
    relative_eef_frame: str = "eef",
    eef_frame: str | None = None,
    static_camera_frame: bool = False,
) -> dict:
    if episode_indices is not None and max_episodes != -1:
        raise ValueError("Specify only one of episode_indices and max_episodes")

    loader = RelativeActionLoader(
        dataset_path,
        embodiment_tag,
        group_key,
        action_delta_indices=action_delta_indices,
        camera_motion_compensation=camera_motion_compensation,
        relative_eef_frame=relative_eef_frame,
        eef_frame=eef_frame,
        static_camera_frame=static_camera_frame,
    )
    action_horizon = len(loader.modality_configs["action"].delta_indices)
    last_action_delta = loader.modality_configs["action"].delta_indices[-1]
    all_episode_lengths = loader.loader.episode_lengths
    if episode_indices is not None:
        selected_episode_ids = sorted(set(int(index) for index in episode_indices))
        invalid_indices = [index for index in selected_episode_ids if index < 0 or index >= len(all_episode_lengths)]
        if invalid_indices:
            raise IndexError(
                f"Episode indices out of range for {dataset_path}: {invalid_indices[:10]} "
                f"(num_episodes={len(all_episode_lengths)})"
            )
    elif max_episodes != -1:
        selected_episode_ids = list(range(min(max_episodes, len(all_episode_lengths))))
    else:
        selected_episode_ids = list(range(len(all_episode_lengths)))
    episode_lengths = [all_episode_lengths[index] for index in selected_episode_ids]
    total_rows = sum(
        max(0, int(episode_length) - last_action_delta) * action_horizon for episode_length in episode_lengths
    )
    quantile_stride = max(1, int(np.ceil(total_rows / REL_STATS_QUANTILE_SAMPLE_SIZE)))

    sample_blocks: list[np.ndarray] = []
    row_offset = 0
    count = 0
    value_sum = None
    squared_sum = None
    value_min = None
    value_max = None
    for episode_id in tqdm(selected_episode_ids, desc=f"Loading trajectories for key {group_key}"):
        chunks = loader.load_relative_actions(episode_id)
        if not chunks:
            continue
        flattened = np.asarray(chunks, dtype=np.float32).reshape(-1, np.asarray(chunks[0]).shape[-1])
        flattened64 = flattened.astype(np.float64, copy=False)
        batch_count = len(flattened64)

        batch_sum = np.sum(flattened64, axis=0)
        batch_squared_sum = np.sum(np.square(flattened64), axis=0)
        batch_min = np.min(flattened64, axis=0)
        batch_max = np.max(flattened64, axis=0)
        if value_sum is None:
            value_sum = batch_sum
            squared_sum = batch_squared_sum
            value_min = batch_min
            value_max = batch_max
        else:
            value_sum += batch_sum
            squared_sum += batch_squared_sum
            value_min = np.minimum(value_min, batch_min)
            value_max = np.maximum(value_max, batch_max)
        count += batch_count

        local_indices = np.arange(batch_count, dtype=np.int64)
        sample_mask = (local_indices + row_offset) % quantile_stride == 0
        if np.any(sample_mask):
            sample_blocks.append(flattened[sample_mask])
        row_offset += batch_count

    if count == 0 or value_sum is None or squared_sum is None or value_min is None or value_max is None:
        raise ValueError(f"No relative action trajectories found for {dataset_path} key={group_key}")
    quantile_sample = np.concatenate(sample_blocks, axis=0)
    mean = value_sum / count
    variance = np.maximum(squared_sum / count - np.square(mean), 0.0)
    return {
        "max": value_max,
        "min": value_min,
        "q01": np.quantile(quantile_sample, 0.01, axis=0),
        "q99": np.quantile(quantile_sample, 0.99, axis=0),
        "mean": mean,
        "std": np.sqrt(variance),
    }


def generate_rel_stats(
    dataset_path: "Path | str",
    embodiment_tag: EmbodimentTag,
    max_episodes: int = -1,
    episode_indices: list[int] | None = None,
    temporal_downsample_stride: int = 1,
    action_index_shift: int = 0,
    action_delta_indices: list[int] | None = None,
    camera_motion_compensation: bool = False,
    relative_eef_frame: str = "eef",
    eef_frame: str | None = None,
    static_camera_frame: bool = False,
) -> dict:
    """Compute (or load cached) chunk-relative action stats for one dataset.

    Storage:
    <dataset>/meta/relative_action_stats__H{H}_ref{ref}_{fmt}_{deltas}[_camcomp-vN].json
    The filename encodes the (action_horizon, reference_index, action_format,
    effective action delta indices) tuple so different configurations coexist.
    The JSON also carries a top-level _meta signature; on load, if the signature
    does not match the current run's expected signature the file is regenerated
    (never silently reused).

    A legacy file at meta/relative_stats.json has no compatible signature, so
    it is left untouched and ignored. Statistics are recomputed under the new
    versioned name.

    ``action_delta_indices`` should come from the effective dataset
    ``ModalityConfig`` when one is available. This keeps statistics aligned
    with runtime action-horizon overrides instead of the static embodiment
    defaults. Statistics are flattened over horizon first, so each stat has
    shape [D], not [H, D].

    ``episode_indices`` supports a representative subset of a merged dataset.
    It is mutually exclusive with ``max_episodes``, which retains its legacy
    behavior of selecting the first N episodes.
    """
    from starVLA.dataloader.gr00t_lerobot.embodiment_configs import MODALITY_CONFIGS

    dataset_path = Path(dataset_path)
    if episode_indices is not None and max_episodes != -1:
        raise ValueError("Specify only one of episode_indices and max_episodes")
    if episode_indices is not None:
        episode_indices = sorted(set(int(index) for index in episode_indices))
    if embodiment_tag.value not in MODALITY_CONFIGS:
        raise ValueError(
            f"[relative_stats] embodiment_tag={embodiment_tag.value} has no MODALITY_CONFIGS entry; "
            "cannot compute relative action statistics."
        )

    action_meta = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_meta.action_configs is None:
        raise ValueError(
            f"[relative_stats] embodiment_tag={embodiment_tag.value} has no action_configs; "
            "cannot compute relative action statistics."
        )

    # Collect the RELATIVE keys and their signatures. All relative keys inside a
    # given tag share the same horizon / reference / format under our convention;
    # the SE(3)-relative pipeline enforces reference_index=0 and action_format
    # matches ActionType.EEF's XYZ_ROTVEC format for now.
    relative_keys, per_key_dim, per_key_fmt = [], {}, None
    uses_rotvec_continuity = False
    for key, cfg in zip(action_meta.modality_keys, action_meta.action_configs):
        if cfg.rep != ActionRepresentation.RELATIVE:
            continue
        relative_keys.append(key)
        uses_rotvec_continuity = uses_rotvec_continuity or (
            cfg.type == ActionType.EEF and cfg.format == ActionFormat.XYZ_ROTVEC
        )
        # Dim depends on format: xyz+rotvec=6, xyz+rot6d=9, DEFAULT for joints=variable.
        if cfg.type.name == "EEF":
            per_key_dim[key] = {"XYZ_ROTVEC": 6, "XYZ_ROT6D": 9}.get(cfg.format.name, -1)
        else:
            per_key_dim[key] = -1  # joints: unknown until we peek at parquet; leave as -1 so signature ignores width
        per_key_fmt = cfg.format  # last one wins; we assume homogeneous inside a tag
    if not relative_keys:
        raise ValueError(
            f"[relative_stats] embodiment_tag={embodiment_tag.value} has no RELATIVE action keys; "
            "disable enable_relative_action for this dataset or add relative action configs."
        )

    base_action_delta_indices = action_meta.delta_indices if action_delta_indices is None else action_delta_indices
    action_delta_indices = _effective_delta_indices(
        base_action_delta_indices,
        temporal_downsample_stride=temporal_downsample_stride,
        action_index_shift=action_index_shift,
    )
    action_horizon = len(action_delta_indices)
    reference_index = 0  # matches RelativeActionTransform default in the pipeline

    expected_meta = _build_meta_signature(
        action_horizon=action_horizon,
        action_delta_indices=action_delta_indices,
        reference_index=reference_index,
        action_format=per_key_fmt,
        d_per_key=per_key_dim,
        rotvec_continuity=uses_rotvec_continuity,
        camera_motion_compensation=camera_motion_compensation,
        relative_eef_frame=relative_eef_frame,
        static_camera_frame=static_camera_frame,
    )

    versioned_rel = _versioned_stats_filename(
        action_horizon,
        reference_index,
        per_key_fmt,
        action_delta_indices,
        camera_motion_compensation=camera_motion_compensation,
        relative_eef_frame=relative_eef_frame,
        static_camera_frame=static_camera_frame,
    )
    stats_path = dataset_path / versioned_rel
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_stats_lock(stats_path):
        # Re-check after taking the lock: another rank may have completed the cache while this rank waited.
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[relative_stats] Failed to read {stats_path} ({exc}); regenerating.")
                stats_path.unlink(missing_ok=True)
            else:
                got_meta = payload.get("_meta") if isinstance(payload, dict) else None
                matches, reason = _stats_meta_matches(got_meta or {}, expected_meta)
                if matches:
                    loaded_stats = {k: v for k, v in payload.items() if k != "_meta"}
                    missing = [k for k in relative_keys if k not in loaded_stats]
                    bad_shapes = []
                    for key in relative_keys:
                        if key not in loaded_stats:
                            continue
                        for stat_name in REL_STATS_STAT_NAMES:
                            if stat_name not in loaded_stats[key]:
                                missing.append(f"{key}.{stat_name}")
                                continue
                            arr = np.asarray(loaded_stats[key][stat_name])
                            if arr.ndim != 1:
                                bad_shapes.append(f"{key}.{stat_name}: shape={arr.shape}")
                    if missing or bad_shapes:
                        print(
                            f"[relative_stats] {stats_path} has incomplete/non-flattened stats "
                            f"(missing={missing}, bad_shapes={bad_shapes}); regenerating."
                        )
                        stats_path.unlink(missing_ok=True)
                    else:
                        print(f"[relative_stats] Loaded {stats_path} (signature OK: {expected_meta})")
                        return loaded_stats
                else:
                    print(f"[relative_stats] {stats_path} signature mismatch ({reason}); regenerating.")
                    stats_path.unlink(missing_ok=True)

        # Legacy files have no compatible signature. Preserve them in place and
        # compute the independently named versioned cache below.
        legacy_path = dataset_path / LEGACY_REL_STATS_FILENAME
        if legacy_path.exists():
            print(
                f"[relative_stats] Legacy {legacy_path} has no compatible signature; "
                f"leaving it untouched and recomputing under {stats_path}."
            )

        stats: dict = {}
        for action_key in sorted(relative_keys):
            print(f"[relative_stats] Generating {dataset_path.name} :: {embodiment_tag.value} :: {action_key}")
            stats[action_key] = calculate_stats_for_key(
                dataset_path,
                embodiment_tag,
                action_key,
                action_delta_indices=action_delta_indices,
                max_episodes=max_episodes,
                episode_indices=episode_indices,
                camera_motion_compensation=camera_motion_compensation,
                relative_eef_frame=relative_eef_frame,
                eef_frame=eef_frame,
                static_camera_frame=static_camera_frame,
            )

        payload = {
            "_meta": {
                **expected_meta,
                "generated_at": _iso_now(),
                "dataset_path": str(dataset_path),
                "episode_sampling": (
                    {
                        "method": "explicit_indices",
                        "count": len(episode_indices),
                        "episode_indices": episode_indices,
                    }
                    if episode_indices is not None
                    else {
                        "method": "first_n" if max_episodes != -1 else "all",
                        "count": max_episodes if max_episodes != -1 else None,
                    }
                ),
            },
            **{k: v for k, v in stats.items()},
        }
        _atomic_json_dump(payload, stats_path)
        print(f"[relative_stats] Wrote {stats_path}")
        return stats


def _iso_now() -> str:
    """Datetime as ISO-8601 with local tz (Date.now-safe alternative)."""
    import time

    lt = time.localtime()
    tz_off = -time.timezone if lt.tm_isdst == 0 else -time.altzone
    sign = "+" if tz_off >= 0 else "-"
    hh, mm = divmod(abs(tz_off) // 60, 60)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f"{sign}{hh:02d}:{mm:02d}"


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    temporal_downsample_stride: int = 1,
    action_index_shift: int = 0,
):
    generate_stats(dataset_path)
    generate_rel_stats(
        dataset_path,
        embodiment_tag,
        temporal_downsample_stride=temporal_downsample_stride,
        action_index_shift=action_index_shift,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
