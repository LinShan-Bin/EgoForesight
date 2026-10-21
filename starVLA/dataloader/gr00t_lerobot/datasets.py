# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import gc
import hashlib
import json
import os
import pickle
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.base_supervision import is_base_supervision_enabled
from starVLA.dataloader.gr00t_lerobot.canonical_action import (
    VALID_RELATIVE_EEF_FRAMES,
    CanonicalSlot,
    get_canonical_action_dim,
    get_canonical_state_dim,
    is_action_alignment_enabled,
    is_state_relative_eef_enabled,
)
from starVLA.dataloader.gr00t_lerobot.canonical_meta import (
    get_canonical_robot_type,
    get_dataset_family,
    get_robot_family,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.relative_stats import generate_rel_stats
from starVLA.dataloader.gr00t_lerobot.sample_quality_index import load_dataset_quality_index
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.state_action.camera_motion import (
    CAMERA_ACTION_TRANSFORM_KEY,
    CAMERA_POSE_COLUMNS,
    CAMERA_REFERENCE_TRANSFORM_KEY,
    CAMERA_SKIP_COMPENSATION_KEY,
    CAMERA_STATE_TRANSFORM_KEY,
    extract_camera_world_transforms,
    extract_robot_base_reference_camera_transforms,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps
from starVLA.quality_index import (
    DatasetQualityFilter,
    quality_tier_name,
    worst_quality_label,
)

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_SUBTASKS_FILENAME = "meta/subtasks.jsonl"
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"
from starVLA.dataloader.gr00t_lerobot.embodiment_configs import MODALITY_CONFIGS
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.types import ActionRepresentation


def _parse_column_to_array(series: pd.Series) -> np.ndarray:
    """Convert a pandas object-dtype Series to a 2-D float64 numpy array.

    Fast path: ``np.array(series.tolist())`` handles the common case where every
    element has the same length.  Falls back to the slower per-row ``np.vstack``
    when the fast path fails (e.g. ragged arrays).
    """
    try:
        arr = np.array(series.tolist(), dtype=np.float64)
    except (ValueError, TypeError):
        arr = np.vstack([np.asarray(x, dtype=np.float64) for x in series])
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _compute_partial_stats(args: tuple) -> dict:
    """Process one parquet file and return per-column partial statistics.

    Returned per column: ``{n, mean, M2, min, max, samples}`` where *M2* is
    the sum of squared deviations from the batch mean (needed by Chan's
    algorithm) and *samples* is a bounded subsample for quantile estimation.
    """
    path, columns, samples_per_file = args
    df = pd.read_parquet(path)
    result = {}
    for col in columns:
        if col not in df.columns:
            continue
        try:
            chunk = _parse_column_to_array(df[col])
        except Exception:
            continue
        n = chunk.shape[0]
        mean = chunk.mean(axis=0)
        M2 = np.sum((chunk - mean) ** 2, axis=0)

        if n <= samples_per_file:
            sampled = chunk
        else:
            idx = np.random.choice(n, samples_per_file, replace=False)
            sampled = chunk[idx]

        result[col] = dict(
            n=n,
            mean=mean,
            M2=M2,
            min=chunk.min(axis=0),
            max=chunk.max(axis=0),
            samples=sampled,
        )
    del df
    return result


def calculate_dataset_statistics(
    parquet_paths: list[Path],
    num_workers: int = 8,
    quantile_samples_per_file: int = 1000,
) -> dict:
    """Calculate dataset statistics for all columns across parquet files.

    Optimised for large-scale datasets (100 GB+):
    * Files are processed **one at a time** — no full-dataset ``pd.concat``.
    * Mean / variance use **Chan's parallel algorithm** (numerically stable,
      single-pass, constant memory per column).
    * Quantiles (q01, q99) are estimated from a **bounded subsample**
      (``quantile_samples_per_file`` rows kept per file).
    * Column conversion uses a bulk ``np.array(series.tolist())`` fast-path
      instead of the per-row ``np.asarray`` + ``np.vstack`` loop.
    * Parquet files can be read in parallel via *num_workers* processes.

    Args:
        parquet_paths: Parquet files to aggregate.
        num_workers: Parallel reader processes.  Set to 1 for sequential.
        quantile_samples_per_file: Max rows kept per file for quantile
            estimation.  Higher → more accurate q01/q99 but more memory.
    """
    from multiprocessing import Pool as _Pool

    sorted_paths = sorted(parquet_paths)
    if not sorted_paths:
        return {}

    first_df = pd.read_parquet(sorted_paths[0])
    all_columns = [c for c in first_df.columns if "task_info" not in c]
    del first_df

    # --- per-column accumulators (Chan's algorithm) -------------------------
    acc: dict = {}  # col -> {n, mean, M2, min, max}
    all_samples: dict = {}  # col -> [array, ...]

    def _merge(partial: dict):
        for col, p in partial.items():
            if col not in acc:
                acc[col] = {k: p[k] for k in ("n", "mean", "M2", "min", "max")}
                all_samples[col] = [p["samples"]]
            else:
                a = acc[col]
                n_a, n_b = a["n"], p["n"]
                n_ab = n_a + n_b
                delta = p["mean"] - a["mean"]
                a["mean"] = a["mean"] + delta * (n_b / n_ab)
                a["M2"] = a["M2"] + p["M2"] + delta**2 * (n_a * n_b / n_ab)
                a["min"] = np.minimum(a["min"], p["min"])
                a["max"] = np.maximum(a["max"], p["max"])
                a["n"] = n_ab
                all_samples[col].append(p["samples"])

    job_args = [(p, all_columns, quantile_samples_per_file) for p in sorted_paths]

    if num_workers > 1:
        with _Pool(processes=num_workers) as pool:
            for partial in tqdm(
                pool.imap_unordered(_compute_partial_stats, job_args),
                total=len(sorted_paths),
                desc="Computing dataset statistics (parallel)",
            ):
                _merge(partial)
    else:
        for args in tqdm(job_args, desc="Computing dataset statistics"):
            _merge(_compute_partial_stats(args))

    # --- finalise -----------------------------------------------------------
    dataset_statistics = {}
    for col, a in tqdm(acc.items(), total=len(acc), desc="Finalising statistics"):
        std = np.sqrt(a["M2"] / a["n"])
        sampled = np.concatenate(all_samples[col], axis=0)
        dataset_statistics[col] = {
            "mean": a["mean"].tolist(),
            "std": std.tolist(),
            "min": a["min"].tolist(),
            "max": a["max"].tolist(),
            "q01": np.quantile(sampled, 0.01, axis=0).tolist(),
            "q99": np.quantile(sampled, 0.99, axis=0).tolist(),
        }
        del sampled

    all_samples.clear()
    gc.collect()
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        enable_relative_action: bool = False,
        control_mode: str | None = None,
        language_embodiment: str | None = None,
        eef_frame: str | None = None,
        robot_type: str | None = None,
        end_effector_type: str | None = None,
        quality_filter: DatasetQualityFilter | None = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg or {}
        self.metadata_only = bool(self.data_cfg.get("metadata_only", False))
        self.enable_relative_action = enable_relative_action
        self.control_mode = control_mode
        self.language_embodiment = language_embodiment
        self.eef_frame = eef_frame
        self.robot_type = robot_type
        self.end_effector_type = end_effector_type
        self.quality_filter = quality_filter
        self.enable_camera_motion_compensation = bool(self.data_cfg.get("enable_camera_motion_compensation", True))
        self.static_camera_frame = bool(self.data_cfg.get("static_camera_frame", False))
        self.relative_eef_frame = str(self.data_cfg.get("relative_eef_frame", "eef")).lower().replace("-", "_")
        if self.relative_eef_frame not in VALID_RELATIVE_EEF_FRAMES:
            raise ValueError(
                f"relative_eef_frame must be one of {sorted(VALID_RELATIVE_EEF_FRAMES)}, "
                f"got {self.relative_eef_frame!r}"
            )
        state_relative_eef = bool(self.data_cfg.get("enable_state_relative_eef", is_state_relative_eef_enabled()))
        self._camera_motion_required = (
            self.enable_camera_motion_compensation
            and self.eef_frame == "camera"
            and state_relative_eef
            and not self.static_camera_frame
        )
        self._camera_spatial_required = state_relative_eef and self.relative_eef_frame == "camera_spatial"
        if self._camera_spatial_required and not self.enable_camera_motion_compensation:
            raise ValueError("relative_eef_frame='camera_spatial' requires camera geometry to be enabled")
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version = self.data_cfg.get("lerobot_version", "v2.0")  # self._indict_lerobot_version(**kwargs)
        self.temporal_downsample_stride = max(1, int(self.data_cfg.get("temporal_downsample_stride", 1)))
        self.base_step_stride = max(1, int(self.data_cfg.get("base_step_stride", self.temporal_downsample_stride)))
        max_episodes_per_dataset = self.data_cfg.get("max_episodes_per_dataset", None)
        if max_episodes_per_dataset in (None, False, "False", "false", ""):
            self.max_episodes_per_dataset = None
        else:
            try:
                self.max_episodes_per_dataset = int(max_episodes_per_dataset)
            except (TypeError, ValueError) as error:
                raise ValueError("max_episodes_per_dataset must be a positive integer or null") from error
            if self.max_episodes_per_dataset <= 0:
                raise ValueError(
                    "max_episodes_per_dataset must be a positive integer, " f"got {self.max_episodes_per_dataset}"
                )

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = transforms if transforms is not None else ComposedModalityTransform(transforms=[])

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
            self.tag_index = EMBODIMENT_TAG_MAPPING[self.tag]
        else:
            self.tag = embodiment_tag
            self.tag_index = EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value]

        self._metadata = self._get_metadata(EmbodimentTag(self.tag), self.enable_relative_action)

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self._subtasks = self._get_subtasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None
        self._sample_quality_index = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        total_trajectories = len(self._trajectory_ids)
        if self.max_episodes_per_dataset is not None:
            retained_trajectories = min(self.max_episodes_per_dataset, total_trajectories)
            self._trajectory_ids = self._trajectory_ids[:retained_trajectories]
            self._trajectory_lengths = self._trajectory_lengths[:retained_trajectories]
            print(
                f"Limited dataset {self.dataset_name} to its first {retained_trajectories} "
                f"of {total_trajectories} episodes"
            )
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        if self.quality_filter is not None and self.quality_filter.binding.video_key not in self._modality_keys["video"]:
            raise ValueError(
                f"Quality-filter video_key={self.quality_filter.binding.video_key!r} is not available in "
                f"dataset {self.dataset_name}; video keys={list(self._modality_keys['video'])}"
            )
        if self.metadata_only:
            # Stats export needs episode counts, transforms, and metadata, but
            # never samples a frame. Avoid expanding every episode into the
            # potentially billion-row [episode, step] index.
            self._all_steps = np.empty((0, 2), dtype=np.int64)
        else:
            self._all_steps = self._filter_steps_to_selected_trajectories(self._get_all_steps())
            self._sample_quality_index = load_dataset_quality_index(self, self.data_cfg, kind="lerobot")
            if self._sample_quality_index is not None:
                self._all_steps = self._filter_steps_to_selected_trajectories(self._sample_quality_index.keys)
                print(
                    f"Loaded per-dataset quality index for {self.dataset_name}: "
                    f"steps={len(self._all_steps)}, pairs={self._sample_quality_index.metadata['total_pairs']}, "
                    f"path={self._sample_quality_index.directory}"
                )
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")
        if self.quality_filter is not None:
            print(
                f"Quality filter enabled for {self.dataset_name}: "
                f"dataset_id={self.quality_filter.binding.dataset_id}, "
                f"index={self.quality_filter.index.source_path}"
            )

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> np.ndarray:
        """The trajectory IDs and base indices for all steps in the dataset.

        Returns:
            np.ndarray of shape (N, 2) with dtype int64.  Each row is
            [trajectory_id, base_index].

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: array([[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[2,2],[2,3]])
        """
        return self._all_steps

    def _filter_steps_to_selected_trajectories(self, steps: np.ndarray) -> np.ndarray:
        """Retain only steps belonging to the configured leading episode subset."""
        steps = np.asarray(steps, dtype=np.int64)
        if self.max_episodes_per_dataset is None or steps.size == 0:
            return steps
        if steps.ndim != 2 or steps.shape[1] != 2:
            raise ValueError(f"Expected steps with shape [N, 2], got {steps.shape}")
        selected_mask = np.isin(steps[:, 0], self.trajectory_ids)
        return np.ascontiguousarray(steps[selected_mask])

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def subtasks(self) -> dict:
        """The subtasks for the dataset."""
        return self._subtasks

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag, enable_relative_action: bool = True) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Args:
            embodiment_tag: The embodiment tag for the dataset.
            enable_relative_action: Whether to compute relative action statistics.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            configured_subkeys = [
                key.split(".", 1)[1] if key.startswith(f"{modality}.") else key
                for key in self.modality_configs[modality].modality_keys
            ]
            for subkey in configured_subkeys:
                if subkey not in le_state_action_meta:
                    raise KeyError(f"Configured modality {modality}.{subkey} is missing from {modality_meta_path}")
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
            default_original_keys = {"state": "observation.state", "action": "action"}
            required_statistics_keys = set()
            for modality_name in ("state", "action"):
                modality = getattr(le_modality_meta, modality_name)
                configured_subkeys = [
                    key.split(".", 1)[1] if key.startswith(f"{modality_name}.") else key
                    for key in self.modality_configs[modality_name].modality_keys
                ]
                for subkey in configured_subkeys:
                    required_statistics_keys.add(modality[subkey].original_key or default_original_keys[modality_name])
            missing_statistics_keys = required_statistics_keys - set(le_statistics)
            if missing_statistics_keys:
                raise ValueError(f"Dataset statistics are missing modality columns: {sorted(missing_statistics_keys)}")
        except (FileNotFoundError, ValidationError, ValueError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            parquet_files_filtered = []
            #  parquet_files[0].name = "episode_033675.parquet" is broken file
            for pf in parquet_files:
                # if "episode_033675.parquet" in pf.name:
                #     continue
                parquet_files_filtered.append(pf)

            le_statistics = calculate_dataset_statistics(parquet_files_filtered)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)

        # Optionally generate relative action statistics
        if enable_relative_action:
            rel_stats = generate_rel_stats(
                self.dataset_path,
                EmbodimentTag(self.tag),
                temporal_downsample_stride=self.temporal_downsample_stride,
                action_index_shift=int(self.data_cfg.get("action_index_shift", 0)),
                action_delta_indices=list(self.modality_configs["action"].delta_indices),
                camera_motion_compensation=(self._camera_motion_required or self._camera_spatial_required),
                relative_eef_frame=self.relative_eef_frame,
                eef_frame=self.eef_frame,
                static_camera_frame=self.static_camera_frame,
            )
            if not rel_stats:
                raise ValueError(
                    f"Relative action is enabled for {self.dataset_name}, but no relative stats were produced."
                )
        else:
            rel_stats = {}
            print("Skipping relative action statistics calculation (enable_relative_action=False)")

        # 这段代码的作用是将 LeRobot 原始格式的扁平统计数据 转换为 按语义分组的层次化统计数据。
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = (
                    state_action_meta.original_key
                    or {
                        "state": "observation.state",
                        "action": "action",
                    }[our_modality]
                )
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # Initialize relative_action statistics only if enabled
        dataset_statistics["relative_action"] = {}
        if enable_relative_action:
            for key, value in rel_stats.items():
                dataset_statistics["relative_action"][key] = value

        # import pdb; pdb.set_trace()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    # TODO auto map key? just map to file_path and file_from_index
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/observation.images.wrist/from_timestamp": episode[
                            "videos/observation.images.wrist/from_timestamp"
                        ],
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # 这里应该可以直接读取到 save index 信息
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> np.ndarray:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            np.ndarray of shape (N, 2) with dtype int64.  Each row is
            [trajectory_id, base_index].
        """
        # Create a hash key based on configuration to ensure cache validity
        config_key = self._get_steps_config_key()

        # Create a unique filename based on config_key
        # steps_filename = f"steps_{config_key}.pkl"
        # @BUG
        # fast get static steps @fangjing --> don't use hash to dynamic sample
        #
        if self.base_step_stride == 1:
            steps_filename = "steps_data_index.pkl"
        else:
            steps_filename = f"steps_data_index_{self.base_step_stride}.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename

        # Try to load cached steps first
        try:
            if steps_path.exists():
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                steps = cached_data["steps"]
                # Backward compatibility: convert old list[tuple] cache to ndarray.
                if isinstance(steps, list):
                    steps = np.array(steps, dtype=np.int64) if steps else np.empty((0, 2), dtype=np.int64)
                return steps
        except (FileNotFoundError, pickle.PickleError, KeyError) as e:
            print(f"Failed to load cached steps: {e}")
            print("Computing steps from scratch...")

        all_steps = None
        if self.base_step_stride > 1:
            base_steps_path = self.dataset_path / "meta" / "steps_data_index.pkl"
            try:
                if base_steps_path.exists():
                    with open(base_steps_path, "rb") as f:
                        base_cached_data = pickle.load(f)
                    base_steps = base_cached_data["steps"]
                    if isinstance(base_steps, list):
                        base_steps = (
                            np.array(base_steps, dtype=np.int64) if base_steps else np.empty((0, 2), dtype=np.int64)
                        )
                    stride_mask = np.remainder(base_steps[:, 1], self.base_step_stride) == 0
                    all_steps = np.ascontiguousarray(base_steps[stride_mask])
                    print(
                        f"Derived {len(all_steps)} stride-{self.base_step_stride} steps from "
                        f"{len(base_steps)} cached base steps"
                    )
                    del stride_mask, base_steps, base_cached_data
            except (FileNotFoundError, pickle.PickleError, KeyError, IndexError, TypeError, ValueError) as e:
                print(f"Failed to derive strided steps from {base_steps_path}: {e}")

        if all_steps is None:
            # Fall back to scanning trajectories only when no reusable base-step cache exists.
            all_steps = self._get_all_steps_single_process()

        # Cache the computed steps with unique filename
        try:
            if self.max_episodes_per_dataset is not None:
                return all_steps
            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }

            # Ensure the meta directory exists
            steps_path.parent.mkdir(parents=True, exist_ok=True)

            with open(steps_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached steps saved to {steps_path}")
        except Exception as e:
            print(f"Failed to cache steps: {e}")

        return all_steps

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "temporal_downsample_stride": self.temporal_downsample_stride,
            "base_step_stride": self.base_step_stride,
        }
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #

    def _get_all_steps_single_process(self) -> np.ndarray:
        # Accumulate as a plain Python list first (fast extend), then convert to
        # a contiguous int64 ndarray once.  This avoids the huge per-object
        # overhead of list[tuple[int,int]] (~56 bytes/entry) and prevents
        # Python reference-counting COW when DataLoader workers fork the process.
        all_steps_list: list[list[int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        has_language_modality = ("language" in self.modality_keys) and (len(self.modality_keys["language"]) > 0)
        lang_key = self.modality_keys["language"][0] if has_language_modality else None

        for trajectory_id, trajectory_length in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths),
            total=len(self.trajectory_ids),
            desc="Getting All Step (fast)",
        ):
            try:
                # 关键：只读一次（如果你必须依赖 parquet 的 language mapping）
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                else:
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                self.curr_traj_data = data

                if has_language_modality:
                    # 只检查一次：base_index=0
                    try:
                        language_instruction = self.get_language(trajectory_id, lang_key, 0)
                        valid = (
                            bool(language_instruction)
                            and len(language_instruction) > 0
                            and (language_instruction[0] not in ["", None])
                        )
                    except Exception:
                        valid = False

                    if not valid:
                        skipped_trajectories += 1
                        continue

                # 直接整段加入（比逐步 append 快很多）
                traj_id_int = int(trajectory_id)

                # 针对 subtask 过滤
                # 判断是否需要按 subtask 过滤
                language_keys = self.modality_keys.get("language", [])
                uses_subtask = (
                    self._subtasks is not None
                    and "subtask_index" in data.columns
                    and any("subtask_description" in k for k in language_keys)
                )
                valid_step_indices = range(0, int(trajectory_length), self.base_step_stride)
                if uses_subtask:
                    subtask_series = data["subtask_index"]
                    valid_subtask_index = set(self._subtasks.index.tolist())
                    valid_step_indices = (
                        i
                        for i in valid_step_indices
                        if (lambda v: (v.item() if hasattr(v, "item") else int(v)))(subtask_series.iloc[i]) >= 0
                        and (lambda v: (v.item() if hasattr(v, "item") else int(v)))(subtask_series.iloc[i])
                        in valid_subtask_index
                    )
                for i in valid_step_indices:
                    all_steps_list.append([traj_id_int, i])
                # 过滤结束

                # all_steps_list.extend(
                #     [traj_id_int, i]
                #     for i in range(0, int(trajectory_length), self.base_step_stride)
                # )
                processed_trajectories += 1

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                continue

        print(f"Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} trajectories")
        print(f"Total steps: {len(all_steps_list)}")
        if not all_steps_list:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(all_steps_list, dtype=np.int64)

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get("action", [])

        # Extract position data
        delta_position_values = None
        position_candidates = ["delta_eef_position"]
        coordinate_candidates = ["x", "y", "z"]

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == "x":
                                    x_data = coord_data
                                elif coord == "y":
                                    y_data = coord_data
                                elif coord == "z":
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.delta_eef_position" in data.columns:
                delta_position_values = data["action.delta_eef_position"].to_numpy().tolist()
            elif all(col in data.columns for col in ["action.x", "action.y", "action.z"]):
                x_vals = data["action.x"].to_numpy()
                y_vals = data["action.y"].to_numpy()
                z_vals = data["action.z"].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ["gripper_close", "gripper"]

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.gripper_close" in data.columns:
                gripper_values = data["action.gripper_close"].to_numpy().tolist()
            elif "action.gripper" in data.columns:
                gripper_values = data["action.gripper"].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities.

        Mirror datasets (where action[t] == state[t] verbatim, e.g. mocap-style
        handpose / robocoin_agilexcobot / tienkung_xsens / etc.) carry a
        nonzero ``action_index_shift`` in data_cfg. We add that offset to
        every action-modality delta index so the model trains on action[t+s..]
        instead of action[t..] -- otherwise it just learns the identity.
        """
        action_index_shift = int(self.data_cfg.get("action_index_shift", 0))
        delta_indices: dict[str, np.ndarray] = {}
        for modality_name, config in self.modality_configs.items():
            extra_shift = action_index_shift if modality_name == "action" else 0
            for key in config.modality_keys:
                delta_indices[key] = (
                    np.array(config.delta_indices, dtype=np.int64) * self.temporal_downsample_stride + extra_shift
                )
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _process_task_text(self, task_text: str) -> str:
        """Process task text to extract English part after @ and validate.

        Args:
            task_text: Original task text, may contain Chinese and English separated by @

        Returns:
            Processed task text (English part only), or None if invalid
        """
        if not isinstance(task_text, str):
            return None

        # Extract English part after @
        if "@" in task_text:
            parts = task_text.split("@", 1)
            if len(parts) > 1:
                english_part = parts[1].strip()
            else:
                english_part = task_text.strip()
        else:
            # If no @, use the whole text
            english_part = task_text.strip()

        # Filter out invalid instructions
        invalid_keywords = ["null", "unqualified", "qualified"]
        if english_part.lower() in invalid_keywords:
            return None

        # Return empty string if result is empty
        if not english_part:
            return None

        return english_part

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            # Process task text: extract English part and filter invalid ones
            df["task"] = df["task"].apply(self._process_task_text)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # 把索引变成一列，列名通常为 'index'
            df = df.rename(columns={"index": "task"})  # 把 'index' 列重命名为 'task'
            df = df[["task_index", "task"]]  # 调整列顺序
            # Process task text: extract English part and filter invalid ones
            df["task"] = df["task"].apply(self._process_task_text)
            return df

    def _get_subtasks(self) -> dict:
        """Get the subtasks for the dataset."""
        subtasks_path = self.dataset_path / LE_ROBOT_SUBTASKS_FILENAME
        # check if the file exists, if not, return None
        if not subtasks_path.exists():
            # check if subtask in episodes.json
            episodes_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(episodes_path, "r") as f:
                episodes = [json.loads(line) for line in f]
            if "subtask" in episodes[0]:
                print(f"Subtask not found in {subtasks_path}, extracting from {episodes_path}")
                self._extract_subtasks(episodes_path, subtasks_path)
            else:
                return None
        with open(subtasks_path, "r") as f:
            subtasks = [json.loads(line) for line in f]
        df = pd.DataFrame(subtasks)
        df["task"] = df["task"].apply(self._process_task_text)
        return df.set_index("subtask_index")

    def _extract_subtasks(self, input_path, output_path):
        """Extract all subtasks from episodes.jsonl and write them to output_path in subtask_index order."""
        all_subtasks = []

        with open(input_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {idx + 1} 行解析失败，已跳过 - {e}")
                    continue

                subtask_list = episode.get("subtask", [])
                for st in subtask_list:
                    subtask_index = st.get("subtask_index")
                    task = st.get("task", "")
                    all_subtasks.append({"subtask_index": subtask_index, "task": task})

        # 按 subtask_index 排序（应对可能的乱序）
        all_subtasks.sort(key=lambda x: x["subtask_index"])

        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_subtasks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"已从 {input_path} 提取 {len(all_subtasks)} 条 subtask，写入 {output_path}")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}")

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        if self._sample_quality_index is not None:
            return self._sample_quality_index.sampling_length
        if self.metadata_only:
            return int(np.sum(self.trajectory_lengths, dtype=np.int64))
        return len(self.all_steps)

    @property
    def sample_quality_index(self):
        return self._sample_quality_index

    def sample_quality_step(self, rng: np.random.Generator) -> tuple[int, int, int]:
        if self._sample_quality_index is None:
            raise RuntimeError(f"Dataset {self.dataset_name} has no per-dataset quality index")
        return self._sample_quality_index.sample(rng)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        row = self.all_steps[index]
        trajectory_id, base_index = int(row[0]), int(row[1])
        data = self.get_step_data(trajectory_id, base_index)

        # Process all video keys dynamically
        images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]

            # Apply image cropping if enabled and the video key is base_view
            # Note: crop_obs_camera functionality has been removed

            # image = Image.fromarray(image).resize((224, 224))
            images.append(image)

        # Get language and action data
        # Language instruction should already be validated in _get_all_steps_single_process
        language_list = data[self.modality_keys["language"][0]]
        # Get the first language instruction (should be valid after filtering)
        language = language_list[0] if language_list and language_list[0] is not None else "Perform the task."

        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1)

        return dict(action=action, image=images, language=language)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        return self.get_step_data_for_modalities(trajectory_id, base_index)

    def get_step_data_for_modalities(
        self,
        trajectory_id: int,
        base_index: int,
        modalities: Sequence[str] | None = None,
    ) -> dict:
        """Load selected raw modalities for one step without applying transforms.

        Loading action separately lets callers reject invalid action chunks before
        triggering video decoding. A subsequent call for the remaining modalities
        reuses the trajectory parquet cached by :meth:`get_trajectory_data`.
        """
        requested_modalities = list(self.modality_keys) if modalities is None else list(modalities)
        unknown_modalities = [modality for modality in requested_modalities if modality not in self.modality_keys]
        if unknown_modalities:
            raise KeyError(
                f"Unknown modalities {unknown_modalities} for dataset {self.dataset_name}; "
                f"available={list(self.modality_keys)}"
            )

        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id  # Ensure the trajectory cache is reusable by the second-stage load.
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in requested_modalities:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        self._inject_camera_motion(data, requested_modalities, base_index)
        return data

    def _inject_camera_motion(
        self,
        data: dict,
        requested_modalities: Sequence[str],
        base_index: int,
    ) -> None:
        """Inject private camera tracks consumed by ``RelativeActionTransform``."""
        if not (self._camera_motion_required or self._camera_spatial_required):
            return
        assert self.curr_traj_data is not None
        if self._camera_spatial_required and self.static_camera_frame:
            if "state" in requested_modalities:
                data[CAMERA_REFERENCE_TRANSFORM_KEY] = np.eye(4, dtype=np.float64)
            return
        if self._camera_spatial_required and self.eef_frame == "robot_base":
            if "state" not in requested_modalities:
                return
            if not self.modality_keys["state"]:
                raise ValueError("camera_spatial relative EEF needs at least one state key")
            reference_key = self.modality_keys["state"][0]
            source_indices = self.delta_indices[reference_key] + int(base_index)
            extracted = extract_robot_base_reference_camera_transforms(
                self.curr_traj_data,
                source_indices,
            )
            if extracted is None:
                raise ValueError(
                    "camera_spatial relative EEF is enabled for a robot-base dataset, but "
                    f"head_camera_to_robot_extrinsics is missing in {self.dataset_name}"
                )
            data[CAMERA_REFERENCE_TRANSFORM_KEY] = extracted[0]
            return

        key_by_role = {
            "state": CAMERA_STATE_TRANSFORM_KEY,
            "action": CAMERA_ACTION_TRANSFORM_KEY,
        }
        for role, output_key in key_by_role.items():
            if role not in requested_modalities:
                continue
            if not self.modality_keys[role]:
                raise ValueError(f"camera-motion compensation needs at least one {role} key")
            reference_key = self.modality_keys[role][0]
            source_indices = self.delta_indices[reference_key] + int(base_index)
            extracted = extract_camera_world_transforms(
                self.curr_traj_data,
                source_indices,
                role=role,
            )
            if extracted is None:
                available = [column for column in CAMERA_POSE_COLUMNS if column in self.curr_traj_data.columns]
                raise ValueError(
                    "Camera-motion compensation is enabled for a camera-frame EEF dataset, but no supported "
                    f"{role} camera trajectory was found in {self.dataset_name}. "
                    f"Expected one of {CAMERA_POSE_COLUMNS}; available camera columns={available}. "
                    "Set enable_camera_motion_compensation=false only if this fallback is intentional."
                )
            data[output_key], _, contains_zero_extrinsic = extracted
            if role == "state" and self._camera_spatial_required:
                data[CAMERA_REFERENCE_TRANSFORM_KEY] = data[output_key]
            data[CAMERA_SKIP_COMPENSATION_KEY] = bool(
                data.get(CAMERA_SKIP_COMPENSATION_KEY, False) or contains_zero_extrinsic
            )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                df = pd.read_parquet(parquet_path)
                self.curr_traj_id = trajectory_id
                self.curr_traj_data = df
                return df
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:  # TODO check detail later
            chunk_index = self.get_episode_chunk(trajectory_id)

            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)

            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()

            # fix timestamp from epis index to file index
            from_timestamp = self.trajectory_ids_to_metadata[trajectory_id][
                "videos/observation.images.wrist/from_timestamp"
            ]
            episode_data["timestamp"] = episode_data["timestamp"] + from_timestamp

            return episode_data

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}")
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video_relpath(self, trajectory_id: int, key: str) -> str:
        """Return the stable dataset-relative path used by a quality index."""
        subkey = key.replace("video.", "", 1) if key.startswith("video.") else key
        return self.get_video_path(trajectory_id, subkey).relative_to(self.dataset_path).as_posix()

    def get_timestamps_for_deltas(
        self,
        trajectory_id: int,
        base_index: int,
        delta_indices: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Return timestamps for exactly the clamped frames that video decoding will use."""
        trajectory_data = self.get_trajectory_data(trajectory_id)
        if "timestamp" not in trajectory_data.columns:
            raise ValueError(f"No timestamp found for trajectory_id={trajectory_id} in {self.dataset_name}")
        if len(trajectory_data) == 0:
            raise ValueError(f"Empty trajectory_id={trajectory_id} in {self.dataset_name}")
        step_indices = np.asarray(delta_indices, dtype=np.int64) + int(base_index)
        step_indices = np.clip(step_indices, 0, len(trajectory_data) - 1)
        return np.asarray(trajectory_data["timestamp"].to_numpy()[step_indices], dtype=np.float64)

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        # import ipdb; ipdb.set_trace()
        video_timestamp = timestamp[step_indices]
        # try:
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,  # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )
        # except Exception as e:
        #     print("\n[VIDEO READ FAIL]")
        #     print("video_path:", video_path.as_posix())
        #     print("trajectory_id:", trajectory_id, "key:", key)
        #     print("base_index:", base_index)
        #     print("step_indices:", step_indices[:10], "...", "len=", len(step_indices))
        #     print("video_timestamp:", video_timestamp[:10], "...", "min/max=", float(np.min(video_timestamp)), float(np.max(video_timestamp)))
        #     raise

    def get_video_frames_for_deltas(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
        delta_indices: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Decode explicit frame deltas through an overridable dataset hook.

        Native v3 datasets can apply independent per-camera timestamp offsets
        here; image-pair loading must not bypass their video routing.
        """
        timestamps = self.get_timestamps_for_deltas(trajectory_id, base_index, delta_indices)
        subkey = key.removeprefix("video.")
        return get_frames_by_timestamps(
            self.get_video_path(trajectory_id, subkey).as_posix(),
            timestamps,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        # We only materialize the rows we need (step_indices), with padding logic.
        col = self.curr_traj_data[le_key]  # pandas Series, each element is array-like

        # Determine feature slice for this subkey
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )

        # Determine padding strategy (same as before)
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]
        pad_mode = "first_last" if state_or_action_cfg.absolute else "zero"

        # Helper: fetch one row -> np.ndarray (D,)
        def _row_to_1d(idx: int) -> np.ndarray:
            v = col.iloc[idx]
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            # scalar -> (1,)
            if v.ndim == 0:
                v = v.reshape(1)
            # if (D,) OK; if (1, D) or something odd, flatten
            if v.ndim > 1:
                v = v.reshape(-1)
            return v

        T = int(max_length)

        # Pre-fetch pad rows if needed
        if pad_mode == "first_last":
            first_row = _row_to_1d(0)[le_indices]
            last_row = _row_to_1d(T - 1)[le_indices]

        out = []
        for si in step_indices:
            if 0 <= int(si) < T:
                out.append(_row_to_1d(int(si))[le_indices])
            else:
                if pad_mode == "zero":
                    out.append(np.zeros(len(le_indices), dtype=np.float32))
                else:
                    # first_last
                    out.append(first_row if int(si) < 0 else last_row)

        data_out = np.stack(out, axis=0)

        return data_out

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):  #
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]  # TODO check v2.0
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        # Get language instructions and filter out None values (invalid instructions)
        if original_key == "subtask_index":
            language_list = self.subtasks.reindex(task_indices)["task"].dropna().tolist()
        else:
            language_list = self.tasks.loc[task_indices]["task"].tolist()
        # Filter out None values (invalid instructions that were filtered in _process_task_text)
        language_list = [lang for lang in language_list if lang is not None]

        return language_list

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, "action") and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = build_export_action_stats(
                self.metadata,
                tag,
                reordered_keys,
                use_relative_action=self.enable_relative_action,
            )

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                # import pdb; pdb.set_trace()

                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(self.metadata.modalities.action, filtered_action_stats.keys())
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats
                add_relative_action_export_stats(
                    tag_stats,
                    self.metadata,
                    tag,
                    reordered_keys,
                    use_relative_action=self.enable_relative_action,
                )

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, "state") and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")

    def save_raw_dataset_statistics(
        self,
        save_path: Path | str,
        format: str = "json",
        raw_statistics: dict | None = None,
    ) -> None:
        """
        Save raw single-dataset statistics in the same flat format as `stats.json`.

        The saved json is keyed by original parquet column names (e.g. "observation.state",
        "action", "episode_index"), and each key contains:
        ["mean", "std", "min", "max", "q01", "q99"].
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute statistics directly from parquet files to match raw stats.json format.
        if raw_statistics is None:
            parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
            parquet_files_filtered = []
            for pf in parquet_files:
                # Keep behavior consistent with metadata-loading path.
                if "episode_033675.parquet" in pf.name:
                    continue
                parquet_files_filtered.append(pf)

            raw_statistics = calculate_dataset_statistics(parquet_files_filtered)

        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(raw_statistics, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Raw single dataset statistics saved to: {save_path}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed

                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        return self.get_step_data_for_modalities(trajectory_id, base_index)

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

_STAT_NAMES = ("mean", "std", "max", "min", "q01", "q99")


def _get_stat_value(stats_obj, stat_name: str):
    if isinstance(stats_obj, dict):
        return stats_obj[stat_name]
    return getattr(stats_obj, stat_name)


def _stat_value_to_json(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _statistical_values_to_dict(stats_obj) -> dict:
    return {stat_name: _stat_value_to_json(_get_stat_value(stats_obj, stat_name)) for stat_name in _STAT_NAMES}


def _validate_flat_statistical_values(stats_obj, context: str) -> None:
    for stat_name in _STAT_NAMES:
        arr = np.asarray(_get_stat_value(stats_obj, stat_name))
        if arr.ndim > 1:
            raise ValueError(
                f"Expected flattened 1-D statistics for {context}.{stat_name}, got shape {arr.shape}. "
                "Regenerate relative action statistics with the horizon-flattened mode."
            )


def get_relative_action_keys_for_tag(tag: str) -> list[str]:
    try:
        action_config = MODALITY_CONFIGS[tag]["action"]
    except KeyError:
        return []
    if action_config.action_configs is None:
        return []
    return [
        key
        for key, action_cfg in zip(action_config.modality_keys, action_config.action_configs)
        if action_cfg.rep == ActionRepresentation.RELATIVE
    ]


def build_export_action_stats(
    metadata: DatasetMetadata,
    tag: str,
    ordered_keys,
    use_relative_action: bool,
) -> dict:
    action_stats = getattr(metadata.statistics, "action", {}) or {}
    relative_stats = getattr(metadata.statistics, "relative_action", {}) or {}
    relative_keys = set(get_relative_action_keys_for_tag(tag))
    if use_relative_action and not relative_keys:
        raise ValueError(f"Relative action is enabled for tag={tag}, but MODALITY_CONFIGS has no RELATIVE action keys.")

    filtered_action_stats = {}
    for key in ordered_keys:
        if key not in action_stats:
            continue
        if use_relative_action and key in relative_keys:
            if key not in relative_stats:
                raise ValueError(
                    f"Relative action statistics missing for tag={tag}, key={key}; "
                    f"available={list(relative_stats.keys())}."
                )
            _validate_flat_statistical_values(relative_stats[key], f"{tag}.relative_action.{key}")
            filtered_action_stats[key] = relative_stats[key]
        else:
            _validate_flat_statistical_values(action_stats[key], f"{tag}.action.{key}")
            filtered_action_stats[key] = action_stats[key]
    return filtered_action_stats


def add_relative_action_export_stats(
    tag_stats: dict,
    metadata: DatasetMetadata,
    tag: str,
    ordered_keys,
    use_relative_action: bool,
) -> None:
    if not use_relative_action:
        return
    used_key_set = set(ordered_keys)
    relative_keys = [key for key in get_relative_action_keys_for_tag(tag) if key in used_key_set]
    if not relative_keys:
        raise ValueError(f"Relative action is enabled for tag={tag}, but none of its RELATIVE keys are used.")
    relative_stats = getattr(metadata.statistics, "relative_action", {}) or {}
    for key in relative_keys:
        if key not in relative_stats:
            raise ValueError(
                f"Relative action statistics missing for tag={tag}, key={key}; "
                f"available={list(relative_stats.keys())}."
            )
        _validate_flat_statistical_values(relative_stats[key], f"{tag}.relative_action.{key}")
        tag_stats[f"relative_action.{key}"] = _statistical_values_to_dict(relative_stats[key])


def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.

    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.

    Returns:
        dict: Combined statistics
    """
    combined_stats = {"mean": [], "std": [], "max": [], "min": [], "q01": [], "q99": []}

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in _STAT_NAMES:
            stat_value = _get_stat_value(subkey_stats, stat_name)
            stat_arr = np.asarray(stat_value)
            if stat_arr.ndim > 1:
                raise ValueError(
                    f"Expected flattened 1-D statistics for {subkey}.{stat_name}, got shape {stat_arr.shape}"
                )
            if stat_arr.ndim == 1:
                combined_stats[stat_name].extend(stat_arr.tolist())
            else:
                combined_stats[stat_name].append(float(stat_arr))

    return combined_stats


def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.

    Returns:
        list[bool]: List of mask values
    """
    mask = []

    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, "shape") and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()

            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True

    return mask


def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        dataset_sampling_alpha: float | None = None,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            dataset_sampling_alpha (float | None): If set, use weight * dataset_length**alpha for dataset
                sampling. If unset, preserve the legacy balance_dataset_weights behavior.
            seed (int): Random seed for sampling.
            data_cfg (dict): Optional data settings. image_pair_only=True requires return_image_pair=True
                and omits the legacy image output and its separate decoding/augmentation pass.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.dataset_sampling_alpha = dataset_sampling_alpha
        if self.dataset_sampling_alpha is not None:
            self.dataset_sampling_alpha = float(self.dataset_sampling_alpha)
            if not np.isfinite(self.dataset_sampling_alpha) or self.dataset_sampling_alpha < 0:
                raise ValueError(
                    "dataset_sampling_alpha must be a finite non-negative number, " f"got {self.dataset_sampling_alpha}"
                )
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None
        self.image_pair_only = self._cfg_get("image_pair_only", False) not in (None, False, "False", "false", "", 0)
        if self.image_pair_only and self._cfg_get("return_image_pair", False) in (None, False, "False", "false", "", 0):
            raise ValueError("image_pair_only=True requires return_image_pair=True")
        self._image_pair_color_jitter = self._build_image_pair_color_jitter()
        max_zero_action_frames = self._cfg_get("max_zero_action_frames", 8)
        if max_zero_action_frames is None or max_zero_action_frames is False:
            self.max_zero_action_frames = None
        else:
            self.max_zero_action_frames = int(max_zero_action_frames)
            if self.max_zero_action_frames < 0:
                raise ValueError(
                    "max_zero_action_frames must be non-negative, or null/false to disable it; "
                    f"got {max_zero_action_frames}"
                )
        if any(getattr(dataset, "is_video_only", False) for dataset in self.datasets):
            if self._cfg_get("return_image_pair", False) in ["False", False, None]:
                raise ValueError("Video-only datasets require return_image_pair=True")

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.asarray(dataset_sampling_weights, dtype=np.float64)
        print(f"Dataset sampling weights: {self._dataset_sampling_weights}")
        print(f"balance_dataset_weights: {self.balance_dataset_weights}")
        print(f"dataset_sampling_alpha: {self.dataset_sampling_alpha}")
        if self.dataset_sampling_alpha is not None:
            self._dataset_sampling_weights *= self._dataset_lengths.astype(np.float64) ** self.dataset_sampling_alpha
        elif self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        print(f"After dataset length weighting: {self._dataset_sampling_weights}")
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print("Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum
        print(f"After normalize weights: {self._dataset_sampling_weights}")
        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            if getattr(dataset, "is_video_only", False):
                self._trajectory_sampling_weights.append(np.empty(0, dtype=np.float64))
                continue
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)
        # print(f'After trajectory sampling weights: {self._trajectory_sampling_weights}')
        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def _sample_step_with_dataset_index(
        self,
        index: int,
    ) -> tuple[int, LeRobotSingleDataset, int, int, int | None]:
        """Sample a dataset and step while retaining the selected dataset index."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        if getattr(dataset, "is_video_only", False):
            if dataset.sample_quality_index is not None:
                trajectory_id, base_index, logical_delta = dataset.sample_quality_pair(rng)
                return int(dataset_index), dataset, trajectory_id, base_index, logical_delta
            trajectory_id, base_index = dataset.sample_step(rng)
            return int(dataset_index), dataset, trajectory_id, base_index, None

        if dataset.sample_quality_index is not None:
            trajectory_id, base_index, logical_delta = dataset.sample_quality_step(rng)
            return int(dataset_index), dataset, trajectory_id, base_index, logical_delta

        # Sample trajectory
        # trajectory_index = rng.choice(
        #     len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        # )
        # trajectory_id = dataset.trajectory_ids[trajectory_index]

        # # Sample step
        # base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        # return dataset, trajectory_id, base_index
        single_step_index = rng.choice(len(dataset.all_steps))
        row = dataset.all_steps[single_step_index]
        trajectory_id, base_index = int(row[0]), int(row[1])
        return int(dataset_index), dataset, trajectory_id, base_index, None

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        _, dataset, trajectory_id, base_index, _ = self._sample_step_with_dataset_index(index)
        return dataset, trajectory_id, base_index

    def _resample_step_in_dataset(
        self,
        dataset_index: int,
        index: int,
        quality_attempt: int,
    ) -> tuple[int, int]:
        """Deterministically resample only the step, preserving mixture selection."""
        dataset = self.datasets[dataset_index]
        seed = safe_hash(("quality_retry", self.epoch, index, self.seed, dataset_index, quality_attempt))
        rng = np.random.default_rng(seed)
        if getattr(dataset, "is_video_only", False):
            return dataset.sample_step(rng)
        row = dataset.all_steps[rng.choice(len(dataset.all_steps))]
        return int(row[0]), int(row[1])

    @staticmethod
    def _discretize_episode_speed(trajectory_length: int, bin_size: int = 500) -> int:
        """Bin episode length to the nearest multiple of bin_size timesteps."""
        return int(round(trajectory_length / bin_size) * bin_size)

    def _get_episode_metadata_dropout_flags(self) -> tuple[bool, bool]:
        """Return whether to include speed and embodiment in the language instruction.

        During training, episode metadata is dropped entirely with 15% probability.
        When not fully dropped, speed and embodiment are each dropped independently
        with 5% probability. Control mode is never dropped.
        """
        include_speed = True
        include_embodiment = True

        if self.mode != "train":
            return include_speed, include_embodiment

        episode_metadata_dropout_prob = 0.15
        speed_dropout_prob = 0.05
        embodiment_dropout_prob = 0.05
        if self.data_cfg is not None:
            episode_metadata_dropout_prob = float(
                self.data_cfg.get("episode_metadata_dropout_prob", episode_metadata_dropout_prob)
            )
            speed_dropout_prob = float(self.data_cfg.get("speed_dropout_prob", speed_dropout_prob))
            embodiment_dropout_prob = float(self.data_cfg.get("embodiment_dropout_prob", embodiment_dropout_prob))

        if random.random() < episode_metadata_dropout_prob:
            return False, False

        if random.random() < speed_dropout_prob:
            include_speed = False
        if random.random() < embodiment_dropout_prob:
            include_embodiment = False

        return include_speed, include_embodiment

    def _build_language_instruction_taskandsubtask(self, dataset, data: dict, trajectory_length: int) -> str:
        """Build hierarchical language like: Task: ... Subtask: ... Speed: ... Control Mode: ... Embodiment: ..."""
        task = None
        subtask = None
        speed_bin_size = 500
        if self.data_cfg is not None:
            speed_bin_size = int(self.data_cfg.get("speed_bin_size", 500))

        language_keys = dataset.modality_keys["language"]
        for language_index, language_key in enumerate(language_keys):
            language_values = data.get(language_key)
            if language_values is None or len(language_values) == 0:
                continue

            language_text = str(language_values[0]).strip()
            if not language_text:
                continue

            if language_index == 0:
                task = language_text
            elif language_index == 1:
                subtask = language_text

        language_parts = []
        if task:
            language_parts.append(("Task", task))
        if subtask:
            language_parts.append(("Subtask", subtask))

        # include_speed, include_embodiment = self._get_episode_metadata_dropout_flags()

        # if include_speed:
        #     discretized_speed = self._discretize_episode_speed(trajectory_length, bin_size=speed_bin_size)
        #     language_parts.append(("Speed", f"{discretized_speed}"))

        # if dataset.control_mode is not None:
        #     language_parts.append(("Control Mode", dataset.control_mode))

        # if include_embodiment and dataset.language_embodiment is not None:
        #     language_parts.append(("Embodiment", dataset.language_embodiment))

        return " ".join(f"{label}: {text.rstrip('.')}." for label, text in language_parts)

    def _build_language_instruction(self, dataset, data: dict, trajectory_length: int) -> str:
        """Build hierarchical language like: Task: ... Subtask: ... Speed: ... Control Mode: ... Embodiment: ..."""
        task = None
        subtask = None
        speed_bin_size = 500
        if self.data_cfg is not None:
            speed_bin_size = int(self.data_cfg.get("speed_bin_size", 500))

        language_keys = dataset.modality_keys["language"]
        for language_index, language_key in enumerate(language_keys):
            language_values = data.get(language_key)
            if language_values is None or len(language_values) == 0:
                continue

            language_text = str(language_values[0]).strip()
            if not language_text:
                continue

            if language_index == 0:
                task = language_text
            elif language_index == 1:
                subtask = language_text

        language_parts = []
        if task:
            language_parts.append(("Task", task))
        if subtask:
            language_parts.append(("Subtask", subtask))

        include_speed, include_embodiment = self._get_episode_metadata_dropout_flags()

        if include_speed:
            discretized_speed = self._discretize_episode_speed(trajectory_length, bin_size=speed_bin_size)
            language_parts.append(("Speed", f"{discretized_speed}"))

        if dataset.control_mode is not None:
            control_mode_text = str(dataset.control_mode)
            prompt_format = str(self._cfg_get("control_mode_prompt_format", "legacy"))
            if prompt_format == "ee_representation_v1" and control_mode_text == "ee":
                action_representation = self._cfg_get("action_representation", None)
                if action_representation == "relative":
                    control_mode_text = "ee state-relative"
                elif action_representation == "absolute":
                    control_mode_text = "ee absolute"
                else:
                    raise ValueError(
                        "EE samples using control_mode_prompt_format='ee_representation_v1' "
                        f"require action_representation relative/absolute, got {action_representation!r}"
                    )
            elif prompt_format not in ("legacy", "ee_representation_v1"):
                raise ValueError(f"Unsupported control_mode_prompt_format={prompt_format!r}")
            language_parts.append(("Control Mode", control_mode_text))

        if include_embodiment and dataset.language_embodiment is not None:
            language_parts.append(("Embodiment", dataset.language_embodiment))

        reference_frame = dataset.eef_frame
        relative_eef_frame = str(self._cfg_get("relative_eef_frame", "eef")).lower().replace("-", "_")
        if (
            dataset.control_mode == "ee"
            and self._cfg_get("action_representation", None) == "relative"
            and relative_eef_frame == "camera_spatial"
        ):
            reference_frame = "camera_spatial"
        if reference_frame is not None:
            language_parts.append(("Reference Frame", reference_frame.replace("_", " ")))

        return " ".join(f"{label}: {text.rstrip('.')}." for label, text in language_parts)

    _getitem_count = 0

    def _cfg_get(self, key: str, default=None):
        data_cfg = getattr(self, "data_cfg", None)
        if data_cfg is None:
            return default
        return data_cfg.get(key, default)

    def _build_image_pair_color_jitter(self) -> torch.nn.Module | None:
        """Build the opt-in color jitter used by the head/wrist image-pair path."""
        from torchvision.transforms.v2 import ColorJitter

        config = self._cfg_get("image_pair_color_jitter", None)
        if config is None or config is False or (isinstance(config, str) and config.lower() in {"", "false", "off"}):
            return None
        if not hasattr(config, "get"):
            raise TypeError("image_pair_color_jitter must be a mapping with enabled/brightness/contrast/saturation/hue")

        enabled = config.get("enabled", True)
        if enabled is False or (isinstance(enabled, str) and enabled.lower() in {"", "false", "off", "0"}):
            return None
        try:
            parameters = {
                name: float(config.get(name, default))
                for name, default in (
                    ("brightness", 0.0),
                    ("contrast", 0.0),
                    ("saturation", 0.0),
                    ("hue", 0.0),
                )
            }
        except (TypeError, ValueError) as error:
            raise ValueError("image_pair_color_jitter parameters must be finite scalar numbers") from error
        if not all(np.isfinite(value) for value in parameters.values()):
            raise ValueError(f"image_pair_color_jitter parameters must be finite, got {parameters}")
        return ColorJitter(**parameters)

    def _apply_image_pair_color_jitter(self, image_pair: np.ndarray) -> np.ndarray:
        """Apply one sampled jitter consistently to all frames and views."""
        if self.mode != "train" or self._image_pair_color_jitter is None:
            return image_pair

        image_pair_tensor = torch.as_tensor(np.asarray(image_pair))
        if image_pair_tensor.ndim != 5 or image_pair_tensor.shape[-1] != 3:
            raise ValueError("image_pair color jitter expects [T,V,H,W,3], " f"got {tuple(image_pair_tensor.shape)}")
        channels_first = image_pair_tensor.permute(0, 1, 4, 2, 3).contiguous()
        original_shape = channels_first.shape
        # ColorJitter samples its factors once per call. Flattening T and V
        # therefore preserves temporal/view consistency for each image pair.
        jittered = self._image_pair_color_jitter(channels_first.flatten(0, 1))
        jittered = jittered.reshape(original_shape).permute(0, 1, 3, 4, 2)
        return jittered.contiguous().cpu().numpy()

    def _get_image_pair_delta_indices(
        self,
        sample_key: tuple | None = None,
        temporal_downsample_stride: int = 1,
        robot_type: str | None = None,
    ) -> list[int]:
        """Sample image-pair deltas in logical steps, then map them to source frames."""
        temporal_downsample_stride = max(1, int(temporal_downsample_stride))
        delta_indices = self._cfg_get("image_pair_delta_indices", [0, 1])
        overrides = self._cfg_get("dataset_config_by_robot_type", {}) or {}
        delta_indices = overrides.get(robot_type, {}).get("image_pair_delta_indices", delta_indices)
        delta_indices = list(delta_indices)
        if len(delta_indices) != 2:
            raise ValueError(f"image_pair_delta_indices must contain exactly two values, got {delta_indices}")
        delta_indices = [int(delta_indices[0]), int(delta_indices[1])]

        random_range = self._cfg_get("image_pair_random_future_delta_range", None)
        if random_range in (None, False, "False"):
            return [delta * temporal_downsample_stride for delta in delta_indices]

        random_range = list(random_range)
        if len(random_range) != 2:
            raise ValueError(f"image_pair_random_future_delta_range must contain exactly two values, got {random_range}")
        min_delta, max_delta = sorted((int(random_range[0]), int(random_range[1])))
        if min_delta < 1:
            raise ValueError(f"image_pair_random_future_delta_range min must be >= 1, got {min_delta}")
        if sample_key is None:
            sample_key = (self.epoch, self.seed, LeRobotMixtureDataset._getitem_count)
        seed = safe_hash(("image_pair_random_future_delta", sample_key))
        rng = np.random.default_rng(seed)
        logical_delta_indices = [delta_indices[0], int(rng.integers(min_delta, max_delta + 1))]
        return [delta * temporal_downsample_stride for delta in logical_delta_indices]

    def _get_image_pair_size(self) -> tuple[int, int]:
        image_size = self._cfg_get("image_size", None)
        if image_size is None:
            image_size = self._cfg_get("default_image_resolution", [3, 224, 224])
        image_size = list(image_size)
        if len(image_size) == 3:
            return int(image_size[1]), int(image_size[2])
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
        raise ValueError(f"image_size/default_image_resolution must be [H, W] or [C, H, W], got {image_size}")

    @staticmethod
    def _parse_hw_size(size, field_name: str) -> tuple[int, int]:
        if isinstance(size, int):
            return int(size), int(size)
        size = list(size)
        if len(size) == 3:
            return int(size[1]), int(size[2])
        if len(size) == 2:
            return int(size[0]), int(size[1])
        raise ValueError(f"{field_name} must be int, [H, W], or [C, H, W], got {size}")

    @staticmethod
    def _as_string_list(value) -> list[str]:
        if value in (None, False, "False"):
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in list(value)]

    @staticmethod
    def _cfg_lookup(config, key: str, default=None):
        if config is None:
            return default
        return config.get(key, default) if hasattr(config, "get") else default

    @staticmethod
    def _video_key_matches(candidate: str, video_key: str) -> bool:
        candidate = str(candidate)
        return (
            candidate == video_key or f"video.{candidate}" == video_key or candidate == video_key.replace("video.", "")
        )

    @classmethod
    def _first_matching_video_key(cls, candidates, video_keys: list[str]) -> str | None:
        for candidate in cls._as_string_list(candidates):
            for video_key in video_keys:
                if cls._video_key_matches(candidate, video_key):
                    return video_key
        return None

    @staticmethod
    def _frame_to_hwc_uint8(frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim != 3:
            raise ValueError(f"Expected image frame [H, W, C] or [C, H, W], got {tuple(frame.shape)}")
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] > 3:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.asarray(frame)
            if frame.size > 0 and np.max(frame) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    @classmethod
    def _resize_image_pair_frame(cls, frame, target_size: tuple[int, int]) -> np.ndarray:
        frame = cls._frame_to_hwc_uint8(frame)
        height, width = target_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        src_height, src_width = frame.shape[:2]
        if src_height <= 0 or src_width <= 0:
            return canvas
        scale = min(height / src_height, width / src_width)
        resized_height = max(1, int(round(src_height * scale)))
        resized_width = max(1, int(round(src_width * scale)))
        resized = np.asarray(Image.fromarray(frame).resize((resized_width, resized_height), Image.BICUBIC))
        y_offset = (height - resized_height) // 2
        x_offset = (width - resized_width) // 2
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        return canvas

    def _get_image_pair_layout(self) -> str:
        return str(self._cfg_get("image_pair_view_layout", self._cfg_get("image_pair_layout", "views"))).lower()

    def _get_head_wrist_sizes(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        head_size = self._parse_hw_size(self._cfg_get("image_pair_head_size", [224, 224]), "image_pair_head_size")
        wrist_size = self._parse_hw_size(
            self._cfg_get("image_pair_wrist_size", [head_size[0] // 2, head_size[1] // 2]),
            "image_pair_wrist_size",
        )
        if head_size[1] != wrist_size[1] * 2:
            raise ValueError(
                "head_wrist image_pair layout expects head_width == wrist_width * 2, "
                f"got head_size={head_size}, wrist_size={wrist_size}"
            )
        output_size = (head_size[0] + wrist_size[0], head_size[1])
        configured_size = self._cfg_get("image_size", None)
        if configured_size is not None and self._cfg_get("image_pair_validate_image_size", True) not in (False, "False"):
            configured_size = self._parse_hw_size(configured_size, "image_size")
            if configured_size != output_size:
                raise ValueError(
                    "head_wrist image_pair layout expects datasets.vla_data.image_size to match "
                    f"head+wrist canvas {output_size}, got {configured_size}"
                )
        return head_size, wrist_size, output_size

    def _get_view_roles_config(self, dataset: LeRobotSingleDataset):
        roles = self._cfg_get("image_pair_view_roles", {}) or {}
        role_maps = (
            self._cfg_get("image_pair_view_roles_by_robot_type", {}) or {},
            self._cfg_get("image_pair_view_roles_by_dataset", {}) or {},
        )
        merged_roles = {}
        if hasattr(roles, "items"):
            merged_roles.update(dict(roles.items()))
        for role_map in role_maps:
            if not hasattr(role_map, "get"):
                continue
            override = role_map.get(dataset.robot_type, None) or role_map.get(dataset.dataset_name, None)
            if override is not None and hasattr(override, "items"):
                merged_roles.update(dict(override.items()))
        return merged_roles

    def _select_head_wrist_video_keys(self, dataset: LeRobotSingleDataset) -> tuple[str, str | None, str | None]:
        video_keys = list(dataset.modality_keys["video"])
        roles_cfg = self._get_view_roles_config(dataset)
        head_key = self._first_matching_video_key(self._cfg_lookup(roles_cfg, "head", []), video_keys)
        left_wrist_key = self._first_matching_video_key(self._cfg_lookup(roles_cfg, "wrist_left", []), video_keys)
        right_wrist_key = self._first_matching_video_key(self._cfg_lookup(roles_cfg, "wrist_right", []), video_keys)

        wrist_keywords = self._as_string_list(self._cfg_get("image_pair_wrist_keywords", ["wrist", "hand"]))
        head_keywords = self._as_string_list(
            self._cfg_get("image_pair_head_keywords", ["head", "ego", "front", "top", "exterior", "high", "main"])
        )
        lower_keys = {key: key.lower() for key in video_keys}
        wrist_keys = [key for key in video_keys if any(keyword.lower() in lower_keys[key] for keyword in wrist_keywords)]
        non_wrist_keys = [key for key in video_keys if key not in wrist_keys]

        if head_key is None:
            head_key = next(
                (key for key in non_wrist_keys if any(keyword.lower() in lower_keys[key] for keyword in head_keywords)),
                None,
            )
        if head_key is None:
            head_key = non_wrist_keys[0] if non_wrist_keys else (video_keys[0] if video_keys else None)
        if head_key is None:
            raise ValueError(f"No video views available for dataset {dataset.dataset_name}")

        if left_wrist_key is None:
            left_wrist_key = next(
                (key for key in wrist_keys if "left" in lower_keys[key] or "_l" in lower_keys[key]),
                None,
            )
        if right_wrist_key is None:
            right_wrist_key = next(
                (key for key in wrist_keys if "right" in lower_keys[key] or "_r" in lower_keys[key]),
                None,
            )

        remaining_wrist_keys = [key for key in wrist_keys if key not in {left_wrist_key, right_wrist_key, head_key}]
        if left_wrist_key is None and remaining_wrist_keys:
            left_wrist_key = remaining_wrist_keys.pop(0)
        if right_wrist_key is None and remaining_wrist_keys:
            right_wrist_key = remaining_wrist_keys.pop(0)
        if right_wrist_key == left_wrist_key:
            right_wrist_key = None
        return head_key, left_wrist_key, right_wrist_key

    def _get_optional_video_frames_for_deltas(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        video_key: str | None,
        base_index: int,
        delta_indices: list[int],
        *,
        required: bool,
    ) -> np.ndarray | None:
        if video_key is None:
            return None
        subkey = video_key.replace("video.", "")
        video_path = dataset.get_video_path(trajectory_id, subkey)
        if not video_path.exists():
            if required:
                raise FileNotFoundError(f"Required head video is missing: {video_path}")
            return None
        return self._get_video_frames_for_deltas(dataset, trajectory_id, video_key, base_index, delta_indices)

    def _build_head_wrist_image_pair(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        base_index: int,
        delta_indices: list[int],
    ) -> np.ndarray:
        head_size, wrist_size, output_size = self._get_head_wrist_sizes()
        head_key, left_wrist_key, right_wrist_key = self._select_head_wrist_video_keys(dataset)
        head_frames = self._get_optional_video_frames_for_deltas(
            dataset, trajectory_id, head_key, base_index, delta_indices, required=True
        )
        left_wrist_frames = self._get_optional_video_frames_for_deltas(
            dataset, trajectory_id, left_wrist_key, base_index, delta_indices, required=False
        )
        right_wrist_frames = self._get_optional_video_frames_for_deltas(
            dataset, trajectory_id, right_wrist_key, base_index, delta_indices, required=False
        )

        image_pair = []
        for frame_idx in range(len(delta_indices)):
            canvas = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
            canvas[: head_size[0], : head_size[1]] = self._resize_image_pair_frame(
                head_frames[frame_idx],
                head_size,
            )
            if left_wrist_frames is not None:
                canvas[
                    head_size[0] : head_size[0] + wrist_size[0],
                    : wrist_size[1],
                ] = self._resize_image_pair_frame(left_wrist_frames[frame_idx], wrist_size)
            if right_wrist_frames is not None:
                canvas[
                    head_size[0] : head_size[0] + wrist_size[0],
                    wrist_size[1] : wrist_size[1] * 2,
                ] = self._resize_image_pair_frame(right_wrist_frames[frame_idx], wrist_size)
            image_pair.append(canvas)
        return np.stack(image_pair, axis=0)[:, None]

    def _get_video_frames_for_deltas(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        video_key: str,
        base_index: int,
        delta_indices: list[int],
    ) -> np.ndarray:
        video_frames_hook = getattr(dataset, "get_video_frames_for_deltas", None)
        if callable(video_frames_hook):
            return video_frames_hook(trajectory_id, video_key, base_index, delta_indices)
        # Retain the legacy duck-typed dataset contract for external adapters.
        trajectory_index = dataset.get_trajectory_index(trajectory_id)
        max_length = dataset.trajectory_lengths[trajectory_index]
        step_indices = np.asarray(delta_indices, dtype=np.int64) + int(base_index)
        step_indices = np.clip(step_indices, 0, max_length - 1)
        assert dataset.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in dataset.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamps = dataset.curr_traj_data["timestamp"].to_numpy()[step_indices]
        subkey = video_key.removeprefix("video.")
        return get_frames_by_timestamps(
            dataset.get_video_path(trajectory_id, subkey).as_posix(),
            timestamps,
            video_backend=dataset.video_backend,
            video_backend_kwargs=dataset.video_backend_kwargs,
        )

    def _build_image_pair(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        base_index: int,
        delta_indices: list[int],
    ) -> np.ndarray:
        if self._get_image_pair_layout() in {"head_wrist", "compose_head_wrist", "head_wrist_grid"}:
            image_pair = self._build_head_wrist_image_pair(dataset, trajectory_id, base_index, delta_indices)
            return self._apply_image_pair_color_jitter(image_pair)

        target_size = self._get_image_pair_size()
        prim_views, wrist_views = [], []
        for video_key in dataset.modality_keys["video"]:
            frames = self._get_video_frames_for_deltas(dataset, trajectory_id, video_key, base_index, delta_indices)
            frames = np.stack(
                [self._resize_image_pair_frame(frame, target_size) for frame in frames],
                axis=0,
            )
            if "wrist" not in video_key and "hand" not in video_key:
                prim_views.append(frames)
            else:
                wrist_views.append(frames)

        ordered_views = prim_views + wrist_views
        if not ordered_views:
            raise ValueError(f"No video views available for dataset {dataset.dataset_name}")
        image_pair = np.stack(ordered_views, axis=1)
        return self._apply_image_pair_color_jitter(image_pair)

    def _build_video_only_image_pair(self, frames: np.ndarray) -> np.ndarray:
        """Apply the configured head/wrist view layout to one raw RGB video stream."""
        if len(frames) != 2:
            raise ValueError(f"Video-only samples require exactly two frames, got {len(frames)}")
        if self._get_image_pair_layout() in {"head_wrist", "compose_head_wrist", "head_wrist_grid"}:
            head_size, _, output_size = self._get_head_wrist_sizes()
            image_pair = []
            for frame in frames:
                canvas = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
                canvas[: head_size[0], : head_size[1]] = self._resize_image_pair_frame(frame, head_size)
                image_pair.append(canvas)
            return np.stack(image_pair, axis=0)[:, None]

        target_size = self._get_image_pair_size()
        resized = [self._resize_image_pair_frame(frame, target_size) for frame in frames]
        return np.stack(resized, axis=0)[:, None]

    @staticmethod
    def _apply_action_temporal_mask(
        action: np.ndarray,
        action_mask: np.ndarray | None,
        valid_frame_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if valid_frame_mask is None:
            return action, action_mask
        valid_frame_mask = np.asarray(valid_frame_mask, dtype=bool)
        if valid_frame_mask.ndim != 1 or valid_frame_mask.shape[0] != action.shape[0]:
            raise ValueError(
                f"valid_frame_mask must be [T] matching action [T, D], "
                f"got mask={valid_frame_mask.shape}, action={action.shape}"
            )
        if valid_frame_mask.all():
            return action, action_mask
        if action_mask is None:
            action_mask = np.ones_like(action, dtype=np.float16)
        action = action.copy()
        action_mask = action_mask.copy()
        action[~valid_frame_mask] = 0
        action_mask[~valid_frame_mask] = 0
        return action, action_mask

    @staticmethod
    def _get_action_valid_frame_mask(
        action_delta_indices: np.ndarray,
        *,
        step: int,
        trajectory_length: int,
        action_future_horizon: int | None,
        action_index_shift: int = 0,
    ) -> np.ndarray:
        """Return valid action targets, respecting boundaries and image horizon."""
        action_delta_indices = np.asarray(action_delta_indices, dtype=np.int64)
        valid_frame_mask = (step + action_delta_indices >= 0) & (step + action_delta_indices < trajectory_length)
        if action_future_horizon is not None:
            # action_index_shift moves targets forward (for example NW uses
            # [4, 8, ...] source-frame deltas). The image-pair horizon is
            # measured from the unshifted logical action sequence, so apply
            # the cutoff before that target shift; otherwise every shifted
            # dataset loses its final valid target.
            valid_frame_mask &= action_delta_indices - int(action_index_shift) < action_future_horizon
        return valid_frame_mask

    def _load_and_validate_raw_action(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        step: int,
        action_future_horizon: int | None,
    ) -> tuple[dict, np.ndarray]:
        """Load an action chunk and reject excessive zeros without touching video data."""
        raw_action_data = dataset.get_step_data_for_modalities(
            trajectory_id,
            step,
            modalities=("action",),
        )
        raw_action_list = [raw_action_data[key] for key in dataset.modality_keys["action"]]
        if isinstance(raw_action_list[0], torch.Tensor):
            raw_action = raw_action_list[0] if len(raw_action_list) == 1 else torch.cat(raw_action_list, dim=1)
        else:
            raw_action = raw_action_list[0] if len(raw_action_list) == 1 else np.concatenate(raw_action_list, axis=1)

        # curr_traj_data was populated by the action-only load, so this avoids the O(N)
        # np.where scan inside get_trajectory_index.
        trajectory_length = len(dataset.curr_traj_data)
        first_action_key = dataset.modality_keys["action"][0]
        valid_frame_mask = self._get_action_valid_frame_mask(
            dataset.delta_indices[first_action_key],
            step=step,
            trajectory_length=trajectory_length,
            action_future_horizon=action_future_horizon,
            action_index_shift=int(dataset.data_cfg.get("action_index_shift", 0)),
        )
        # A loaded per-dataset index has already evaluated this exact action
        # horizon/delta pair. Keep the runtime check only for the legacy path
        # so deterministic zero-action filtering cannot trigger outer retries.
        if dataset.sample_quality_index is None and valid_frame_mask.any() and self.max_zero_action_frames is not None:
            if isinstance(raw_action, torch.Tensor):
                valid_mask_t = torch.from_numpy(valid_frame_mask).to(device=raw_action.device)
                zero_action_mask = torch.all(raw_action[valid_mask_t] == 0, dim=-1)
                zero_action_count = int(torch.count_nonzero(zero_action_mask).item())
            else:
                zero_action_mask = np.all(raw_action[valid_frame_mask] == 0, axis=-1)
                zero_action_count = int(np.count_nonzero(zero_action_mask))
            if zero_action_count > self.max_zero_action_frames:
                raise ValueError(
                    "Found too many all-zero raw action frames in the non-padded action chunk "
                    f"for tag {dataset.tag_index}: {zero_action_count} exceeds "
                    f"max_zero_action_frames={self.max_zero_action_frames}"
                )
        return raw_action_data, valid_frame_mask

    def _infer_end_effector_type(self, dataset: LeRobotSingleDataset) -> str:
        declared = dataset.end_effector_type
        if declared not in (None, "", "unknown"):
            return str(declared)
        concat_transform = self._find_concat_transform(dataset)
        slots = set(getattr(concat_transform, "action_canonical_slots", {}).values())
        if slots & {"LEFT_DEX_HAND", "RIGHT_DEX_HAND", "LEFT_DEX_JOINT", "RIGHT_DEX_JOINT"}:
            return "hand"
        if slots & {"LEFT_GRIPPER", "RIGHT_GRIPPER"}:
            return "gripper"
        return "unknown"

    def _build_sample_meta(self, dataset: LeRobotSingleDataset) -> dict:
        canonical_robot_type = get_canonical_robot_type(dataset.robot_type)
        meta = {
            # Shards that share an embodiment tag are one logical dataset for
            # both normalization and action-encoder metadata conditioning.
            "dataset_id": dataset.tag,
            # Schema-only aliases must reuse the Stage-2 metadata embedding.
            # The action-schema mask carries the complementary-slot change.
            "robot_type": canonical_robot_type,
            "dataset_family": get_dataset_family(dataset.robot_type),
            "robot_family": get_robot_family(dataset.robot_type),
            "control_mode": dataset.control_mode or "unknown",
            "end_effector_type": self._infer_end_effector_type(dataset),
            "embodiment_tag": dataset.tag_index,
        }
        if dataset.control_mode == "ee":
            action_representation = self._cfg_get("action_representation", None)
            if action_representation is not None:
                meta["action_representation"] = str(action_representation)
        return meta

    @staticmethod
    def _get_task_index_for_step(dataset: LeRobotSingleDataset, step: int) -> int:
        """Read the task id for a sampled base step from the cached trajectory parquet."""
        trajectory = dataset.curr_traj_data
        if trajectory is None or "task_index" not in trajectory.columns:
            columns = [] if trajectory is None else list(trajectory.columns)
            raise ValueError(
                f"Task-filtered base supervision requires task_index in {dataset.dataset_name}; "
                f"available columns={columns}"
            )
        row_index = min(max(int(step), 0), len(trajectory) - 1)
        value = trajectory["task_index"].iloc[row_index]
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise ValueError(
                f"Expected scalar task_index in {dataset.dataset_name}, got shape {np.asarray(value).shape}"
            )
        return int(array[0])

    @staticmethod
    def _apply_base_supervision_policy(
        dataset: LeRobotSingleDataset,
        data: dict,
        step: int,
    ) -> dict:
        """Zero BASE_VELOCITY labels/mask outside a dataset's supervised tasks."""
        policy = dataset.data_cfg.get("base_supervision_policy")
        if policy is None:
            return data
        task_index = None
        if policy is not True:
            task_index = LeRobotMixtureDataset._get_task_index_for_step(dataset, step)
        if is_base_supervision_enabled(policy, task_index):
            return data

        action = data.get("action")
        action_mask = data.get("action_mask")
        base_slot = CanonicalSlot.BASE_VELOCITY
        if action is None or action_mask is None:
            raise ValueError(
                f"Base supervision for {dataset.dataset_name} requires canonical action/action_mask outputs"
            )
        if action.shape[-1] < base_slot.end or action_mask.shape[-1] < base_slot.end:
            raise ValueError(
                f"Canonical action dim is too small for BASE_VELOCITY [{base_slot.start}:{base_slot.end}]: "
                f"action={action.shape}, mask={action_mask.shape}"
            )
        action[..., base_slot.start : base_slot.end] = 0
        action_mask[..., base_slot.start : base_slot.end] = False
        return data

    def _sample_quality_eligible_video_pair(
        self,
        dataset_index: int,
        dataset,
        video_index: int,
        sample_seed: int,
        index: int,
        reference_delta: int,
    ) -> tuple[np.ndarray, int]:
        """Decode an accepted video-only frame pair while preserving mixture selection."""
        quality_filter = dataset.quality_filter
        if quality_filter is None:
            raise ValueError(f"Dataset {dataset.dataset_name} has no quality filter")

        rejected_counts: dict[str, int] = defaultdict(int)
        for quality_attempt in range(quality_filter.max_resample_attempts):
            if quality_attempt > 0:
                video_index, sample_seed = self._resample_step_in_dataset(dataset_index, index, quality_attempt)
            video_relpath = dataset.get_video_relpath(video_index)
            if not quality_filter.index.has_video(quality_filter.binding.dataset_id, video_relpath):
                rejected_counts["UNKNOWN"] += 1
                continue
            frames, timestamps, decoded_relpath = dataset.get_frame_pair_with_metadata(
                video_index,
                sample_seed,
                reference_delta,
            )
            frame_labels = quality_filter.index.lookup(
                quality_filter.binding.dataset_id,
                decoded_relpath,
                timestamps,
            )
            sample_label = worst_quality_label(frame_labels)
            accept_seed = safe_hash(
                (
                    "video_quality_accept",
                    self.epoch,
                    index,
                    self.seed,
                    dataset_index,
                    video_index,
                    sample_seed,
                    quality_attempt,
                )
            )
            unit_random = float(np.random.default_rng(accept_seed).random())
            if quality_filter.accepts(sample_label, unit_random):
                return frames, sample_label
            rejected_counts[quality_tier_name(sample_label)] += 1

        raise RuntimeError(
            f"Unable to sample an accepted quality tier from video-only dataset={dataset.dataset_name!r} "
            f"after {quality_filter.max_resample_attempts} attempts; rejected={dict(rejected_counts)}, "
            f"index={quality_filter.index.source_path}"
        )

    def _sample_quality_eligible_step(
        self,
        dataset_index: int,
        dataset: LeRobotSingleDataset,
        trajectory_id: int,
        step: int,
        index: int,
    ) -> tuple[int, int, list[int], int]:
        """Find an accepted two-frame sample without changing the selected mixture dataset."""
        quality_filter = dataset.quality_filter
        if quality_filter is None:
            raise ValueError(f"Dataset {dataset.dataset_name} has no quality filter")
        if self._cfg_get("return_image_pair", False) in (None, False, "False"):
            raise ValueError("A clip quality filter requires return_image_pair=True to define the two sampled frames")

        rejected_counts: dict[str, int] = defaultdict(int)
        for quality_attempt in range(quality_filter.max_resample_attempts):
            if quality_attempt > 0:
                trajectory_id, step = self._resample_step_in_dataset(dataset_index, index, quality_attempt)
            delta_indices = self._get_image_pair_delta_indices(
                sample_key=(
                    self.epoch,
                    index,
                    self.seed,
                    dataset.dataset_name,
                    trajectory_id,
                    step,
                    quality_attempt,
                ),
                temporal_downsample_stride=dataset.temporal_downsample_stride,
                robot_type=getattr(dataset, "robot_type", None),
            )
            timestamps = dataset.get_timestamps_for_deltas(trajectory_id, step, delta_indices)
            video_relpath = dataset.get_video_relpath(trajectory_id, quality_filter.binding.video_key)
            frame_labels = quality_filter.index.lookup(
                quality_filter.binding.dataset_id,
                video_relpath,
                timestamps,
            )
            sample_label = worst_quality_label(frame_labels)
            accept_seed = safe_hash(
                (
                    "quality_accept",
                    self.epoch,
                    index,
                    self.seed,
                    dataset_index,
                    trajectory_id,
                    step,
                    quality_attempt,
                )
            )
            unit_random = float(np.random.default_rng(accept_seed).random())
            if quality_filter.accepts(sample_label, unit_random):
                return trajectory_id, step, delta_indices, sample_label
            rejected_counts[quality_tier_name(sample_label)] += 1

        raise RuntimeError(
            f"Unable to sample an accepted quality tier from dataset={dataset.dataset_name!r} after "
            f"{quality_filter.max_resample_attempts} attempts; rejected={dict(rejected_counts)}, "
            f"index={quality_filter.index.source_path}"
        )

    @classmethod
    def _apply_non_video_transforms(cls, transform: ModalityTransform, data: dict) -> dict:
        """Keep action/state/language processing without decoding legacy images.

        Do not edit the dataset's shared transform stack: its original metadata,
        train/eval state, and statistics-export behavior must remain available.
        """
        from starVLA.dataloader.gr00t_lerobot.transform.gr00ttransform import GR00TTransform

        if isinstance(transform, ComposedModalityTransform):
            for index, child in enumerate(transform.transforms):
                try:
                    data = cls._apply_non_video_transforms(child, data)
                except Exception as exc:
                    raise ValueError(f"Error applying non-video transform {index} to data: {exc}") from exc
            return data

        targets = getattr(transform, "apply_to", ())
        video_targets = [key for key in targets if key == "video" or key.startswith("video.")]
        if video_targets:
            if len(video_targets) != len(targets):
                raise ValueError(
                    f"image_pair_only cannot skip mixed video/non-video transform {type(transform).__name__}; "
                    "split its video processing into a separate transform"
                )
            return data
        if isinstance(transform, GR00TTransform):
            # apply() only needs pixels to infer the batch dimension. Mixture
            # samples are unbatched; retain all existing state/action padding,
            # canonical masks, and embodiment metadata through apply_single().
            return transform.apply_single(data)
        return transform(data)

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """

        LeRobotMixtureDataset._getitem_count += 1
        if LeRobotMixtureDataset._getitem_count % 50000 == 0:
            gc.collect()  # Force garbage collection every 50000 samples to manage memory
        max_retries = 50
        last_exception = None

        for attempt in range(max_retries):
            try:
                dataset_index, dataset, trajectory_id, step, indexed_logical_delta = (
                    self._sample_step_with_dataset_index(index)
                )
                if getattr(dataset, "is_video_only", False):
                    quality_sample_label = None
                    if indexed_logical_delta is not None:
                        frames, timestamps, video_relpath = dataset.get_indexed_frame_pair_with_metadata(
                            trajectory_id,
                            step,
                            indexed_logical_delta,
                        )
                        if dataset.quality_filter is not None:
                            frame_labels = dataset.quality_filter.index.lookup(
                                dataset.quality_filter.binding.dataset_id,
                                video_relpath,
                                timestamps,
                            )
                            quality_sample_label = worst_quality_label(frame_labels)
                    else:
                        image_pair_delta_indices = self._get_image_pair_delta_indices(
                            sample_key=(self.epoch, index, self.seed, dataset.dataset_name, trajectory_id, step),
                            temporal_downsample_stride=getattr(dataset, "temporal_downsample_stride", 1),
                            robot_type=getattr(dataset, "robot_type", None),
                        )
                        reference_delta = abs(int(image_pair_delta_indices[1]) - int(image_pair_delta_indices[0]))
                    if indexed_logical_delta is None and dataset.quality_filter is None:
                        frames = dataset.get_frame_pair(trajectory_id, step, reference_delta)
                    elif indexed_logical_delta is None:
                        frames, quality_sample_label = self._sample_quality_eligible_video_pair(
                            dataset_index,
                            dataset,
                            trajectory_id,
                            step,
                            index,
                            reference_delta,
                        )
                    image_pair = self._build_video_only_image_pair(frames)
                    sample = {
                        "lang": str(self._cfg_get("video_only_language", "Observe the human activity.")),
                        "embodiment_tag": dataset.tag_index,
                        "dataset_id": dataset.tag,
                        "action_valid": False,
                        "is_video_only": True,
                    }
                    if not self.image_pair_only:
                        sample["image"] = [Image.fromarray(image_pair[0, 0]).resize((224, 224))]
                    sample["image_pair"] = image_pair
                    if self._cfg_get("include_meta", False) not in ["False", False, None]:
                        sample.update(self._build_sample_meta(dataset))
                    if (
                        quality_sample_label is not None
                        and dataset.quality_filter is not None
                        and dataset.quality_filter.return_quality_label
                    ):
                        sample["quality_label"] = quality_tier_name(quality_sample_label)
                    return sample

                image_pair_delta_indices = None
                quality_sample_label = None
                if indexed_logical_delta is not None:
                    image_pair_delta_indices = [
                        0,
                        indexed_logical_delta * dataset.temporal_downsample_stride,
                    ]
                    if dataset.quality_filter is not None:
                        timestamps = dataset.get_timestamps_for_deltas(
                            trajectory_id,
                            step,
                            image_pair_delta_indices,
                        )
                        video_relpath = dataset.get_video_relpath(
                            trajectory_id,
                            dataset.quality_filter.binding.video_key,
                        )
                        frame_labels = dataset.quality_filter.index.lookup(
                            dataset.quality_filter.binding.dataset_id,
                            video_relpath,
                            timestamps,
                        )
                        quality_sample_label = worst_quality_label(frame_labels)
                elif dataset.quality_filter is not None:
                    trajectory_id, step, image_pair_delta_indices, quality_sample_label = (
                        self._sample_quality_eligible_step(
                            dataset_index,
                            dataset,
                            trajectory_id,
                            step,
                            index,
                        )
                    )
                key = dataset.modality_keys["video"][0].replace("video.", "")
                video_path = dataset.get_video_path(trajectory_id, key)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video path does not exist: {video_path}")

                action_future_horizon = None
                if image_pair_delta_indices is None and self._cfg_get("return_image_pair", False) not in [
                    "False",
                    False,
                    None,
                ]:
                    image_pair_delta_indices = self._get_image_pair_delta_indices(
                        sample_key=(self.epoch, index, self.seed, dataset.dataset_name, trajectory_id, step),
                        temporal_downsample_stride=dataset.temporal_downsample_stride,
                        robot_type=getattr(dataset, "robot_type", None),
                    )
                if image_pair_delta_indices is not None and self._cfg_get(
                    "image_pair_random_future_delta_range", None
                ) not in (None, False, "False"):
                    action_future_horizon = max(image_pair_delta_indices) - min(image_pair_delta_indices)
                # Invalid NW hand-pose chunks are common. Reject them before requesting any video modality so the
                # retry path only pays for parquet/action access, not video decoding.
                raw_data, valid_frame_mask = self._load_and_validate_raw_action(
                    dataset,
                    trajectory_id,
                    step,
                    action_future_horizon,
                )
                remaining_modalities = tuple(
                    modality
                    for modality in dataset.modality_keys
                    if modality != "action" and not (self.image_pair_only and modality == "video")
                )
                remaining_data = dataset.get_step_data_for_modalities(
                    trajectory_id,
                    step,
                    modalities=remaining_modalities,
                )
                skip_camera_compensation = bool(
                    raw_data.get(CAMERA_SKIP_COMPENSATION_KEY, False)
                    or remaining_data.get(CAMERA_SKIP_COMPENSATION_KEY, False)
                )
                raw_data.update(remaining_data)
                if skip_camera_compensation:
                    raw_data[CAMERA_SKIP_COMPENSATION_KEY] = True
                data = (
                    self._apply_non_video_transforms(dataset.transforms, raw_data)
                    if self.image_pair_only
                    else dataset.transforms(raw_data)
                )
                data = self._apply_base_supervision_policy(dataset, data, step)
                # Process video: check if ConcatTransform has been applied
                prim_images = []
                wrist_views = []

                if not self.image_pair_only and "video" in data and not any(k.startswith("video.") for k in data.keys()):
                    # Videos have been concatenated by ConcatTransform
                    # data["video"] shape: [T, V, H, W, C] where V is number of views
                    video_data = data["video"]
                    num_views = video_data.shape[1]  # Number of video views

                    for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
                        # Extract the view at view_idx
                        image = video_data[0, view_idx, :, :, :]  # Get first frame [H, W, C]
                        # image = Image.fromarray(image)

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)
                elif not self.image_pair_only:
                    # Videos are still individual keys (no ConcatTransform applied)
                    for video_key in dataset.modality_keys["video"]:
                        image = data[video_key][0]  # Get first frame
                        image = Image.fromarray(image).resize((224, 224))

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)

                all_images = prim_images + wrist_views

                # Get language data
                prompt_stype = self.data_cfg.get("prompt_style", "task_and_subtask_with_others")
                if prompt_stype == "task_or_subtask":
                    if len(dataset.modality_keys["language"]) == 1:
                        language = data[dataset.modality_keys["language"][0]][0]
                    else:
                        # print(f"Language: {data[dataset.modality_keys['language'][-1]]}")
                        language = data[dataset.modality_keys["language"][-1]][0]
                        # print(f"Language: {language}")
                elif prompt_stype == "task_and_subtask":
                    trajectory_index = dataset.get_trajectory_index(trajectory_id)
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                    language = self._build_language_instruction_taskandsubtask(dataset, data, trajectory_length)
                else:
                    trajectory_index = dataset.get_trajectory_index(trajectory_id)
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                    language = self._build_language_instruction(dataset, data, trajectory_length)

                # print(f"language2: {language}")
                def to_numpy_float16(x):
                    if isinstance(x, torch.Tensor):
                        return x.cpu().numpy().astype(np.float16)
                    else:
                        return x.astype(np.float16)

                # Get action data: check if ConcatTransform has been applied
                if "action" in data and not any(k.startswith("action.") for k in data.keys()):
                    # Actions have been concatenated by ConcatTransform
                    action = to_numpy_float16(data["action"])
                else:
                    # Individual action keys
                    action = []
                    for action_key in dataset.modality_keys["action"]:
                        action.append(data[action_key])
                    action = np.concatenate(action, axis=1).astype(np.float16)
                # Note: zero-action check is intentionally not repeated here.
                # The raw action has already been validated above (on non-padded frames).
                # Re-checking after float16 conversion risks false positives because small
                # but valid values may round to exactly 0 in float16 precision.

                # Get action_mask if available
                action_mask = data.get("action_mask", None)
                if action_mask is not None:
                    action_mask = to_numpy_float16(action_mask)
                action, action_mask = self._apply_action_temporal_mask(action, action_mask, valid_frame_mask)

                sample = dict(
                    action=action,
                    lang=language,
                    action_mask=action_mask,
                    action_valid=bool(action_mask is None or np.asarray(action_mask, dtype=bool).any()),
                    dataset_id=dataset.tag,
                    embodiment_tag=dataset.tag_index,
                )
                if not self.image_pair_only:
                    sample["image"] = all_images
                if self._cfg_get("return_image_pair", False) not in ["False", False, None]:
                    sample["image_pair"] = self._build_image_pair(dataset, trajectory_id, step, image_pair_delta_indices)
                if self._cfg_get("include_meta", False) not in ["False", False, None]:
                    sample.update(self._build_sample_meta(dataset))
                if (
                    quality_sample_label is not None
                    and dataset.quality_filter is not None
                    and dataset.quality_filter.return_quality_label
                ):
                    sample["quality_label"] = quality_tier_name(quality_sample_label)

                # Get state data if needed
                state = None
                if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
                    if "state" in data and not any(k.startswith("state.") for k in data.keys()):
                        # States have been concatenated
                        # state = data["state"].astype(np.float16)
                        state = to_numpy_float16(data["state"])
                    else:
                        # Individual state keys
                        state = []
                        for state_key in dataset.modality_keys["state"]:
                            state.append(data[state_key])
                        state = np.concatenate(state, axis=1).astype(np.float16)

                    sample["state"] = state
                    state_mask = data.get("state_mask")
                    if state_mask is None:
                        state_mask = np.ones_like(state, dtype=bool)
                    sample["state_mask"] = np.asarray(state_mask, dtype=bool)

                return sample
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print("Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = int(safe_hash(("data_retry", self.epoch, index, self.seed, attempt)) % len(self))
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(
                f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}"
            )
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print("Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys - use union of all keys from all datasets
        all_modality_keys = set()
        for task_stats in per_task_stats:
            all_modality_keys.update(task_stats.keys())

        # For each modality key, only include datasets that have this key
        for modality in all_modality_keys:
            # Find which datasets have this modality
            datasets_with_modality = []
            weights_for_modality = []

            for task_idx, task_stats in enumerate(per_task_stats):
                if modality in task_stats:
                    datasets_with_modality.append(task_idx)
                    weights_for_modality.append(normalized_weights[task_idx])

            # Skip if no dataset has this modality (shouldn't happen)
            if not datasets_with_modality:
                continue

            # Re-normalize weights for this modality
            weights_for_modality = np.array(weights_for_modality)
            weights_for_modality = weights_for_modality / weights_for_modality.sum()

            # Number of dimensions (from first dataset that has this modality)
            # num_dims = len(per_task_stats[datasets_with_modality[0]][modality]["mean"])

            first_mean = np.asarray(per_task_stats[datasets_with_modality[0]][modality]["mean"], dtype=np.float64)

            stats_shape = first_mean.shape

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(stats_shape)
            weighted_squares = np.zeros(stats_shape)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for i, task_idx in enumerate(datasets_with_modality):
                w_i = weights_for_modality[i]
                task_stats = per_task_stats[task_idx]
                stats = task_stats[modality]
                means = np.asarray(stats["mean"], dtype=np.float64)
                stds = np.asarray(stats["std"], dtype=np.float64)
                if means.shape != stats_shape or stds.shape != stats_shape:
                    raise ValueError(
                        f"Shape mismatch for {modality}: expected {stats_shape}, "
                        f"got mean={means.shape}, std={stds.shape}"
                    )

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(np.asarray(stats["min"], dtype=np.float64))
                max_list.append(np.asarray(stats["max"], dtype=np.float64))
                q01_list.append(np.asarray(stats["q01"], dtype=np.float64))
                q99_list.append(np.asarray(stats["q99"], dtype=np.float64))

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.stack(min_list, axis=0), axis=0).tolist()
            overall_max = np.max(np.stack(max_list, axis=0), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.stack(q01_list, axis=0)
            q99_array = np.stack(q99_list, axis=0)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=weights_for_modality).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=weights_for_modality).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        merged_metadata = {}

        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )

        # Merge relative_action statistics if available
        relative_action_stats_list = [m["statistics"].get("relative_action", {}) for m in metadata_dicts]
        if any(relative_action_stats_list):  # If at least one dataset has relative_action stats
            # Filter out empty dicts and corresponding weights
            valid_indices = [i for i, stats in enumerate(relative_action_stats_list) if stats]
            valid_relative_action_stats = [relative_action_stats_list[i] for i in valid_indices]
            valid_weights = [dataset_sampling_weights[i] for i in valid_indices]

            # import pdb; pdb.set_trace()

            if valid_relative_action_stats:
                dataset_statistics["relative_action"] = LeRobotMixtureDataset.compute_overall_statistics(
                    per_task_stats=valid_relative_action_stats,
                    dataset_sampling_weights=valid_weights,
                    percentile_mixing_method=percentile_mixing_method,
                )

        merged_metadata["statistics"] = dataset_statistics

        # ---- Merge modality configs ----
        modality_configs = defaultdict(list)  # store as list of dicts (not set of strings)
        for md in metadata_dicts:
            for modality, configs in md["modalities"].items():
                modality_configs[modality].append(configs)

        merged_metadata["modalities"] = {}

        for modality, cfg_list in modality_configs.items():
            if modality != "video":
                # Strict: all non-video modality configs must match exactly
                cfg_strs = {json.dumps(c, sort_keys=True) for c in cfg_list}
                assert len(cfg_strs) == 1, f"Multiple modality configs for modality {modality}: {list(cfg_strs)}"
                merged_metadata["modalities"][modality] = json.loads(next(iter(cfg_strs)))
                continue

            # ---- video: allow different resolutions, unify ----
            # 1) ensure the set of video keys is consistent (optional but recommended)
            key_sets = [tuple(sorted(c.keys())) for c in cfg_list]
            key_set_strs = {json.dumps(ks) for ks in key_sets}
            assert len(key_set_strs) == 1, f"Multiple video key sets found: {list(key_set_strs)}"

            # 2) pick canonical config by smallest total pixel area (sum over views)
            def total_pixels(video_cfg: dict) -> int:
                s = 0
                for _, v in video_cfg.items():
                    w, h = v["resolution"]
                    s += int(w) * int(h)
                return s

            canonical = min(cfg_list, key=total_pixels)

            # 3) (optional) normalize fps/channels if you want; here we keep canonical values
            merged_metadata["modalities"]["video"] = canonical

            # helpful warning
            cfg_strs = {json.dumps(c, sort_keys=True) for c in cfg_list}
            if len(cfg_strs) > 1:
                print("[WARN] Multiple video modality configs found; " "unified to canonical (smallest resolution).")

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValueError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # A tag identifies one logical dataset and may contain multiple
        # physical shards. Merge only within a tag; never across tags.
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        all_metadata_weights: dict[str, list[float]] = {}
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights.tolist()):
            if getattr(dataset, "is_video_only", False):
                continue
            all_metadatas.setdefault(dataset.tag, []).append(dataset.metadata)
            all_metadata_weights.setdefault(dataset.tag, []).append(weight)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=all_metadata_weights[tag],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            if not getattr(dataset, "is_video_only", False):
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Canonical mode: stats are laid out by globally-unique slot,
        # not by per-embodiment key. Diverge from the legacy path entirely.
        if is_action_alignment_enabled():
            return self._save_dataset_statistics_canonical(save_path, format=format)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            if getattr(dataset, "is_video_only", False):
                continue
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by logical dataset tag.
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, "action") and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                any_dataset_has_relative_action = any(
                    dataset.enable_relative_action
                    for dataset in self.datasets
                    if dataset.tag == tag and not getattr(dataset, "is_video_only", False)
                )

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = build_export_action_stats(
                    merged_metadata,
                    tag,
                    reordered_keys,
                    use_relative_action=any_dataset_has_relative_action,
                )

                if filtered_action_stats:

                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats
                    add_relative_action_export_stats(
                        tag_stats,
                        merged_metadata,
                        tag,
                        reordered_keys,
                        use_relative_action=any_dataset_has_relative_action,
                    )

            # Process state statistics
            if hasattr(merged_metadata.statistics, "state") and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def _save_dataset_statistics_canonical(self, save_path: "Path", *, format: str = "json") -> None:
        """Canonical variant of save_dataset_statistics.

        Output JSON shape (per logical dataset / embodiment tag):

            tag_stats = {
              "action": {
                "mean": [D floats],
                "std":  [D floats],
                "min":  [D floats],
                "max":  [D floats],
                "q01":  [D floats],
                "q99":  [D floats],
                "mask": [D bools],           # True on filled slot dims
                "slot_names": [D strings],   # canonical slot name per dim
                "norm_modes": [D strings],   # q99/min_max; empty on placeholders
              },
              "state": {...},                # same layout for state
              "num_transitions": int,
              "num_trajectories": int,
            }

        Slot regions not filled by any dataset for this embodiment receive
        placeholder stats (mean=0, std=1, min=-1, max=1, q01=-1, q99=1) and
        mask=False, so downstream normalization never divides by zero and
        gradient masking aligns with what reaches the model.
        """
        statistics_data: dict = {}

        # Physical shards under one tag share their merged logical-dataset
        # distribution and the globally shared canonical slot layout.
        for tag, merged_metadata in self.merged_metadata.items():
            datasets_for_tag = [
                dataset
                for dataset in self.datasets
                if dataset.tag == tag and not getattr(dataset, "is_video_only", False)
            ]
            use_relative_action = any(dataset.enable_relative_action for dataset in datasets_for_tag)
            relative_action_keys = set(get_relative_action_keys_for_tag(tag))
            relative_action_stats = getattr(merged_metadata.statistics, "relative_action", {}) or {}
            tag_stats: dict = {}

            for modality_name, dim, slot_attr in (
                ("action", get_canonical_action_dim(), "action_canonical_slots"),
                ("state", get_canonical_state_dim(), "state_canonical_slots"),
            ):
                # Initialize placeholder layout at the same effective dim used by ConcatTransform.
                stats_arr = {
                    "mean": np.zeros(dim, dtype=np.float64),
                    "std": np.ones(dim, dtype=np.float64),
                    "min": -np.ones(dim, dtype=np.float64),
                    "max": np.ones(dim, dtype=np.float64),
                    "q01": -np.ones(dim, dtype=np.float64),
                    "q99": np.ones(dim, dtype=np.float64),
                }
                mask = np.zeros(dim, dtype=bool)
                slot_names = [""] * dim
                norm_modes = [""] * dim

                modality_stats = getattr(merged_metadata.statistics, modality_name, None) or {}

                for dataset in datasets_for_tag:
                    concat_tf = self._find_concat_transform(dataset)
                    if concat_tf is None:
                        continue
                    slot_map = getattr(concat_tf, slot_attr, None) or {}
                    slot_indices_map = getattr(concat_tf, f"{modality_name}_canonical_slot_indices", None) or {}
                    dataset_norm_modes = self._normalization_modes_for_dataset(dataset)
                    for full_key, slot_name in slot_map.items():
                        slot = CanonicalSlot[slot_name] if isinstance(slot_name, str) else slot_name
                        # full_key format: "action.foo" / "state.bar" -> "foo" / "bar"
                        sub_key = full_key.split(".", 1)[1] if "." in full_key else full_key
                        if modality_name == "action" and use_relative_action and sub_key in relative_action_keys:
                            if sub_key not in relative_action_stats:
                                raise ValueError(
                                    f"Relative action statistics missing for tag={tag}, key={sub_key}; "
                                    f"available={list(relative_action_stats.keys())}."
                                )
                            sk_stats = relative_action_stats[sub_key]
                        elif sub_key in modality_stats:
                            sk_stats = modality_stats[sub_key]
                        else:
                            continue
                        if slot.start >= dim:
                            continue
                        _validate_flat_statistical_values(sk_stats, f"{tag}.{modality_name}.{sub_key}")
                        source_dim = np.asarray(sk_stats.mean).reshape(-1).shape[0]
                        configured_slot_indices = slot_indices_map.get(full_key)
                        if configured_slot_indices is None:
                            n_filled = min(source_dim, slot.size, dim - slot.start)
                            destinations = np.arange(slot.start, slot.start + n_filled, dtype=np.int64)
                        else:
                            slot_indices = np.asarray(configured_slot_indices, dtype=np.int64).reshape(-1)
                            if len(slot_indices) != source_dim:
                                raise ValueError(
                                    f"Canonical slot indices for tag={tag}, key={full_key} must match "
                                    f"the {source_dim} source statistics dims, got {len(slot_indices)} indices"
                                )
                            if len(np.unique(slot_indices)) != len(slot_indices):
                                raise ValueError(
                                    f"Canonical slot indices for tag={tag}, key={full_key} contain duplicates: "
                                    f"{slot_indices.tolist()}"
                                )
                            if (slot_indices < 0).any() or (slot_indices >= slot.size).any():
                                raise ValueError(
                                    f"Canonical slot indices for tag={tag}, key={full_key} exceed "
                                    f"slot {slot.spec.name} capacity {slot.size}: {slot_indices.tolist()}"
                                )
                            destinations = slot.start + slot_indices
                            if (destinations >= dim).any():
                                raise ValueError(
                                    f"Canonical destinations for tag={tag}, key={full_key} exceed dimension {dim}: "
                                    f"{destinations.tolist()}"
                                )

                        # Each stat is shape (d_src,). Scatter it exactly as the
                        # runtime ConcatTransform scatters the corresponding field.
                        for name in ("mean", "std", "min", "max", "q01", "q99"):
                            val = np.asarray(getattr(sk_stats, name), dtype=np.float64).reshape(-1)
                            stats_arr[name][destinations] = val[: len(destinations)]
                        mask[destinations] = True
                        mode = dataset_norm_modes.get(full_key)
                        if mode not in {"q99", "min_max"}:
                            raise ValueError(
                                f"Normalization mode missing for tag={tag}, key={full_key}; "
                                f"available={sorted(dataset_norm_modes)}"
                            )
                        for canonical_index in destinations.tolist():
                            existing_mode = norm_modes[canonical_index]
                            if existing_mode and existing_mode != mode:
                                raise ValueError(
                                    "Conflicting normalization modes for canonical dimension "
                                    f"{canonical_index} in tag={tag}: {existing_mode!r} vs {mode!r}"
                                )
                            slot_names[canonical_index] = slot.spec.name
                            norm_modes[canonical_index] = mode

                tag_stats[modality_name] = {
                    "mean": stats_arr["mean"].tolist(),
                    "std": stats_arr["std"].tolist(),
                    "min": stats_arr["min"].tolist(),
                    "max": stats_arr["max"].tolist(),
                    "q01": stats_arr["q01"].tolist(),
                    "q99": stats_arr["q99"].tolist(),
                    "mask": mask.tolist(),
                    "slot_names": slot_names,
                    "norm_modes": norm_modes,
                }

            tag_stats.update(self._get_dataset_counts(tag))
            statistics_data[tag] = tag_stats

        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Canonical dataset statistics saved to: {save_path}")

    @staticmethod
    def _find_concat_transform(dataset):
        """Locate the ConcatTransform inside this dataset's transform stack."""
        composed = getattr(dataset, "transforms", None)
        if composed is None:
            return None
        for tf in getattr(composed, "transforms", []) or []:
            if tf.__class__.__name__ == "ConcatTransform":
                return tf
        return None

    @staticmethod
    def _normalization_modes_for_dataset(dataset) -> dict[str, str]:
        """Collect the modes actually captured by a dataset's transform pipeline."""

        modes: dict[str, str] = {}
        for transform in getattr(dataset.transforms, "transforms", []) or []:
            transform_modes = getattr(transform, "normalization_modes", None)
            if not transform_modes:
                continue
            for key, mode in transform_modes.items():
                existing_mode = modes.get(key)
                if existing_mode is not None and existing_mode != mode:
                    raise ValueError(
                        f"Dataset {dataset.dataset_name} assigns conflicting normalization modes "
                        f"to {key}: {existing_mode!r} vs {mode!r}"
                    )
                modes[key] = str(mode)
        return modes

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.

        Args:
            tag: Logical dataset / embodiment tag.

        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if getattr(dataset, "is_video_only", False):
                continue
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {"num_transitions": num_transitions, "num_trajectories": num_trajectories}

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.

        Args:
            load_path (Path | str): Path to the statistics file

        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == ".json":
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif load_path.suffix.lower() == ".pkl":
            import pickle

            with open(load_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.

        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        raise ValueError(
            "cached_statistics_path currently expects a full merged DatasetMetadata cache, "
            "but exported dataset_statistics.json only contains flattened model statistics. "
            "Recompute metadata from source datasets instead."
        )
