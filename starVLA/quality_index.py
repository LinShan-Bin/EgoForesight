# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add clip-quality index loading, lookup, and dataset binding support.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

QUALITY_INDEX_VERSION = 1
QUALITY_INDEX_COLUMNS = (
    "dataset_id",
    "video_relpath",
    "video_key",
    "start_sec",
    "end_sec",
    "label",
    "clip_id",
    "source_run",
)
QUALITY_TIER_TO_CODE = {"A": 0, "B": 1, "C": 2, "D": 3, "UNKNOWN": 4}
QUALITY_CODE_TO_TIER = {value: key for key, value in QUALITY_TIER_TO_CODE.items()}
UNKNOWN_QUALITY_CODE = QUALITY_TIER_TO_CODE["UNKNOWN"]
_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_quality_config_path(path: Path | str) -> Path:
    """Resolve a quality-filter config from an absolute path, cwd, or repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.exists():
        resolved = candidate.resolve()
    else:
        resolved = (_REPO_ROOT / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Quality-filter config does not exist: {resolved}")
    return resolved


def resolve_path_from_config(path: Path | str, config_path: Path) -> Path:
    """Resolve a path relative to the quality-filter config directory."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def normalize_video_relpath(value: Path | str) -> str:
    """Return a stable POSIX relative video path and reject parent traversal."""
    normalized = PurePosixPath(str(value).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"video_relpath must be relative and may not contain '..': {value}")
    text = normalized.as_posix()
    if text in {"", "."}:
        raise ValueError(f"video_relpath may not be empty: {value}")
    return text


def video_relpath_from_root(video_path: Path | str, dataset_root: Path | str) -> str:
    """Convert an on-disk video path to the canonical path used by the index."""
    path = Path(video_path)
    root = Path(dataset_root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Video path {path} is not under dataset root {root}") from exc
    return normalize_video_relpath(relative)


def quality_tier_name(label: int) -> str:
    """Convert an integer quality label to its public tier name."""
    try:
        return QUALITY_CODE_TO_TIER[int(label)]
    except KeyError as exc:
        raise ValueError(f"Unknown quality label code: {label}") from exc


def worst_quality_label(labels: np.ndarray | list[int] | tuple[int, ...]) -> int:
    """Return the worst label according to A < B < C < D < UNKNOWN."""
    values = np.asarray(labels, dtype=np.uint8)
    return UNKNOWN_QUALITY_CODE if values.size == 0 else int(values.max())


class QualityIndex:
    """Immutable interval index implementing (dataset, video, timestamp) -> quality label."""

    def __init__(self, table: pa.Table, *, source_path: Path | None = None) -> None:
        missing_columns = sorted(set(QUALITY_INDEX_COLUMNS) - set(table.column_names))
        if missing_columns:
            raise ValueError(f"Quality index is missing required columns: {missing_columns}")

        table = table.select(QUALITY_INDEX_COLUMNS).sort_by(
            [("dataset_id", "ascending"), ("video_relpath", "ascending"), ("start_sec", "ascending")]
        )
        self.source_path = source_path
        self._starts = np.asarray(table["start_sec"].to_numpy(zero_copy_only=False), dtype=np.float64)
        self._ends = np.asarray(table["end_sec"].to_numpy(zero_copy_only=False), dtype=np.float64)
        self._labels = np.asarray(table["label"].to_numpy(zero_copy_only=False), dtype=np.uint8)
        dataset_ids = table["dataset_id"].to_pylist()
        video_relpaths = [normalize_video_relpath(value) for value in table["video_relpath"].to_pylist()]

        invalid_intervals = np.flatnonzero(
            ~np.isfinite(self._starts) | ~np.isfinite(self._ends) | (self._ends <= self._starts)
        )
        if invalid_intervals.size:
            raise ValueError(f"Quality index contains invalid intervals at rows {invalid_intervals[:10].tolist()}")
        invalid_labels = sorted(set(int(value) for value in self._labels) - set(QUALITY_CODE_TO_TIER))
        if invalid_labels:
            raise ValueError(f"Quality index contains invalid label codes: {invalid_labels}")

        self._video_slices: dict[tuple[str, str], tuple[int, int]] = {}
        self._dataset_label_durations: dict[str, np.ndarray] = {}
        start = 0
        while start < len(dataset_ids):
            key = (str(dataset_ids[start]), video_relpaths[start])
            end = start + 1
            while end < len(dataset_ids) and (str(dataset_ids[end]), video_relpaths[end]) == key:
                end += 1
            if np.any(self._starts[start + 1 : end] < self._ends[start : end - 1]):
                raise ValueError(f"Quality index contains overlapping intervals for dataset/video {key}")
            self._video_slices[key] = (start, end)
            durations = self._dataset_label_durations.setdefault(key[0], np.zeros(len(QUALITY_CODE_TO_TIER)))
            np.add.at(durations, self._labels[start:end], self._ends[start:end] - self._starts[start:end])
            start = end

        self._dataset_ids = frozenset(key[0] for key in self._video_slices)
        for array in (self._starts, self._ends, self._labels):
            array.flags.writeable = False

    @classmethod
    def from_parquet(cls, path: Path | str) -> "QualityIndex":
        path = Path(path)
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.schema_arrow.metadata or {}
        raw_version = metadata.get(b"quality_index_version", str(QUALITY_INDEX_VERSION).encode())
        if int(raw_version.decode()) != QUALITY_INDEX_VERSION:
            raise ValueError(
                f"Unsupported quality index version {raw_version!r} in {path}; expected {QUALITY_INDEX_VERSION}"
            )
        return cls(parquet_file.read(columns=list(QUALITY_INDEX_COLUMNS)), source_path=path)

    def has_dataset(self, dataset_id: str) -> bool:
        return dataset_id in self._dataset_ids

    def has_video(self, dataset_id: str, video_relpath: Path | str) -> bool:
        return (dataset_id, normalize_video_relpath(video_relpath)) in self._video_slices

    def weighted_duration(self, dataset_id: str, tier_weights: tuple[float, ...]) -> float:
        """Return seconds retained by a tier-weighted rejection filter."""
        durations = self._dataset_label_durations.get(dataset_id)
        if durations is None:
            return 0.0
        weights = np.asarray(tier_weights, dtype=np.float64)
        if weights.shape != durations.shape:
            raise ValueError(f"Expected {durations.size} tier weights, got {weights.size}")
        max_weight = float(weights.max(initial=0.0))
        if max_weight <= 0:
            return 0.0
        return float(np.dot(durations, weights / max_weight))

    def lookup(self, dataset_id: str, video_relpath: Path | str, timestamps: np.ndarray | list[float]) -> np.ndarray:
        """Look up timestamp labels. Unmatched videos or time gaps return UNKNOWN."""
        timestamps_array = np.asarray(timestamps, dtype=np.float64)
        result = np.full(timestamps_array.shape, UNKNOWN_QUALITY_CODE, dtype=np.uint8)
        video_slice = self._video_slices.get((dataset_id, normalize_video_relpath(video_relpath)))
        if video_slice is None or timestamps_array.size == 0:
            return result

        start, end = video_slice
        starts = self._starts[start:end]
        ends = self._ends[start:end]
        labels = self._labels[start:end]
        interval_indices = np.searchsorted(starts, timestamps_array, side="right") - 1
        safe_indices = np.maximum(interval_indices, 0)
        covered = (
            (interval_indices >= 0)
            & (timestamps_array >= starts[safe_indices])
            & (timestamps_array < ends[safe_indices])
        )
        result[covered] = labels[safe_indices[covered]]
        return result


@dataclass(frozen=True)
class QualityDatasetBinding:
    """Explicit binding from one mixture entry to one index dataset."""

    mixture_name: str
    dataset_id: str
    video_key: str


@dataclass(frozen=True)
class QualityFilterSpec:
    """Parsed standalone quality-filter configuration."""

    config_path: Path
    index_path: Path
    bindings: Mapping[str, QualityDatasetBinding]
    tier_weights: tuple[float, float, float, float, float]
    max_resample_attempts: int
    return_quality_label: bool

    def binding_for(self, mixture_name: str) -> QualityDatasetBinding | None:
        return self.bindings.get(mixture_name)


@dataclass(frozen=True)
class DatasetQualityFilter:
    """Runtime quality filter attached only to an explicitly configured dataset."""

    index: QualityIndex
    binding: QualityDatasetBinding
    tier_weights: tuple[float, float, float, float, float]
    max_resample_attempts: int
    return_quality_label: bool

    def accept_probability(self, label: int) -> float:
        max_weight = max(self.tier_weights)
        if max_weight <= 0:
            return 0.0
        return float(self.tier_weights[int(label)] / max_weight)

    def accepts(self, label: int, unit_random: float = 0.0) -> bool:
        return unit_random < self.accept_probability(label)


@lru_cache(maxsize=8)
def load_quality_index(path: str) -> QualityIndex:
    return QualityIndex.from_parquet(Path(path))


@lru_cache(maxsize=8)
def load_quality_filter_spec(path: str) -> QualityFilterSpec:
    config_path = resolve_quality_config_path(path)
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Quality-filter config must contain a mapping: {config_path}")
    version = int(raw.get("version", 0))
    if version != QUALITY_INDEX_VERSION:
        raise ValueError(f"Unsupported quality-filter config version {version}; expected {QUALITY_INDEX_VERSION}")

    index_path_value = raw.get("index_path")
    if not index_path_value:
        raise ValueError(f"quality-filter config is missing index_path: {config_path}")
    index_path = resolve_path_from_config(str(index_path_value), config_path)

    raw_datasets = raw.get("datasets", {})
    if not isinstance(raw_datasets, dict):
        raise ValueError("quality-filter datasets must be a mapping keyed by full mixture dataset name")
    bindings = {}
    for mixture_name, dataset_config in raw_datasets.items():
        if not isinstance(dataset_config, dict):
            raise ValueError(f"Invalid quality-filter dataset binding for {mixture_name}: {dataset_config}")
        dataset_id = str(dataset_config.get("dataset_id", "")).strip()
        video_key = str(dataset_config.get("video_key", "")).strip()
        if not dataset_id or not video_key:
            raise ValueError(f"Dataset binding {mixture_name} requires dataset_id and video_key")
        explicit_names = [str(mixture_name)]
        raw_aliases = dataset_config.get("mixture_aliases", [])
        if isinstance(raw_aliases, str):
            explicit_names.append(raw_aliases)
        elif isinstance(raw_aliases, (list, tuple)):
            explicit_names.extend(str(alias) for alias in raw_aliases)
        else:
            raise ValueError(f"mixture_aliases for {mixture_name} must be a string or list")
        for explicit_name in explicit_names:
            if explicit_name in bindings:
                raise ValueError(f"Duplicate explicit quality-filter binding for mixture dataset {explicit_name!r}")
            bindings[explicit_name] = QualityDatasetBinding(
                mixture_name=explicit_name,
                dataset_id=dataset_id,
                video_key=video_key,
            )

    raw_weights = raw.get("tier_weights", {})
    tier_weights = tuple(float(raw_weights.get(tier, 0.0)) for tier in ("A", "B", "C", "D", "UNKNOWN"))
    if any(weight < 0 for weight in tier_weights) or max(tier_weights, default=0.0) <= 0:
        raise ValueError(f"tier_weights must contain at least one positive, non-negative weight: {raw_weights}")

    return QualityFilterSpec(
        config_path=config_path,
        index_path=index_path,
        bindings=bindings,
        tier_weights=tier_weights,
        max_resample_attempts=max(1, int(raw.get("max_resample_attempts", 64))),
        return_quality_label=bool(raw.get("return_quality_label", True)),
    )


def get_dataset_quality_filter(config_path: Path | str, mixture_name: str) -> DatasetQualityFilter | None:
    """Build a runtime filter only when the full mixture name is explicitly bound."""
    spec = load_quality_filter_spec(str(resolve_quality_config_path(config_path)))
    binding = spec.binding_for(mixture_name)
    if binding is None:
        return None
    quality_index = load_quality_index(str(spec.index_path))
    if not quality_index.has_dataset(binding.dataset_id):
        raise ValueError(
            f"Quality index {spec.index_path} has no rows for configured dataset_id={binding.dataset_id!r} "
            f"(mixture={mixture_name!r})"
        )
    return DatasetQualityFilter(
        index=quality_index,
        binding=binding,
        tier_weights=spec.tier_weights,
        max_resample_attempts=spec.max_resample_attempts,
        return_quality_label=spec.return_quality_label,
    )
