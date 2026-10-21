# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Validate resumed canonical normalization before replacing run statistics.

"""Fail closed when runtime normalizers disagree with a resume snapshot."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist

_NUMERIC_FIELDS = ("min", "max", "mean", "std", "q01", "q99")
_NORMALIZATION_FIELDS = ("norm_modes", "normalization_modes")


def _normalization_modes(statistics: dict, context: str) -> list | None:
    modes = [statistics[field] for field in _NORMALIZATION_FIELDS if field in statistics]
    if len(modes) == 2 and modes[0] != modes[1]:
        raise ValueError(f"{context}: conflicting norm_modes and normalization_modes")
    return modes[0] if modes else None


def _compare_schema(expected: dict, actual: dict, width: int, context: str) -> None:
    for field in ("mask", "slot_names", "norm_modes"):
        if field == "norm_modes":
            expected_values = _normalization_modes(expected, f"expected {context}")
            actual_values = _normalization_modes(actual, f"runtime {context}")
            if expected_values is None:
                continue
        else:
            if field not in expected:
                continue
            expected_values, actual_values = expected[field], actual.get(field)
        expected_array = np.asarray(expected_values)
        actual_array = np.asarray(actual_values)
        if expected_array.shape != (width,) or actual_array.shape != (width,):
            raise ValueError(
                f"{context}.{field}: expected and runtime schemas must have shape ({width},), "
                f"got {expected_array.shape} and {actual_array.shape}"
            )
        if field == "mask" and (expected_array.dtype.kind != "b" or actual_array.dtype.kind != "b"):
            raise ValueError(f"{context}.mask: canonical masks must contain booleans")
        if field != "mask" and (expected_array.dtype.kind not in "US" or actual_array.dtype.kind not in "US"):
            raise ValueError(f"{context}.{field}: canonical schema values must be strings")
        mismatch = np.flatnonzero(expected_array != actual_array)
        if mismatch.size:
            index = int(mismatch[0])
            raise ValueError(
                f"{context}.{field}[{index}]: expected {expected_values[index]!r}, " f"runtime {actual_values[index]!r}"
            )


def compare_canonical_statistics(expected: dict, actual: dict) -> None:
    """Compare all expected normalizers, including padding, while ignoring dataset counts.

    Existing snapshots may omit optional schema fields. When present, masks,
    normalization modes, and slot names must match exactly. Numeric fields use
    the same fixed tolerance for active slots and inactive placeholders.
    Additional runtime tags are allowed, but every expected tag is required.
    """
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Expected statistics must be a nonempty mapping of canonical dataset tags")
    if not isinstance(actual, dict):
        raise ValueError("Runtime statistics must be a mapping of canonical dataset tags")
    missing_tags = sorted(set(expected) - set(actual))
    if missing_tags:
        raise ValueError(f"Runtime statistics are missing expected dataset tags: {missing_tags}")

    for tag, expected_tag in expected.items():
        actual_tag = actual[tag]
        if not isinstance(expected_tag, dict) or not isinstance(actual_tag, dict):
            raise ValueError(f"{tag}: expected and runtime tag statistics must be mappings")
        for modality in ("action", "state"):
            context = f"{tag}.{modality}"
            expected_stats, actual_stats = expected_tag.get(modality), actual_tag.get(modality)
            if not isinstance(expected_stats, dict) or not isinstance(actual_stats, dict):
                raise ValueError(f"{context}: expected and runtime canonical statistics must both exist")
            width = None
            for field in _NUMERIC_FIELDS:
                if field not in expected_stats or field not in actual_stats:
                    raise ValueError(f"{context}.{field}: required normalization statistic is missing")
                expected_values = np.asarray(expected_stats[field], dtype=np.float64)
                actual_values = np.asarray(actual_stats[field], dtype=np.float64)
                if expected_values.ndim != 1 or expected_values.size == 0:
                    raise ValueError(f"{context}.{field}: expected statistics must be a nonempty flat array")
                width = expected_values.size if width is None else width
                if expected_values.shape != (width,) or actual_values.shape != (width,):
                    raise ValueError(
                        f"{context}.{field}: expected and runtime statistics must have shape ({width},), "
                        f"got {expected_values.shape} and {actual_values.shape}"
                    )
                if not np.isfinite(expected_values).all() or not np.isfinite(actual_values).all():
                    raise ValueError(f"{context}.{field}: normalization statistics must be finite")
                if not np.allclose(actual_values, expected_values, rtol=1e-5, atol=1e-6):
                    index = int(np.flatnonzero(~np.isclose(actual_values, expected_values, rtol=1e-5, atol=1e-6))[0])
                    raise ValueError(
                        f"{context}.{field}[{index}]: expected {expected_values[index]:.9g}, "
                        f"runtime {actual_values[index]:.9g} (rtol=1e-5, atol=1e-6)"
                    )
            _compare_schema(expected_stats, actual_stats, width, context)


def validate_resume_statistics(dataset: Any, expected_path: Path | str) -> None:
    """Validate already-built metadata on every rank before writing run statistics.

    The dataset export only serializes loaded metadata. Temporary files are
    local to each process, and validation errors are shared before rank zero
    can replace the existing run statistics. No dataset frames are traversed.
    """
    expected_path = Path(expected_path)
    local_error = None
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="starvla_resume_statistics_") as temporary_directory:
            actual_path = Path(temporary_directory) / "dataset_statistics.json"
            dataset.save_dataset_statistics(actual_path)
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
        compare_canonical_statistics(expected, actual)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"

    errors = [local_error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(
            f"Resume statistics validation failed against {expected_path}; "
            "existing run statistics were not overwritten. " + "; ".join(failures[:8])
        )
