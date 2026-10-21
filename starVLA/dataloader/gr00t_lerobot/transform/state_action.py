# EgoForesight modifications Copyright 2026 Chubin Zhang.
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

import functools
import random
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..relative_action_stats.types import ActionFormat, ActionType
from ..schema import DatasetMetadata, RotationType, StateActionMetadata
from ..state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from ..state_action.batch_ops import (
    compute_camera_spatial_xyz_rotvec_chunks,
    compute_relative_xyz_rotvec_chunks,
    recover_camera_spatial_xyz_rotvec_actions,
)
from ..state_action.camera_motion import (
    CAMERA_ACTION_TRANSFORM_KEY,
    CAMERA_REFERENCE_TRANSFORM_KEY,
    CAMERA_SKIP_COMPENSATION_KEY,
    CAMERA_STATE_TRANSFORM_KEY,
)
from ..state_action.pose import EndEffectorPose, JointPose
from .base import InvertibleModalityTransform, ModalityTransform


# Rotation transformer adapted from Diffusion Policy.
# Copyright 2023 Columbia Artificial Intelligence and Robotics Lab; MIT.
class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        # Only custom configs requesting representation conversion need PyTorch3D.
        # Released canonical relative-action recipes already carry xyz/rotvec.
        try:
            import pytorch3d.transforms as pt
        except ImportError as exc:
            raise ImportError("RotationTransform requires the optional pytorch3d package") from exc

        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)
        # 检测是否为 2D 统计数据（用于 relative action）
        self.is_2d_stats = any(v.ndim == 2 for v in self.statistics.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # 对于 2D 统计数据，使用元素级别的归一化
        if self.is_2d_stats:
            return self._forward_2d(x)

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (max[..., mask] - min[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """对于 2D 统计数据 [T, D]，使用元素级别的归一化"""
        if self.mode == "min_max":
            min_val = self.statistics["min"].to(x.dtype)
            max_val = self.statistics["max"].to(x.dtype)
            diff = max_val - min_val
            # 使用 torch.where 避免除零
            normalized = torch.where(diff != 0, 2 * (x - min_val) / diff - 1, torch.zeros_like(x))
            return normalized
        elif self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            diff = q99 - q01
            normalized = torch.where(diff != 0, 2 * (x - q01) / diff - 1, x)
            return torch.clamp(normalized, -1, 1)
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            normalized = torch.where(std != 0, (x - mean) / std, x)
            return normalized
        else:
            raise ValueError(f"2D stats not supported for mode: {self.mode}")

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(default_factory=dict, description="The input dtypes for each state key.")
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]

            ##### add ######

            # 允许标量 / list / ndarray
            value = np.asarray(value)

            # 把 (T,) 变成 (T,1)
            if value.ndim == 1:
                value = value[:, None]
            elif value.ndim == 0:
                value = value.reshape(1, 1)

            ##### add ######

            assert isinstance(value, np.ndarray), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
            else:
                # Default to float32 to ensure consistent dtype across samples
                if data[key].dtype == torch.float64:
                    data[key] = data[key].to(torch.float32)
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )

    relative_action_keys: list[str] = Field(
        default_factory=list, description="The relative action keys to be transformed."
    )
    action_horizon: int = Field(default=16, description="The action chunk length for relative action statistics.")

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def _slice_relative_stats(self, raw_stats: dict, action_horizon: int) -> dict:
        """
        Validate OpenPI-style relative action statistics.

        Relative action stats are computed after flattening the horizon axis, so
        every statistic must be a 1-D [D] vector. Older per-horizon [H, D] caches
        are rejected instead of silently changing the normalization behavior.
        """
        processed = {}
        for stat_name, stat_value in raw_stats.items():
            arr = np.asarray(stat_value)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected flattened relative action stats [D] for {stat_name}, "
                    f"got shape {arr.shape}. Regenerate relative stats with horizon-flattened mode."
                )
            processed[stat_name] = arr.tolist()
        return processed

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(config, StateActionMetadata), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata.
        # Missing metadata is allowed for synthesized keys (e.g. wrist_pose produced
        # by EefPoseAlignTransform) -- they carry no rotation/continuity flags to
        # look up, and the downstream normalization_modes loop still asserts stats
        # if a normalization mode is registered for the key.
        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                if not hasattr(modality_metadata, modality):
                    continue
                bucket = getattr(modality_metadata, modality)
                if state_key not in bucket:
                    continue
                self.modality_metadata[key] = bucket[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(dataset_statistics, modality), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[state_key].model_dump()

        ### Relative action statistics
        for key in self.relative_action_keys:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "Relative action keys should have two parts: 'modality.key'"
            _, relative_action_key = split_key
            if not hasattr(dataset_statistics, "relative_action") or not dataset_statistics.relative_action:
                raise ValueError(
                    f"[StateActionTransform] relative_action stats are required for {key}, "
                    "but dataset metadata has none."
                )
            if relative_action_key not in dataset_statistics.relative_action:
                raise ValueError(
                    f"[StateActionTransform] relative_action stats missing for {relative_action_key}; "
                    f"available={list(dataset_statistics.relative_action.keys())}."
                )
            raw_stats = dataset_statistics.relative_action[relative_action_key].model_dump()
            processed_stats = self._slice_relative_stats(raw_stats, self.action_horizon)
            self.normalization_statistics[f"relative_{key}"] = processed_stats

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(from_rep=from_rep.value, to_rep=to_rep.value)

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif not self.modality_metadata[key].continuous and self.normalization_modes[key] != "binary":
                raise ValueError(f"{key} is not continuous, so it should be normalized using `binary` mode")
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]

            self._normalizers[key] = Normalizer(mode=self.normalization_modes[key], statistics=statistics)

        # Initialize normalizers for relative action keys
        for key in self.relative_action_keys:
            relative_key = f"relative_{key}"
            if relative_key not in self.normalization_statistics:
                raise ValueError(f"Missing prepared relative action statistics for {relative_key}")
            # 使用与原 action key 相同的 normalization mode
            mode = self.normalization_modes.get(key, "min_max")
            self._normalizers[relative_key] = Normalizer(
                mode=mode, statistics=self.normalization_statistics[relative_key]
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            # Rotate the state
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state.
            if key in self.relative_action_keys:
                rel_norm = self._normalizers.get(f"relative_{key}")
                if rel_norm is None:
                    raise ValueError(f"Missing relative action normalizer for {key}")
                state = rel_norm.forward(state)
            elif key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            # Unnormalize the state.
            if key in self.relative_action_keys:
                rel_norm = self._normalizers.get(f"relative_{key}")
                if rel_norm is None:
                    raise ValueError(f"Missing relative action normalizer for {key}")
                state = rel_norm.inverse(state)
            elif key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(..., description="Standard deviation of the noise to be added to the state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data


class RelativeActionTransform(InvertibleModalityTransform):
    """
    Make action chunks state-relative:
        a_rel[t+k] = a_abs[t+k] - s_ref
    where s_ref is taken from state at reference_index (default -1, i.e., last).

    This is meant to run AFTER StateActionToTensor, BEFORE StateActionTransform(normalization).
    """

    apply_to: list[str] = Field(..., description="Action keys to convert to state-relative.")
    state_key_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping action key -> state key. Default: replace 'action.' with 'state.'",
    )
    relative_action_info: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relative action information.",
    )
    reference_index: int = Field(
        default=-1,
        description="Which timestep in state chunk to use as reference. -1 means last.",
    )
    relative_eef_frame: str = Field(
        default="eef",
        description="Relative EEF convention: body-frame 'eef' or current-camera 'camera_spatial'.",
    )
    in_place: bool = Field(default=True, description="Overwrite the original action key in data.")
    out_key_map: dict[str, str] = Field(default_factory=dict)

    _input_dtypes: dict[str, torch.dtype] = PrivateAttr(default_factory=dict)
    _action_info_map: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    @field_validator("relative_eef_frame")
    @classmethod
    def validate_relative_eef_frame(cls, value: str) -> str:
        normalized = str(value).lower().replace("-", "_")
        if normalized not in {"eef", "camera_spatial"}:
            raise ValueError("relative_eef_frame must be 'eef' or 'camera_spatial'")
        return normalized

    def model_post_init(self, __context: Any) -> None:
        # Build a mapping from action key to its action info
        for info in self.relative_action_info:
            self._action_info_map[info["key"]] = info

    def _get_action_info(self, akey: str) -> dict[str, Any] | None:
        """Get action info (type, format) for a given action key."""
        return self._action_info_map.get(akey)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        modalities = dataset_metadata.modalities

        for akey in self.apply_to:
            # 1) 默认自动映射：action.xxx -> state.xxx
            skey = self.state_key_map.get(akey)
            if skey is None:
                skey = akey.replace("action.", "state.", 1)
                self.state_key_map[akey] = skey

            # 2) 强校验：确保 state key 存在（防 silent wrong）
            assert skey.startswith("state."), f"Mapped state key must start with 'state.': {skey}"
            _, a_sub = akey.split(".", 1)
            _, s_sub = skey.split(".", 1)

            # Some keys (e.g. wrist_pose after EefPoseAlignTransform) are
            # synthesized at runtime and never appear in modality.json.
            # Skip the schema-existence check for those; apply() still verifies
            # the data at runtime.
            a_in_meta = hasattr(modalities, "action") and a_sub in modalities.action
            s_in_meta = hasattr(modalities, "state") and s_sub in modalities.state
            if not a_in_meta or not s_in_meta:
                continue

            # 3) （推荐）维度一致性检查：避免 action/state 维度不对齐
            a_meta = modalities.action[a_sub]
            s_meta = modalities.state[s_sub]
            assert (
                len(a_meta.shape) == 1 and len(s_meta.shape) == 1
            ), f"Expected 1D vectors, got {akey}:{a_meta.shape}, {skey}:{s_meta.shape}"
            assert (
                a_meta.shape[0] == s_meta.shape[0]
            ), f"Dim mismatch {akey}:{a_meta.shape[0]} vs {skey}:{s_meta.shape[0]}"

    def _pick_ref(self, state: torch.Tensor) -> torch.Tensor:
        # state: [Ts, D] or [D]
        if state.ndim == 1:
            return state
        if state.ndim != 2:
            raise ValueError(f"state must be [T,D] or [D], got {tuple(state.shape)}")
        return state[self.reference_index]

    def _camera_frame_tracks(
        self,
        data: dict[str, Any],
        action_length: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the current/future ``T_world_from_camera`` tracks, if injected."""
        if bool(data.get(CAMERA_SKIP_COMPENSATION_KEY, False)):
            return None, None
        has_state_camera = CAMERA_STATE_TRANSFORM_KEY in data
        has_action_camera = CAMERA_ACTION_TRANSFORM_KEY in data
        if not has_state_camera and not has_action_camera:
            return None, None
        if has_state_camera != has_action_camera:
            raise KeyError(
                "Camera-motion compensation requires both current and future camera transforms; "
                f"found state={has_state_camera}, action={has_action_camera}"
            )

        state_camera = data[CAMERA_STATE_TRANSFORM_KEY]
        action_camera = data[CAMERA_ACTION_TRANSFORM_KEY]
        if isinstance(state_camera, torch.Tensor):
            state_camera = state_camera.detach().cpu().numpy()
        if isinstance(action_camera, torch.Tensor):
            action_camera = action_camera.detach().cpu().numpy()
        state_camera = np.asarray(state_camera, dtype=np.float64)
        action_camera = np.asarray(action_camera, dtype=np.float64)
        if state_camera.shape == (4, 4):
            reference_camera = state_camera
        elif state_camera.ndim == 3 and state_camera.shape[-2:] == (4, 4):
            reference_camera = state_camera[self.reference_index]
        else:
            raise ValueError(f"{CAMERA_STATE_TRANSFORM_KEY} must be [Ts,4,4] or [4,4], got {state_camera.shape}")
        if action_camera.shape == (4, 4) and action_length == 1:
            action_camera = action_camera[None, ...]
        if action_camera.shape != (action_length, 4, 4):
            raise ValueError(f"{CAMERA_ACTION_TRANSFORM_KEY} must be {(action_length, 4, 4)}, got {action_camera.shape}")
        return reference_camera, action_camera

    def _reference_camera_transform(
        self,
        data: dict[str, Any],
        pose_reference_camera: np.ndarray | None,
    ) -> np.ndarray:
        """Return the current ``T_world_from_camera`` used as the spatial basis."""
        if bool(data.get(CAMERA_SKIP_COMPENSATION_KEY, False)):
            raise ValueError("camera_spatial relative EEF requires valid camera extrinsics for every sample")
        value = data.get(CAMERA_REFERENCE_TRANSFORM_KEY, pose_reference_camera)
        if value is None:
            raise KeyError(
                "camera_spatial relative EEF requires the current observation-camera transform; "
                f"missing {CAMERA_REFERENCE_TRANSFORM_KEY}"
            )
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if value.shape == (4, 4):
            return value
        if value.ndim == 3 and value.shape[-2:] == (4, 4):
            return value[self.reference_index]
        raise ValueError(f"{CAMERA_REFERENCE_TRANSFORM_KEY} must be [Ts,4,4] or [4,4], got {value.shape}")

    def _camera_recovery_frames(
        self,
        data: dict[str, Any],
        action_length: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return optional reference/output storage-frame transforms for recovery.

        A state-frame transform makes the default recovered output use the
        current state's storage frame. Action-frame transforms request an exact
        round trip into each original future action frame.
        """
        reference_frame = data.get(CAMERA_STATE_TRANSFORM_KEY)
        output_frames = data.get(CAMERA_ACTION_TRANSFORM_KEY)

        if isinstance(reference_frame, torch.Tensor):
            reference_frame = reference_frame.detach().cpu().numpy()
        if reference_frame is not None:
            reference_frame = np.asarray(reference_frame, dtype=np.float64)
            if reference_frame.shape == (4, 4):
                pass
            elif reference_frame.ndim == 3 and reference_frame.shape[-2:] == (4, 4):
                reference_frame = reference_frame[self.reference_index]
            else:
                raise ValueError(
                    f"{CAMERA_STATE_TRANSFORM_KEY} must be [Ts,4,4] or [4,4], " f"got {reference_frame.shape}"
                )

        if isinstance(output_frames, torch.Tensor):
            output_frames = output_frames.detach().cpu().numpy()
        if output_frames is not None:
            output_frames = np.asarray(output_frames, dtype=np.float64)
            if output_frames.shape == (4, 4) and action_length == 1:
                output_frames = output_frames[None, ...]
            if output_frames.shape != (action_length, 4, 4):
                raise ValueError(
                    f"{CAMERA_ACTION_TRANSFORM_KEY} must be {(action_length, 4, 4)}, " f"got {output_frames.shape}"
                )

        return reference_frame, output_frames

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for akey in self.apply_to:
            if akey not in data:
                continue
            skey = self.state_key_map.get(akey, akey.replace("action.", "state.", 1))
            if skey not in data:
                raise KeyError(f"Missing state key {skey} needed for relative action of {akey}")

            action = data[akey]
            state = data[skey]
            assert isinstance(action, torch.Tensor), f"{akey} must be torch.Tensor"
            assert isinstance(state, torch.Tensor), f"{skey} must be torch.Tensor"

            if akey not in self._input_dtypes:
                self._input_dtypes[akey] = action.dtype

            ref = self._pick_ref(state)  # [D]

            # Get action info for this key
            action_info = self._get_action_info(akey)

            if action_info is None:
                # Fallback to simple subtraction if no action info
                if action.ndim == 1:
                    out = action - ref
                elif action.ndim == 2:
                    out = action - ref.unsqueeze(0)
                else:
                    raise ValueError(f"{akey} must be [T,D] or [D], got {tuple(action.shape)}")
            else:
                # Use proper relative action computation based on type and format
                action_type = action_info["type"]
                action_format = action_info["format"]

                # Convert tensors to numpy for processing
                ref_np = ref.cpu().numpy() if isinstance(ref, torch.Tensor) else ref
                if action.ndim == 1:
                    actions_np = action.cpu().numpy().reshape(1, -1)
                    squeeze_output = True
                elif action.ndim == 2:
                    actions_np = action.cpu().numpy()
                    squeeze_output = False
                else:
                    raise ValueError(f"{akey} must be [T,D] or [D], got {tuple(action.shape)}")

                if action_type == ActionType.EEF:
                    # Determine rotation type based on action format
                    if action_format == ActionFormat.XYZ_ROT6D:
                        # 9D: xyz (3) + rot6d (6)
                        rotation_type = "rot6d"
                        translation_dim = 3
                    elif action_format == ActionFormat.XYZ_ROTVEC:
                        # 6D: xyz (3) + rotvec/axis-angle (3)
                        rotation_type = "rotvec"
                        translation_dim = 3
                    else:
                        raise ValueError(f"Unsupported action format for EEF: {action_format}")

                    reference_camera, action_cameras = self._camera_frame_tracks(data, len(actions_np))
                    if action_format == ActionFormat.XYZ_ROTVEC:
                        frame_kwargs = {
                            "world_from_ref_frames": None if reference_camera is None else reference_camera[None, ...],
                            "world_from_action_frames": action_cameras,
                        }
                        if self.relative_eef_frame == "camera_spatial":
                            spatial_camera = self._reference_camera_transform(data, reference_camera)
                            out_np = compute_camera_spatial_xyz_rotvec_chunks(
                                actions=actions_np,
                                refs=np.asarray(ref_np, dtype=np.float64).reshape(1, -1),
                                delta_indices=np.arange(len(actions_np), dtype=np.int64),
                                world_from_reference_cameras=spatial_camera[None, ...],
                                **frame_kwargs,
                            )[0]
                        else:
                            out_np = compute_relative_xyz_rotvec_chunks(
                                actions=actions_np,
                                refs=np.asarray(ref_np, dtype=np.float64).reshape(1, -1),
                                delta_indices=np.arange(len(actions_np), dtype=np.int64),
                                **frame_kwargs,
                            )[0]
                    else:
                        if self.relative_eef_frame == "camera_spatial":
                            raise ValueError("camera_spatial relative EEF currently requires XYZ_ROTVEC")
                        reference_frame = EndEffectorPose(
                            translation=ref_np[:translation_dim],
                            rotation=ref_np[translation_dim:],
                            rotation_type=rotation_type,
                        )
                        action_poses = [
                            EndEffectorPose(
                                translation=pose[:translation_dim],
                                rotation=pose[translation_dim:],
                                rotation_type=rotation_type,
                            )
                            for pose in actions_np
                        ]
                        if reference_camera is not None:
                            reference_frame = EndEffectorPose(homogeneous=reference_camera @ reference_frame.homogeneous)
                            assert action_cameras is not None
                            action_poses = [
                                EndEffectorPose(homogeneous=camera @ pose.homogeneous)
                                for camera, pose in zip(action_cameras, action_poses)
                            ]
                        traj = EndEffectorActionChunk(action_poses).relative_chunking(reference_frame=reference_frame)
                        out_np = np.stack([pose.xyz_rot6d for pose in traj.poses], dtype=np.float32)

                elif action_type == ActionType.NON_EEF:
                    reference_frame = JointPose(ref_np)
                    traj = JointActionChunk([JointPose(m) for m in actions_np]).relative_chunking(
                        reference_frame=reference_frame
                    )
                    out_np = np.stack([p.joints for p in traj.poses], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown ActionType: {action_type}")

                # Convert back to torch tensor
                if squeeze_output:
                    out_np = out_np.squeeze(0)
                out = torch.from_numpy(out_np).to(action.device)

            if self.in_place:
                data[akey] = out
            else:
                data[self.out_key_map[akey]] = out
        # Camera poses are internal inputs to this transform, not model modalities.
        data.pop(CAMERA_STATE_TRANSFORM_KEY, None)
        data.pop(CAMERA_ACTION_TRANSFORM_KEY, None)
        data.pop(CAMERA_REFERENCE_TRANSFORM_KEY, None)
        data.pop(CAMERA_SKIP_COMPENSATION_KEY, None)
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert relative actions back to absolute poses.

        Camera-spatial recovery requires the current observation-camera
        transform under ``CAMERA_REFERENCE_TRANSFORM_KEY`` or the state-frame
        camera track. Camera-frame sources may additionally provide action-frame
        tracks for an exact moving-camera round trip.
        """
        for akey in self.apply_to:
            in_key = akey if self.in_place else self.out_key_map[akey]
            if in_key not in data:
                continue
            skey = self.state_key_map.get(akey, akey.replace("action.", "state.", 1))
            if skey not in data:
                raise KeyError(f"Missing state key {skey} needed for unapply of {akey}")

            action_rel = data[in_key]
            state = data[skey]
            assert isinstance(action_rel, torch.Tensor)
            assert isinstance(state, torch.Tensor)

            ref = self._pick_ref(state)

            # Get action info for this key
            action_info = self._get_action_info(akey)

            if action_info is None:
                # Fallback to simple addition if no action info
                if action_rel.ndim == 1:
                    action_abs = action_rel + ref
                elif action_rel.ndim == 2:
                    action_abs = action_rel + ref.unsqueeze(0)
                else:
                    raise ValueError(f"{in_key} must be [T,D] or [D], got {tuple(action_rel.shape)}")
            else:
                # Use proper absolute action computation based on type and format
                action_type = action_info["type"]
                action_format = action_info["format"]

                # Convert tensors to numpy for processing
                ref_np = ref.cpu().numpy() if isinstance(ref, torch.Tensor) else ref
                if action_rel.ndim == 1:
                    actions_rel_np = (
                        action_rel.cpu().numpy().reshape(1, -1)
                        if isinstance(action_rel, torch.Tensor)
                        else action_rel.reshape(1, -1)
                    )
                    squeeze_output = True
                elif action_rel.ndim == 2:
                    actions_rel_np = action_rel.cpu().numpy() if isinstance(action_rel, torch.Tensor) else action_rel
                    squeeze_output = False
                else:
                    raise ValueError(f"{in_key} must be [T,D] or [D], got {tuple(action_rel.shape)}")

                if action_type == ActionType.EEF:
                    # Determine rotation type based on action format
                    if action_format == ActionFormat.XYZ_ROT6D:
                        rotation_type = "rot6d"
                        translation_dim = 3
                    elif action_format == ActionFormat.XYZ_ROTVEC:
                        rotation_type = "rotvec"
                        translation_dim = 3
                    else:
                        raise ValueError(f"Unsupported action format for EEF: {action_format}")

                    if self.relative_eef_frame == "camera_spatial":
                        if action_format != ActionFormat.XYZ_ROTVEC:
                            raise ValueError("camera_spatial relative EEF recovery requires XYZ_ROTVEC")
                        reference_storage_frame, output_storage_frames = self._camera_recovery_frames(
                            data,
                            len(actions_rel_np),
                        )
                        reference_camera = self._reference_camera_transform(
                            data,
                            reference_storage_frame,
                        )
                        out_np = recover_camera_spatial_xyz_rotvec_actions(
                            relative_actions=actions_rel_np,
                            reference_pose=np.asarray(ref_np, dtype=np.float64),
                            world_from_reference_camera=reference_camera,
                            world_from_reference_frame=reference_storage_frame,
                            world_from_output_frames=output_storage_frames,
                        )
                    else:
                        reference_frame = EndEffectorPose(
                            translation=ref_np[:translation_dim],
                            rotation=ref_np[translation_dim:],
                            rotation_type=rotation_type,
                        )

                        # Create relative action chunk
                        relative_chunk = EndEffectorActionChunk(
                            [
                                EndEffectorPose(
                                    translation=m[:translation_dim],
                                    rotation=m[translation_dim:],
                                    rotation_type=rotation_type,
                                )
                                for m in actions_rel_np
                            ]
                        )

                        # Convert to absolute
                        absolute_chunk = relative_chunk.to_absolute_chunking(reference_frame=reference_frame)

                        # Convert poses to the correct format based on action_format
                        if action_format == ActionFormat.XYZ_ROT6D:
                            out_np = np.stack([p.xyz_rot6d for p in absolute_chunk.poses], dtype=np.float32)
                        elif action_format == ActionFormat.XYZ_ROTVEC:
                            out_np = np.stack([p.xyz_rotvec for p in absolute_chunk.poses], dtype=np.float32)
                        else:
                            raise ValueError(f"Unsupported action format: {action_format}")

                elif action_type == ActionType.NON_EEF:
                    reference_frame = JointPose(ref_np)

                    # Create relative action chunk
                    relative_chunk = JointActionChunk([JointPose(m) for m in actions_rel_np])

                    # Convert to absolute
                    absolute_chunk = relative_chunk.to_absolute_chunking(reference_frame=reference_frame)
                    out_np = np.stack([p.joints for p in absolute_chunk.poses], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown ActionType: {action_type}")

                # Convert back to torch tensor
                if squeeze_output:
                    out_np = out_np.squeeze(0)
                action_abs = torch.from_numpy(out_np).to(action_rel.device)

            data[akey] = action_abs.to(self._input_dtypes.get(akey, action_abs.dtype))
        if self.relative_eef_frame == "camera_spatial":
            data.pop(CAMERA_STATE_TRANSFORM_KEY, None)
            data.pop(CAMERA_ACTION_TRANSFORM_KEY, None)
            data.pop(CAMERA_REFERENCE_TRANSFORM_KEY, None)
            data.pop(CAMERA_SKIP_COMPENSATION_KEY, None)
        return data
