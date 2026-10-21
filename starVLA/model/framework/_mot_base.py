# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Default CUDA MoT inputs to the GPU processor while retaining explicit opt-out.

from __future__ import annotations

import math
import re
import time
from contextlib import nullcontext
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.mot.action_codec import ActionCodec
from starVLA.model.modules.mot.dinov3_tokens import DinoV3TokenEncoder
from starVLA.model.modules.mot.ema import (
    EMATeacher,
    cosine_representation_loss,
)
from starVLA.model.modules.mot.flow_matching import (
    interpolate,
    make_dual_flow_match_sample,
    make_flow_match_sample,
    masked_l1,
    masked_mean,
    masked_mse,
    sample_beta_timesteps,
    sample_timesteps,
)
from starVLA.model.modules.mot.foresight_forcing import (
    ForesightTimestepSchedule,
    sample_foresight_timesteps,
    scale_teacher_action_timesteps,
)
from starVLA.model.modules.mot.layers import MoTOutput
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"false", "0", "no", "off", ""}
    return bool(value)


def _require_tensor_condition(condition: torch.Tensor, message: str) -> None:
    """Fail closed on a scalar tensor condition without synchronizing CUDA."""

    if condition.ndim != 0:
        raise ValueError(f"Mask validation condition must be scalar, got {tuple(condition.shape)}")
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
    elif hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class MoTFrameworkBase(baseframework):
    """
    Joint future-vision and action rectified-flow model.

    Qwen3.5 supplies trainable multimodal context tokens, frozen DINOv3-B
    supplies the current/future patch-token space, and two private transformer
    experts jointly predict future visual features and an action chunk.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            raise ValueError("Qwen35MoT requires a hierarchical model configuration")
        self.config = config
        framework_config = config.framework
        trainer_config = _cfg_get(config, "trainer", {})
        self.freeze_vlm_for_vla = _as_bool(_cfg_get(trainer_config, "freeze_vlm_for_vla", False))
        self.flow_config = framework_config.flow
        self.action_loss_type = str(_cfg_get(self.flow_config, "action_loss_type", "mse")).lower().strip()
        if self.action_loss_type not in {"mse", "l1"}:
            raise ValueError("flow.action_loss_type must be 'mse' or 'l1', " f"got {self.action_loss_type!r}")
        self.foresight_config = _cfg_get(self.flow_config, "foresight_forcing", {})

        self.qwen_vl_interface = self._build_vlm_model(config)
        self.action_codec = self._build_action_codec(framework_config.action_model)

        dino_config = framework_config.dino
        self.dino_encoder = self._build_dino_encoder(
            dino_config,
        )
        self.spatial_pool_size = int(_cfg_get(dino_config, "spatial_pool_size", 1))
        if self.spatial_pool_size <= 0:
            raise ValueError(f"dino.spatial_pool_size must be positive, got {self.spatial_pool_size}")

        mot_config = dict(framework_config.mot)
        mot_config["action_dim"] = self.action_codec.flow_dim
        robot_state_config = dict(_cfg_get(framework_config.mot, "robot_state", {}))
        robot_state_config.setdefault(
            "state_dim",
            int(_cfg_get(framework_config.action_model, "state_dim", 130)),
        )
        mot_config["robot_state"] = robot_state_config
        self.vlm_feature_layers = self._parse_vlm_feature_layers(
            _cfg_get(framework_config.mot, "vlm_feature_layers", -1),
            depth=int(mot_config.get("depth", 16)),
        )
        context_dim = int(mot_config.get("context_dim", self.qwen_vl_interface.hidden_size))
        if context_dim != self.qwen_vl_interface.hidden_size:
            raise ValueError(
                f"MoT context_dim={context_dim} does not match Qwen3.5 hidden_size="
                f"{self.qwen_vl_interface.hidden_size}"
            )
        self.mot_model = self._build_mot_model(mot_config)
        self.robot_state_enabled = self.mot_model.robot_state_enabled
        self.state_dim = (
            self.mot_model.state_encoder.state_dim
            if self.mot_model.state_encoder is not None
            else int(_cfg_get(framework_config.action_model, "state_dim", 130))
        )
        self.chunk_size = self.action_codec.chunk_size
        self.inference_steps = int(_cfg_get(self.flow_config, "inference_steps", 4))
        if self.inference_steps <= 0:
            raise ValueError(f"flow.inference_steps must be positive, got {self.inference_steps}")
        self.predict_future_vision = _as_bool(_cfg_get(self.flow_config, "predict_future_vision", True))
        if not self.predict_future_vision:
            self.mot_model.future_type.requires_grad_(False)
            self.mot_model.vision_output.requires_grad_(False)
            self.mot_model.vision_final_norm.requires_grad_(False)

        if _as_bool(_cfg_get(_cfg_get(self.flow_config, "self_flow", {}), "enabled", False)):
            raise ValueError("This release supports ordinary flow matching and Foresight Forcing only")
        self.foresight_enabled = _as_bool(_cfg_get(self.foresight_config, "enabled", False))
        self._foresight_teacher: EMATeacher | None = None
        self._foresight_representation_weight = 0.0
        self.foresight_teacher_enabled = _as_bool(_cfg_get(self.foresight_config, "teacher_enabled", True))
        self.foresight_alpha_teacher = float(_cfg_get(self.foresight_config, "alpha_teacher", 1.0))
        if not math.isfinite(self.foresight_alpha_teacher) or not 0.0 < self.foresight_alpha_teacher <= 1.0:
            raise ValueError(
                "flow.foresight_forcing.alpha_teacher must lie in (0, 1], " f"got {self.foresight_alpha_teacher}"
            )
        reliability_config = _cfg_get(self.foresight_config, "dataset_action_reliability", {}) or {}
        try:
            reliability_items = dict(reliability_config).items()
        except (TypeError, ValueError) as error:
            raise TypeError("flow.foresight_forcing.dataset_action_reliability must be a mapping") from error
        self.dataset_action_reliability = {
            str(dataset_id): float(reliability) for dataset_id, reliability in reliability_items
        }
        self._unconfigured_reliability_ids: set[str] = set()
        for dataset_id, reliability in self.dataset_action_reliability.items():
            if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ValueError(
                    "Foresight Forcing dataset reliability must lie in [0, 1], " f"got {dataset_id!r}: {reliability}"
                )
        foresight_decay = float(_cfg_get(self.foresight_config, "ema_decay", 0.999))
        if not 0.0 <= foresight_decay < 1.0:
            raise ValueError("flow.foresight_forcing.ema_decay must lie in [0, 1), " f"got {foresight_decay}")
        foresight_representation_weight = float(_cfg_get(self.foresight_config, "representation_weight", 0.1))
        if not math.isfinite(foresight_representation_weight) or foresight_representation_weight < 0.0:
            raise ValueError(
                "flow.foresight_forcing.representation_weight must be finite and non-negative, "
                f"got {foresight_representation_weight}"
            )
        self._foresight_representation_weight = foresight_representation_weight
        self.vision_rep_projector: nn.Module | None = None
        self.action_rep_projector: nn.Module | None = None
        if self.foresight_enabled and self.foresight_teacher_enabled:
            student_layer = int(_cfg_get(self.foresight_config, "student_layer", 5))
            teacher_layer = int(_cfg_get(self.foresight_config, "teacher_layer", 11))
            if not 1 <= student_layer < teacher_layer <= self.mot_model.depth:
                raise ValueError(
                    f"Foresight Forcing layers must satisfy 1 <= student_layer < teacher_layer <= mot.depth, got {student_layer}, {teacher_layer}, depth={self.mot_model.depth}"
                )
            projector_hidden = int(
                _cfg_get(self.foresight_config, "projector_hidden_size", self.mot_model.hidden_size * 2)
            )
            self.action_rep_projector = self._build_rep_projector(projector_hidden)

    def _build_rep_projector(self, projector_hidden: int) -> nn.Module:
        if projector_hidden <= 0:
            raise ValueError(f"projector_hidden_size must be positive, got {projector_hidden}")
        return nn.Sequential(
            nn.Linear(self.mot_model.hidden_size, projector_hidden),
            nn.SiLU(),
            nn.Linear(projector_hidden, self.mot_model.hidden_size),
        )

    @property
    def vision_expert_parameter_count(self) -> int:
        """Return parameters owned by the vision path."""
        count = self.mot_model.vision_parameter_count()
        if self.vision_rep_projector is not None:
            count += sum(parameter.numel() for parameter in self.vision_rep_projector.parameters())
        return count

    @staticmethod
    def _image_to_chw_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        if torch.is_tensor(image):
            tensor = image.detach().cpu()
        else:
            tensor = torch.as_tensor(np.asarray(image))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        if tensor.ndim != 3:
            raise ValueError(f"Expected an image with 3 dimensions, got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3) and tensor.shape[-1] not in (1, 3):
            chw = tensor
        elif tensor.shape[-1] in (1, 3):
            chw = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Could not identify RGB channels in image shape {tuple(tensor.shape)}")
        if chw.shape[0] == 1:
            chw = chw.repeat(3, 1, 1)
        chw = chw.float()
        if chw.numel() and chw.max() > 2.0:
            chw = chw / 255.0
        return chw.clamp_(0.0, 1.0)

    @classmethod
    def _views_to_tensor(cls, views: Any) -> torch.Tensor:
        if isinstance(views, (list, tuple)):
            return torch.stack([cls._image_to_chw_tensor(view) for view in views], dim=0)
        if torch.is_tensor(views):
            tensor = views.detach().cpu()
        else:
            tensor = torch.as_tensor(np.asarray(views))
        if tensor.ndim == 3:
            return cls._image_to_chw_tensor(tensor).unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected views [V,H,W,C] or [V,C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[1] in (1, 3) and tensor.shape[-1] not in (1, 3):
            view_list = [tensor[index] for index in range(tensor.shape[0])]
        else:
            view_list = [tensor[index] for index in range(tensor.shape[0])]
        return torch.stack([cls._image_to_chw_tensor(view) for view in view_list], dim=0)

    @classmethod
    def _image_pair_to_tensor(cls, image_pair: Any) -> torch.Tensor:
        if isinstance(image_pair, (list, tuple)):
            if len(image_pair) != 2:
                raise ValueError("image_pair must contain exactly the current and future frame")
            return torch.stack([cls._views_to_tensor(frame) for frame in image_pair], dim=0)
        if torch.is_tensor(image_pair):
            tensor = image_pair.detach().cpu()
        else:
            tensor = torch.as_tensor(np.asarray(image_pair))
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 5 or tensor.shape[0] != 2:
            raise ValueError("image_pair must be [2,V,H,W,C] or [2,V,C,H,W], " f"got {tuple(tensor.shape)}")
        # Dataset RGB pairs can stay compact through batching and transfer. DINO
        # converts uint8 to normalized floats on its device; other image formats
        # retain the existing conversion and validation below.
        if tensor.dtype == torch.uint8:
            if tensor.shape[2] == 3 and tensor.shape[-1] not in (1, 3):
                return tensor
            if tensor.shape[-1] == 3:
                return tensor.permute(0, 1, 4, 2, 3)
        return torch.stack(
            [cls._views_to_tensor(tensor[frame_index]) for frame_index in range(2)],
            dim=0,
        )

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        if image.dtype == torch.uint8:
            return Image.fromarray(image.detach().cpu().permute(1, 2, 0).numpy())
        image = image.detach().float().cpu().clamp(0, 1)
        array = image.permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))

    @staticmethod
    def _sequence_to_tensor(
        value: Any,
        horizon: int,
        width: int,
        *,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected a sequence [T,D], got {tuple(tensor.shape)}")
        output = tensor.new_full((horizon, width), fill_value)
        copy_horizon = min(horizon, tensor.shape[0])
        copy_width = min(width, tensor.shape[1])
        output[:copy_horizon, :copy_width] = tensor[:copy_horizon, :copy_width]
        return output

    def _current_state_to_tensor(self, value: Any, *, field_name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape != (1, self.state_dim):
            raise ValueError(
                f"{field_name} must be the current canonical state [1,{self.state_dim}], " f"got {tuple(tensor.shape)}"
            )
        if field_name == "state":
            tensor = tensor.to(dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError("state contains non-finite values")
            return tensor
        return tensor.bool()

    @staticmethod
    def _collect_meta(examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
        fields = (
            "embodiment_tag",
            "end_effector_type",
            "dataset_id",
            "dataset_family",
            "robot_family",
            "robot_type",
            "control_mode",
            "action_representation",
        )
        meta: dict[str, list[Any]] = {}
        for field in fields:
            values = []
            has_value = False
            for example in examples:
                nested = example.get("meta", {}) or {}
                value = example.get(field, nested.get(field))
                values.append(value)
                has_value = has_value or value is not None
            if has_value:
                meta[field] = values
        return meta

    @staticmethod
    def _coerce_action_valid(value: Any, *, default: bool) -> bool:
        """Normalize the sample-level action-validity mask to one boolean."""

        if value is None:
            return default
        validity = torch.as_tensor(value)
        if validity.numel() != 1:
            raise ValueError("action_valid must contain one sample-level boolean, " f"got shape {tuple(validity.shape)}")
        return bool(validity.item())

    def _batch_examples(
        self,
        examples: list[dict[str, Any]] | dict[str, Any],
        *,
        require_action: bool,
        duplicate_current: bool = False,
    ) -> dict[str, Any]:
        if isinstance(examples, dict):
            examples = [examples]
        if not isinstance(examples, list) or not examples:
            raise ValueError("Qwen35MoT received an empty or invalid example batch")

        action_horizon = self.action_codec.chunk_size
        max_training_horizon = getattr(self.action_codec, "max_training_chunk_size", action_horizon)
        if require_action and max_training_horizon != action_horizon:
            for index, example in enumerate(examples):
                if "action" not in example:
                    continue
                shape = torch.as_tensor(example["action"]).shape
                if (
                    len(shape) != 2
                    or shape[1] != self.action_codec.action_dim
                    or not 0 < shape[0] <= max_training_horizon
                ):
                    raise ValueError(f"Invalid variable-horizon action shape {tuple(shape)} at batch index {index}")
                action_horizon = max(action_horizon, shape[0])

        pairs = []
        current_frames = []
        for example in examples:
            if "image_pair" in example:
                pair = self._image_pair_to_tensor(example["image_pair"])
            elif "image" in example:
                image = example["image"]
                if isinstance(image, (list, tuple)):
                    if len(image) != 1:
                        raise ValueError(
                            "Inference without image_pair accepts exactly one head/wrist-composite image; "
                            "compose multiple camera views before calling predict_action"
                        )
                    image = image[0]
                frame = self._image_to_chw_tensor(image).unsqueeze(0)
                pair = torch.stack((frame, frame), dim=0)
            else:
                raise KeyError("Each example must contain image_pair or one head/wrist-composite image")
            if duplicate_current:
                pair = torch.stack((pair[0], pair[0]), dim=0)
            pairs.append(pair)
            current_frames.append(pair[0, 0])

        # Avoid promoting raw uint8 pixels to unnormalized [0, 255] floats when
        # a batch also includes a legacy float/PIL sample.
        all_uint8 = all(pair.dtype == torch.uint8 for pair in pairs)
        if not all_uint8:
            pairs = [pair.float().div_(255.0) if pair.dtype == torch.uint8 else pair for pair in pairs]
        device = next(self.mot_model.parameters()).device
        image_pair = torch.stack(pairs, dim=0).to(device=device, non_blocking=True)
        qwen_config = _cfg_get(_cfg_get(getattr(self, "config", None), "framework", {}), "qwenvl", {})
        if all_uint8 and _as_bool(_cfg_get(qwen_config, "image_processor_on_gpu", device.type == "cuda")):
            # Reuse the current-frame bytes already uploaded for DINO. The
            # torchvision-backed Qwen processor accepts CHW tensors directly.
            current_images = list(image_pair[:, 0, 0].unbind(0))
        else:
            current_images = [self._tensor_to_pil(frame) for frame in current_frames]
        batch: dict[str, Any] = {
            "image_pair": image_pair,
            "current_images": current_images,
            "instructions": [str(example.get("lang", example.get("instruction", ""))) for example in examples],
            "meta": self._collect_meta(examples),
        }
        action_valid_values: list[bool] | None = None
        if require_action:
            missing = [index for index, example in enumerate(examples) if "action" not in example]
            if missing and not self.foresight_enabled:
                raise ValueError(f"Training examples are missing action labels at batch indices {missing}")
            actions = []
            masks = []
            action_valid_values = []
            for example in examples:
                has_action = "action" in example
                if has_action:
                    action = torch.as_tensor(example["action"])
                    supplied_mask = example.get("action_mask")
                    default_valid = bool(supplied_mask is None or torch.as_tensor(supplied_mask).bool().any().item())
                else:
                    action = torch.zeros(
                        action_horizon,
                        self.action_codec.action_dim,
                        dtype=torch.float32,
                    )
                    supplied_mask = None
                    default_valid = False
                action_valid = self._coerce_action_valid(
                    example.get("action_valid"),
                    default=default_valid,
                )
                if action_valid and not has_action:
                    raise ValueError("action_valid=true requires an action label")
                actions.append(
                    self._sequence_to_tensor(
                        action,
                        action_horizon,
                        self.action_codec.action_dim,
                    )
                )
                mask = supplied_mask
                if mask is None:
                    mask = (
                        torch.ones_like(action, dtype=torch.float32)
                        if has_action
                        else torch.zeros_like(action, dtype=torch.float32)
                    )
                mask = self._sequence_to_tensor(
                    mask,
                    action_horizon,
                    self.action_codec.action_dim,
                )
                if not action_valid:
                    mask.zero_()
                masks.append(mask)
                action_valid_values.append(action_valid)
            batch["action"] = torch.stack(actions, dim=0).to(device=device)
            batch["action_mask"] = torch.stack(masks, dim=0).to(device=device)
            batch["action_valid"] = torch.tensor(
                action_valid_values,
                device=device,
                dtype=torch.bool,
            )
        if self.robot_state_enabled:
            missing_state = []
            missing_mask = []
            for index, example in enumerate(examples):
                may_omit = bool(
                    require_action
                    and self.foresight_enabled
                    and action_valid_values is not None
                    and not action_valid_values[index]
                )
                if "state" not in example and not may_omit:
                    missing_state.append(index)
                if "state_mask" not in example and not may_omit:
                    missing_mask.append(index)
            if missing_state or missing_mask:
                raise ValueError(
                    "Robot-state conditioning requires state and state_mask for every example; "
                    f"missing state at {missing_state}, missing state_mask at {missing_mask}"
                )
            zero_state = torch.zeros(1, self.state_dim, dtype=torch.float32)
            zero_state_mask = torch.zeros(1, self.state_dim, dtype=torch.bool)
            batch["state"] = torch.stack(
                [
                    self._current_state_to_tensor(
                        example.get("state", zero_state),
                        field_name="state",
                    )
                    for example in examples
                ],
                dim=0,
            ).to(device=device)
            batch["state_mask"] = torch.stack(
                [
                    self._current_state_to_tensor(
                        example.get("state_mask", zero_state_mask),
                        field_name="state_mask",
                    )
                    for example in examples
                ],
                dim=0,
            ).to(device=device)
        return batch

    def _pool_vision_tokens(
        self,
        tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        if self.spatial_pool_size == 1:
            return tokens
        patch_height, patch_width = spatial_shape
        feature_map = tokens.reshape(tokens.shape[0], patch_height, patch_width, tokens.shape[-1])
        feature_map = feature_map.permute(0, 3, 1, 2)
        feature_map = F.avg_pool2d(
            feature_map,
            kernel_size=self.spatial_pool_size,
            stride=self.spatial_pool_size,
        )
        return feature_map.flatten(2).transpose(1, 2)

    def _encode_condition(
        self,
        batch: dict[str, Any],
        *,
        current_only: bool = False,
        timing: dict[str, float] | None = None,
        precomputed_qwen_context: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        precomputed_context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.mot_model.parameters()).device
        if timing is not None:
            self._synchronize(device)
            qwen_preprocess_start = time.perf_counter()
        if precomputed_qwen_context is None:
            if precomputed_context_mask is not None:
                raise ValueError("precomputed_context_mask requires precomputed_qwen_context")
            qwen_config = self.config.framework.qwenvl
            processor_kwargs: dict[str, Any] = {}
            max_context_tokens = _cfg_get(qwen_config, "max_context_tokens", None)
            if max_context_tokens is not None:
                processor_kwargs.update(truncation=True, max_length=int(max_context_tokens))
            if _as_bool(_cfg_get(qwen_config, "image_processor_on_gpu", device.type == "cuda")):
                processor_kwargs["images_kwargs"] = {"device": device, "disable_grouping": False}
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch["current_images"],
                instructions=batch["instructions"],
                **processor_kwargs,
            )
            if timing is not None:
                self._synchronize(device)
                qwen_preprocess_end = time.perf_counter()
            if self.freeze_vlm_for_vla:
                with torch.no_grad():
                    qwen_context = self._encode_vlm_context(qwen_inputs)
                if isinstance(qwen_context, torch.Tensor):
                    qwen_context = qwen_context.detach()
                else:
                    qwen_context = tuple(layer_context.detach() for layer_context in qwen_context)
            else:
                qwen_context = self._encode_vlm_context(qwen_inputs)
            context_mask = qwen_inputs.get("attention_mask")
        else:
            if precomputed_context_mask is None:
                raise ValueError("precomputed_qwen_context requires precomputed_context_mask")
            qwen_context = precomputed_qwen_context
            context_mask = precomputed_context_mask
            if timing is not None:
                qwen_preprocess_end = qwen_preprocess_start
        if timing is not None:
            self._synchronize(device)
            qwen_encode_end = time.perf_counter()
        context_reference = qwen_context if isinstance(qwen_context, torch.Tensor) else qwen_context[0]
        if context_mask is None:
            context_mask = torch.ones(
                context_reference.shape[:2],
                device=context_reference.device,
                dtype=torch.bool,
            )
        elif context_mask.shape != context_reference.shape[:2]:
            raise RuntimeError(
                f"Qwen attention mask shape {tuple(context_mask.shape)} does not match context "
                f"{tuple(context_reference.shape[:2])}"
            )

        if current_only:
            current_output = self.dino_encoder.encode_current(batch["image_pair"])
            current_vision = self._pool_vision_tokens(current_output.tokens, current_output.spatial_shape)
            future_vision = current_vision
        else:
            dino_output = self.dino_encoder(batch["image_pair"])
            current_vision = self._pool_vision_tokens(dino_output.current_tokens, dino_output.spatial_shape)
            future_vision = self._pool_vision_tokens(dino_output.future_tokens, dino_output.spatial_shape)
        if timing is not None:
            self._synchronize(device)
            dino_encode_end = time.perf_counter()
            timing.update(
                {
                    "qwen_preprocess_ms": (qwen_preprocess_end - qwen_preprocess_start) * 1000.0,
                    "vlm_encode_ms": (qwen_encode_end - qwen_preprocess_end) * 1000.0,
                    "dino_encode_ms": (dino_encode_end - qwen_encode_end) * 1000.0,
                }
            )
        return qwen_context, context_mask.bool(), current_vision, future_vision

    def _joint_supervised_vlm_context(
        self,
        qwen_inputs: Mapping[str, Any],
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute FAST loss and detached prompt-only MoT context in one Qwen forward."""

        prompt_mask = qwen_inputs.get("prompt_attention_mask")
        attention_mask = qwen_inputs.get("attention_mask")
        labels = qwen_inputs.get("labels")
        if not all(isinstance(value, torch.Tensor) for value in (prompt_mask, attention_mask, labels)):
            raise ValueError("Joint FAST/MoT inputs require tensor prompt_attention_mask, attention_mask, and labels")
        if prompt_mask.shape != attention_mask.shape or labels.shape != attention_mask.shape:
            raise ValueError(
                "Joint FAST/MoT masks and labels must share [B,S], got "
                f"prompt={tuple(prompt_mask.shape)}, attention={tuple(attention_mask.shape)}, "
                f"labels={tuple(labels.shape)}"
            )
        prompt_mask = prompt_mask.bool()
        attention_mask = attention_mask.bool()
        supervised_mask = labels.ne(-100)
        _require_tensor_condition(
            ~(prompt_mask & ~attention_mask).any(),
            "Joint FAST/MoT prompt mask includes padded positions",
        )
        _require_tensor_condition(
            ~(supervised_mask & ~attention_mask).any(),
            "Joint FAST/MoT labels supervise padded positions",
        )
        _require_tensor_condition(
            ~(prompt_mask & supervised_mask).any(),
            "Joint FAST/MoT prompt and supervised masks overlap",
        )
        active_suffix = attention_mask & ~prompt_mask
        suffix_seen = active_suffix.to(dtype=torch.long).cumsum(dim=-1).gt(0)
        _require_tensor_condition(
            ~(prompt_mask & suffix_seen).any(),
            "Joint FAST/MoT prompt positions must form a causal prefix before every FAST suffix",
        )
        prompt_seen = prompt_mask.to(dtype=torch.long).cumsum(dim=-1).gt(0)
        prompt_gap_seen = ((~prompt_mask) & prompt_seen).to(dtype=torch.long).cumsum(dim=-1).gt(0)
        _require_tensor_condition(
            ~(prompt_mask & prompt_gap_seen).any(),
            "Joint FAST/MoT prompt positions must form one contiguous causal span",
        )
        _require_tensor_condition(
            prompt_mask.any(dim=-1).all(),
            "Every joint FAST/MoT sample must retain at least one prompt token",
        )
        _require_tensor_condition(
            supervised_mask.any(),
            "Joint FAST/MoT batch has no supervised FAST tokens",
        )

        max_context_tokens = _cfg_get(self.config.framework.qwenvl, "max_context_tokens", None)
        if max_context_tokens is not None:
            max_context_tokens = int(max_context_tokens)
            if max_context_tokens <= 0:
                raise ValueError(f"framework.qwenvl.max_context_tokens must be positive, got {max_context_tokens}")
            prompt_ordinal = prompt_mask.to(dtype=torch.long).cumsum(dim=-1)
            prompt_mask = prompt_mask & prompt_ordinal.le(max_context_tokens)

        qwen_output, captured_context = self.qwen_vl_interface.forward_with_selected_hidden_states(
            hidden_layers=self.vlm_feature_layers,
            required_decoder_layer_type=self._joint_context_decoder_layer_type(),
            **qwen_inputs,
        )
        fast_loss = qwen_output.get("loss") if isinstance(qwen_output, Mapping) else getattr(qwen_output, "loss", None)
        if not isinstance(fast_loss, torch.Tensor):
            raise ValueError("Joint FAST/MoT Qwen forward did not return a tensor language-model loss")

        qwen_context: torch.Tensor | tuple[torch.Tensor, ...]
        if self.vlm_feature_layers is None:
            if len(captured_context) != 1:
                raise RuntimeError(
                    "Final-state MoT conditioning expected one captured Qwen context, " f"got {len(captured_context)}"
                )
            qwen_context = captured_context[0]
            context_layers = (qwen_context,)
        else:
            if len(captured_context) != len(self.vlm_feature_layers):
                raise RuntimeError(
                    f"Expected {len(self.vlm_feature_layers)} captured Qwen contexts, " f"got {len(captured_context)}"
                )
            qwen_context = tuple(captured_context)
            context_layers = qwen_context

        for layer_context in context_layers:
            if layer_context.shape[:2] != prompt_mask.shape:
                raise RuntimeError(
                    f"Captured Qwen context {tuple(layer_context.shape[:2])} does not match prompt mask "
                    f"{tuple(prompt_mask.shape)}"
                )

        # Causal Qwen states at prompt positions cannot attend to the later
        # FAST suffix. Pack only those states so MoT neither sees nor computes
        # projections/attention for discrete-action or padding positions.
        prompt_lengths = prompt_mask.sum(dim=-1, dtype=torch.long)
        packed_length = int(prompt_lengths.max().item())
        if packed_length <= 0:
            raise RuntimeError("Joint FAST/MoT prompt packing received no context tokens")
        packed_mask = (
            torch.arange(
                packed_length,
                device=prompt_mask.device,
            )
            .unsqueeze(0)
            .lt(prompt_lengths.unsqueeze(1))
        )
        prompt_starts = prompt_mask.to(dtype=torch.long).argmax(dim=-1)
        source_sequence = prompt_starts.unsqueeze(1) + torch.arange(
            packed_length,
            device=prompt_mask.device,
        ).unsqueeze(0)
        source_sequence = source_sequence.clamp_max(prompt_mask.shape[1] - 1)
        packed_layers = []
        for layer_context in context_layers:
            gather_index = source_sequence.unsqueeze(-1).expand(-1, -1, layer_context.shape[-1])
            packed_context = layer_context.detach().gather(dim=1, index=gather_index)
            packed_context = packed_context.masked_fill(~packed_mask.unsqueeze(-1), 0)
            packed_layers.append(packed_context)
        packed_context: torch.Tensor | tuple[torch.Tensor, ...]
        packed_context = packed_layers[0] if self.vlm_feature_layers is None else tuple(packed_layers)
        return packed_context, packed_mask, fast_loss

    def _autocast_context(self) -> Any:
        device = next(self.mot_model.parameters()).device
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _get_foresight_teacher(self) -> EMATeacher:
        if not self.foresight_enabled or not self.foresight_teacher_enabled:
            raise RuntimeError("Foresight teacher requested while flow.foresight_forcing teacher is disabled")
        if self._foresight_teacher is None:
            decay = float(_cfg_get(self.foresight_config, "ema_decay", 0.999))
            self._foresight_teacher = EMATeacher(self.mot_model, decay=decay)
            logger.info(f"Initialized Foresight Forcing EMA teacher with decay={decay}")
        student_parameter = next(self.mot_model.parameters())
        self._foresight_teacher.to(device=student_parameter.device)
        return self._foresight_teacher

    def set_foresight_representation_weight(self, weight: float) -> None:
        """Set the effective runtime weight for the Foresight representation objective."""

        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Foresight representation weight must be finite and non-negative, got {weight}")
        self._foresight_representation_weight = weight
        # Materialize the shadow copy before the first optimizer update even
        # when its forward is intentionally skipped during loss warmup.
        if self.foresight_enabled and self.foresight_teacher_enabled and self._foresight_teacher is None:
            self._get_foresight_teacher()

    def update_self_flow_teacher(self) -> None:
        """Update active training-only EMA teachers after a synchronized optimizer step."""

        if self.foresight_enabled and self.foresight_teacher_enabled and self._foresight_teacher is not None:
            self._foresight_teacher.update(self.mot_model)

    def expects_training_ema_state(self) -> bool:
        """Return whether this training configuration owns an EMA teacher."""

        return self.foresight_enabled and self.foresight_teacher_enabled

    def training_ema_state_dict(self) -> dict[str, Any]:
        """Return explicitly checkpointed EMA state without registering it on the model."""

        state: dict[str, Any] = {"format_version": 1}
        if self._foresight_teacher is not None:
            state["foresight_forcing"] = {
                "decay": self._foresight_teacher.decay,
                "module": self._foresight_teacher.state_dict(),
            }
        return state

    def load_training_ema_state_dict(self, state_dict: Mapping[str, Any]) -> tuple[str, ...]:
        """Restore configured EMA teachers from a trainer sidecar checkpoint."""

        if int(state_dict.get("format_version", 0)) != 1:
            raise ValueError("Unsupported EMA checkpoint format_version " f"{state_dict.get('format_version')!r}")
        loaded = []
        foresight_state = state_dict.get("foresight_forcing")
        if foresight_state is not None and self.foresight_enabled and self.foresight_teacher_enabled:
            teacher = self._get_foresight_teacher()
            teacher.decay = float(foresight_state["decay"])
            teacher.load_state_dict(foresight_state["module"], strict=True)
            loaded.append("foresight_forcing")
        return tuple(loaded)

    def _foresight_reliabilities(
        self,
        metadata: dict[str, list[Any]],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[str], torch.Tensor]:
        dataset_ids = metadata.get("dataset_id")
        if (
            dataset_ids is None
            or len(dataset_ids) != batch_size
            or any(dataset_id is None for dataset_id in dataset_ids)
        ):
            raise ValueError("Foresight Forcing requires one dataset_id for every training sample")
        normalized_ids = [str(dataset_id) for dataset_id in dataset_ids]
        default_reliability = self.dataset_action_reliability.get("*", 1.0)
        missing_ids = {
            dataset_id
            for dataset_id in normalized_ids
            if dataset_id not in self.dataset_action_reliability
            and "*" not in self.dataset_action_reliability
            and dataset_id not in self._unconfigured_reliability_ids
        }
        if missing_ids:
            logger.warning(
                "Foresight Forcing dataset reliability is unconfigured for %s; using q_d=%s",
                sorted(missing_ids),
                default_reliability,
            )
            self._unconfigured_reliability_ids.update(missing_ids)
        reliability = torch.tensor(
            [self.dataset_action_reliability.get(dataset_id, default_reliability) for dataset_id in normalized_ids],
            device=device,
            dtype=torch.float32,
        )
        return normalized_ids, reliability

    @staticmethod
    def _foresight_metric_slug(dataset_id: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("_")
        return slug or "unknown"

    def _foresight_timestep_metrics(
        self,
        schedule: ForesightTimestepSchedule,
        dataset_ids: list[str],
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_v = schedule.vision_timesteps.detach()
        t_a = schedule.action_timesteps.detach()
        gap = t_a - t_v
        metrics = {
            "foresight/t_v_mean": t_v.mean(),
            "foresight/t_a_mean": t_a.mean(),
            "foresight/timestep_gap_mean": gap.mean(),
            "foresight/action_after_vision_fraction": (t_a > t_v).float().mean(),
            "foresight/teacher_edge_fraction": schedule.teacher_edge_mask.float().mean(),
            "foresight/reliability_mean": reliability.detach().mean(),
            "foresight/alpha_mean": schedule.alpha.detach().mean(),
        }
        if not _as_bool(_cfg_get(self.foresight_config, "log_per_dataset_metrics", True)):
            return metrics
        for dataset_id in sorted(set(dataset_ids)):
            indices = torch.tensor(
                [value == dataset_id for value in dataset_ids],
                device=t_v.device,
                dtype=torch.bool,
            )
            prefix = f"foresight/dataset/{self._foresight_metric_slug(dataset_id)}"
            metrics.update(
                {
                    f"{prefix}/sample_fraction": indices.float().mean(),
                    f"{prefix}/reliability": reliability[indices].mean(),
                    f"{prefix}/alpha": schedule.alpha[indices].mean(),
                    f"{prefix}/t_v_mean": t_v[indices].mean(),
                    f"{prefix}/t_a_mean": t_a[indices].mean(),
                    f"{prefix}/timestep_gap_mean": gap[indices].mean(),
                    f"{prefix}/action_after_vision_fraction": (t_a[indices] > t_v[indices]).float().mean(),
                }
            )
        return metrics

    def _action_flow_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the configured action velocity regression loss."""

        loss_fn = masked_l1 if self.action_loss_type == "l1" else masked_mse
        return loss_fn(prediction, target, mask)

    def _reliability_weighted_action_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        """Apply dataset reliability weighting to the configured action loss."""

        reliability_shape = (reliability.shape[0],) + (1,) * (prediction.ndim - 1)
        error = prediction.float() - target.float()
        elementwise_loss = error.abs() if self.action_loss_type == "l1" else error.square()
        weighted_loss = elementwise_loss * reliability.float().reshape(reliability_shape)
        return masked_mean(weighted_loss, valid_mask)

    def forward(
        self,
        examples: list[dict[str, Any]] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if examples is None:
            raise ValueError("Qwen35MoT.forward requires examples")
        fast_action_inputs = kwargs.pop("fast_action_inputs", None)
        if kwargs:
            raise TypeError(f"Qwen35MoT.forward received unexpected keyword arguments: {sorted(kwargs)}")
        if fast_action_inputs is not None and not isinstance(fast_action_inputs, Mapping):
            raise TypeError("fast_action_inputs must be a mapping of tokenized Qwen inputs")
        batch = self._batch_examples(examples, require_action=True)
        with self._autocast_context():
            fast_action_loss = None
            precomputed_context = None
            precomputed_context_mask = None
            if fast_action_inputs is not None:
                (
                    precomputed_context,
                    precomputed_context_mask,
                    fast_action_loss,
                ) = self._joint_supervised_vlm_context(fast_action_inputs)
            context, context_mask, current_vision, clean_vision = self._encode_condition(
                batch,
                current_only=not self.predict_future_vision,
                precomputed_qwen_context=precomputed_context,
                precomputed_context_mask=precomputed_context_mask,
            )
            if not self.predict_future_vision:
                clean_vision = current_vision[:, :0]
            clean_action = self.action_codec.encode(
                batch["action"],
                batch["meta"],
                batch["action_mask"],
            )
            action_token_mask = self.action_codec.token_mask(batch["action_mask"], clean_action)
            action_loss_mask = self.action_codec.flow_loss_mask(batch["action_mask"], clean_action)
            robot_state = batch.get("state")
            robot_state_mask = batch.get("state_mask")
            if self.foresight_enabled:
                action_valid = batch["action_valid"].bool()
                action_token_mask = action_token_mask & action_valid[:, None]
                if action_loss_mask is None:
                    action_loss_mask = action_valid
                else:
                    valid_shape = (action_valid.shape[0],) + (1,) * (action_loss_mask.ndim - 1)
                    action_loss_mask = action_loss_mask * action_valid.reshape(valid_shape).to(action_loss_mask)
                output = self._foresight_forward(
                    context,
                    context_mask,
                    current_vision,
                    robot_state,
                    robot_state_mask,
                    clean_vision,
                    clean_action,
                    action_token_mask,
                    action_loss_mask,
                    action_valid,
                    batch["meta"],
                )
            else:
                output = self._ordinary_flow_forward(
                    context,
                    context_mask,
                    current_vision,
                    robot_state,
                    robot_state_mask,
                    clean_vision,
                    clean_action,
                    action_token_mask,
                    action_loss_mask,
                )
            if fast_action_loss is not None:
                output["vlm_fast_loss"] = fast_action_loss
            return output

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def encode_future_vision_target(
        self,
        examples: list[dict[str, Any]] | dict[str, Any],
    ) -> torch.Tensor:
        """Encode held-out future frames without exposing them to the policy condition."""
        if not self.predict_future_vision:
            raise RuntimeError("Future DINO prediction is disabled by flow.predict_future_vision=false")
        was_training = self.training
        self.eval()
        try:
            batch = self._batch_examples(examples, require_action=False)
            with self._autocast_context():
                future_output = self.dino_encoder.encode_future(batch["image_pair"])
                future_vision = self._pool_vision_tokens(
                    future_output.tokens,
                    future_output.spatial_shape,
                )
            return future_vision.float().cpu()
        finally:
            self.train(was_training)

    def _prepare_inference_mot_cache(
        self,
        context: torch.Tensor | tuple[torch.Tensor, ...],
        context_mask: torch.Tensor,
        current_vision: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Build optional model-specific state reused by every flow step."""

        del self, context, context_mask, current_vision, batch
        return {}

    def _integrate_flow(
        self,
        *,
        context: torch.Tensor | tuple[torch.Tensor, ...],
        context_mask: torch.Tensor,
        current_vision: torch.Tensor,
        vision_state: torch.Tensor,
        action_state: torch.Tensor,
        action_token_mask: torch.Tensor | None,
        batch: dict[str, Any],
        inference_mot_kwargs: dict[str, Any],
        unconditional_mask: torch.Tensor | None,
        unconditional_kwargs: dict[str, Any] | None,
        steps: int,
        visual_timestep_alpha: float,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate initialized flow noise without sampling or crossing the CPU boundary.

        Keeping RNG and observation encoding outside this tensor-only boundary
        permits optional serving-time capture of the entire original flow loop.
        """
        device = action_state.device
        action_time_grid = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=device,
            dtype=torch.float32,
        )
        vision_time_grid = action_time_grid
        if visual_timestep_alpha != 1.0:
            vision_time_grid = torch.pow(vision_time_grid, visual_timestep_alpha)
        with self._autocast_context():
            for step_index in range(steps):
                action_timestep = action_time_grid[step_index].expand(action_state.shape[0])
                vision_timestep = vision_time_grid[step_index].expand(action_state.shape[0])
                action_delta_time = (action_time_grid[step_index + 1] - action_time_grid[step_index]).to(vision_state)
                vision_delta_time = (vision_time_grid[step_index + 1] - vision_time_grid[step_index]).to(vision_state)
                output = self.mot_model(
                    context=context,
                    context_mask=context_mask,
                    current_vision=current_vision,
                    future_vision=vision_state,
                    action=action_state,
                    vision_timesteps=vision_timestep,
                    action_timesteps=action_timestep,
                    action_token_mask=action_token_mask,
                    robot_state=batch.get("state"),
                    robot_state_mask=batch.get("state_mask"),
                    **inference_mot_kwargs,
                )
                action_velocity = output.action_velocity
                vision_velocity = output.vision_velocity
                if unconditional_mask is not None:
                    # Match training dropout: remove VLM context only, retaining
                    # current vision, robot state and action-schema metadata.
                    unconditional = self.mot_model(
                        context=context,
                        context_mask=unconditional_mask,
                        current_vision=current_vision,
                        future_vision=vision_state,
                        action=action_state,
                        vision_timesteps=vision_timestep,
                        action_timesteps=action_timestep,
                        action_token_mask=action_token_mask,
                        robot_state=batch.get("state"),
                        robot_state_mask=batch.get("state_mask"),
                        **unconditional_kwargs,
                    )
                    action_velocity = unconditional.action_velocity + cfg_scale * (
                        action_velocity - unconditional.action_velocity
                    )
                    if self.predict_future_vision:
                        vision_velocity = unconditional.vision_velocity + cfg_scale * (
                            vision_velocity - unconditional.vision_velocity
                        )
                if self.predict_future_vision:
                    vision_state = vision_state + vision_delta_time * vision_velocity
                action_state = self._update_inference_action_state(
                    action_state,
                    action_velocity,
                    action_delta_time,
                    batch,
                )
            decoded_action = self._decode_inference_action_state(action_state, batch)
        return decoded_action, vision_state

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict[str, Any]] | dict[str, Any],
        *,
        return_timing: bool = False,
        num_flow_steps: int | None = None,
        seed: int | None = None,
        visual_timestep_alpha: float | None = None,
        cfg_scale: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        cfg_scale = getattr(self, "inference_cfg_scale", 1.0) if cfg_scale is None else cfg_scale
        if (
            isinstance(cfg_scale, bool)
            or not isinstance(cfg_scale, (int, float))
            or not math.isfinite(cfg_scale)
            or cfg_scale < 0
        ):
            raise ValueError("cfg_scale must be a finite non-negative number")
        if cfg_scale != 1.0 and (
            not getattr(self, "cfg_context_dropout_enabled", False) or getattr(self, "cfg_context_dropout_prob", 0) <= 0
        ):
            raise ValueError("CFG requires a policy trained with context dropout")
        if visual_timestep_alpha is None:
            visual_timestep_alpha = getattr(
                self,
                "inference_visual_timestep_alpha",
                1.0,
            )
        if not isinstance(visual_timestep_alpha, (int, float)):
            raise TypeError(
                f"visual_timestep_alpha must be a float-like value, got " f"{type(visual_timestep_alpha).__name__}"
            )
        visual_timestep_alpha = float(visual_timestep_alpha)
        if not math.isfinite(visual_timestep_alpha) or visual_timestep_alpha <= 0.0:
            raise ValueError(f"visual_timestep_alpha must be finite and positive, got {visual_timestep_alpha}")

        was_training = self.training
        self.eval()
        device = next(self.mot_model.parameters()).device
        try:
            batch = self._batch_examples(examples, require_action=False, duplicate_current=True)
            self._synchronize(device)
            total_start = time.perf_counter()
            context_start = total_start
            condition_timing: dict[str, float] = {}
            with self._autocast_context():
                context, context_mask, current_vision, _ = self._encode_condition(
                    batch,
                    current_only=True,
                    timing=condition_timing if return_timing else None,
                )
            self._synchronize(device)
            context_end = time.perf_counter()

            steps = int(num_flow_steps or self.inference_steps)
            if steps <= 0:
                raise ValueError(f"num_flow_steps must be positive, got {steps}")
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
            if self.predict_future_vision:
                vision_state = torch.randn(
                    current_vision.shape,
                    device=device,
                    dtype=current_vision.dtype,
                    generator=generator,
                )
            else:
                vision_state = current_vision[:, :0]
            action_state, action_token_mask = self._build_inference_action_state(
                batch,
                current_vision,
                generator,
            )
            inference_mot_kwargs = self._inference_mot_kwargs(context, batch)
            reusable_mot_kwargs = self._prepare_inference_mot_cache(
                context,
                context_mask,
                current_vision,
                batch,
            )
            overlap = set(inference_mot_kwargs).intersection(reusable_mot_kwargs)
            if overlap:
                raise RuntimeError(f"Inference MoT kwargs and cache kwargs overlap: {sorted(overlap)}")
            inference_mot_kwargs.update(reusable_mot_kwargs)

            unconditional_mask = torch.zeros_like(context_mask) if cfg_scale != 1.0 else None
            unconditional_kwargs = None
            if unconditional_mask is not None:
                unconditional_kwargs = self._inference_mot_kwargs(context, batch)
                unconditional_kwargs.update(
                    self._prepare_inference_mot_cache(context, unconditional_mask, current_vision, batch)
                )

            self._synchronize(device)
            flow_start = time.perf_counter()
            decoded_action, vision_state = self._integrate_flow(
                context=context,
                context_mask=context_mask,
                current_vision=current_vision,
                vision_state=vision_state,
                action_state=action_state,
                action_token_mask=action_token_mask,
                batch=batch,
                inference_mot_kwargs=inference_mot_kwargs,
                unconditional_mask=unconditional_mask,
                unconditional_kwargs=unconditional_kwargs,
                steps=steps,
                visual_timestep_alpha=visual_timestep_alpha,
                cfg_scale=cfg_scale,
            )
            self._synchronize(device)
            flow_end = time.perf_counter()

            timing = {
                **condition_timing,
                "context_encode_ms": (context_end - context_start) * 1000.0,
                "flow_4step_ms": (flow_end - flow_start) * 1000.0 if steps == 4 else None,
                "flow_ms": (flow_end - flow_start) * 1000.0,
                "total_ms": (flow_end - total_start) * 1000.0,
                "num_flow_steps": steps,
            }
            result: dict[str, Any] = {"normalized_actions": decoded_action.float().cpu().numpy()}
            if self.predict_future_vision:
                # Inference outputs cross the websocket/msgpack boundary, so
                # keep every returned array transport-safe and consistent with
                # normalized_actions.
                result["predicted_future_vision"] = vision_state.float().cpu().numpy()
            if return_timing:
                result["timing"] = timing
            return result
        finally:
            self.train(was_training)
