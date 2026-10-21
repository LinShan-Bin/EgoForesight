# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add schema-aware MoT2 with CFG context dropout and random FF teacher timesteps.

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import torch

from starVLA.model.framework._mot_base import MoTFrameworkBase, _as_bool, _cfg_get
from starVLA.model.modules.mot.action_codec import ActionCodec
from starVLA.model.modules.mot.dinov3_tokens_mot2 import DinoV3TokenEncoder2
from starVLA.model.modules.mot.flow_matching import (
    FlowMatchSample,
    sample_beta_timesteps,
    sample_logit_normal_timesteps,
    sample_timesteps,
)
from starVLA.model.modules.mot.foresight_forcing import (
    ForesightTimestepSchedule,
    sample_foresight_timesteps,
    sample_uniform_teacher_action_timesteps,
    scale_teacher_action_timesteps,
)
from starVLA.model.modules.mot.layers import MoTOutput
from starVLA.model.modules.mot.modeling_mot3 import MoT3Model
from starVLA.model.modules.mot.schema import (
    masked_action_flow_sample,
    masked_cosine_representation_loss,
    masked_mse,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


class _MoT2Condition(tuple):
    """Eight VLM feature tensors plus raw metadata/schema for model-owned encoding."""

    def __new__(
        cls,
        layers: tuple[torch.Tensor, ...],
        metadata: dict[str, Any],
        action_schema_mask: torch.Tensor,
    ) -> "_MoT2Condition":
        instance = super().__new__(cls, layers)
        instance.metadata = metadata
        instance.action_schema_mask = action_schema_mask
        return instance


@FRAMEWORK_REGISTRY.register("Qwen35MoT3")
class Qwen35MoT3(MoTFrameworkBase):
    """Qwen3.5 + MoT3 with schema-masked flow matching and Foresight Forcing."""

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        if config is None:
            raise ValueError("Qwen35MoT3 requires a hierarchical model configuration")
        encoding = str(_cfg_get(config.framework.action_model, "encoding", "raw"))
        if encoding != "raw":
            raise ValueError(
                "Qwen35MoT3 requires action_model.encoding='raw': an elementwise canonical "
                "action schema requires raw canonical action coordinates"
            )
        super().__init__(config=config, **kwargs)
        if self.action_codec.encoding != "raw":
            raise ValueError(
                "Qwen35MoT3 requires action_model.encoding='raw': an elementwise canonical "
                "action schema requires raw canonical action coordinates"
            )
        validate_hidden_layers = getattr(self.qwen_vl_interface, "validate_hidden_layers", None)
        if callable(validate_hidden_layers):
            validate_hidden_layers(
                self.vlm_feature_layers,
                required_decoder_layer_type="full_attention",
            )

        cfg_training = _cfg_get(self.flow_config, "classifier_free_guidance", {}) or {}
        self.cfg_context_dropout_enabled = _as_bool(_cfg_get(cfg_training, "enabled", False))
        self.cfg_context_dropout_prob = float(_cfg_get(cfg_training, "context_dropout_prob", 0.0))
        if not math.isfinite(self.cfg_context_dropout_prob) or not 0.0 <= self.cfg_context_dropout_prob <= 1.0:
            raise ValueError(
                "flow.classifier_free_guidance.context_dropout_prob must lie in [0, 1], "
                f"got {self.cfg_context_dropout_prob}"
            )

        teacher_ratio = _cfg_get(self.foresight_config, "teacher_timestep_ratio", None)
        if teacher_ratio is None:
            self.foresight_teacher_ratio_distribution = "fixed"
            self.foresight_teacher_ratio_min = self.foresight_alpha_teacher
            self.foresight_teacher_ratio_max = self.foresight_alpha_teacher
        else:
            self.foresight_teacher_ratio_distribution = (
                str(_cfg_get(teacher_ratio, "distribution", "uniform")).lower().strip()
            )
            self.foresight_teacher_ratio_min = float(_cfg_get(teacher_ratio, "min", 0.0))
            self.foresight_teacher_ratio_max = float(_cfg_get(teacher_ratio, "max", 1.0))
        if self.foresight_teacher_ratio_distribution not in {"fixed", "uniform"}:
            raise ValueError(
                "flow.foresight_forcing.teacher_timestep_ratio.distribution must be "
                f"'fixed' or 'uniform', got {self.foresight_teacher_ratio_distribution!r}"
            )
        if (
            not math.isfinite(self.foresight_teacher_ratio_min)
            or not math.isfinite(self.foresight_teacher_ratio_max)
            or not 0.0 <= self.foresight_teacher_ratio_min <= self.foresight_teacher_ratio_max <= 1.0
        ):
            raise ValueError(
                "Foresight teacher timestep ratio bounds must satisfy 0 <= min <= max <= 1, "
                f"got min={self.foresight_teacher_ratio_min}, max={self.foresight_teacher_ratio_max}"
            )

    @staticmethod
    def _build_vlm_model(config: Any) -> torch.nn.Module:
        return get_vlm_model(config)

    @staticmethod
    def _build_action_codec(action_config: Any) -> ActionCodec:
        return ActionCodec(action_config)

    @staticmethod
    def _build_dino_encoder(dino_config: Any) -> DinoV3TokenEncoder2:
        return DinoV3TokenEncoder2(
            dino_model_variant=str(_cfg_get(dino_config, "model_variant", "dinov3_vitb16")),
            dino_model_path=str(_cfg_get(dino_config, "model_path", "./third_party/dinov3")),
            dino_weights_path=_cfg_get(dino_config, "weights_path", None),
            dino_dim=int(_cfg_get(dino_config, "feature_dim", 768)),
            patch_size=int(_cfg_get(dino_config, "patch_size", 16)),
            view_index=int(_cfg_get(dino_config, "view_index", 0)),
            freeze=_as_bool(_cfg_get(dino_config, "freeze", True)),
        )

    @staticmethod
    def _build_mot_model(mot_config: Any) -> MoT3Model:
        return MoT3Model(mot_config)

    @staticmethod
    def _parse_vlm_feature_layers(value: Any, *, depth: int) -> tuple[int, ...]:
        if depth != 16:
            raise ValueError(f"Qwen35MoT3 requires mot.depth=16, got {depth}")
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, int):
            raise ValueError(
                "Qwen35MoT3 requires exactly eight increasing mot.vlm_feature_layers, "
                "for example [4,8,12,16,20,24,28,32]"
            )
        layers = tuple(int(layer) for layer in value)
        if len(layers) != 8:
            raise ValueError(f"Qwen35MoT3 requires exactly 8 VLM feature layers, got {len(layers)}")
        if any(layer < 0 for layer in layers):
            raise ValueError(f"Qwen35MoT3 VLM feature layers must be non-negative, got {layers}")
        if any(left >= right for left, right in pairwise(layers)):
            raise ValueError(f"Qwen35MoT3 VLM feature layers must be strictly increasing, got {layers}")
        return layers

    def _sample_training_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        modality: str | None = None,
    ) -> torch.Tensor:
        if modality not in {None, "vision", "action"}:
            raise ValueError(f"modality must be None, 'vision', or 'action', got {modality!r}")

        def flow_value(name: str, default: Any) -> Any:
            if modality is not None:
                modality_value = _cfg_get(self.flow_config, f"{modality}_{name}", None)
                if modality_value is not None:
                    return modality_value
            return _cfg_get(self.flow_config, name, default)

        distribution = str(flow_value("timestep_distribution", "beta")).lower().strip().replace("-", "_")
        if distribution == "uniform":
            return sample_timesteps(batch_size, device=device, dtype=dtype)
        if distribution == "beta":
            return sample_beta_timesteps(
                batch_size,
                alpha=float(flow_value("beta_alpha", 1.5)),
                beta=float(flow_value("beta_beta", 1.0)),
                device=device,
                dtype=dtype,
            )
        if distribution in {"logit_normal", "logitnormal"}:
            return sample_logit_normal_timesteps(
                batch_size,
                mean=float(flow_value("logit_normal_mean", 0.0)),
                std=float(flow_value("logit_normal_std", 1.0)),
                device=device,
                dtype=dtype,
            )
        raise ValueError(
            "flow timestep distribution must be 'uniform', 'beta', or 'logit_normal', "
            f"got {distribution!r} for modality={modality!r}"
        )

    def _uses_modality_timestep_distributions(self) -> bool:
        fields = ("timestep_distribution", "beta_alpha", "beta_beta", "logit_normal_mean", "logit_normal_std")
        return any(
            _cfg_get(self.flow_config, f"{modality}_{field}", None) is not None
            for modality in ("vision", "action")
            for field in fields
        )

    @staticmethod
    def _validated_schema(
        value: Any,
        *,
        expected_shape: tuple[int, int],
        batch_index: int,
    ) -> torch.Tensor:
        try:
            schema = torch.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Qwen35MoT3 action schema at batch index {batch_index} is not tensor-like") from error
        if schema.shape != expected_shape:
            raise ValueError(
                "Qwen35MoT3 action schema must have exact shape "
                f"[{expected_shape[0]},{expected_shape[1]}], got {tuple(schema.shape)} "
                f"at batch index {batch_index}"
            )
        if schema.dtype != torch.bool:
            valid = torch.isfinite(schema) & ((schema == 0) | (schema == 1))
            message = (
                "Qwen35MoT3 action schema must contain only boolean or exact 0/1 values, "
                f"at batch index {batch_index}"
            )
            if schema.device.type == "cpu":
                if not bool(valid.all()):
                    raise ValueError(message)
            elif hasattr(torch, "_assert_async"):
                torch._assert_async(valid.all(), message)
            else:
                raise RuntimeError(
                    "Qwen35MoT3 requires torch._assert_async to validate CUDA action schemas "
                    "without a device-to-host synchronization"
                )
        return schema.bool()

    def _normalize_schema_example(
        self,
        example: dict[str, Any],
        *,
        batch_index: int,
        required: bool,
    ) -> dict[str, Any]:
        has_canonical = "action_schema_mask" in example
        has_alias = "action_mask" in example
        if not has_canonical and not has_alias:
            if required:
                raise ValueError(
                    "Qwen35MoT3 requires an explicit action_schema_mask or action_mask " f"at batch index {batch_index}"
                )
            return example

        expected_shape = (self.action_codec.chunk_size, self.action_codec.action_dim)
        value = example["action_schema_mask"] if has_canonical else example["action_mask"]
        max_training_horizon = getattr(self.action_codec, "max_training_chunk_size", self.action_codec.chunk_size)
        if required and max_training_horizon != self.action_codec.chunk_size:
            action_shape = torch.as_tensor(value).shape
            if (
                len(action_shape) != 2
                or action_shape[1] != self.action_codec.action_dim
                or not 0 < action_shape[0] <= max_training_horizon
            ):
                raise ValueError(
                    f"Invalid variable-horizon action shape {tuple(action_shape)} at batch index {batch_index}"
                )
            expected_shape = tuple(action_shape)
            if "action" in example and torch.as_tensor(example["action"]).shape != action_shape:
                raise ValueError(f"Variable-horizon action and mask shapes differ at batch index {batch_index}")
        schema = self._validated_schema(
            value,
            expected_shape=expected_shape,
            batch_index=batch_index,
        )
        if has_canonical and has_alias:
            alias = self._validated_schema(
                example["action_mask"],
                expected_shape=expected_shape,
                batch_index=batch_index,
            ).to(device=schema.device)
            equal = torch.eq(schema, alias).all()
            message = (
                "Qwen35MoT3 received conflicting action_schema_mask and action_mask " f"at batch index {batch_index}"
            )
            if equal.device.type == "cpu":
                if not bool(equal):
                    raise ValueError(message)
            elif hasattr(torch, "_assert_async"):
                torch._assert_async(equal, message)
            else:
                raise RuntimeError(
                    "Qwen35MoT3 requires torch._assert_async to compare CUDA action schemas "
                    "without a device-to-host synchronization"
                )
        normalized = dict(example)
        normalized["action_mask"] = schema
        return normalized

    def _batch_examples(
        self,
        examples: list[dict[str, Any]] | dict[str, Any],
        *,
        require_action: bool,
        duplicate_current: bool = False,
    ) -> dict[str, Any]:
        if isinstance(examples, dict):
            example_list = [examples]
        elif isinstance(examples, list):
            example_list = examples
        else:
            raise ValueError("Qwen35MoT3 received an empty or invalid example batch")

        if require_action:
            normalized_examples = [
                self._normalize_schema_example(
                    example,
                    batch_index=index,
                    required="action" in example,
                )
                for index, example in enumerate(example_list)
            ]
            batch = super()._batch_examples(
                normalized_examples,
                require_action=True,
                duplicate_current=duplicate_current,
            )
            batch["action_mask"] = batch["action_mask"].bool()
            return batch

        normalized_examples: list[dict[str, Any]] = []
        missing_schema = []
        for index, example in enumerate(example_list):
            if "action_schema_mask" not in example and "action_mask" not in example:
                missing_schema.append(index)
                normalized_examples.append(example)
                continue
            normalized_examples.append(
                self._normalize_schema_example(
                    example,
                    batch_index=index,
                    required=True,
                )
            )
        if missing_schema:
            raise ValueError(
                "Qwen35MoT3.predict_action requires an explicit action_schema_mask or action_mask "
                f"for every example; missing at batch indices {missing_schema}"
            )

        batch = super()._batch_examples(
            normalized_examples,
            require_action=False,
            duplicate_current=duplicate_current,
        )
        device = next(self.mot_model.parameters()).device
        horizon = self.action_codec.chunk_size
        if getattr(self.action_codec, "max_training_chunk_size", horizon) != horizon:
            horizon = max(horizon, *(example["action_mask"].shape[0] for example in normalized_examples))
        batch["action_mask"] = torch.stack(
            [
                self._sequence_to_tensor(
                    example["action_mask"],
                    horizon,
                    self.action_codec.action_dim,
                )
                for example in normalized_examples
            ],
            dim=0,
        ).to(device=device, dtype=torch.bool)
        return batch

    def _encode_condition(
        self,
        batch: dict[str, Any],
        *,
        current_only: bool = False,
        timing: dict[str, float] | None = None,
        precomputed_qwen_context: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        precomputed_context_mask: torch.Tensor | None = None,
    ) -> tuple[_MoT2Condition, torch.Tensor, torch.Tensor, torch.Tensor]:
        context, context_mask, current_vision, future_vision = super()._encode_condition(
            batch,
            current_only=current_only,
            timing=timing,
            precomputed_qwen_context=precomputed_qwen_context,
            precomputed_context_mask=precomputed_context_mask,
        )
        if isinstance(context, torch.Tensor):
            raise RuntimeError("Qwen35MoT3 expected eight layerwise Qwen hidden states")
        context = tuple(context)
        context_mask = context_mask.to(device=context[0].device, dtype=torch.bool)
        if self.training and self.cfg_context_dropout_enabled and self.cfg_context_dropout_prob > 0.0:
            drop_context = (
                torch.rand(
                    (context_mask.shape[0], 1),
                    device=context_mask.device,
                    dtype=torch.float32,
                )
                < self.cfg_context_dropout_prob
            )
            context_mask = context_mask & ~drop_context
        schema_mask = batch.get("action_mask")
        if schema_mask is None:
            raise ValueError("Qwen35MoT3 requires an action schema mask for metadata conditioning")
        return (
            _MoT2Condition(context, batch.get("meta", {}), schema_mask),
            context_mask,
            current_vision,
            future_vision,
        )

    def _encode_vlm_context(
        self,
        qwen_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        context = self.qwen_vl_interface.encode_context(
            hidden_layers=self.vlm_feature_layers,
            selected_hidden_states_only=True,
            required_decoder_layer_type="full_attention",
            **qwen_inputs,
        )
        if isinstance(context, torch.Tensor):
            raise RuntimeError("Qwen35MoT3 expected eight selected VLM hidden states")
        return tuple(context)

    def _joint_context_decoder_layer_type(self) -> str:
        return "full_attention"

    @staticmethod
    def _condition_kwargs(
        context: tuple[torch.Tensor, ...],
        *,
        detach: bool = False,
    ) -> dict[str, Any]:
        metadata = getattr(context, "metadata", None)
        action_schema_mask = getattr(context, "action_schema_mask", None)
        if metadata is None or action_schema_mask is None:
            raise RuntimeError("MoT2 condition is missing raw metadata/action schema")
        if detach:
            action_schema_mask = action_schema_mask.detach()
        return {
            "metadata": metadata,
            "action_schema_mask": action_schema_mask,
        }

    @staticmethod
    def _vision_flow_sample(
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> FlowMatchSample:
        mask = torch.ones(clean.shape[:2], device=clean.device, dtype=torch.bool)
        return masked_action_flow_sample(clean, timesteps, mask, noise=noise)

    def _ordinary_flow_forward(
        self,
        context: tuple[torch.Tensor, ...],
        context_mask: torch.Tensor,
        current_vision: torch.Tensor,
        robot_state: torch.Tensor | None,
        robot_state_mask: torch.Tensor | None,
        clean_vision: torch.Tensor,
        clean_action: torch.Tensor,
        action_token_mask: torch.Tensor,
        action_loss_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if action_loss_mask is None:
            raise ValueError("Qwen35MoT3 training requires an explicit action schema mask")
        if self._uses_modality_timestep_distributions():
            vision_timesteps = self._sample_training_timesteps(
                clean_action.shape[0],
                device=clean_action.device,
                dtype=torch.float32,
                modality="vision",
            )
            action_timesteps = self._sample_training_timesteps(
                clean_action.shape[0],
                device=clean_action.device,
                dtype=torch.float32,
                modality="action",
            )
        else:
            shared_timesteps = self._sample_training_timesteps(
                clean_action.shape[0],
                device=clean_action.device,
                dtype=torch.float32,
            )
            vision_timesteps = shared_timesteps
            action_timesteps = shared_timesteps
        if self.predict_future_vision:
            vision_sample = self._vision_flow_sample(clean_vision, vision_timesteps)
            future_vision = vision_sample.noisy
        else:
            vision_sample = None
            future_vision = clean_vision
        action_sample = masked_action_flow_sample(clean_action, action_timesteps, action_loss_mask)
        output = self.mot_model(
            context=context,
            context_mask=context_mask,
            current_vision=current_vision,
            future_vision=future_vision,
            action=action_sample.noisy,
            vision_timesteps=vision_timesteps,
            action_timesteps=action_sample.timesteps,
            action_token_mask=action_token_mask,
            robot_state=robot_state,
            robot_state_mask=robot_state_mask,
            **self._condition_kwargs(context),
        )
        action_loss = self._action_flow_loss(output.action_velocity, action_sample.velocity, action_loss_mask)
        total_loss = float(_cfg_get(self.flow_config, "action_loss_weight", 1.0)) * action_loss
        losses = {"loss": total_loss, "action_loss": action_loss}
        if vision_sample is not None:
            vision_loss = masked_mse(output.vision_velocity, vision_sample.velocity)
            losses["vision_flow_loss"] = vision_loss
            losses["loss"] = total_loss + float(_cfg_get(self.flow_config, "vision_loss_weight", 0.1)) * vision_loss
        return losses

    def _sample_foresight_schedule(
        self,
        reliability: torch.Tensor,
    ) -> ForesightTimestepSchedule:
        return sample_foresight_timesteps(
            reliability,
            alpha_min=float(_cfg_get(self.foresight_config, "alpha_min", 0.5)),
            alpha_max=float(_cfg_get(self.foresight_config, "alpha_max", 3.0)),
            vision_timestep_power=_cfg_get(self.foresight_config, "vision_timestep_power", None),
            teacher_edge_prob=float(_cfg_get(self.foresight_config, "teacher_edge_prob", 0.02)),
        )

    def _foresight_forward(
        self,
        context: tuple[torch.Tensor, ...],
        context_mask: torch.Tensor,
        current_vision: torch.Tensor,
        robot_state: torch.Tensor | None,
        robot_state_mask: torch.Tensor | None,
        clean_vision: torch.Tensor,
        clean_action: torch.Tensor,
        action_token_mask: torch.Tensor,
        action_loss_mask: torch.Tensor,
        action_valid: torch.Tensor,
        metadata: dict[str, list[Any]],
    ) -> dict[str, torch.Tensor]:
        dataset_ids, reliability = self._foresight_reliabilities(
            metadata,
            batch_size=clean_action.shape[0],
            device=clean_action.device,
        )
        schedule = self._sample_foresight_schedule(reliability)
        if self.foresight_teacher_ratio_distribution == "uniform":
            teacher_action_timesteps, teacher_timestep_ratio = sample_uniform_teacher_action_timesteps(
                schedule.action_timesteps,
                ratio_min=self.foresight_teacher_ratio_min,
                ratio_max=self.foresight_teacher_ratio_max,
            )
        else:
            teacher_action_timesteps = scale_teacher_action_timesteps(
                schedule.action_timesteps,
                self.foresight_alpha_teacher,
            )
            teacher_timestep_ratio = torch.full_like(
                schedule.action_timesteps,
                self.foresight_alpha_teacher,
                dtype=torch.float32,
            )
        if self.predict_future_vision:
            vision_sample = self._vision_flow_sample(clean_vision, schedule.vision_timesteps)
            noisy_vision = vision_sample.noisy
        else:
            vision_sample = None
            noisy_vision = clean_vision
        action_sample = masked_action_flow_sample(
            clean_action,
            schedule.action_timesteps,
            action_loss_mask,
        )
        student_layer = int(_cfg_get(self.foresight_config, "student_layer", 5))
        teacher_layer = int(_cfg_get(self.foresight_config, "teacher_layer", 11))
        representation_active = self.foresight_teacher_enabled and self._foresight_representation_weight > 0.0
        student_output = self.mot_model(
            context=context,
            context_mask=context_mask,
            current_vision=current_vision,
            future_vision=noisy_vision,
            action=action_sample.noisy,
            vision_timesteps=schedule.vision_timesteps,
            action_timesteps=schedule.action_timesteps,
            action_token_mask=action_token_mask,
            robot_state=robot_state,
            robot_state_mask=robot_state_mask,
            feature_layers=(student_layer,) if representation_active else (),
            **self._condition_kwargs(context),
        )
        raw_action_loss = self._action_flow_loss(
            student_output.action_velocity,
            action_sample.velocity,
            action_loss_mask,
        )
        weighted_action_loss = self._reliability_weighted_action_loss(
            student_output.action_velocity,
            action_sample.velocity,
            action_loss_mask,
            reliability,
        )
        if representation_active:
            teacher = self._get_foresight_teacher()
            teacher_action = (
                action_sample.noisy
                if self.foresight_teacher_ratio_distribution == "fixed" and self.foresight_alpha_teacher == 1.0
                else masked_action_flow_sample(
                    clean_action,
                    teacher_action_timesteps,
                    action_loss_mask,
                    noise=action_sample.noise,
                ).noisy
            )
            teacher_output: MoTOutput = teacher(
                context=tuple(layer_context.detach() for layer_context in context),
                context_mask=context_mask,
                current_vision=current_vision.detach(),
                future_vision=clean_vision,
                action=teacher_action,
                vision_timesteps=torch.zeros_like(schedule.vision_timesteps),
                action_timesteps=teacher_action_timesteps,
                action_token_mask=action_token_mask,
                robot_state=robot_state.detach() if robot_state is not None else None,
                robot_state_mask=robot_state_mask,
                feature_layers=(teacher_layer,),
                **self._condition_kwargs(context, detach=True),
            )
            if self.action_rep_projector is None:
                raise RuntimeError("Foresight Forcing action representation projector was not initialized")
            action_rep_loss = masked_cosine_representation_loss(
                self.action_rep_projector(student_output.action_features[student_layer]),
                teacher_output.action_features[teacher_layer],
                action_token_mask,
            )
        else:
            action_rep_loss = student_output.action_velocity.float().sum() * 0.0

        total_loss = (
            float(_cfg_get(self.flow_config, "action_loss_weight", 1.0)) * weighted_action_loss
            + self._foresight_representation_weight * action_rep_loss
        )
        losses: dict[str, torch.Tensor] = {
            "loss": total_loss,
            "action_loss": weighted_action_loss,
            "raw_action_flow_loss": raw_action_loss,
            "reliability_weighted_action_flow_loss": weighted_action_loss,
            "foresight_action_rep_loss": action_rep_loss,
            "foresight_representation_weight": student_output.action_velocity.new_tensor(
                self._foresight_representation_weight,
                dtype=torch.float32,
            ),
            "foresight/alpha_teacher": teacher_timestep_ratio.mean(),
            "foresight/teacher_timestep_ratio_min": teacher_timestep_ratio.min(),
            "foresight/teacher_timestep_ratio_max": teacher_timestep_ratio.max(),
            "foresight/teacher_t_a_mean": teacher_action_timesteps.mean(),
            "foresight/teacher_action_timestep_reduction_mean": (
                (schedule.action_timesteps - teacher_action_timesteps).mean()
            ),
            "foresight/action_valid_fraction": action_valid.float().mean(),
        }
        if vision_sample is not None:
            # Clean edge vision calibrates action conditioning, not vision flow.
            vision_loss = masked_mse(
                student_output.vision_velocity,
                vision_sample.velocity,
                (~schedule.teacher_edge_mask)[:, None],
            )
            losses["vision_flow_loss"] = vision_loss
            losses["loss"] = total_loss + float(_cfg_get(self.flow_config, "vision_loss_weight", 0.1)) * vision_loss
        losses.update(self._foresight_timestep_metrics(schedule, dataset_ids, reliability))
        return losses

    def _build_inference_action_state(
        self,
        batch: dict[str, Any],
        current_vision: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        schema = batch.get("action_mask")
        if schema is None:
            raise ValueError("Qwen35MoT3 inference requires action_schema_mask")
        flow_tokens = self.action_codec.num_flow_tokens
        if getattr(self.action_codec, "max_training_chunk_size", self.chunk_size) != self.chunk_size:
            flow_tokens = schema.shape[1]
        action_state = torch.randn(
            (
                current_vision.shape[0],
                flow_tokens,
                self.action_codec.flow_dim,
            ),
            device=current_vision.device,
            dtype=current_vision.dtype,
            generator=generator,
        )
        flow_schema = self.action_codec.flow_loss_mask(schema, action_state)
        if flow_schema is None:
            raise RuntimeError("Qwen35MoT3 could not map the action schema into flow space")
        flow_schema = flow_schema.bool()
        batch["_flow_action_schema_mask"] = flow_schema
        action_state = torch.where(flow_schema, action_state, torch.zeros_like(action_state))
        return action_state, self.action_codec.token_mask(schema, action_state)

    def _inference_mot_kwargs(
        self,
        context: torch.Tensor | tuple[torch.Tensor, ...],
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        del batch
        if isinstance(context, torch.Tensor):
            raise RuntimeError("Qwen35MoT3 inference requires eight VLM context tensors")
        metadata = getattr(context, "metadata", None)
        action_schema_mask = getattr(context, "action_schema_mask", None)
        if metadata is None or action_schema_mask is None:
            raise RuntimeError("MoT2 inference condition is missing raw metadata/action schema")
        reference = context[0]
        meta_tokens, meta_mask = self.mot_model.encode_metadata(
            metadata,
            action_schema_mask,
            batch_size=reference.shape[0],
            device=reference.device,
            dtype=reference.dtype,
        )
        return {"meta_tokens": meta_tokens, "meta_mask": meta_mask}

    @staticmethod
    def _update_inference_action_state(
        action_state: torch.Tensor,
        action_velocity: torch.Tensor,
        delta_time: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        flow_schema = batch["_flow_action_schema_mask"].bool()
        action_velocity = torch.where(
            flow_schema,
            action_velocity,
            torch.zeros_like(action_velocity),
        )
        updated = action_state + delta_time * action_velocity
        return torch.where(flow_schema, updated, torch.zeros_like(updated))

    def _decode_inference_action_state(
        self,
        action_state: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        action = self.action_codec.decode(action_state, horizon=self.chunk_size)
        schema = batch["action_mask"].to(device=action.device, dtype=torch.bool)
        return torch.where(schema, action, torch.zeros_like(action))
