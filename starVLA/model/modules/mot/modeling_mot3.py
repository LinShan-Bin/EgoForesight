# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Support explicit spatial interpolation of pretrained MoT3 vision positions.

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.mot.flow_matching import FlowMatchSample
from starVLA.model.modules.mot.layers import (
    AdaLNRMSNormZero,
    AdaptiveRMSNorm,
    MoTOutput,
    RMSNorm,
    SwiGLU,
    _as_bool,
    _cfg_get,
    _merge_heads,
    _split_heads,
)

DEFAULT_QK_NORM_EPS = 1.0e-6


def _parse_vision_grid_size(value: Any, name: str) -> tuple[int, int] | None:
    """Validate an optional spatial grid without inferring checkpoint geometry."""

    if value is None:
        return None
    try:
        dimensions = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be [height, width] with positive integer dimensions, got {value!r}") from error
    if len(dimensions) != 2 or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in dimensions
    ):
        raise ValueError(f"{name} must be [height, width] with positive integer dimensions, got {value!r}")
    return dimensions


def _assert_async(condition: torch.Tensor, message: str) -> None:
    """Assert a scalar device predicate without a CUDA-to-Python value read."""

    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
        return
    if condition.device.type != "cpu":
        raise RuntimeError("MoT3 requires torch._assert_async for non-blocking CUDA validation")
    if not bool(condition):
        raise ValueError(message)


def _validate_binary_mask(mask: torch.Tensor, name: str) -> None:
    """Validate a binary mask without reading a CUDA predicate on the host."""

    if mask.dtype == torch.bool:
        return
    _assert_async(torch.isfinite(mask).all(), f"{name} must contain only finite values")
    _assert_async(
        ((mask == 0) | (mask == 1)).all(),
        f"{name} must contain only boolean or exact 0/1 values",
    )


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)!r}")
    if mask.ndim > reference.ndim:
        raise ValueError(f"mask rank {mask.ndim} cannot broadcast to reference rank {reference.ndim}")
    shape = tuple(mask.shape) + (1,) * (reference.ndim - mask.ndim)
    reshaped = mask.reshape(shape)
    try:
        torch.broadcast_shapes(reshaped.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to reference shape {tuple(reference.shape)}"
        ) from error
    return reshaped.expand_as(reference)


def _expand_timesteps(timesteps: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    time = torch.as_tensor(timesteps, device=reference.device, dtype=reference.dtype)
    if time.ndim > reference.ndim:
        raise ValueError(f"timesteps rank {time.ndim} cannot broadcast to reference rank {reference.ndim}")
    if time.ndim > 0 and time.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"timestep batch {time.shape[0]} does not match reference batch {reference.shape[0]}")
    shape = tuple(time.shape) + (1,) * (reference.ndim - time.ndim)
    time = time.reshape(shape)
    try:
        torch.broadcast_shapes(time.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"timesteps shape {tuple(time.shape)} cannot broadcast to reference shape {tuple(reference.shape)}"
        ) from error
    _assert_async(torch.isfinite(time).all(), "timesteps must contain only finite values")
    _assert_async((time >= 0.0).all(), "timesteps must lie in [0, 1]")
    _assert_async((time <= 1.0).all(), "timesteps must lie in [0, 1]")
    return time


def masked_action_flow_sample(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    action_schema_mask: torch.Tensor,
    noise: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchSample:
    """Build an RF sample while keeping every inactive action element exactly zero."""

    if not isinstance(clean, torch.Tensor) or not clean.is_floating_point() or clean.ndim == 0:
        raise TypeError("clean must be a floating-point tensor with a batch dimension")
    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    if noise.shape != clean.shape or noise.device != clean.device:
        raise ValueError(
            f"noise must match clean shape/device, got {tuple(noise.shape)} on {noise.device} "
            f"for {tuple(clean.shape)} on {clean.device}"
        )

    action_schema_mask = action_schema_mask.to(device=clean.device)
    _validate_binary_mask(action_schema_mask, "action_schema_mask")
    active = _broadcast_mask(action_schema_mask.bool(), clean)
    clean = torch.where(active, clean, torch.zeros_like(clean))
    noise = torch.where(active, noise, torch.zeros_like(noise))
    time = _expand_timesteps(timesteps, clean)
    noisy = clean * (1.0 - time) + noise * time
    return FlowMatchSample(
        noisy=noisy,
        velocity=noise - clean,
        timesteps=timesteps,
        noise=noise,
    )


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a masked mean without converting a device predicate to Python bool."""

    if mask is None:
        return values.mean()
    weights = _broadcast_mask(mask.to(device=values.device, dtype=values.dtype), values)
    _assert_async((weights.detach() >= 0.0).all(), "mask weights must be non-negative")
    weighted_values = torch.where(
        weights > 0.0,
        values * weights,
        torch.zeros_like(values),
    )
    return weighted_values.sum() / weights.sum().clamp_min(1.0)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable masked MSE without a GPU-to-Python mask check."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}"
        )
    return masked_mean((prediction.float() - target.float()).square(), mask)


def masked_cosine_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student and teacher features must match, got {tuple(student_features.shape)} and "
            f"{tuple(teacher_features.shape)}"
        )
    similarity = F.cosine_similarity(
        student_features.float(),
        teacher_features.detach().clone().float(),
        dim=-1,
        eps=eps,
    )
    return masked_mean(1.0 - similarity, mask)


class TimestepEmbedder2(nn.Module):
    """Sinusoidal timestep embedding using LDA's 0..1000 input scale."""

    def __init__(
        self,
        hidden_size: int,
        frequency_size: int = 256,
        timestep_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(timestep_scale) or timestep_scale <= 0.0:
            raise ValueError(f"timestep_scale must be finite and positive, got {timestep_scale}")
        self.frequency_size = int(frequency_size)
        self.timestep_scale = float(timestep_scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        original_shape = timesteps.shape
        flat_timesteps = timesteps.reshape(-1).float() * self.timestep_scale
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=flat_timesteps.device) / max(half, 1)
        )
        angles = flat_timesteps[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_size % 2:
            embedding = F.pad(embedding, (0, 1))
        output = self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))
        return output.reshape(*original_shape, output.shape[-1])


# Public name mirrors the original modeling module while retaining a distinct class implementation.
TimestepEmbedder = TimestepEmbedder2


class PostAttentionGamma(nn.Module):
    """Generate an identity-initialized post-attention scale from a token condition."""

    def __init__(self, hidden_size: int, rank: int | None = None) -> None:
        super().__init__()
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive when provided, got {rank}")
        self.rank = rank
        if rank is None:
            self.modulation: nn.Module = nn.Linear(hidden_size, hidden_size)
        else:
            self.modulation = nn.Sequential(
                nn.Linear(hidden_size, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, hidden_size),
            )
        self.reset_parameters()

    @property
    def output_projection(self) -> nn.Linear:
        if isinstance(self.modulation, nn.Linear):
            return self.modulation
        output_projection = self.modulation[-1]
        if not isinstance(output_projection, nn.Linear):
            raise TypeError("Post-attention gamma modulation must end with nn.Linear")
        return output_projection

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.output_projection.weight)
        nn.init.ones_(self.output_projection.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if self.rank is None:
            return self.modulation(F.silu(condition))
        return self.modulation(condition)


class RobotStateEncoder2(nn.Module):
    """Encode canonical robot state with asynchronous finite-value validation."""

    def __init__(self, state_dim: int, action_dim: int, encoder_hidden_size: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if min(self.state_dim, self.action_dim, int(encoder_hidden_size)) <= 0:
            raise ValueError("state_dim, action_dim, and encoder_hidden_size must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim * 2, int(encoder_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(encoder_hidden_size), self.action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[1:] != (1, self.state_dim):
            raise ValueError(f"robot_state must be [B,1,{self.state_dim}], got {tuple(state.shape)}")
        if state_mask.shape != state.shape:
            raise ValueError(
                f"robot_state_mask shape {tuple(state_mask.shape)} does not match state {tuple(state.shape)}"
            )
        _assert_async(torch.isfinite(state).all(), "robot_state contains non-finite values")
        state_mask = state_mask.bool()
        masked_state = state * state_mask.to(state.dtype)
        encoder_input = torch.cat((masked_state, state_mask.to(state.dtype)), dim=-1)
        token = self.encoder(encoder_input.to(dtype=self.encoder[0].weight.dtype))
        return token, state_mask.any(dim=-1)


class MetadataEncoder(nn.Module):
    """Hash metadata and encode the canonical action schema into condition tokens."""

    def __init__(
        self,
        config: Any,
        *,
        action_dim: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        fields = _cfg_get(config, "fields", ())
        if isinstance(fields, str):
            fields = [field.strip() for field in fields.split(",") if field.strip()]
        self.fields = tuple(str(field) for field in fields)
        self.num_buckets = int(_cfg_get(config, "num_buckets", 4096))
        self.action_dim = int(action_dim)
        embedding_dim = int(_cfg_get(config, "embedding_dim", min(256, model_dim)))
        if self.num_buckets < 2:
            raise ValueError(f"metadata.num_buckets must be at least 2, got {self.num_buckets}")
        if embedding_dim <= 0:
            raise ValueError(f"metadata.embedding_dim must be positive, got {embedding_dim}")

        self.value_embedding = nn.Embedding(self.num_buckets, embedding_dim, padding_idx=0)
        self.field_embedding = nn.Embedding(max(len(self.fields), 1), embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, model_dim, bias=False)
        self.schema_encoder = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, model_dim),
        )
        self.metadata_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.schema_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.metadata_type, std=0.02)
        nn.init.normal_(self.schema_type, std=0.02)
        # Cache only immutable IDs, never the outputs of trainable embeddings.
        self._metadata_id_cache: OrderedDict = OrderedDict()

    def _apply(self, fn, recurse: bool = True):
        self._metadata_id_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _bucket(self, field: str, value: Any) -> int:
        if value is None:
            return 0
        value = getattr(value, "value", value)
        payload = f"{field}\0{value}".encode("utf-8", errors="replace")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % (self.num_buckets - 1) + 1

    def _metadata_ids(
        self,
        metadata: Mapping[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fields:
            empty = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            return empty, empty.bool()
        ids: list[list[int]] = []
        for sample_index in range(batch_size):
            row = []
            for field in self.fields:
                values = metadata.get(field)
                if isinstance(values, (list, tuple)):
                    value = values[sample_index] if sample_index < len(values) else None
                elif torch.is_tensor(values):
                    if values.device.type != "cpu":
                        raise ValueError(
                            f"metadata field {field!r} must contain host values, not a {values.device} tensor"
                        )
                    if values.ndim == 0:
                        value = values.item()
                    elif sample_index < values.shape[0]:
                        sample_value = values[sample_index]
                        if sample_value.numel() != 1:
                            raise ValueError(
                                f"metadata field {field!r} tensor entries must be scalar, got "
                                f"shape {tuple(sample_value.shape)}"
                            )
                        value = sample_value.item()
                    else:
                        value = None
                else:
                    value = values
                row.append(self._bucket(field, value))
            ids.append(row)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (device, tuple(tuple(row) for row in ids))
        cached = self._metadata_id_cache.get(key)
        if cached is not None:
            self._metadata_id_cache.move_to_end(key)
            return cached
        # Evaluation can populate this cache before training reuses the same IDs.
        # Autograd must be able to save them for the embedding's backward pass.
        with torch.inference_mode(False):
            token_ids = torch.tensor(ids, dtype=torch.long, pin_memory=device.type == "cuda")
            token_ids = token_ids.to(device=device, non_blocking=True)
            cached = (token_ids, token_ids.ne(0))
        self._metadata_id_cache[key] = cached
        if len(self._metadata_id_cache) > 32:
            self._metadata_id_cache.popitem(last=False)
        return cached

    def forward(
        self,
        metadata: Mapping[str, Any],
        action_schema_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, metadata_mask = self._metadata_ids(metadata, batch_size, device)
        if token_ids.shape[1]:
            field_ids = torch.arange(token_ids.shape[1], device=device)
            values = self.value_embedding(token_ids) + self.field_embedding(field_ids)[None]
            metadata_tokens = self.value_projection(values)
            metadata_tokens = metadata_tokens + self.metadata_type.to(metadata_tokens)
        else:
            metadata_tokens = self.metadata_type.new_empty(batch_size, 0, self.metadata_type.shape[-1])

        if (
            action_schema_mask.ndim != 3
            or action_schema_mask.shape[0] != batch_size
            or action_schema_mask.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "action_schema_mask must be [B,T,action_dim], got "
                f"{tuple(action_schema_mask.shape)} for B={batch_size}, "
                f"action_dim={self.action_dim}"
            )
        action_schema_mask = action_schema_mask.to(device=device)
        _validate_binary_mask(action_schema_mask, "action_schema_mask")
        schema = action_schema_mask.bool().any(dim=1).to(dtype=self.schema_encoder[0].weight.dtype)
        schema_token = self.schema_encoder(schema)[:, None]
        schema_token = schema_token + self.schema_type.to(schema_token)
        schema_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        tokens = torch.cat((metadata_tokens, schema_token), dim=1).to(dtype=dtype)
        return tokens, torch.cat((metadata_mask, schema_mask), dim=1)


class MoT3Block(nn.Module):
    """MM-DiT block with AdaLN joint/FFN branches and LDA-style LayerNorm cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        vision_intermediate_size: int,
        action_intermediate_size: int,
        timestep_rank: int,
        qk_norm_eps: float = DEFAULT_QK_NORM_EPS,
        adaln_rank: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        if not math.isfinite(qk_norm_eps) or qk_norm_eps <= 0.0:
            raise ValueError(f"qk_norm_eps must be finite and positive, got {qk_norm_eps}")
        if adaln_rank is not None and adaln_rank <= 0:
            raise ValueError(f"adaln_rank must be positive when provided, got {adaln_rank}")
        # Retain the constructor argument for configuration compatibility. New
        # MoT3 modulation uses the explicit optional adaln_rank instead.
        del timestep_rank
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.qk_norm_eps = float(qk_norm_eps)

        self.vision_joint_norm = AdaLNRMSNormZero(hidden_size, rank=adaln_rank)
        self.action_joint_norm = AdaLNRMSNormZero(hidden_size, rank=adaln_rank)
        self.vision_cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.action_cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.vision_post_attn_gamma = PostAttentionGamma(hidden_size, rank=adaln_rank)
        self.action_post_attn_gamma = PostAttentionGamma(hidden_size, rank=adaln_rank)
        self.vision_ffn_norm = AdaLNRMSNormZero(hidden_size, rank=adaln_rank)
        self.action_ffn_norm = AdaLNRMSNormZero(hidden_size, rank=adaln_rank)
        self.condition_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Match SD3 MM-DiT QK normalization: each Q/K projection source has an
        # independent learnable RMS scale over the per-head channel dimension.
        self.vision_joint_q_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.vision_joint_k_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.action_joint_q_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.action_joint_k_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.vision_cross_q_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.vision_cross_k_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.action_cross_q_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)
        self.action_cross_k_norm = RMSNorm(self.head_dim, eps=self.qk_norm_eps)

        self.vision_joint_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.action_joint_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.vision_joint_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.action_joint_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.vision_cross_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vision_cross_kv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.vision_cross_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.action_cross_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.action_cross_kv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.action_cross_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.vision_ffn = SwiGLU(hidden_size, vision_intermediate_size)
        self.action_ffn = SwiGLU(hidden_size, action_intermediate_size)

    @staticmethod
    def _key_only_mask(
        key_mask: torch.Tensor | None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        del dtype
        if key_mask is None:
            return None
        if key_mask.ndim != 2:
            raise ValueError(f"key_mask must be [B,K], got {tuple(key_mask.shape)}")
        return key_mask[:, None, None, :].bool()

    def _attend_heads(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        value_heads: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=self._key_only_mask(key_mask, query_heads.dtype),
            dropout_p=0.0,
        )
        return _merge_heads(attended)

    def prepare_condition_kv(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project read-only conditioning for this block using the original kernels."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MoT3 conditioning cache is inference-only; use eval and inference_mode")
        normalized = self.condition_norm(condition)
        vision_k, vision_v = self.vision_cross_kv(normalized).chunk(2, dim=-1)
        action_k, action_v = self.action_cross_kv(normalized).chunk(2, dim=-1)
        return (
            self.vision_cross_k_norm(_split_heads(vision_k, self.num_heads)),
            _split_heads(vision_v, self.num_heads),
            self.action_cross_k_norm(_split_heads(action_k, self.num_heads)),
            _split_heads(action_v, self.num_heads),
        )

    def forward(
        self,
        condition: torch.Tensor,
        vision: torch.Tensor,
        action: torch.Tensor,
        vision_time: torch.Tensor,
        action_time: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        condition_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition_kv is not None and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("MoT3 conditioning cache is inference-only; use eval and inference_mode")
        batch_size = vision.shape[0]
        if condition_mask is None:
            condition_mask = torch.ones(condition.shape[:2], device=condition.device, dtype=torch.bool)
        if action_mask is None:
            action_mask = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)

        vision_joint_input, vision_joint_gate = self.vision_joint_norm(vision, vision_time)
        action_joint_input, action_joint_gate = self.action_joint_norm(action, action_time)
        vision_q, vision_k, vision_v = self.vision_joint_qkv(vision_joint_input).chunk(3, dim=-1)
        action_q, action_k, action_v = self.action_joint_qkv(action_joint_input).chunk(3, dim=-1)
        joint_query = torch.cat(
            (
                self.vision_joint_q_norm(_split_heads(vision_q, self.num_heads)),
                self.action_joint_q_norm(_split_heads(action_q, self.num_heads)),
            ),
            dim=2,
        )
        joint_key = torch.cat(
            (
                self.vision_joint_k_norm(_split_heads(vision_k, self.num_heads)),
                self.action_joint_k_norm(_split_heads(action_k, self.num_heads)),
            ),
            dim=2,
        )
        joint_value = torch.cat(
            (_split_heads(vision_v, self.num_heads), _split_heads(action_v, self.num_heads)),
            dim=2,
        )
        vision_mask = torch.ones(batch_size, vision.shape[1], device=vision.device, dtype=torch.bool)
        joint_mask = torch.cat((vision_mask, action_mask.bool()), dim=1)
        joint_output = self._attend_heads(joint_query, joint_key, joint_value, joint_mask)
        vision_joint, action_joint = joint_output.split((vision.shape[1], action.shape[1]), dim=1)
        vision = vision + vision_joint_gate * self.vision_joint_out(vision_joint)
        action = action + action_joint_gate * self.action_joint_out(action_joint)

        if condition_kv is None:
            normalized_condition = self.condition_norm(condition)
            vision_condition_k, vision_condition_v = self.vision_cross_kv(normalized_condition).chunk(2, dim=-1)
        vision_cross_input = self.vision_cross_norm(vision)
        vision_cross = self._attend_heads(
            self.vision_cross_q_norm(_split_heads(self.vision_cross_q(vision_cross_input), self.num_heads)),
            (
                self.vision_cross_k_norm(_split_heads(vision_condition_k, self.num_heads))
                if condition_kv is None
                else condition_kv[0]
            ),
            _split_heads(vision_condition_v, self.num_heads) if condition_kv is None else condition_kv[1],
            condition_mask,
        )
        vision = vision + self.vision_cross_out(vision_cross)

        if condition_kv is None:
            action_condition_k, action_condition_v = self.action_cross_kv(normalized_condition).chunk(2, dim=-1)
        action_cross_input = self.action_cross_norm(action)
        action_cross = self._attend_heads(
            self.action_cross_q_norm(_split_heads(self.action_cross_q(action_cross_input), self.num_heads)),
            (
                self.action_cross_k_norm(_split_heads(action_condition_k, self.num_heads))
                if condition_kv is None
                else condition_kv[2]
            ),
            _split_heads(action_condition_v, self.num_heads) if condition_kv is None else condition_kv[3],
            condition_mask,
        )
        action = action + self.action_cross_out(action_cross)

        # Match LDA's placement: scale the complete residual stream after both
        # joint and cross attention, before entering the private FFN branch.
        vision = vision * self.vision_post_attn_gamma(vision_time)
        action = action * self.action_post_attn_gamma(action_time)

        vision_ffn_input, vision_ffn_gate = self.vision_ffn_norm(vision, vision_time)
        action_ffn_input, action_ffn_gate = self.action_ffn_norm(action, action_time)
        vision = vision + vision_ffn_gate * self.vision_ffn(vision_ffn_input)
        action = action + action_ffn_gate * self.action_ffn(action_ffn_input)
        return vision, action


class MoT3Model(nn.Module):
    """MoT3 backbone with read-only current-vision context and future/action generation streams."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(_cfg_get(config, "hidden_size", 1536))
        num_heads = int(_cfg_get(config, "num_heads", 24))
        depth = int(_cfg_get(config, "depth", 16))
        vision_dim = int(_cfg_get(config, "vision_dim", 768))
        action_dim = int(_cfg_get(config, "action_dim", 130))
        context_dim = int(_cfg_get(config, "context_dim", 2560))
        vision_intermediate_size = int(_cfg_get(config, "vision_intermediate_size", 6144))
        action_intermediate_size = int(_cfg_get(config, "action_intermediate_size", 2048))
        timestep_rank = int(_cfg_get(config, "timestep_rank", 128))
        configured_adaln_rank = _cfg_get(config, "adaln_rank", None)
        adaln_rank = None if configured_adaln_rank is None else int(configured_adaln_rank)
        if adaln_rank is not None and adaln_rank <= 0:
            raise ValueError(f"adaln_rank must be positive when provided, got {adaln_rank}")
        qk_norm_eps = float(_cfg_get(config, "qk_norm_eps", DEFAULT_QK_NORM_EPS))
        if not math.isfinite(qk_norm_eps) or qk_norm_eps <= 0.0:
            raise ValueError(f"qk_norm_eps must be finite and positive, got {qk_norm_eps}")
        max_vision_tokens = int(_cfg_get(config, "max_vision_tokens", 480))
        max_action_tokens = int(_cfg_get(config, "max_action_tokens", 64))
        self.pretrained_vision_grid_size = _parse_vision_grid_size(
            _cfg_get(config, "pretrained_vision_grid_size", None), "pretrained_vision_grid_size"
        )
        self.vision_grid_size = _parse_vision_grid_size(_cfg_get(config, "vision_grid_size", None), "vision_grid_size")
        if (self.pretrained_vision_grid_size is None) != (self.vision_grid_size is None):
            raise ValueError("pretrained_vision_grid_size and vision_grid_size must be configured together")
        if self.vision_grid_size is not None and math.prod(self.vision_grid_size) != max_vision_tokens:
            raise ValueError(
                f"vision_grid_size={self.vision_grid_size} must contain max_vision_tokens={max_vision_tokens} positions"
            )
        gradient_checkpointing = _as_bool(_cfg_get(config, "gradient_checkpointing", True))
        self.num_vlm_features = int(_cfg_get(config, "num_vlm_features", 8))
        self.layers_per_vlm_feature = int(_cfg_get(config, "layers_per_vlm_feature", 2))
        if self.num_vlm_features != 8 or self.layers_per_vlm_feature != 2 or depth != 16:
            raise ValueError(
                "MoT3 requires num_vlm_features=8, layers_per_vlm_feature=2, and depth=16; "
                f"got {self.num_vlm_features}, {self.layers_per_vlm_feature}, and {depth}"
            )

        robot_state_config = _cfg_get(config, "robot_state", {})
        robot_state_enabled = _as_bool(_cfg_get(robot_state_config, "enabled", False))
        self.hidden_size = hidden_size
        self.depth = depth
        self.vision_dim = vision_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.max_vision_tokens = max_vision_tokens
        self.max_action_tokens = max_action_tokens
        pretrained_action_tokens = _cfg_get(config, "pretrained_action_tokens", None)
        self.pretrained_action_tokens = None if pretrained_action_tokens is None else int(pretrained_action_tokens)
        if self.pretrained_action_tokens is not None and self.pretrained_action_tokens < 2:
            raise ValueError("pretrained_action_tokens must be at least 2 for temporal interpolation")
        self.gradient_checkpointing = gradient_checkpointing
        self.robot_state_enabled = robot_state_enabled
        self.qk_norm_eps = qk_norm_eps
        self.adaln_rank = adaln_rank
        self.compact_vision_condition = _as_bool(_cfg_get(config, "compact_vision_condition", False))

        self.context_projection = nn.Linear(context_dim, hidden_size, bias=False)
        self.current_context_projection = nn.Linear(vision_dim, hidden_size, bias=False)
        self.vision_input = nn.Linear(vision_dim, hidden_size, bias=False)
        self.action_input = nn.Linear(action_dim, hidden_size, bias=False)
        self.vision_output = nn.Linear(hidden_size, vision_dim, bias=False)
        self.action_output = nn.Linear(hidden_size, action_dim, bias=False)

        self.context_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.future_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.state_position = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.vision_position = nn.Parameter(torch.zeros(1, max_vision_tokens, hidden_size))
        self.action_position = nn.Parameter(torch.zeros(1, max_action_tokens, hidden_size))
        self.clean_condition_embedding = nn.Parameter(torch.empty(hidden_size))
        self.noisy_condition_embedding = nn.Parameter(torch.empty(hidden_size))
        self.timestep_embedder = TimestepEmbedder2(
            hidden_size,
            timestep_scale=float(_cfg_get(config, "timestep_scale", 1000.0)),
        )
        self.metadata_encoder = MetadataEncoder(
            _cfg_get(config, "metadata", {}),
            action_dim=action_dim,
            model_dim=hidden_size,
        )

        self.state_encoder: RobotStateEncoder2 | None = None
        if self.robot_state_enabled:
            tokenization = str(_cfg_get(robot_state_config, "tokenization", "single_token"))
            if tokenization != "single_token":
                raise ValueError(f"robot_state.tokenization must be 'single_token', got {tokenization!r}")
            self.state_encoder = RobotStateEncoder2(
                state_dim=int(_cfg_get(robot_state_config, "state_dim", 130)),
                action_dim=action_dim,
                encoder_hidden_size=int(_cfg_get(robot_state_config, "encoder_hidden_size", 512)),
            )

        self.blocks = nn.ModuleList(
            MoT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                vision_intermediate_size=vision_intermediate_size,
                action_intermediate_size=action_intermediate_size,
                timestep_rank=timestep_rank,
                adaln_rank=adaln_rank,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(depth)
        )
        self.vision_final_norm = AdaptiveRMSNorm(hidden_size, rank=adaln_rank)
        self.action_final_norm = AdaptiveRMSNorm(hidden_size, rank=adaln_rank)
        self._initialize_weights()

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Validate legacy modulation and optionally resize pretrained spatial positions."""

        position_key = prefix + "vision_position"
        pretrained_position = state_dict.get(position_key)
        if (
            self.pretrained_vision_grid_size is not None
            and self.vision_grid_size is not None
            and pretrained_position is not None
            and pretrained_position.shape != self.vision_position.shape
        ):
            expected_shape = (1, math.prod(self.pretrained_vision_grid_size), self.hidden_size)
            if tuple(pretrained_position.shape) != expected_shape or not pretrained_position.is_floating_point():
                error_msgs.append(
                    f"Cannot resize {position_key}: pretrained_vision_grid_size={self.pretrained_vision_grid_size} "
                    f"requires floating-point positions with shape {expected_shape}, got "
                    f"{tuple(pretrained_position.shape)} with dtype {pretrained_position.dtype}."
                )
            else:
                # Checkpoints flatten [height, width] in row-major order. Interpolate
                # only the spatial axes, retaining every learned embedding channel.
                feature_map = pretrained_position.float().reshape(1, *self.pretrained_vision_grid_size, self.hidden_size)
                resized_position = F.interpolate(
                    feature_map.permute(0, 3, 1, 2),
                    size=self.vision_grid_size,
                    mode="bicubic",
                    align_corners=False,
                )
                state_dict = dict(state_dict)
                state_dict[position_key] = resized_position.flatten(2).transpose(1, 2).to(pretrained_position.dtype)

        action_position_key = prefix + "action_position"
        pretrained_action_position = state_dict.get(action_position_key)
        if (
            self.pretrained_action_tokens is not None
            and pretrained_action_position is not None
            and pretrained_action_position.shape != self.action_position.shape
        ):
            expected_shape = (1, self.pretrained_action_tokens, self.hidden_size)
            if (
                tuple(pretrained_action_position.shape) != expected_shape
                or not pretrained_action_position.is_floating_point()
            ):
                error_msgs.append(
                    f"Cannot resize {action_position_key}: expected floating-point positions with shape "
                    f"{expected_shape}, got {tuple(pretrained_action_position.shape)} "
                    f"with dtype {pretrained_action_position.dtype}."
                )
            else:
                # Preserve horizon endpoints while interpolating each learned channel.
                # Expanded checkpoints already match the target and bypass this path.
                state_dict = dict(state_dict)
                state_dict[action_position_key] = (
                    F.interpolate(
                        pretrained_action_position.float().transpose(1, 2),
                        size=self.max_action_tokens,
                        mode="linear",
                        align_corners=True,
                    )
                    .transpose(1, 2)
                    .to(pretrained_action_position.dtype)
                )

        internal_output_suffixes = tuple(
            f"blocks.{block_index}.{modality}_{branch}_norm.modulation.2.weight"
            for block_index in range(self.depth)
            for modality in ("vision", "action")
            for branch in ("joint", "ffn")
        )
        legacy_keys = [
            key
            for key, value in state_dict.items()
            if key.startswith(prefix)
            and key.removeprefix(prefix).endswith(internal_output_suffixes)
            and value.shape[0] == self.hidden_size * 2
        ]
        if legacy_keys:
            examples = ", ".join(legacy_keys[:3])
            error_msgs.append(
                "Legacy low-rank MoT3 checkpoints without residual gates are incompatible with gated "
                f"AdaLN-Zero ({examples}). Start from a checkpoint produced by the updated MoT3 model."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _initialize_weights(self) -> None:
        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Apply general initialization first; special DiT initializers below
        # deliberately override the relevant projections.
        self.apply(initialize_linear)
        for parameter in (
            self.vision_position,
            self.action_position,
            self.context_type,
            self.current_type,
            self.future_type,
            self.state_type,
            self.action_type,
            self.state_position,
            self.clean_condition_embedding,
            self.noisy_condition_embedding,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

        for module in self.timestep_embedder.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        adaptive_norms = [
            adaptive_norm
            for block in self.blocks
            for adaptive_norm in (
                block.vision_joint_norm,
                block.action_joint_norm,
                block.vision_ffn_norm,
                block.action_ffn_norm,
            )
        ]
        adaptive_norms.extend((self.vision_final_norm, self.action_final_norm))
        for adaptive_norm in adaptive_norms:
            nn.init.zeros_(adaptive_norm.output_projection.weight)
            nn.init.zeros_(adaptive_norm.output_projection.bias)

        for block in self.blocks:
            block.vision_post_attn_gamma.reset_parameters()
            block.action_post_attn_gamma.reset_parameters()

        nn.init.zeros_(self.vision_output.weight)
        nn.init.zeros_(self.action_output.weight)

    @staticmethod
    def _normalize_token_time(
        timesteps: torch.Tensor,
        batch_size: int,
        token_count: int,
    ) -> torch.Tensor:
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]
        if timesteps.ndim != 2 or timesteps.shape[0] != batch_size:
            raise ValueError(f"timesteps must be [B] or [B,N], got {tuple(timesteps.shape)}")
        if timesteps.shape[1] == 1:
            timesteps = timesteps.expand(-1, token_count)
        if timesteps.shape[1] != token_count:
            raise ValueError(f"Expected {token_count} token timesteps, got {timesteps.shape[1]}")
        return timesteps

    def _clean_condition(
        self,
        batch_size: int,
        token_count: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Expand the learned clean-token condition without a timestep embedding."""

        return self.clean_condition_embedding.to(reference).view(1, 1, -1).expand(batch_size, token_count, -1)

    def _noisy_condition(
        self,
        timesteps: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the learned noisy-token condition with per-token timestep embeddings."""

        time_embedding = self.timestep_embedder(timesteps).to(reference)
        return self.noisy_condition_embedding.to(reference).view(1, 1, -1) + time_embedding

    def encode_metadata(
        self,
        metadata: Mapping[str, Any],
        action_schema_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.metadata_encoder(
            metadata,
            action_schema_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _prepare_context(
        self,
        context: Sequence[torch.Tensor],
        current_vision: torch.Tensor,
        context_mask: torch.Tensor | None,
        meta_tokens: torch.Tensor | None,
        meta_mask: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        reference_shape = context[0].shape
        batch_size = current_vision.shape[0]
        if context_mask is None:
            context_mask = torch.ones(reference_shape[:2], device=context[0].device, dtype=torch.bool)
        elif context_mask.shape != reference_shape[:2]:
            raise ValueError(
                f"context_mask must be [B,S]={tuple(reference_shape[:2])}, got " f"{tuple(context_mask.shape)}"
            )
        context_mask = context_mask.to(device=context[0].device, dtype=torch.bool)
        context_tokens_by_stage = tuple(
            self.context_projection(layer_context) + self.context_type.to(layer_context) for layer_context in context
        )
        if meta_tokens is not None:
            if meta_tokens.ndim != 3 or meta_tokens.shape[0] != batch_size or meta_tokens.shape[-1] != self.hidden_size:
                raise ValueError(f"meta_tokens must be [B,M,D], got {tuple(meta_tokens.shape)}")
            if meta_mask is None:
                meta_mask = torch.ones(meta_tokens.shape[:2], device=meta_tokens.device, dtype=torch.bool)
            elif meta_mask.shape != meta_tokens.shape[:2]:
                raise ValueError(
                    f"meta_mask must be [B,M]={tuple(meta_tokens.shape[:2])}, got " f"{tuple(meta_mask.shape)}"
                )
            context_tokens_by_stage = tuple(
                torch.cat((stage_context, meta_tokens.to(stage_context)), dim=1)
                for stage_context in context_tokens_by_stage
            )
            context_mask = torch.cat(
                (context_mask, meta_mask.to(device=context_mask.device, dtype=torch.bool)),
                dim=1,
            )

        current_context = self.current_context_projection(current_vision)
        current_context = (
            current_context
            + self.current_type.to(current_context)
            + self.vision_position[:, : current_context.shape[1]].to(current_context)
        )
        context_tokens_by_stage = tuple(
            torch.cat((stage_context, current_context.to(stage_context)), dim=1)
            for stage_context in context_tokens_by_stage
        )
        current_context_mask = torch.ones(
            current_context.shape[:2],
            device=context_mask.device,
            dtype=torch.bool,
        )
        context_mask = torch.cat((context_mask, current_context_mask), dim=1)

        return context_tokens_by_stage, context_mask

    def prepare_inference_cache(
        self,
        *,
        context: Sequence[torch.Tensor],
        current_vision: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        meta_tokens: torch.Tensor | None = None,
        meta_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Build read-only conditioning once for one observation's flow integration.

        Never persist this cache across observations or parameter updates. It is
        deliberately rebuilt inside each whole-flow graph replay, not at capture
        time alone. Training and gradient-enabled use are forbidden.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MoT3 conditioning cache is inference-only; use eval and inference_mode")
        contexts, mask = self._prepare_context(context, current_vision, context_mask, meta_tokens, meta_mask)
        condition_kv = tuple(
            block.prepare_condition_kv(contexts[index // self.layers_per_vlm_feature])
            for index, block in enumerate(self.blocks)
        )
        return {"contexts": contexts, "mask": mask, "condition_kv": condition_kv}

    def forward(
        self,
        context: Sequence[torch.Tensor],
        current_vision: torch.Tensor,
        future_vision: torch.Tensor,
        action: torch.Tensor,
        vision_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action_token_mask: torch.Tensor | None = None,
        robot_state: torch.Tensor | None = None,
        robot_state_mask: torch.Tensor | None = None,
        feature_layers: tuple[int, ...] = (),
        meta_tokens: torch.Tensor | None = None,
        meta_mask: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        action_schema_mask: torch.Tensor | None = None,
        inference_cache: dict[str, Any] | None = None,
    ) -> MoTOutput:
        if isinstance(context, torch.Tensor):
            raise ValueError("MoT3 requires exactly eight layerwise VLM context tensors")
        if metadata is None:
            metadata = getattr(context, "metadata", None)
        if action_schema_mask is None:
            action_schema_mask = getattr(context, "action_schema_mask", None)
        context = tuple(context)
        if len(context) != self.num_vlm_features:
            raise ValueError(f"MoT3 requires {self.num_vlm_features} VLM features, got {len(context)}")
        reference_shape = context[0].shape
        if len(reference_shape) != 3 or reference_shape[-1] != self.context_dim:
            raise ValueError(f"Each VLM context must be [B,S,{self.context_dim}], got {tuple(reference_shape)}")
        if any(layer_context.shape != reference_shape for layer_context in context):
            raise ValueError("All VLM context tensors must share one [B,S,D] shape")
        if any(layer_context.device != context[0].device for layer_context in context):
            raise ValueError("All VLM context tensors must reside on one device")

        batch_size = future_vision.shape[0]
        if current_vision.shape[0] != batch_size or reference_shape[0] != batch_size:
            raise ValueError("context/current/future vision batch dimensions must match")
        if current_vision.ndim != 3 or current_vision.shape[-1] != self.vision_dim:
            raise ValueError(f"current_vision must be [B,N,{self.vision_dim}], got {tuple(current_vision.shape)}")
        if future_vision.ndim != 3 or future_vision.shape[-1] != self.vision_dim:
            raise ValueError(f"future_vision must be [B,N,{self.vision_dim}], got {tuple(future_vision.shape)}")
        if action.ndim != 3 or action.shape[0] != batch_size or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must be [B,T,{self.action_dim}], got {tuple(action.shape)}")
        if current_vision.shape[1] > self.max_vision_tokens or future_vision.shape[1] > self.max_vision_tokens:
            raise ValueError(f"vision token count exceeds max_vision_tokens={self.max_vision_tokens}")
        if action.shape[1] > self.max_action_tokens:
            raise ValueError(f"action token count exceeds max_action_tokens={self.max_action_tokens}")

        if meta_tokens is None and metadata is not None:
            if action_schema_mask is None:
                raise ValueError("metadata conditioning requires action_schema_mask")
            if action_schema_mask.shape != action.shape:
                raise ValueError(
                    "action_schema_mask must exactly match raw action shape, got "
                    f"{tuple(action_schema_mask.shape)} for action {tuple(action.shape)}"
                )
            meta_tokens, meta_mask = self.encode_metadata(
                metadata,
                action_schema_mask,
                batch_size=batch_size,
                device=context[0].device,
                dtype=context[0].dtype,
            )
        if inference_cache is None:
            context_tokens_by_stage, context_mask = self._prepare_context(
                context, current_vision, context_mask, meta_tokens, meta_mask
            )
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("MoT3 conditioning cache is inference-only; use eval and inference_mode")
            context_tokens_by_stage = inference_cache["contexts"]
            context_mask = inference_cache["mask"]
            if len(inference_cache["condition_kv"]) != len(self.blocks):
                raise ValueError("MoT3 conditioning cache must contain one K/V tuple per block")

        vision_time_tokens = future_vision.shape[1]
        if (
            self.compact_vision_condition
            and vision_time_tokens
            and (vision_timesteps.ndim == 1 or (vision_timesteps.ndim == 2 and vision_timesteps.shape[1] == 1))
        ):
            # One timestep per sample: run the timestep/AdaLN MLPs once, then
            # broadcast modulation over vision tokens. Explicit per-token time
            # retains the original path; clean robot-state conditions are unchanged.
            vision_time_tokens = 1
        vision_timesteps = self._normalize_token_time(vision_timesteps, batch_size, vision_time_tokens)
        action_timesteps = self._normalize_token_time(action_timesteps, batch_size, action.shape[1])

        vision_tokens = self.vision_input(future_vision)
        vision_tokens = (
            vision_tokens
            + self.future_type.to(vision_tokens)
            + self.vision_position[:, : vision_tokens.shape[1]].to(vision_tokens)
        )
        vision_time = self._noisy_condition(vision_timesteps, vision_tokens)

        state_input = None
        state_token_mask = None
        if self.robot_state_enabled:
            if robot_state is None or robot_state_mask is None or self.state_encoder is None:
                raise ValueError("robot_state and robot_state_mask are required by this MoT3 configuration")
            state_input, state_token_mask = self.state_encoder(robot_state, robot_state_mask)
            state_input = state_input.to(action)
            state_token_mask = state_token_mask.to(device=action.device)
        elif robot_state is not None or robot_state_mask is not None:
            raise ValueError("robot_state inputs were provided while robot-state conditioning is disabled")

        action_tokens = self.action_input(action)
        action_tokens = (
            action_tokens
            + self.action_type.to(action_tokens)
            + self.action_position[:, : action.shape[1]].to(action_tokens)
        )
        if action_token_mask is None:
            action_token_mask = torch.ones(action_tokens.shape[:2], device=action.device, dtype=torch.bool)
        elif action_token_mask.shape != action.shape[:2]:
            raise ValueError(
                f"action_token_mask must be [B,T]={tuple(action.shape[:2])}, got " f"{tuple(action_token_mask.shape)}"
            )
        action_token_mask = action_token_mask.to(device=action.device, dtype=torch.bool)
        num_state_tokens = 0
        if state_input is not None:
            state_tokens = self.action_input(state_input)
            state_tokens = state_tokens + self.state_type.to(state_tokens) + self.state_position.to(state_tokens)
            action_tokens = torch.cat((state_tokens, action_tokens), dim=1)
            action_token_mask = torch.cat((state_token_mask.bool(), action_token_mask.bool()), dim=1)
            num_state_tokens = state_tokens.shape[1]
        action_time = torch.cat(
            (
                self._clean_condition(batch_size, num_state_tokens, action_tokens),
                self._noisy_condition(action_timesteps, action_tokens),
            ),
            dim=1,
        )

        vision_features: dict[int, torch.Tensor] = {}
        action_features: dict[int, torch.Tensor] = {}
        requested_layers = set(feature_layers)
        for layer_index, block in enumerate(self.blocks, start=1):
            stage_index = (layer_index - 1) // self.layers_per_vlm_feature
            block_args = (
                context_tokens_by_stage[stage_index],
                vision_tokens,
                action_tokens,
                vision_time,
                action_time,
                context_mask.bool(),
                action_token_mask.bool(),
            )
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                vision_tokens, action_tokens = checkpoint(block, *block_args, use_reentrant=False)
            elif inference_cache is not None:
                vision_tokens, action_tokens = block(
                    *block_args, condition_kv=inference_cache["condition_kv"][layer_index - 1]
                )
            else:
                vision_tokens, action_tokens = block(*block_args)
            if layer_index in requested_layers:
                vision_features[layer_index] = vision_tokens
                action_features[layer_index] = action_tokens[:, num_state_tokens:]

        future_tokens = vision_tokens
        future_time = vision_time
        predicted_action_tokens = action_tokens[:, num_state_tokens:]
        predicted_action_time = action_time[:, num_state_tokens:]
        if future_tokens.shape[1]:
            vision_velocity = self.vision_output(self.vision_final_norm(future_tokens, future_time))
        else:
            vision_velocity = future_vision
        action_velocity = self.action_output(self.action_final_norm(predicted_action_tokens, predicted_action_time))
        return MoTOutput(
            vision_velocity=vision_velocity,
            action_velocity=action_velocity,
            vision_features=vision_features,
            action_features=action_features,
        )

    def vision_parameter_count(self) -> int:
        total = sum(parameter.numel() for parameter in self.current_context_projection.parameters())
        total += sum(parameter.numel() for parameter in self.vision_input.parameters())
        total += sum(parameter.numel() for parameter in self.vision_output.parameters())
        total += self.vision_position.numel() + self.current_type.numel() + self.future_type.numel()
        for block in self.blocks:
            modules = (
                block.vision_joint_norm,
                block.vision_cross_norm,
                block.vision_ffn_norm,
                block.vision_joint_qkv,
                block.vision_joint_q_norm,
                block.vision_joint_k_norm,
                block.vision_joint_out,
                block.vision_cross_q,
                block.vision_cross_kv,
                block.vision_cross_q_norm,
                block.vision_cross_k_norm,
                block.vision_cross_out,
                block.vision_post_attn_gamma,
                block.vision_ffn,
            )
            total += sum(parameter.numel() for module in modules for parameter in module.parameters())
        total += sum(parameter.numel() for parameter in self.vision_final_norm.parameters())
        return total
