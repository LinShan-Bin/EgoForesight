# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Prepare image layout outside the Qwen training forward to avoid CUDA scalar reads.
# The vision-attention implementation is adapted from Transformers Qwen3_5VisionAttention:
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# That portion is licensed under Apache-2.0; see licenses/Transformers-LICENSE.txt.
# Changes: accept precomputed host lengths and retain the upstream projections/rotary/attention math.

"""Optional Qwen3.5 image-only compute path; no activations or trainable outputs are cached.

Instance methods are replaced, not Transformers classes or installed packages. Calls
without a prepared layout retain the original implementation (including generation).
"""

from __future__ import annotations

import importlib
from types import MethodType
from typing import Any, Mapping

import torch

LAYOUT_KEY = "_starvla_image_layout"


def supports_image_layout_optimization(model: torch.nn.Module | None) -> bool:
    """Identify the dense Transformers implementation supported by the layout adapter."""
    return type(model).__name__ == "Qwen3_5Model" and type(model).__module__.endswith("modeling_qwen3_5")


def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda" and value.device.type == "cpu":
        return value.pin_memory().to(device, non_blocking=True)
    return value.to(device)


@torch.no_grad()
def prepare_image_layout(
    model: torch.nn.Module, inputs: Mapping[str, Any], device: torch.device
) -> dict[str, Any] | None:
    """Derive integer image indices, mRoPE and segment lengths from processor outputs."""
    if inputs.get("pixel_values") is None or inputs.get("pixel_values_videos") is not None:
        return None
    if any(inputs.get(key) is None for key in ("input_ids", "image_grid_thw", "mm_token_type_ids")):
        return None
    host = {
        key: inputs[key].detach().cpu() if inputs.get(key) is not None else None
        for key in ("input_ids", "image_grid_thw", "attention_mask", "mm_token_type_ids")
    }
    grid = host["image_grid_thw"]
    merge = model.visual.spatial_merge_size
    if grid.ndim != 2 or grid.shape[1] != 3 or not len(grid) or (grid <= 0).any():
        raise ValueError("Qwen image_grid_thw must contain positive [time, height, width] rows")
    if (grid[:, 1:] % merge).any():
        raise ValueError("Qwen image grid must be divisible by spatial_merge_size")
    image_mask = host["input_ids"].eq(model.config.image_token_id)
    indices = image_mask.flatten().nonzero().flatten()
    if indices.numel() != int((grid.prod(-1) // (merge * merge)).sum()):
        raise ValueError("Qwen image token count does not match image_grid_thw")
    position_ids, rope_deltas = model.get_rope_index(**host)
    lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
    cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
    layout = {
        "indices": indices,
        "image_mask": image_mask,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        "cu_seqlens": cu_seqlens,
    }
    layout = {key: _to_device(value, device) for key, value in layout.items()}
    layout["max_seqlen"] = int(lengths.max())
    return layout


def _vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    max_seqlen = kwargs.pop("_starvla_vision_max_seqlen", None)
    implementation = importlib.import_module(type(self).__module__)
    if max_seqlen is None or not implementation.is_flash_attention_requested(self.config):
        return type(self).forward(self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
    # Keep the upstream QKV/rotary/projection math. Only the FA length arguments
    # change from CUDA scalars to host integers (also saved by FA for backward).
    seq_length = hidden_states.shape[0]
    query, key, value = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    query, key = implementation.apply_rotary_pos_emb_vision(query, key, cos, sin)
    query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value))
    attention = implementation.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, implementation.eager_attention_forward
    )
    output, _ = attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )
    return self.proj(output.reshape(seq_length, -1).contiguous())


def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
    layout = kwargs.pop(LAYOUT_KEY, None)
    original = type(self).forward
    if (
        layout is None
        or args
        or kwargs.get("pixel_values") is None
        or kwargs.get("pixel_values_videos") is not None
        or kwargs.get("past_key_values") is not None
        or kwargs.get("inputs_embeds") is not None
    ):
        return original(self, *args, **kwargs)
    input_ids = kwargs.pop("input_ids")
    if input_ids.shape != layout["image_mask"].shape:
        raise ValueError("Prepared Qwen image layout has a different input shape")
    torch._assert_async(
        input_ids.eq(self.config.image_token_id).eq(layout["image_mask"]).all(),
        "Prepared Qwen image layout has stale image token positions",
    )
    embeddings = self.get_input_embeddings()(input_ids)
    pixels = kwargs.pop("pixel_values").to(dtype=self.visual.dtype)
    vision = self.visual(
        pixels,
        grid_thw=kwargs["image_grid_thw"],
        return_dict=True,
        cu_seqlens=layout["cu_seqlens"],
        _starvla_vision_max_seqlen=layout["max_seqlen"],
    )
    image_embeddings = vision.pooler_output.to(device=embeddings.device, dtype=embeddings.dtype)
    # Unlike masked_scatter backward, index_copy backward needs no nonzero on CUDA.
    embeddings = embeddings.flatten(0, 1).index_copy(0, layout["indices"], image_embeddings).view_as(embeddings)
    if kwargs.get("position_ids") is None:
        kwargs["position_ids"] = layout["position_ids"]
        self.rope_deltas = layout["rope_deltas"]
    return original(self, input_ids=None, inputs_embeds=embeddings, **kwargs)


def enable_image_layout_optimization(model: torch.nn.Module) -> None:
    """Enable the guarded instance-local fast path without changing checkpoint keys."""
    if not supports_image_layout_optimization(model):
        raise ValueError("optimize_image_layout currently requires the Transformers Qwen3_5Model implementation")
    for block in model.visual.blocks:
        attention = block.attn
        if type(attention).__name__ != "Qwen3_5VisionAttention":
            raise ValueError("optimize_image_layout requires Qwen3_5VisionAttention blocks")
        attention.forward = MethodType(_vision_attention_forward, attention)
    model.forward = MethodType(_model_forward, model)
