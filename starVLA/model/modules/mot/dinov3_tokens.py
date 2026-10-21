# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add a frozen DINOv3 patch-token encoder for MoT vision inputs.

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn as nn

from starVLA.model.modules.mot.dino_loader import load_dino_encoder

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DinoV3TokenOutput(NamedTuple):
    """Patch tokens for the current and future frames."""

    current_tokens: torch.Tensor
    future_tokens: torch.Tensor
    spatial_shape: tuple[int, int]


class DinoV3FrameOutput(NamedTuple):
    """Patch tokens for one selected frame."""

    tokens: torch.Tensor
    spatial_shape: tuple[int, int]


class DinoV3TokenEncoder(nn.Module):
    """
    Encode a current/future image pair into normalized DINOv3 patch tokens.

    The input must use the head/wrist image-pair layout ``[B, 2, V, C, H, W]``.
    A single composite view is selected before both frames are encoded in one
    backbone call. Class and storage tokens are excluded by consuming
    DINOv3's ``x_norm_patchtokens`` output.
    """

    def __init__(
        self,
        dino_model_variant: str = "dinov3_vitb16",
        dino_model_path: str = "./third_party/dinov3",
        dino_weights_path: str | None = "./third_party/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        dino_dim: int = 768,
        patch_size: int = 16,
        view_index: int = 0,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if dino_dim <= 0:
            raise ValueError(f"dino_dim must be positive, got {dino_dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if view_index < 0:
            raise ValueError(f"view_index must be non-negative, got {view_index}")

        self.dino_dim = int(dino_dim)
        self.patch_size = int(patch_size)
        self.view_index = int(view_index)
        self.freeze = bool(freeze)
        self.dino_encoder = load_dino_encoder(
            dino_model_type="dinov3",
            dino_model_variant=dino_model_variant,
            dino_model_path=dino_model_path,
            dino_weights_path=dino_weights_path,
        )
        self.dino_encoder.requires_grad_(not self.freeze)
        if self.freeze:
            self.dino_encoder.eval()

        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> DinoV3TokenEncoder:
        """Preserve evaluation mode for a frozen backbone."""
        super().train(mode)
        if self.freeze:
            self.dino_encoder.eval()
        return self

    def _prepare_images(
        self,
        image_pair: torch.Tensor,
        frame_indices: tuple[int, ...] = (0, 1),
    ) -> tuple[torch.Tensor, int, int, int]:
        if not torch.is_tensor(image_pair):
            raise TypeError(f"image_pair must be a torch.Tensor, got {type(image_pair).__name__}")
        if image_pair.ndim != 6:
            raise ValueError(f"image_pair must have shape [B, 2, V, C, H, W], got {tuple(image_pair.shape)}")

        batch_size, num_frames, num_views, channels, height, width = image_pair.shape
        if num_frames != 2:
            raise ValueError(f"image_pair must contain exactly 2 frames (current and future), got {num_frames}")
        if channels != 3:
            raise ValueError(f"image_pair must contain 3-channel RGB images, got C={channels}")
        if self.view_index >= num_views:
            raise ValueError(f"view_index={self.view_index} is out of range for V={num_views}")
        if not frame_indices or any(index not in (0, 1) for index in frame_indices):
            raise ValueError(f"frame_indices must select current/future indices 0 or 1, got {frame_indices}")
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"image height and width must be at least patch_size={self.patch_size}, got H={height}, W={width}"
            )

        images = image_pair[:, frame_indices, self.view_index]
        images = images.reshape(batch_size * len(frame_indices), channels, height, width)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        elif torch.is_floating_point(images):
            images = images.float()
            if not torch.isfinite(images).all():
                raise ValueError("floating-point image_pair must contain only finite values")
            if images.numel() > 0 and (images.min() < 0 or images.max() > 1):
                raise ValueError("floating-point image_pair values must be in [0, 1]")
        else:
            raise TypeError(f"image_pair must use uint8 or floating-point values in [0, 1], got {images.dtype}")

        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        try:
            backbone_dtype = next(self.dino_encoder.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        return images.to(dtype=backbone_dtype), batch_size, height, width

    def _forward_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.dino_encoder.eval()
        grad_context = torch.no_grad() if self.freeze else nullcontext()
        with grad_context:
            if not hasattr(self.dino_encoder, "forward_features"):
                raise RuntimeError("DINOv3 backbone must provide forward_features()")
            output = self.dino_encoder.forward_features(images)

        if isinstance(output, list):
            if not output:
                raise RuntimeError("DINOv3 forward_features() returned an empty list")
            output = output[0]
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise RuntimeError("DINOv3 forward_features() must return a dict containing x_norm_patchtokens")
        patch_tokens = output["x_norm_patchtokens"]
        if not torch.is_tensor(patch_tokens) or patch_tokens.ndim != 3:
            shape = tuple(patch_tokens.shape) if torch.is_tensor(patch_tokens) else type(patch_tokens).__name__
            raise RuntimeError(f"x_norm_patchtokens must have shape [N, L, D], got {shape}")
        return patch_tokens

    def _encode_frames(
        self,
        image_pair: torch.Tensor,
        frame_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        images, batch_size, height, width = self._prepare_images(image_pair, frame_indices)
        patch_tokens = self._forward_patch_tokens(images)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        expected_tokens = patch_height * patch_width
        expected_batch = batch_size * len(frame_indices)

        if patch_tokens.shape[0] != expected_batch:
            raise RuntimeError(
                f"DINOv3 produced batch size {patch_tokens.shape[0]}, expected {expected_batch} "
                f"for frame_indices={frame_indices}"
            )
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DINOv3 produced {patch_tokens.shape[1]} patch tokens, expected {expected_tokens} "
                f"for image size {(height, width)} and patch_size={self.patch_size}"
            )
        if patch_tokens.shape[2] != self.dino_dim:
            raise RuntimeError(f"DINOv3 produced dim={patch_tokens.shape[2]}, expected dino_dim={self.dino_dim}")

        patch_tokens = patch_tokens.reshape(batch_size, len(frame_indices), expected_tokens, self.dino_dim)
        return patch_tokens, (patch_height, patch_width)

    def encode_current(self, image_pair: torch.Tensor) -> DinoV3FrameOutput:
        """Encode only the current frame, avoiding a redundant future-frame pass at inference."""
        patch_tokens, spatial_shape = self._encode_frames(image_pair, (0,))
        return DinoV3FrameOutput(tokens=patch_tokens[:, 0], spatial_shape=spatial_shape)

    def encode_future(self, image_pair: torch.Tensor) -> DinoV3FrameOutput:
        """Encode only the held-out future frame for open-loop evaluation."""
        patch_tokens, spatial_shape = self._encode_frames(image_pair, (1,))
        return DinoV3FrameOutput(tokens=patch_tokens[:, 0], spatial_shape=spatial_shape)

    def forward(self, image_pair: torch.Tensor) -> DinoV3TokenOutput:
        """
        Return current tokens, future tokens, and their ``(patch_height, patch_width)`` grid.
        """
        patch_tokens, spatial_shape = self._encode_frames(image_pair, (0, 1))
        return DinoV3TokenOutput(
            current_tokens=patch_tokens[:, 0],
            future_tokens=patch_tokens[:, 1],
            spatial_shape=spatial_shape,
        )
