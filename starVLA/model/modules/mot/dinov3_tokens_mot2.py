# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Validate MoT2 DINO inputs without synchronizing CUDA tensors to Python.

from __future__ import annotations

import torch

from starVLA.model.modules.mot.dinov3_tokens import DinoV3TokenEncoder


class DinoV3TokenEncoder2(DinoV3TokenEncoder):
    """MoT2 DINO encoder with device-side floating-point range assertions."""

    @staticmethod
    def _assert_async(condition: torch.Tensor, message: str) -> None:
        if hasattr(torch, "_assert_async"):
            torch._assert_async(condition, message)
            return
        # Older PyTorch versions do not expose the asynchronous assertion. They
        # are unsupported for the CUDA training path, but retain useful CPU
        # validation for lightweight unit tests.
        if condition.device.type != "cpu":
            raise RuntimeError("MoT2 requires torch._assert_async for non-blocking CUDA input validation")
        if not bool(condition):
            raise ValueError(message)

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

        # A Python tuple index constructs a CPU LongTensor and synchronously
        # uploads it even when the image batch already resides on CUDA.
        if frame_indices == (0, 1):
            images = image_pair[:, :, self.view_index]
        elif len(frame_indices) == 1:
            index = frame_indices[0]
            images = image_pair[:, index : index + 1, self.view_index]
        else:
            # Preserve arbitrary ordering and repeated frames without CPU indices.
            images = torch.stack([image_pair[:, index, self.view_index] for index in frame_indices], dim=1)
        images = images.reshape(batch_size * len(frame_indices), channels, height, width)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        elif torch.is_floating_point(images):
            images = images.float()
            self._assert_async(
                (torch.isfinite(images) & (images >= 0.0) & (images <= 1.0)).all(),
                "floating-point image_pair must contain only finite values in [0, 1]",
            )
        else:
            raise TypeError(f"image_pair must use uint8 or floating-point values in [0, 1], got {images.dtype}")

        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        try:
            backbone_dtype = next(self.dino_encoder.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        return images.to(dtype=backbone_dtype), batch_size, height, width
