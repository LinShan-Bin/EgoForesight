# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Share exact pick-fruit image and normalization transforms with serving.

"""Pure array transforms shared by cache preparation, evaluation and deployment."""

import numpy as np


def normalize_values(values: np.ndarray, statistics: dict) -> np.ndarray:
    """Apply the training quantile normalization, clipping and schema mask."""
    normalized = (values - np.asarray(statistics["center"])) / np.asarray(statistics["scale"])
    return (np.clip(normalized, -1, 1) * np.asarray(statistics["mask"])).astype(np.float32)


def denormalize_values(values: np.ndarray, statistics: dict) -> np.ndarray:
    """Invert once, exactly as offline evaluation does; do not clip predictions."""
    return (
        (values * np.asarray(statistics["scale"]) + np.asarray(statistics["center"])) * np.asarray(statistics["mask"])
    ).astype(np.float32)


def letterbox_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize RGB with INTER_AREA and centered black padding, preserving aspect ratio."""
    import cv2

    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    new_width, new_height = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def make_rgb_canvas(images: dict[str, np.ndarray]) -> np.ndarray:
    """Compose the training 384x320 head/left-wrist/right-wrist canvas."""
    canvas = np.zeros((384, 320, 3), dtype=np.uint8)
    canvas[:256] = letterbox_rgb(images["head"], 256, 320)
    canvas[256:, :160] = letterbox_rgb(images["wrist_left"], 128, 160)
    canvas[256:, 160:] = letterbox_rgb(images["wrist_right"], 128, 160)
    return canvas
