# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add CLAP++ open-loop metrics and dual-arm action visualization.

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_DUAL_ARM_INDICES = tuple(range(14))

CANONICAL_DUAL_ARM_INDICES = (7, 8, 9, 10, 11, 12, 13, 36, 37, 38, 39, 40, 41, 42)

DUAL_ARM_DOF_NAMES = ("x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z", "gripper")

DUAL_ARM_LABELS = tuple(f"left_{name}" for name in DUAL_ARM_DOF_NAMES) + tuple(
    f"right_{name}" for name in DUAL_ARM_DOF_NAMES
)


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _as_btd(value) -> np.ndarray:
    array = _to_numpy(value)
    if array is None:
        raise ValueError("Expected an action array, got None")
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected action shape [B, T, D] or [T, D], got {array.shape}")
    return array


def _dual_arm_index_spec(action_dim: int) -> list[tuple[int, str, str, str]]:
    if action_dim >= max(CANONICAL_DUAL_ARM_INDICES) + 1:
        indices = CANONICAL_DUAL_ARM_INDICES
    else:
        indices = COMPACT_DUAL_ARM_INDICES[: min(action_dim, len(COMPACT_DUAL_ARM_INDICES))]
    specs = []
    for position, index in enumerate(indices):
        side = "left" if position < 7 else "right"
        dof_name = DUAL_ARM_DOF_NAMES[position % 7]
        specs.append((index, side, dof_name, DUAL_ARM_LABELS[position]))
    return specs


def _masked_mse_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    axis=None,
) -> np.ndarray:
    loss = (prediction.astype(np.float32) - target.astype(np.float32)) ** 2
    if mask is None:
        return loss.mean(axis=axis)
    mask = mask.astype(np.float32)
    numerator = (loss * mask).sum(axis=axis)
    denominator = mask.sum(axis=axis)
    return numerator / np.maximum(denominator, 1.0)


def _choose_sample(mask: np.ndarray | None, batch_size: int) -> int:
    if mask is None:
        return 0
    valid_counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(valid_counts)) if valid_counts.size else 0


def _infer_token_grid(num_tokens: int, image_hw: tuple[int, int] | None = None) -> tuple[int, int]:
    if num_tokens <= 0:
        return 0, 0
    target_ratio = float(image_hw[0]) / max(1.0, float(image_hw[1])) if image_hw is not None else 1.0
    candidates = []
    for grid_h in range(1, int(np.sqrt(num_tokens)) + 1):
        if num_tokens % grid_h != 0:
            continue
        grid_w = num_tokens // grid_h
        candidates.append((grid_h, grid_w))
        if grid_h != grid_w:
            candidates.append((grid_w, grid_h))
    if not candidates:
        return 1, num_tokens
    return min(candidates, key=lambda shape: abs(float(shape[0]) / max(1.0, float(shape[1])) - target_ratio))


def _fit_joint_pca_rgb(feature_maps: list[np.ndarray]) -> list[np.ndarray] | None:
    valid_maps = [features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False) for features in feature_maps]
    valid_maps = [features for features in valid_maps if features.size and features.shape[-1] > 0]
    if not valid_maps:
        return None
    combined = np.concatenate(valid_maps, axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    component_count = min(3, centered.shape[0], centered.shape[1])
    if component_count <= 0:
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:component_count].T
    except np.linalg.LinAlgError:
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        components = eigenvectors[:, -component_count:][:, ::-1]
    projections = []
    for features in valid_maps:
        projected = (features - mean) @ components
        if component_count < 3:
            projected = np.pad(projected, ((0, 0), (0, 3 - component_count)))
        projections.append(projected.astype(np.float32, copy=False))
    all_projected = np.concatenate(projections, axis=0)
    low = np.percentile(all_projected, 1, axis=0, keepdims=True)
    high = np.percentile(all_projected, 99, axis=0, keepdims=True)
    scale = high - low
    stable = scale[0] >= 1.0e-6
    rgb_maps = []
    for projected in projections:
        rgb = np.empty_like(projected[:, :3], dtype=np.float32)
        if np.any(stable):
            rgb[:, stable] = (projected[:, stable] - low[:, stable]) / scale[:, stable]
        if np.any(~stable):
            rgb[:, ~stable] = 0.5
        rgb_maps.append(np.clip(rgb, 0.0, 1.0))
    return rgb_maps


def _feature_rgb_to_image(
    rgb_tokens: np.ndarray,
    image_hw: tuple[int, int] | None,
    output_size: tuple[int, int] | None,
):
    from PIL import Image

    grid_h, grid_w = _infer_token_grid(rgb_tokens.shape[0], image_hw)
    if grid_h <= 0 or grid_w <= 0:
        return None
    token_image = (np.clip(rgb_tokens, 0.0, 1.0).reshape(grid_h, grid_w, 3) * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(token_image, mode="RGB")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    if output_size is None:
        scale = max(1, int(256 / max(grid_h, grid_w)))
        output_size = (grid_w * scale, grid_h * scale)
    return image.resize(output_size, resampling)


def _fit_image_to_tile(image, tile_size: tuple[int, int]):
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    tile_width, tile_height = tile_size
    scale = min(tile_width / max(1, image.width), tile_height / max(1, image.height))
    resized_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(resized_size, resampling)
    tile = Image.new("RGB", tile_size, (0, 0, 0))
    tile.paste(resized, ((tile_width - resized.width) // 2, (tile_height - resized.height) // 2))
    return tile


def _compose_labeled_grid(panels: list[tuple[str, object]], columns: int = 2):
    from PIL import Image, ImageDraw

    if not panels:
        return None
    columns = max(1, min(columns, len(panels)))
    label_height = 28
    padding = 8
    tile_width = max(160, max(image.width for _, image in panels))
    tile_height = max(image.height for _, image in panels)
    rows = int(np.ceil(len(panels) / columns))
    canvas_width = columns * tile_width + (columns - 1) * padding
    canvas_height = rows * (tile_height + label_height) + (rows - 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        x = col * (tile_width + padding)
        y = row * (tile_height + label_height + padding)
        draw.rectangle((x, y, x + tile_width, y + label_height), fill=(26, 26, 26))
        draw.text((x + 8, y + 7), label, fill=(240, 240, 240))
        canvas.paste(_fit_image_to_tile(image, (tile_width, tile_height)), (x, y + label_height))
    return canvas


def _plot_single_dof(
    ax,
    timesteps,
    gt,
    predictions,
    title: str,
    chunk_boundaries: tuple[int, ...] = (),
) -> None:
    ax.plot(timesteps, gt, label="Ground Truth", linewidth=2.0, alpha=0.9, color="#2E86AB")
    for label, values, style, color in predictions:
        mse = float(np.mean((gt - values) ** 2))
        ax.plot(
            timesteps,
            values,
            label=f"{label} (MSE {mse:.4f})",
            linewidth=1.8,
            alpha=0.85,
            linestyle=style,
            color=color,
        )
    for boundary in chunk_boundaries:
        if 0 < boundary < len(timesteps):
            ax.axvline(boundary, color="#555555", linewidth=1.0, alpha=0.45, linestyle="-.")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=7)


def _plot_mse_bars(ax, labels: list[str], pred_names: list[str], per_dof_mses: list[np.ndarray], title: str) -> None:
    x = np.arange(len(labels))
    width = min(0.8 / max(1, len(per_dof_mses)), 0.35)
    offsets = (np.arange(len(per_dof_mses)) - (len(per_dof_mses) - 1) / 2.0) * width
    colors = ("#A23B72", "#F18F01", "#C73E1D")
    for idx, (name, values) in enumerate(zip(pred_names, per_dof_mses)):
        ax.bar(x + offsets[idx], values, width, label=name, alpha=0.85, color=colors[idx % len(colors)])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    if any(np.any(values > 0) for values in per_dof_mses):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def save_dual_arm_action_plot(
    artifact_dir: Path,
    output: dict,
    max_timesteps: int = 500,
    *,
    prediction_label: str = "Prediction",
    figure_title: str = "Open-loop Action Prediction vs Ground Truth",
    filename: str = "action_pred_gt_dual_arm_xyz_rotvec_gripper.png",
    chunk_boundaries: tuple[int, ...] = (),
) -> Path | None:
    if "action" not in output or "recon_actions" not in output:
        return None

    import os

    mpl_cache_dir = artifact_dir / ".matplotlib"
    xdg_cache_dir = artifact_dir / ".cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = _as_btd(output["action"])
    predictions = []
    for key, label, style, color in (
        ("recon_actions", prediction_label, "--", "#A23B72"),
        ("mld_recon_actions", "MLD VAE Recon", ":", "#F18F01"),
    ):
        if key in output:
            predictions.append((key, label, style, color, _as_btd(output[key])))
    if not predictions:
        return None

    mask = _to_numpy(output.get("action_mask"))
    if mask is not None and mask.ndim == 2:
        mask = mask[None]
    common_batch = min([gt.shape[0], *[pred.shape[0] for *_, pred in predictions]])
    common_horizon = min([gt.shape[1], *[pred.shape[1] for *_, pred in predictions]])
    common_dim = min([gt.shape[2], *[pred.shape[2] for *_, pred in predictions]])
    gt = gt[:common_batch, :common_horizon, :common_dim]
    predictions = [
        (key, label, style, color, pred[:common_batch, :common_horizon, :common_dim])
        for key, label, style, color, pred in predictions
    ]
    if mask is not None:
        mask = mask[:common_batch, :common_horizon, :common_dim]
    specs = _dual_arm_index_spec(common_dim)
    if not specs:
        return None
    indices = [spec[0] for spec in specs]
    labels = [spec[3] for spec in specs]
    selected_mask = mask[:, :, indices] if mask is not None else None
    sample_idx = _choose_sample(selected_mask, gt.shape[0])
    timesteps = np.arange(min(gt.shape[1], max_timesteps))

    pred_names = []
    per_dof_mses = []
    selected_predictions = []
    selected_gt = gt[:, :, indices]
    for _, label, style, color, pred in predictions:
        selected_pred = pred[:, :, indices]
        selected_predictions.append((label, style, color, selected_pred))
        pred_names.append(label)
        per_dof_mses.append(_masked_mse_numpy(selected_pred, selected_gt, selected_mask, axis=(0, 1)))

    fig = plt.figure(figsize=(18, 26))
    grid = fig.add_gridspec(8, 2, hspace=0.45, wspace=0.28, height_ratios=[1.25] + [1.0] * 7)
    ax_mse = fig.add_subplot(grid[0, :])
    _plot_mse_bars(ax_mse, labels, pred_names, per_dof_mses, "Dual-arm xyz + rotvec + gripper MSE")

    for dof_idx, dof_name in enumerate(DUAL_ARM_DOF_NAMES):
        for col, side in enumerate(("left", "right")):
            ax = fig.add_subplot(grid[dof_idx + 1, col])
            matching = [idx for idx, spec in enumerate(specs) if spec[1] == side and spec[2] == dof_name]
            if not matching:
                ax.axis("off")
                continue
            selected_idx = matching[0]
            gt_curve = selected_gt[sample_idx, : len(timesteps), selected_idx]
            pred_curves = [
                (label, selected_pred[sample_idx, : len(timesteps), selected_idx], style, color)
                for label, style, color, selected_pred in selected_predictions
            ]
            _plot_single_dof(
                ax,
                timesteps,
                gt_curve,
                pred_curves,
                f"{side.title()} {dof_name}",
                chunk_boundaries=chunk_boundaries,
            )

    fig.suptitle(figure_title, fontsize=16, fontweight="bold")
    save_path = artifact_dir / filename
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
