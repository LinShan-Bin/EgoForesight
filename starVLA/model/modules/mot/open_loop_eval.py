# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add deterministic state-conditioned RoboTwin evaluation for Qwen3.5 MoT.

"""Fixed-episode open-loop evaluation for the Qwen3.5 MoT VLA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.mot.plotting import (
    _compose_labeled_grid,
    _feature_rgb_to_image,
    _fit_joint_pca_rgb,
    save_dual_arm_action_plot,
)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _to_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def chunk_start_indices(trajectory_length: int, chunk_size: int) -> list[int]:
    """Return non-overlapping policy query points, including the final partial chunk."""
    if trajectory_length <= 0:
        raise ValueError(f"trajectory_length must be positive, got {trajectory_length}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return list(range(0, trajectory_length, chunk_size))


def _feature_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Future-feature prediction and target must have the same shape, "
            f"got prediction={prediction.shape}, target={target.shape}"
        )
    prediction_tensor = torch.from_numpy(prediction).float()
    target_tensor = torch.from_numpy(target).float()
    return {
        "future_feature_mse": float(F.mse_loss(prediction_tensor, target_tensor)),
        "future_feature_cosine": float(F.cosine_similarity(prediction_tensor, target_tensor, dim=-1).mean()),
    }


def _masked_action_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(
            "Action prediction, target, and mask must have identical shapes, "
            f"got prediction={prediction.shape}, target={target.shape}, mask={mask.shape}"
        )
    squared_error = np.square(prediction.astype(np.float32) - target.astype(np.float32))
    mask = mask.astype(np.float32)
    valid_values = float(mask.sum())
    if valid_values <= 0:
        raise ValueError("Open-loop action mask contains no valid action values")
    return {
        "action_masked_mse": float((squared_error * mask).sum() / valid_values),
        "action_valid_values": valid_values,
    }


def save_future_feature_plot(
    artifact_dir: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    chunk_starts: list[int],
    chunk_size: int,
    trajectory_length: int,
    image_hw: tuple[int, int],
) -> Path:
    """Render all query points in one shared PCA color coordinate system."""
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"Expected future feature arrays [N,L,D], got prediction={prediction.shape}, target={target.shape}"
        )
    if prediction.shape != target.shape or prediction.shape[0] != len(chunk_starts):
        raise ValueError(
            "Feature arrays must match each other and chunk_starts, "
            f"got prediction={prediction.shape}, target={target.shape}, chunks={len(chunk_starts)}"
        )

    feature_maps: list[np.ndarray] = []
    labels: list[str] = []
    for chunk_index, start in enumerate(chunk_starts):
        future_step = min(start + chunk_size, trajectory_length - 1)
        feature_maps.extend((target[chunk_index], prediction[chunk_index]))
        labels.extend((f"GT future t={future_step}", f"Pred future t={future_step}"))

    rgb_maps = _fit_joint_pca_rgb(feature_maps)
    if rgb_maps is None:
        raise RuntimeError("Could not fit joint PCA for future-feature visualization")
    output_size = (image_hw[1], image_hw[0])
    panels = []
    for label, rgb_tokens in zip(labels, rgb_maps, strict=True):
        image = _feature_rgb_to_image(rgb_tokens, image_hw, output_size)
        if image is None:
            raise RuntimeError(f"Could not reshape DINO tokens for panel {label!r}")
        panels.append((label, image))
    canvas = _compose_labeled_grid(panels, columns=2)
    if canvas is None:
        raise RuntimeError("Could not compose future-feature visualization")

    save_path = artifact_dir / "future_feature_pred_gt.png"
    canvas.save(save_path)
    return save_path


def save_episode_image_pair(artifact_dir: Path, image_pair: np.ndarray) -> Path | None:
    """Save the first-view current/future RGB pair used by one episode evaluation."""
    try:
        from PIL import Image
    except ImportError:
        return None

    array = np.asarray(image_pair)
    if array.ndim == 6 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 5:
        array = array[:, 0]
    if array.ndim != 4 or array.shape[0] < 2:
        return None
    frames = []
    for label, frame in zip(("Current RGB", "Future RGB"), array[:2], strict=True):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0 + 0.5).astype(np.uint8)
        frames.append((label, Image.fromarray(frame)))
    canvas = _compose_labeled_grid(frames, columns=2)
    if canvas is None:
        return None
    save_path = artifact_dir / "input_current_future.png"
    canvas.save(save_path)
    return save_path


class MoTRoboTwinOpenLoopEvaluator:
    """Evaluate one fixed clean and randomized RoboTwin episode at chunk boundaries."""

    SPLIT_ORDER = ("clean", "randomized")

    def __init__(
        self,
        *,
        open_loop_config: Any,
        data_config: Any,
        training_mixture: LeRobotMixtureDataset,
        chunk_size: int,
        output_dir: str | Path,
    ) -> None:
        self.config = open_loop_config
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        self.output_dir = Path(output_dir)
        self.seed = int(_cfg_get(open_loop_config, "seed", 42))
        self.num_flow_steps = int(_cfg_get(open_loop_config, "num_flow_steps", 4))
        if self.num_flow_steps <= 0:
            raise ValueError(f"open_loop.num_flow_steps must be positive, got {self.num_flow_steps}")
        self.query_batch_size = int(_cfg_get(open_loop_config, "query_batch_size", 1))
        if self.query_batch_size <= 0:
            raise ValueError(f"open_loop.query_batch_size must be positive, got {self.query_batch_size}")
        self.eval_names = tuple(self.SPLIT_ORDER)

        data_dict = OmegaConf.to_container(data_config, resolve=True)
        if not isinstance(data_dict, dict):
            raise TypeError(f"datasets.vla_data must resolve to a mapping, got {type(data_dict).__name__}")
        data_dict.update(
            {
                "action_horizon": self.chunk_size,
                "return_image_pair": True,
                "image_pair_delta_indices": [0, self.chunk_size],
                "image_pair_random_future_delta_range": None,
                "dataset_quality_index_enabled": False,
                "dataset_quality_index_required": False,
                "quality_filter_config": None,
                "max_zero_action_frames": None,
            }
        )
        self.data_config = OmegaConf.create(data_dict)

        split_datasets = []
        self.datasets: dict[str, Any] = {}
        self.episode_ids: dict[str, int] = {}
        for split_name in self.SPLIT_ORDER:
            split_config = _cfg_get(open_loop_config, split_name, None)
            if split_config is None:
                raise ValueError(f"open_loop config is missing the {split_name!r} split")
            dataset = make_LeRobotSingleDataset(
                Path(str(_cfg_get(split_config, "data_root_dir", self.data_config.data_root_dir))),
                str(_cfg_get(split_config, "data_name")),
                str(_cfg_get(split_config, "robot_type", "my_robotwin")),
                temporal_downsample_stride=int(_cfg_get(split_config, "temporal_downsample_stride", 1)),
                base_step_stride=int(_cfg_get(split_config, "base_step_stride", 1)),
                data_cfg=self.data_config,
                control_mode=str(_cfg_get(split_config, "control_mode", "ee")),
                language_embodiment=str(_cfg_get(split_config, "language_embodiment", "aloha")),
                eef_frame=str(_cfg_get(split_config, "eef_frame", "robot_base")),
                action_index_shift=int(_cfg_get(split_config, "action_index_shift", 0)),
            )
            episode_id = int(_cfg_get(split_config, "episode_index", 0))
            available_episode_ids = {int(value) for value in dataset.trajectory_ids}
            if episode_id not in available_episode_ids:
                raise ValueError(
                    f"Episode {episode_id} is not present in {dataset.dataset_name}; "
                    f"first available IDs={sorted(available_episode_ids)[:10]}"
                )
            self.datasets[split_name] = dataset
            self.episode_ids[split_name] = episode_id
            split_datasets.append((dataset, 1.0))

        self.sample_builder = LeRobotMixtureDataset(
            split_datasets,
            mode="test",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            data_cfg=self.data_config,
        )
        self._apply_training_statistics(training_mixture)
        for dataset in self.datasets.values():
            dataset.transforms.eval()

    def _apply_training_statistics(self, training_mixture: LeRobotMixtureDataset) -> None:
        merged_metadata = getattr(training_mixture, "merged_metadata", None)
        if not isinstance(merged_metadata, dict):
            raise TypeError("Open-loop evaluation requires the original LeRobotMixtureDataset with merged_metadata")
        for split_name, dataset in self.datasets.items():
            if dataset.tag not in merged_metadata:
                raise KeyError(
                    f"Training mixture has no merged statistics for eval tag={dataset.tag!r} "
                    f"({split_name}); available={list(merged_metadata)}"
                )
            dataset.set_transforms_metadata(merged_metadata[dataset.tag])

    @staticmethod
    def _language(
        mixture: LeRobotMixtureDataset,
        dataset: Any,
        data: dict[str, Any],
        trajectory_length: int,
    ) -> str:
        prompt_style = str(mixture.data_cfg.get("prompt_style", "task_and_subtask_with_others"))
        if prompt_style == "task_or_subtask":
            language_keys = dataset.modality_keys["language"]
            language_key = language_keys[0] if len(language_keys) == 1 else language_keys[-1]
            return str(data[language_key][0])
        if prompt_style == "task_and_subtask":
            return mixture._build_language_instruction_taskandsubtask(dataset, data, trajectory_length)
        return mixture._build_language_instruction(dataset, data, trajectory_length)

    def _build_example(
        self,
        dataset: Any,
        trajectory_id: int,
        step: int,
        trajectory_length: int,
    ) -> dict[str, Any]:
        raw_data, valid_frame_mask = self.sample_builder._load_and_validate_raw_action(
            dataset,
            trajectory_id,
            step,
            action_future_horizon=self.chunk_size * dataset.temporal_downsample_stride,
        )
        remaining_modalities = tuple(modality for modality in dataset.modality_keys if modality != "action")
        raw_data.update(
            dataset.get_step_data_for_modalities(
                trajectory_id,
                step,
                modalities=remaining_modalities,
            )
        )
        data = dataset.transforms(raw_data)
        data = self.sample_builder._apply_base_supervision_policy(dataset, data, step)

        if "action" in data and not any(key.startswith("action.") for key in data):
            action = _to_numpy(data["action"])
        else:
            action = np.concatenate(
                [_to_numpy(data[key]) for key in dataset.modality_keys["action"]],
                axis=1,
            )
        action_mask_value = data.get("action_mask")
        action_mask = (
            np.ones_like(action, dtype=np.float32) if action_mask_value is None else _to_numpy(action_mask_value)
        )
        action, action_mask = self.sample_builder._apply_action_temporal_mask(
            action,
            action_mask,
            valid_frame_mask,
        )
        future_delta = self.chunk_size * dataset.temporal_downsample_stride
        image_pair = self.sample_builder._build_image_pair(
            dataset,
            trajectory_id,
            step,
            [0, future_delta],
        )
        language = self._language(self.sample_builder, dataset, data, trajectory_length)
        sample = {
            "action": action.astype(np.float32, copy=False),
            "action_mask": action_mask.astype(np.float32, copy=False),
            "image_pair": image_pair,
            "lang": language,
            "trajectory_id": trajectory_id,
            "frame_index": step,
        }
        if _cfg_get(self.data_config, "include_state", False) not in (False, "False", None):
            if "state" not in data:
                raise KeyError("Open-loop example is missing transformed robot state")
            state = _to_numpy(data["state"]).astype(np.float32, copy=False)
            state_mask_value = data.get("state_mask")
            if state_mask_value is None:
                state_mask = np.ones_like(state, dtype=bool)
            else:
                state_mask = _to_numpy(state_mask_value).astype(bool, copy=False)
            sample["state"] = state
            sample["state_mask"] = state_mask
        sample.update(self.sample_builder._build_sample_meta(dataset))
        return sample

    def _run_episode(self, model: Any, split_name: str) -> tuple[dict[str, float], dict[str, Any]]:
        dataset = self.datasets[split_name]
        episode_id = self.episode_ids[split_name]
        trajectory_index = dataset.get_trajectory_index(episode_id)
        trajectory_length = int(dataset.trajectory_lengths[trajectory_index])
        chunk_starts = chunk_start_indices(trajectory_length, self.chunk_size)

        predicted_actions = []
        target_actions = []
        action_masks = []
        predicted_features = []
        target_features = []
        preview_image_pair = None
        predicts_future_vision: bool | None = None
        split_offset = self.eval_names.index(split_name) * 1_000_000
        for batch_start in range(0, len(chunk_starts), self.query_batch_size):
            batch_steps = chunk_starts[batch_start : batch_start + self.query_batch_size]
            examples = [self._build_example(dataset, episode_id, step, trajectory_length) for step in batch_steps]
            if preview_image_pair is None:
                preview_image_pair = np.asarray(examples[0]["image_pair"])
            prediction = model.predict_action(
                examples,
                num_flow_steps=self.num_flow_steps,
                seed=self.seed + split_offset + batch_start,
            )
            chunk_predicts_future_vision = "predicted_future_vision" in prediction
            if predicts_future_vision is None:
                predicts_future_vision = chunk_predicts_future_vision
            elif predicts_future_vision != chunk_predicts_future_vision:
                raise RuntimeError("Future-vision prediction availability changed within one episode")
            batch_prediction_actions = _to_numpy(prediction["normalized_actions"])
            if chunk_predicts_future_vision:
                batch_prediction_features = _to_numpy(prediction["predicted_future_vision"])
                batch_target_features = _to_numpy(model.encode_future_vision_target(examples))

            for batch_index, (step, example) in enumerate(zip(batch_steps, examples, strict=True)):
                valid_length = min(self.chunk_size, trajectory_length - step)
                predicted_actions.append(batch_prediction_actions[batch_index, :valid_length])
                target_actions.append(example["action"][:valid_length])
                action_masks.append(example["action_mask"][:valid_length])
                if chunk_predicts_future_vision:
                    predicted_features.append(batch_prediction_features[batch_index])
                    target_features.append(batch_target_features[batch_index])

        prediction_action = np.concatenate(predicted_actions, axis=0)
        target_action = np.concatenate(target_actions, axis=0)
        action_mask = np.concatenate(action_masks, axis=0)

        metrics = {
            **_masked_action_metrics(prediction_action, target_action, action_mask),
            "episode_length": float(trajectory_length),
            "num_chunks": float(len(chunk_starts)),
        }
        output = {
            "trajectory_id": episode_id,
            "trajectory_length": trajectory_length,
            "chunk_starts": chunk_starts,
            "action": target_action,
            "action_mask": action_mask,
            "recon_actions": prediction_action,
            "preview_image_pair": preview_image_pair,
        }
        if predicts_future_vision:
            prediction_feature = np.stack(predicted_features, axis=0)
            target_feature = np.stack(target_features, axis=0)
            metrics.update(_feature_metrics(prediction_feature, target_feature))
            output.update(
                {
                    "target_features": target_feature,
                    "recon_features": prediction_feature,
                }
            )
        return metrics, output

    def _save_episode_artifacts(
        self,
        *,
        step: int,
        split_name: str,
        metrics: dict[str, float],
        output: dict[str, Any],
    ) -> dict[str, Path]:
        episode_id = int(output["trajectory_id"])
        artifact_dir = self.output_dir / "open_loop" / f"step_{step:09d}" / split_name / f"episode_{episode_id:06d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        saved_arrays = {
            "chunk_starts": np.asarray(output["chunk_starts"], dtype=np.int64),
            "action": output["action"],
            "action_mask": output["action_mask"],
            "predicted_action": output["recon_actions"],
        }
        if "recon_features" in output:
            saved_arrays.update(
                {
                    "target_future_feature": output["target_features"],
                    "predicted_future_feature": output["recon_features"],
                }
            )
        np.savez_compressed(artifact_dir / "pred_gt.npz", **saved_arrays)
        action_plot = save_dual_arm_action_plot(
            artifact_dir,
            output,
            max_timesteps=int(output["trajectory_length"]),
            prediction_label="MoT Pred",
            figure_title=f"Open-loop Action Prediction vs Ground Truth (chunk={self.chunk_size})",
            filename="action_pred_gt.png",
            chunk_boundaries=tuple(int(value) for value in output["chunk_starts"][1:]),
        )
        artifacts = {}
        if output.get("preview_image_pair") is not None:
            image_pair_plot = save_episode_image_pair(
                artifact_dir,
                np.asarray(output["preview_image_pair"]),
            )
            if image_pair_plot is not None:
                artifacts[f"{split_name}/input_current_future"] = image_pair_plot
        if "recon_features" in output:
            feature_plot = save_future_feature_plot(
                artifact_dir,
                prediction=output["recon_features"],
                target=output["target_features"],
                chunk_starts=output["chunk_starts"],
                chunk_size=self.chunk_size,
                trajectory_length=int(output["trajectory_length"]),
                image_hw=tuple(int(value) for value in self.data_config.image_size),
            )
            artifacts[f"{split_name}/future_feature_pred_gt"] = feature_plot
        if action_plot is not None:
            artifacts[f"{split_name}/action_pred_gt"] = action_plot
        return artifacts

    def run(self, model: Any, step: int) -> tuple[dict[str, float], dict[str, Path]]:
        """Run both fixed episodes and persist per-step metrics and plots."""
        all_metrics: dict[str, float] = {}
        all_artifacts: dict[str, Path] = {}
        for split_name in self.eval_names:
            print(
                f"[open_loop] Evaluating {split_name} "
                f"(episode={self.episode_ids[split_name]}, chunk={self.chunk_size})",
                flush=True,
            )
            metrics, output = self._run_episode(model, split_name)
            all_metrics.update({f"{split_name}/{key}": value for key, value in metrics.items()})
            all_artifacts.update(
                self._save_episode_artifacts(
                    step=step,
                    split_name=split_name,
                    metrics=metrics,
                    output=output,
                )
            )
            print(f"[open_loop] Finished {split_name}: {metrics}", flush=True)

        step_dir = self.output_dir / "open_loop" / f"step_{step:09d}"
        with (step_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2)
        return all_metrics, all_artifacts


class MoTTaskOpenLoopEvaluator(MoTRoboTwinOpenLoopEvaluator):
    """Evaluate arbitrary fixed episodes from one dataset in a prepared training mixture."""

    def __init__(
        self,
        *,
        training_mixture: LeRobotMixtureDataset,
        episode_ids: Mapping[str, int],
        chunk_size: int,
        output_dir: str | Path,
        dataset_index: int = 0,
        seed: int = 42,
        num_flow_steps: int = 4,
        query_batch_size: int = 1,
    ) -> None:
        if not episode_ids:
            raise ValueError("episode_ids must contain at least one named episode")
        if dataset_index < 0 or dataset_index >= len(training_mixture.datasets):
            raise IndexError(
                f"dataset_index={dataset_index} is out of range for " f"{len(training_mixture.datasets)} datasets"
            )

        self.config = OmegaConf.create(
            {
                "seed": seed,
                "num_flow_steps": num_flow_steps,
                "query_batch_size": query_batch_size,
            }
        )
        self.chunk_size = int(chunk_size)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.num_flow_steps = int(num_flow_steps)
        self.query_batch_size = int(query_batch_size)
        if self.chunk_size <= 0 or self.num_flow_steps <= 0 or self.query_batch_size <= 0:
            raise ValueError("chunk_size, num_flow_steps, and query_batch_size must all be positive")

        dataset = training_mixture.datasets[dataset_index]
        available_episode_ids = {int(value) for value in dataset.trajectory_ids}
        normalized_episode_ids = {str(name): int(episode_id) for name, episode_id in episode_ids.items()}
        missing = [
            episode_id for episode_id in normalized_episode_ids.values() if episode_id not in available_episode_ids
        ]
        if missing:
            raise ValueError(f"Episodes are not present in {dataset.dataset_name}: {missing[:10]}")

        self.data_config = training_mixture.data_cfg
        self.sample_builder = training_mixture
        self.eval_names = tuple(normalized_episode_ids)
        self.datasets = {name: dataset for name in self.eval_names}
        self.episode_ids = normalized_episode_ids
        dataset.transforms.eval()
