# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Align MoT3 RoboCasa inputs with training, including aligned relative EEF conditioning.
"""RoboCasa client for canonical MoT3 GR1 absolute-joint checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from deployment.model_server.tools.temporal_ensemble import TemporalEnsembler
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class PolicyWrapper:
    """Send compact simulator observations and consume 29-D absolute joint chunks."""

    def __init__(
        self,
        policy_ckpt_path: str,
        host: str = "127.0.0.1",
        port: int = 10095,
        n_action_steps: int = 16,
        temporal_ensemble: str = "disabled",
        temporal_half_life_steps: float = 4.0,
        **_: Any,
    ) -> None:
        self.input_image_size, self.input_image_layout, model_chunk_size = self._load_inference_contract(
            policy_ckpt_path
        )
        self.temporal_ensemble = temporal_ensemble
        self.temporal_half_life_steps = temporal_half_life_steps
        if temporal_ensemble != "disabled":
            TemporalEnsembler(temporal_ensemble, temporal_half_life_steps)
        self.reset()
        self.client = WebsocketClientPolicy(host, port)
        self.n_action_steps = int(n_action_steps)
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
        if self.n_action_steps > model_chunk_size:
            raise ValueError(f"n_action_steps={self.n_action_steps} exceeds checkpoint chunk_size={model_chunk_size}")
        action_space = self.client.get_server_metadata().get("action_space") or {}
        self.aligned_auxiliary_actions = action_space.get("name") == "robocasa_gr1_aug_aligned"
        if action_space.get("name") not in {"robocasa_gr1_joint", "robocasa_gr1_aug_aligned"}:
            raise RuntimeError(
                "Policy server did not load the RoboCasa GR1 joint adapter; " f"server action_space={action_space}"
            )

    def reset(self) -> None:
        """Clear per-environment temporal predictions and execution clocks."""
        self._ensemblers: list[TemporalEnsembler] = []
        self._action_steps = np.zeros(0, dtype=np.int64)

    def advance(self, executed_steps: list[int]) -> None:
        """Advance each environment by the number of actions actually executed."""
        if self.temporal_ensemble == "disabled":
            return
        steps = np.asarray(executed_steps, dtype=np.int64)
        if steps.shape != self._action_steps.shape or np.any(steps < 0):
            raise ValueError(f"Invalid executed steps {steps} for clocks {self._action_steps}")
        self._action_steps += steps
        for ensembler, step in zip(self._ensemblers, self._action_steps, strict=True):
            ensembler.discard_before(int(step))

    def _select_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.temporal_ensemble == "disabled":
            return actions[:, : self.n_action_steps]
        if not self._ensemblers:
            self._ensemblers = [
                TemporalEnsembler(self.temporal_ensemble, self.temporal_half_life_steps) for _ in actions
            ]
            self._action_steps = np.zeros(len(actions), dtype=np.int64)
        if len(actions) != len(self._ensemblers):
            raise ValueError("Environment batch size changed without resetting temporal history")
        selected = []
        for ensembler, step, chunk in zip(self._ensemblers, self._action_steps, actions, strict=True):
            ensembler.add(chunk, start_step=int(step))
            selected.append(ensembler.fuse(start_step=int(step), num_steps=self.n_action_steps))
        return np.stack(selected)

    @staticmethod
    def _load_inference_contract(policy_ckpt_path: str) -> tuple[tuple[int, int], str, int]:
        """Recover image geometry and action horizon from the checkpoint's effective config."""
        checkpoint_path = Path(policy_ckpt_path).expanduser().resolve()
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        try:
            framework = config["framework"]
            action_config = framework["action_model"]
            mot_config = framework["mot"]
            dino_config = framework["dino"]
            data_config = config["datasets"]["vla_data"]
            chunk_size = int(action_config["chunk_size"])
            representation = str(action_config["action_space"]["representation"])
            max_vision_tokens = int(mot_config["max_vision_tokens"])
            patch_size = int(dino_config["patch_size"])
            spatial_pool_size = int(dino_config["spatial_pool_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Incomplete RoboCasa inference contract in {config_path}") from error
        valid_contracts = {
            ("robocasa_1000_joint_absolute", "absolute"),
            ("robocasa_1000_aug_aligned", "relative"),
        }
        if (data_config.get("data_mix"), representation) not in valid_contracts:
            raise ValueError(
                "Expected an absolute-joint RoboCasa checkpoint, got "
                f"data_mix={data_config.get('data_mix')!r}, representation={representation!r}"
            )

        # Older saved configs omitted CLI-only image overrides. Recover only the
        # two RoboCasa geometries whose token counts make the intended layout
        # unambiguous, then run the same strict validation as explicit configs.
        legacy_vision_contracts = {
            120: ((384, 320), "head_wrist"),
            260: ((416, 640), "views"),
        }
        legacy_vision_contract = legacy_vision_contracts.get(max_vision_tokens)
        configured_image_size = data_config.get("image_size")
        image_pair_layout = str(data_config.get("image_pair_view_layout", "")).lower()
        if configured_image_size is None or not image_pair_layout:
            if legacy_vision_contract is None:
                raise ValueError(
                    "Checkpoint omits image geometry and no legacy RoboCasa contract is registered for "
                    f"max_vision_tokens={max_vision_tokens}: {config_path}"
                )
            legacy_image_size, legacy_image_pair_layout = legacy_vision_contract
            if configured_image_size is None:
                configured_image_size = legacy_image_size
            if not image_pair_layout:
                image_pair_layout = legacy_image_pair_layout
        try:
            input_image_size = tuple(int(value) for value in configured_image_size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid checkpoint image_size={configured_image_size!r} in {config_path}") from error
        if (
            len(input_image_size) != 2
            or min(input_image_size) <= 0
            or max_vision_tokens <= 0
            or patch_size <= 0
            or spatial_pool_size <= 0
        ):
            raise ValueError(
                "Invalid checkpoint vision geometry: "
                f"image_size={input_image_size}, max_vision_tokens={max_vision_tokens}, "
                f"patch_size={patch_size}, spatial_pool_size={spatial_pool_size}"
            )
        token_stride = patch_size * spatial_pool_size
        if any(value % token_stride for value in input_image_size):
            raise ValueError(
                f"Checkpoint image_size={input_image_size} is not divisible by DINO token stride {token_stride}"
            )
        derived_vision_grid = tuple(value // token_stride for value in input_image_size)
        if derived_vision_grid[0] * derived_vision_grid[1] != max_vision_tokens:
            raise ValueError(
                f"image_size={input_image_size} yields DINO grid {derived_vision_grid}, which does not match "
                f"max_vision_tokens={max_vision_tokens}"
            )
        configured_vision_grid = mot_config.get("vision_grid_size")
        if configured_vision_grid is not None:
            configured_vision_grid = tuple(int(value) for value in configured_vision_grid)
            if configured_vision_grid != derived_vision_grid:
                raise ValueError(
                    f"Configured vision_grid_size={configured_vision_grid} does not match image-derived "
                    f"grid {derived_vision_grid}"
                )

        if image_pair_layout in {"head_wrist", "compose_head_wrist", "head_wrist_grid"}:
            head_size = tuple(int(value) for value in data_config.get("image_pair_head_size", (256, 320)))
            wrist_size = tuple(int(value) for value in data_config.get("image_pair_wrist_size", (128, 160)))
            if input_image_size != (384, 320) or head_size != (256, 320) or wrist_size != (128, 160):
                raise ValueError(
                    "RoboCasa head_wrist inference supports only the Stage-2 layout: "
                    f"image_size={input_image_size}, head_size={head_size}, wrist_size={wrist_size}"
                )
            input_image_layout = "train_head_wrist"
        elif image_pair_layout == "views" and input_image_size == (416, 640):
            # Retain inference compatibility with older 260-token checkpoints.
            input_image_layout = "raw_ego_resize_pad"
        else:
            raise ValueError(
                "No train-consistent RoboCasa preprocessing is registered for "
                f"layout={image_pair_layout!r}, image_size={input_image_size}"
            )
        if chunk_size <= 0:
            raise ValueError(f"Checkpoint chunk_size must be positive, got {chunk_size}")
        return input_image_size, input_image_layout, chunk_size

    def _compose_training_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the train-preprocessed image canvas supplied by the simulator wrapper."""
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"Expected HWC RGB image, got {array.shape}")
        array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.asarray(array, dtype=np.float32)
            if array.size and float(np.nanmax(array)) <= 1.0 + 1e-6:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.shape[:2] != self.input_image_size:
            raise ValueError(
                "RoboCasa input must be prepared from render_cache with the training PIL letterbox; "
                f"expected {self.input_image_size}, got {array.shape[:2]}"
            )
        return np.ascontiguousarray(array)

    @staticmethod
    def _training_prompt(task_description: str, aligned_auxiliary_actions: bool = False) -> str:
        task = str(task_description).strip().rstrip(".")
        if task.startswith("Task:"):
            return task + "."
        # Runtime has no demonstration trajectory length from which to derive
        # the training Speed bin. Omitting both optional episode fields exactly
        # matches the explicit 15% metadata-dropout branch seen during training.
        if aligned_auxiliary_actions:
            return f"Task: {task}. Control Mode: ee state-relative. Reference Frame: camera spatial."
        return f"Task: {task}. Control Mode: joint. Reference Frame: robot base."

    @staticmethod
    def _current(observations: dict[str, Any], key: str) -> np.ndarray:
        value = np.asarray(observations[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[1] < 1:
            raise ValueError(f"{key} must be [B,H,D], got {value.shape}")
        return value[:, -1]

    def step(self, observations: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        images = np.asarray(observations["video.ego_view"])
        if images.ndim != 5 or images.shape[1] < 1:
            raise ValueError(f"video.ego_view must be [B,H,H,W,C], got {images.shape}")
        task_values = np.asarray(observations["annotation.human.coarse_action"], dtype=object).reshape(-1)
        batch_size = images.shape[0]
        if task_values.size == 1 and batch_size > 1:
            task_values = np.repeat(task_values, batch_size)
        if task_values.size != batch_size:
            raise ValueError(f"Language batch has {task_values.size} items but image batch has {batch_size}")

        compact_state = np.concatenate(
            (
                self._current(observations, "state.left_arm"),
                self._current(observations, "state.right_arm"),
                self._current(observations, "state.left_hand"),
                self._current(observations, "state.right_hand"),
                self._current(observations, "state.waist"),
            ),
            axis=-1,
        )
        if compact_state.shape != (batch_size, 29):
            raise ValueError(f"Expected compact GR1 state [B,29], got {compact_state.shape}")

        examples = []
        for batch_index in range(batch_size):
            examples.append(
                {
                    "image": [self._compose_training_image(images[batch_index, -1])],
                    "lang": self._training_prompt(
                        str(task_values[batch_index]), getattr(self, "aligned_auxiliary_actions", False)
                    ),
                    "robocasa_joint_state": compact_state[batch_index],
                }
            )
        response = self.client.predict_action({"examples": examples})
        absolute_actions = np.asarray(response["data"]["absolute_actions"], dtype=np.float32)
        if absolute_actions.ndim != 3 or absolute_actions.shape[0] != batch_size:
            raise ValueError(f"absolute_actions must be [B,T,29], got {absolute_actions.shape}")
        if absolute_actions.shape[1] < self.n_action_steps or absolute_actions.shape[2] != 29:
            raise ValueError(f"Need at least [B,{self.n_action_steps},29] actions, got {absolute_actions.shape}")
        actions = self._select_actions(absolute_actions)
        return {
            "actions": {
                "action.left_arm": actions[..., 0:7],
                "action.right_arm": actions[..., 7:14],
                "action.left_hand": actions[..., 14:20],
                "action.right_hand": actions[..., 20:26],
                "action.waist": actions[..., 26:29],
            }
        }


__all__ = ["PolicyWrapper"]
