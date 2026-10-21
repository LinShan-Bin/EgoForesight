# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Chubin Zhang in 2026.
# Modification: Add canonical MoT RoboCasa camera-spatial relative-EEF client.
"""RoboCasa client for canonical MoT camera-spatial relative-EEF checkpoints."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class PolicyWrapper:
    """Send compact absolute EEF state and consume server-recovered action chunks."""

    def __init__(
        self,
        policy_ckpt_path: str,
        host: str = "127.0.0.1",
        port: int = 10095,
        n_action_steps: int = 16,
        **_: Any,
    ) -> None:
        del policy_ckpt_path
        self.client = WebsocketClientPolicy(host, port)
        self.n_action_steps = int(n_action_steps)
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
        action_space = self.client.get_server_metadata().get("action_space") or {}
        if action_space.get("name") != "robocasa_gr1_eef":
            raise RuntimeError(
                "Policy server did not load the RoboCasa GR1 EEF adapter; " f"server action_space={action_space}"
            )
        if action_space.get("relative_eef_frame") != "camera_spatial":
            raise RuntimeError(f"RoboCasa EEF server uses an incompatible relative frame: {action_space}")

    @staticmethod
    def _compose_training_image(image: np.ndarray) -> np.ndarray:
        """Match the training head_wrist 384x320 canvas with blank wrist views."""
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

        target_height, target_width = 256, 320
        source_height, source_width = array.shape[:2]
        scale = min(target_height / source_height, target_width / source_width)
        resized_height = max(1, int(round(source_height * scale)))
        resized_width = max(1, int(round(source_width * scale)))
        resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((384, 320, 3), dtype=np.uint8)
        y_offset = (target_height - resized_height) // 2
        x_offset = (target_width - resized_width) // 2
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        return canvas

    @staticmethod
    def _training_prompt(task_description: str) -> str:
        task = str(task_description).strip().rstrip(".")
        if task.startswith("Task:"):
            return task + "."
        return (
            f"Task: {task}. Control Mode: ee state-relative. "
            "Embodiment: fourier gr1. Reference Frame: camera spatial."
        )

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
            raise ValueError(f"Expected RoboCasa EEF state [B,29], got {compact_state.shape}")

        examples = [
            {
                "image": [self._compose_training_image(images[index, -1])],
                "lang": self._training_prompt(str(task_values[index])),
                "robocasa_eef_state": compact_state[index],
            }
            for index in range(batch_size)
        ]
        response = self.client.predict_action({"examples": examples})
        absolute_actions = np.asarray(response["data"]["absolute_actions"], dtype=np.float32)
        if absolute_actions.ndim != 3 or absolute_actions.shape[0] != batch_size:
            raise ValueError(f"absolute_actions must be [B,T,29], got {absolute_actions.shape}")
        if absolute_actions.shape[1] < self.n_action_steps or absolute_actions.shape[2] != 29:
            raise ValueError(f"Need at least [B,{self.n_action_steps},29] actions, got {absolute_actions.shape}")
        actions = absolute_actions[:, : self.n_action_steps]
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
