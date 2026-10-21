# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Execute aligned RoboCasa arm EEF targets with joint-controlled hands and waist.
"""RoboCasa client for mixed EEF-arm and joint-hand execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.Robocasa_tabletop.eval_files.model2robocasa_joint_mot import (
    PolicyWrapper as JointPolicyWrapper,
)


class PolicyWrapper(JointPolicyWrapper):
    """Keep the training joint state while requesting aligned EEF arm decoding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        action_space = self.client.get_server_metadata().get("action_space") or {}
        if action_space.get("name") != "robocasa_gr1_aug_aligned":
            raise RuntimeError(
                "Mixed EEF/joint execution requires the aligned RoboCasa action adapter; "
                f"server action_space={action_space}"
            )
        if action_space.get("static_camera_frame") is not True:
            raise RuntimeError(f"Aligned EEF/joint execution requires the training static camera frame: {action_space}")
        execution_modes = action_space.get("execution_modes", [])
        if "aligned_eef_joint" not in execution_modes:
            raise RuntimeError(f"Policy server does not advertise aligned_eef_joint execution: {action_space}")

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

        compact_joint_state = np.concatenate(
            (
                self._current(observations, "state.left_arm"),
                self._current(observations, "state.right_arm"),
                self._current(observations, "state.left_hand"),
                self._current(observations, "state.right_hand"),
                self._current(observations, "state.waist"),
            ),
            axis=-1,
        )
        eef_reference = np.concatenate(
            (
                self._current(observations, "state.left_eef_pose"),
                self._current(observations, "state.right_eef_pose"),
            ),
            axis=-1,
        )
        if compact_joint_state.shape != (batch_size, 29):
            raise ValueError(f"Expected compact GR1 joint state [B,29], got {compact_joint_state.shape}")
        if eef_reference.shape != (batch_size, 14):
            raise ValueError(f"Expected two aligned EEF references [B,14], got {eef_reference.shape}")

        examples = [
            {
                "image": [self._compose_training_image(images[index, -1])],
                "lang": self._training_prompt(str(task_values[index]), aligned_auxiliary_actions=True),
                "robocasa_joint_state": compact_joint_state[index],
                "robocasa_eef_reference": eef_reference[index],
            }
            for index in range(batch_size)
        ]
        response = self.client.predict_action({"examples": examples})
        actions = np.asarray(response["data"]["eef_joint_actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != batch_size:
            raise ValueError(f"eef_joint_actions must be [B,T,29], got {actions.shape}")
        if actions.shape[1] < self.n_action_steps or actions.shape[2] != 29:
            raise ValueError(f"Need at least [B,{self.n_action_steps},29] actions, got {actions.shape}")
        actions = actions[:, : self.n_action_steps]
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
