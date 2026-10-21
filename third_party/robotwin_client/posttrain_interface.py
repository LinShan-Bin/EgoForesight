# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications Copyright 2026 Chubin Zhang.
# Modification: Match augmented RoboTwin training state, images and camera-space EEF targets.
"""RoboTwin posttrain transport client; normalization is owned by the model server."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.temporal_ensemble import TemporalEnsembler, TemporalPoseEnsembler
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.action_space import resolve_action_space_spec


def resize_training_view(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Match LeRobotMixtureDataset._resize_image_pair_frame: bicubic, centered padding."""
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = target_size
    source_h, source_w = frame.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if source_h <= 0 or source_w <= 0:
        return canvas
    scale = min(height / source_h, width / source_w)
    resized_h, resized_w = max(1, round(source_h * scale)), max(1, round(source_w * scale))
    resized = np.asarray(Image.fromarray(frame).resize((resized_w, resized_h), Image.BICUBIC))
    y, x = (height - resized_h) // 2, (width - resized_w) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def camera_pose(world_pose: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Convert simulator xyz/wxyz into OpenCV xyz/rotvec, exactly as training data."""
    pose = np.asarray(world_pose, dtype=np.float64)
    return np.r_[
        w2c[:3, :3] @ pose[:3] + w2c[:3, 3],
        (Rotation.from_matrix(w2c[:3, :3]) * Rotation.from_quat(pose[[4, 5, 6, 3]])).as_rotvec(),
    ]


def world_target(camera_target: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Convert a compact camera xyz/rotvec/gripper target to simulator xyz/wxyz/gripper."""
    c2w = np.linalg.inv(w2c)
    quat = (Rotation.from_matrix(c2w[:3, :3]) * Rotation.from_rotvec(camera_target[3:6])).as_quat()
    return np.r_[c2w[:3, :3] @ camera_target[:3] + c2w[:3, 3], quat[[3, 0, 1, 2]], camera_target[6]]


def training_prompt(instruction: str, prompt_format: str, eef_representation: str) -> str:
    """Match the RoboTwin training prompt for the checkpoint's EEF representation."""
    if eef_representation not in {"relative", "absolute"}:
        raise ValueError(f"Unsupported EEF representation: {eef_representation}")
    control = "ee"
    if prompt_format == "ee_representation_v1":
        control = "ee state-relative" if eef_representation == "relative" else "ee absolute"
    reference_frame = "camera spatial" if eef_representation == "relative" else "camera"
    instruction = str(instruction).strip().rstrip(".")
    return f"Task: {instruction}. Control Mode: {control}. Embodiment: aloha. Reference Frame: {reference_frame}."


class PosttrainClient:
    """Use identical 14-D joint state for both joint and EEF control."""

    def __init__(self, args: dict) -> None:
        cfg = OmegaConf.load(Path(args["policy_ckpt_path"]).parents[1] / "config.yaml")
        stats = json.loads((Path(args["policy_ckpt_path"]).parents[1] / "dataset_statistics.json").read_text())
        spec = resolve_action_space_spec(SimpleNamespace(config=cfg, norm_stats=stats))
        if not spec or spec["name"] != "robotwin_aug_aligned":
            raise ValueError("Posttrain client requires robotwin_aug_aligned checkpoint metadata")
        self.eef_representation = str(spec["representation"])
        if self.eef_representation not in {"relative", "absolute"} or spec["joint_representation"] != "absolute":
            raise ValueError("Posttrain client requires relative/absolute EEF and absolute joint targets")
        self.control_mode = args.get("control_mode", "joint")
        if self.control_mode not in {"joint", "eef"}:
            raise ValueError("control_mode must be joint or eef")
        self.temporal_ensemble = str(args.get("temporal_ensemble", "disabled"))
        self.temporal_half_life_steps = float(args.get("temporal_half_life_steps", 4.0))
        if self.temporal_ensemble != "disabled":
            if self.control_mode == "eef" and self.eef_representation != "absolute":
                raise ValueError("EEF temporal fusion requires absolute-EEF checkpoint metadata")
            TemporalEnsembler(self.temporal_ensemble, self.temporal_half_life_steps)
        data = cfg.datasets.vla_data
        self.head_size = tuple(data.get("image_pair_head_size", [256, 320]))
        self.wrist_size = tuple(data.get("image_pair_wrist_size", [128, 160]))
        if self.head_size[1] != 2 * self.wrist_size[1]:
            raise ValueError("Training head/wrist image widths are inconsistent")
        self.prompt_format = str(data.get("control_mode_prompt_format", "legacy"))
        self.steps = int(args.get("action_execution_steps", 24))
        if not 0 < self.steps <= int(cfg.framework.action_model.chunk_size):
            raise ValueError("Execution steps must be within the trained action chunk")
        self.client = WebsocketClientPolicy(args.get("host", "127.0.0.1"), int(args["port"]))
        self.episode_counter = -1
        self.reset()

    def reset(self) -> None:
        """Discard the previous episode's chunk."""
        self.actions = None
        self.index = 0
        self.request_index = 0
        self.episode_seed = 0
        self.action_step = 0
        self.ensembler = (
            (TemporalPoseEnsembler if self.control_mode == "eef" else TemporalEnsembler)(
                self.temporal_ensemble, self.temporal_half_life_steps
            )
            if self.temporal_ensemble != "disabled"
            else None
        )

    def set_episode_seed(self, seed: int) -> None:
        """Record the simulator seed for reproducible per-request policy sampling."""
        self.episode_seed = seed
        self.episode_counter += 1
        self.client.prompt_log.set_episode(self.episode_counter, seed)

    def needs_observation(self) -> bool:
        """Render only for a new policy chunk, unless the episode is recorded."""
        return self.actions is None or self.index >= self.steps

    def step(self, task: object, observation: dict) -> None:
        """Request a chunk, then execute one absolute simulator target."""
        if self.needs_observation():
            cameras = observation["observation"]
            head, left, right = [cameras[key]["rgb"] for key in ("head_camera", "left_camera", "right_camera")]
            hh, hw = self.head_size
            wh, ww = self.wrist_size
            image = np.concatenate(
                (
                    resize_training_view(head, (hh, hw)),
                    np.concatenate(
                        (resize_training_view(left, (wh, ww)), resize_training_view(right, (wh, ww))), axis=1
                    ),
                ),
                axis=0,
            )
            prompt = training_prompt(task.get_instruction(), self.prompt_format, self.eef_representation)
            example = {
                "image": [image],
                "lang": prompt,
                "robotwin_joint_state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            }
            if self.control_mode == "eef":
                raw = np.asarray(cameras["head_camera"]["extrinsic_cv"], dtype=np.float64)
                self.w2c = np.eye(4)
                self.w2c[:3, :4] = raw[:3, :4]
                example["robotwin_ee_pose"] = np.concatenate(
                    [camera_pose(task.get_arm_pose(arm), self.w2c) for arm in ("left", "right")]
                ).astype(np.float32)
            response = self.client.predict_action(
                {
                    "examples": [example],
                    "robotwin_control_mode": self.control_mode,
                    "robotwin_sample_seed": self.episode_seed * 100000 + self.request_index,
                }
            )
            output = response.get("data") or {}
            if output.get("control_mode") != self.control_mode:
                raise RuntimeError(f'Policy control mode mismatch or inference error: {response.get("error")}')
            actions = np.asarray(output.get("absolute_actions"), dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 14:
                raise ValueError(f"Expected absolute actions [1,T,14], got {actions.shape}")
            if actions.shape[1] < self.steps or not np.isfinite(actions).all():
                raise ValueError("Invalid policy action chunk")
            if self.ensembler is not None:
                chunk = actions[0]
                if self.control_mode == "eef":
                    chunk = np.stack(
                        [np.r_[world_target(row[:7], self.w2c), world_target(row[7:], self.w2c)] for row in chunk]
                    )
                self.ensembler.add(chunk, start_step=self.action_step)
                self.actions = self.ensembler.fuse(start_step=self.action_step, num_steps=self.steps)
            else:
                self.actions = actions[0]
            self.index = 0
            self.request_index += 1
        action = self.actions[self.index]
        self.index += 1
        self.action_step += 1
        if self.ensembler is not None:
            self.ensembler.discard_before(self.action_step)
        if self.control_mode == "joint":
            task.take_action(action, action_type="qpos")
        elif self.ensembler is not None:
            task.take_action(action, action_type="ee")
        else:
            task.take_action(
                np.r_[world_target(action[:7], self.w2c), world_target(action[7:], self.w2c)], action_type="ee"
            )


def get_model(usr_args: dict) -> PosttrainClient:
    """Build the simulator client."""
    return PosttrainClient(usr_args)


def reset_model(model: PosttrainClient) -> None:
    """Reset episode state."""
    model.reset()


def eval(task: object, model: PosttrainClient, observation: dict) -> None:  # noqa: A001 - RoboTwin plugin API
    """Execute one action."""
    model.step(task, observation)
