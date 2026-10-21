# EgoForesight modifications Copyright 2026 Chubin Zhang.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implemented by Chubin Zhang in 2026.
# Modification: Add train-consistent preprocessing and joint, EEF, and mixed EEF/joint evaluation.
import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
import tyro
from gymnasium import spaces
from PIL import Image
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from examples.Robocasa_tabletop.eval_files.base_config import BasePolicy, ModalityConfig
from examples.Robocasa_tabletop.eval_files.wrappers.eef_pose_wrapper import (
    RobocasaGr1AlignedEefJointWrapper,
    RobocasaGr1EefPoseWrapper,
)
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

logger = logging.getLogger(__name__)

BOTTLE_TASKS = {"PnPWineToCabinetClose", "PnPBottleToCabinetClose"}


def is_pnp_task(env: Any) -> bool:
    """Include derived PnP tasks and multi-object PnP without matching unrelated tasks."""
    return any(
        "pnp" in cls.__name__.lower() or cls.__module__.rsplit(".", 1)[-1] == "tabletop_multi_pnp"
        for cls in type(env).__mro__
    )


def _source_regions(env: Any, name: str, reference: tuple) -> list:
    """Resolve explicit or random source layers using the pre-settling sampled height."""
    source, index = reference
    container = env.objects[source]
    if index != -1:
        return [container.spawns[index]]
    # The sampler resolves -1 internally without saving the chosen index in the config.
    # Reconstruct its height from the immutable sampled pose, never the settled pose.
    object_bottom = env.object_placements[name][0][2] + env.objects[name].bottom_offset[2]
    source_z = env.object_placements[source][0][2]
    heights = np.array([source_z + container.get_spawn_bottom_offset(spawn)[2] for spawn in container.spawns])
    if not len(heights):
        raise RuntimeError(f"Missing source spawn regions for {name} on {source}")
    differences = np.abs(heights - object_bottom)
    closest = int(np.argmin(differences))
    if differences[closest] > 0.05:
        raise RuntimeError(f"Cannot resolve sampled source layer for {name} on {source}")
    # Regions at the same height represent the same layer; retain all of them.
    return [
        spawn for spawn, height in zip(container.spawns, heights, strict=True) if abs(height - heights[closest]) < 1e-4
    ]


def pnp_initial_state_issues(
    env: Any,
    *,
    max_linear_speed: float = 0.1,
    max_angular_speed: float = 1.0,
    max_displacement: float = 0.25,
) -> list[str]:
    """Return rejection reasons without stepping or modifying the simulator.

    Speeds are m/s and rad/s. Displacement is measured from the sampled position.
    Only task objects and their containers receive motion and support checks.
    Distractors are ignored; support uses initial placement configs, not goal conditions.
    """
    from robocasa.utils import object_utils as ou

    issues = []
    data = env.sim.data
    for name, obj in env.objects.items():
        if name.startswith("distractor_"):
            continue
        if name not in env.object_placements:
            raise RuntimeError(f"Missing initial placement for required PnP object {name}")
        body = env.obj_body_id[name]
        position = np.asarray(data.body_xpos[body])
        velocity = np.asarray(data.get_body_xvelp(obj.root_body))
        angular_velocity = np.asarray(data.get_body_xvelr(obj.root_body))
        if not all(np.isfinite(value).all() for value in (position, velocity, angular_velocity)):
            issues.append(f"{name}:nonfinite_state")
            continue
        if np.linalg.norm(velocity) > max_linear_speed or np.linalg.norm(angular_velocity) > max_angular_speed:
            issues.append(f"{name}:moving(v={np.linalg.norm(velocity):.3f},w={np.linalg.norm(angular_velocity):.3f})")
        initial_position = np.asarray(env.object_placements[name][0])
        if np.linalg.norm(position - initial_position) > max_displacement:
            issues.append(f"{name}:displaced")

    for cfg in env.object_cfgs:
        name = cfg["name"]
        if name.startswith("distractor_") or name not in env.objects:
            continue
        placement = cfg.get("placement") or {}
        reference = (placement.get("sample_args") or {}).get("reference")
        if isinstance(reference, str) or (isinstance(reference, tuple) and len(reference) == 2):
            source = reference if isinstance(reference, str) else reference[0]
            regions = None if isinstance(reference, str) else _source_regions(env, name, reference)
            if not ou.check_obj_in_receptacle(env, name, source, spawn_regions=regions):
                issues.append(f"{name}:outside_source_region({source})")
            if not env.check_contact(env.objects[name], env.objects[source]):
                issues.append(f"{name}:unsupported_by_source({source})")
        elif placement.get("fixture") is not None:
            fixture = env.get_fixture(placement["fixture"], ref=placement.get("ref"))
            if not ou.check_obj_fixture_contact(env, name, fixture):
                issues.append(f"{name}:unsupported_by_fixture({fixture.name})")
        else:
            raise RuntimeError(f"Unsupported initial placement for PnP object {name}: {placement}")

    if type(env).__name__ in BOTTLE_TASKS:
        alignment = float(np.asarray(data.body_xmat[env.obj_body_id["obj"]]).reshape(3, 3)[2, 2])
        if not np.isfinite(alignment) or alignment < np.cos(np.deg2rad(20.0)):
            issues.append(f"obj:toppled(up={alignment:.3f})")
    return issues


class UprightBottleResetWrapper(gym.Wrapper):
    """Resample toppled target bottles after RoboCasa's built-in settling phase."""

    def __init__(self, env: gym.Env, max_attempts: int = 20, max_tilt_degrees: float = 20.0) -> None:
        super().__init__(env)
        if max_attempts < 1 or not 0 < max_tilt_degrees < 90:
            raise ValueError("Expected max_attempts >= 1 and 0 < max_tilt_degrees < 90")
        self.max_attempts = max_attempts
        self.min_up_alignment = float(np.cos(np.deg2rad(max_tilt_degrees)))

    def initial_state_issues(self, sim_env: Any) -> list[str]:
        """Check the target bottle's local +Z alignment with world up."""
        body_id = sim_env.obj_body_id["obj"]
        alignment = float(np.asarray(sim_env.sim.data.body_xmat[body_id]).reshape(3, 3)[2, 2])
        if np.isfinite(alignment) and alignment >= self.min_up_alignment:
            return []
        return [f"obj:toppled(up={alignment:.6f})"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Any, dict]:
        """Keep the requested seed first; derive reproducible replacement seeds on rejection."""
        for attempt in range(self.max_attempts):
            candidate_seed = seed
            if seed is not None and attempt:
                candidate_seed = int(np.random.SeedSequence([seed, attempt, 0x57494E45]).generate_state(1)[0])
            obs, info = self.env.reset(seed=candidate_seed, options=options)
            sim_env = self.env.unwrapped.env
            issues = self.initial_state_issues(sim_env)
            if not issues:
                return obs, {
                    **info,
                    "initial_state_skipped": attempt,
                    "initial_state_seed": candidate_seed if candidate_seed is not None else -1,
                }
            logger.warning(
                "INITIAL_STATE_SKIP task=%s seed=%s attempt=%d reasons=%s",
                type(sim_env).__name__,
                candidate_seed,
                attempt + 1,
                ";".join(issues),
            )
        raise RuntimeError(
            f"No valid initialization for {type(sim_env).__name__} "
            f"after {self.max_attempts} attempts (seed={seed}): {';'.join(issues)}"
        )


class PnPInitialStateResetWrapper(UprightBottleResetWrapper):
    """Reject invalid PnP scenes before policy, video, and action-history wrappers."""

    def initial_state_issues(self, sim_env: Any) -> list[str]:
        """Validate initial support, source layer, displacement, and motion."""
        return pnp_initial_state_issues(sim_env)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2  # 和10 是什么关系
    fps: int = 10  # BUG 数据集中应该是 20?
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440
    stop_on_success: bool = True


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    base_seed: Optional[int] = None  # Base seed for reproducibility, each episode uses base_seed + episode_id
    action_format: str = "relative_eef"
    input_image_size: Optional[Tuple[int, int]] = None
    input_image_layout: str = "legacy_head_wrist"
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class TrainConsistentEgoViewWrapper(gym.ObservationWrapper):
    """Build the model canvas directly from the raw simulator egoview as training does."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_size: Tuple[int, int],
        content_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(env)
        self.output_size = tuple(int(value) for value in output_size)
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError(f"output_size must be positive [H,W], got {self.output_size}")
        self.content_size = tuple(int(value) for value in (content_size or self.output_size))
        if (
            len(self.content_size) != 2
            or min(self.content_size) <= 0
            or any(content > output for content, output in zip(self.content_size, self.output_size))
        ):
            raise ValueError(
                f"content_size must fit inside output_size, got content={self.content_size}, output={self.output_size}"
            )
        observation_spaces = dict(env.observation_space.spaces)
        observation_spaces["video.ego_view"] = spaces.Box(
            low=0,
            high=255,
            shape=(*self.output_size, 3),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Dict(observation_spaces)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raw_image = getattr(self.env.unwrapped, "render_cache", None)
        if raw_image is None:
            raise RuntimeError("RoboCasa raw egoview is unavailable from render_cache")
        raw_image = np.asarray(raw_image)
        if raw_image.ndim != 3 or raw_image.shape[-1] < 3:
            raise ValueError(f"Expected raw HWC egoview, got {raw_image.shape}")
        raw_image = np.asarray(raw_image[..., :3], dtype=np.uint8)

        target_height, target_width = self.content_size
        source_height, source_width = raw_image.shape[:2]
        scale = min(target_height / source_height, target_width / source_width)
        resized_height = max(1, int(round(source_height * scale)))
        resized_width = max(1, int(round(source_width * scale)))
        resized = np.asarray(Image.fromarray(raw_image).resize((resized_width, resized_height), Image.BICUBIC))
        content_canvas = np.zeros((*self.content_size, 3), dtype=np.uint8)
        y_offset = (target_height - resized_height) // 2
        x_offset = (target_width - resized_width) // 2
        content_canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        canvas = np.zeros((*self.output_size, 3), dtype=np.uint8)
        canvas[:target_height, :target_width] = content_canvas
        output = dict(observation)
        output["video.ego_view"] = canvas
        return output


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[BasePolicy] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None
        self.episode_counter = 0  # Track episode count for seed generation

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if (
            "video.ego_view" not in observations and "video.ego_view_bg_crop_pad_res256_freq20" in observations
        ):  # BUG @JinhuiYE here only one viwes
            observations["video.ego_view"] = observations.pop("video.ego_view_bg_crop_pad_res256_freq20")
        return self.model.step(observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the model."""
        return self.model.get_modality_config()

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig, model: Optional[BasePolicy] = None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.

        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")

        # Set base seed for reproducibility
        if config.base_seed is not None:
            import random

            random.seed(config.base_seed)
            np.random.seed(config.base_seed)
            print(f"🎲 Set base random seed to {config.base_seed} for reproducible episodes")

        start_time = time.time()
        print(f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments")
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Reset episode counter
        self.episode_counter = 0

        # Initial environment reset with seed
        reset_seed = config.base_seed + self.episode_counter if config.base_seed is not None else None
        if reset_seed is not None:
            print(f"🔄 Episode {self.episode_counter}: Resetting with seed {reset_seed}")
        obs, _ = self.env.reset(seed=reset_seed)
        if hasattr(self.model, "reset"):
            self.model.reset()
        if hasattr(self.model, "client") and hasattr(self.model.client, "prompt_log"):
            self.model.client.prompt_log.set_episode(self.episode_counter, reset_seed)
        self.episode_counter += 1

        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            if hasattr(self.model, "advance"):
                self.model.advance([int(n) for n in env_infos["executed_steps"]])
            # Update episode tracking
            need_manual_reset = False
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    print(
                        f"EVAL_PROGRESS env={config.env_name} completed={completed_episodes}/{config.n_episodes} "
                        f"successes={int(sum(episode_successes))} "
                        f"success_rate={np.mean(episode_successes):.6f}",
                        flush=True,
                    )
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    need_manual_reset = True

            # Manual reset with proper seed after episode completion
            if need_manual_reset and completed_episodes < config.n_episodes:
                reset_seed = config.base_seed + self.episode_counter if config.base_seed is not None else None
                if reset_seed is not None:
                    print(f"🔄 Episode {self.episode_counter}: Resetting with seed {reset_seed}")
                next_obs, _ = self.env.reset(seed=reset_seed)
                if hasattr(self.model, "reset"):
                    self.model.reset()
                if hasattr(self.model, "client") and hasattr(self.model.client, "prompt_log"):
                    self.model.client.prompt_log.set_episode(self.episode_counter, reset_seed)
                self.episode_counter += 1

            obs = next_obs
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        print(f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds")
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_model(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the model."""
        # Get actions from the model
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    if is_pnp_task(env.unwrapped.env):
        env = PnPInitialStateResetWrapper(env)
    if config.input_image_layout == "train_head_wrist":
        if config.input_image_size != (384, 320):
            raise ValueError(f"train_head_wrist requires input_image_size=(384, 320), got {config.input_image_size}")
        env = TrainConsistentEgoViewWrapper(
            env,
            output_size=config.input_image_size,
            content_size=(256, 320),
        )
    elif config.input_image_layout == "raw_ego_resize_pad":
        if config.input_image_size is None:
            raise ValueError("raw_ego_resize_pad requires input_image_size")
        env = TrainConsistentEgoViewWrapper(env, output_size=config.input_image_size)
    elif config.input_image_layout != "legacy_head_wrist":
        raise ValueError(f"Unsupported input_image_layout: {config.input_image_layout!r}")
    if config.action_format == "relative_eef":
        env = RobocasaGr1EefPoseWrapper(env, n_action_steps=config.multistep.n_action_steps)
    elif config.action_format == "aligned_eef_joint":
        env = RobocasaGr1AlignedEefJointWrapper(env, n_action_steps=config.multistep.n_action_steps)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
        stop_on_success=config.multistep.stop_on_success,
    )
    return env


def run_evaluation(
    env_name: str,
    model: BasePolicy,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    base_seed: Optional[int] = None,
    action_format: str = "relative_eef",
    input_image_size: Optional[Tuple[int, int]] = None,
    input_image_layout: str = "legacy_head_wrist",
    stop_on_success: bool = True,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
        base_seed: Base seed for reproducibility, each episode uses base_seed + episode_id
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        base_seed=base_seed,
        action_format=action_format,
        input_image_size=input_image_size,
        input_image_layout=input_image_layout,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps, stop_on_success=stop_on_success
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: tuple[int, int] = (224, 224)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # RoboCasa Gym environment id.
    env_name: str = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360  #
    n_action_steps: int = 3
    stop_on_success: bool = True
    temporal_ensemble: str = "disabled"
    temporal_half_life_steps: float = 4.0
    action_format: str = "relative_eef"

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = (
        "experiments/1029_qwenGR00T_fourier_gr1_unified_1000_"
        "PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/"
        "steps_20000_pytorch_model.pt.log/gr1_unified/logs/"
        "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    )

    seed: Optional[int] = None  # Random seed; None preserves the simulator's stochastic reset behavior.

    pretrained_path: str = (
        "results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt"
    )


def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()

    if args.temporal_ensemble != "disabled" and args.action_format != "absolute_joint":
        raise ValueError("Temporal fusion currently supports absolute_joint only; EEF requires geometric fusion")
    if args.action_format == "absolute_joint":
        from examples.Robocasa_tabletop.eval_files.model2robocasa_joint_mot import PolicyWrapper
    elif args.action_format == "relative_eef":
        from examples.Robocasa_tabletop.eval_files.model2robocasa_eef_mot import PolicyWrapper
    elif args.action_format == "aligned_eef_joint":
        from examples.Robocasa_tabletop.eval_files.model2robocasa_aligned_eef_joint_mot import PolicyWrapper
    else:
        raise ValueError(
            "action_format must be 'absolute_joint', 'relative_eef', or 'aligned_eef_joint', "
            f"got {args.action_format!r}"
        )

    model = PolicyWrapper(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble=args.temporal_ensemble,
        temporal_half_life_steps=args.temporal_half_life_steps,
    )
    run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=None if args.video_out_path in ("", "None", None) else args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        stop_on_success=args.stop_on_success,
        base_seed=args.seed,
        action_format=args.action_format,
        input_image_size=getattr(model, "input_image_size", None),
        input_image_layout=getattr(model, "input_image_layout", "legacy_head_wrist"),
    )


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
