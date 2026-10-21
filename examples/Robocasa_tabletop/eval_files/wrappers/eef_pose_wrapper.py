# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Chubin Zhang in 2026.
# Modification: Track world EEF goals against measured waist motion inside the physics control loop.
"""RoboCasa GR1 FK/IK wrapper for camera-frame EEF policies."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from examples.Robocasa_tabletop.eval_files.wrappers.gr1_arm_ik import Gr1ArmIKSolver
from starVLA.dataloader.gr00t_lerobot.state_action.robocasa_frames import PSEUDO_FROM_BASE_ROTATION


class RobocasaGr1EefPoseWrapper(gym.Wrapper):
    """Expose camera-frame EEF state/action while retaining the BASIC joint controller.

    The registered ``gr1_unified`` environments expose seven arm joints under
    ``state/action.{left,right}_arm``. EEF checkpoints instead consume and emit
    ``xyz + quaternion_xyzw`` poses. This wrapper performs FK for observations
    and uses RoboSuite's Mink solver to map recovered absolute EEF targets back
    to arm-joint commands.

    A predicted chunk is expressed in the camera frame observed at the start of
    that chunk. The camera-to-world transform is therefore latched on the first
    action and reused for all open-loop actions in the chunk.
    """

    _CAMERA_AXIS_CORRECTION = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, env: gym.Env, *, n_action_steps: int, camera_name: str = "egoview") -> None:
        super().__init__(env)
        if n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        self.n_action_steps = int(n_action_steps)
        self.camera_name = str(camera_name)
        self._action_index = 0
        self._chunk_camera_to_world: np.ndarray | None = None
        self._world_eef_target: np.ndarray | None = None

        groot_env = env.unwrapped
        if not hasattr(groot_env, "env"):
            raise TypeError("RoboCasa EEF wrapper requires a GrootRoboCasaEnv")
        self._robosuite_env = groot_env.env
        self._build_ik_solver()

    def _build_ik_solver(self) -> None:
        """Bind IK and its joint/site mappings to the current MuJoCo simulation."""
        if len(self._robosuite_env.robots) != 1:
            raise ValueError("RoboCasa GR1 EEF wrapper currently supports exactly one robot")
        robot = self._robosuite_env.robots[0]
        controllers = robot.composite_controller.part_controllers
        if "right" not in controllers or "left" not in controllers:
            raise ValueError(f"GR1 BASIC controller is missing arm controllers: {list(controllers)}")

        self._right_joint_names = list(controllers["right"].joint_names)
        self._left_joint_names = list(controllers["left"].joint_names)
        if "torso" not in controllers:
            raise ValueError("GR1 EEF tracking requires the independently controlled torso")
        for name in ("right", "left"):
            controller = controllers[name]
            if controller.input_type != "absolute" or controller.interpolator is not None:
                raise ValueError("GR1 EEF tracking requires absolute arm controllers without interpolation")
            if not np.all(np.isfinite(controller.kp)) or not np.all(controller.kp > 0):
                raise ValueError("GR1 EEF velocity feedforward requires positive finite arm stiffness")
        self._waist_qpos_indexes = np.array(controllers["torso"].qpos_index, dtype=np.int64)
        self._arm_kd_over_kp = np.concatenate(
            [controllers[name].kd / controllers[name].kp for name in ("right", "left")]
        )
        if len(self._right_joint_names) != 7 or len(self._left_joint_names) != 7:
            raise ValueError(
                "RoboCasa GR1 EEF IK expects two seven-joint arms, got "
                f"right={len(self._right_joint_names)}, left={len(self._left_joint_names)}"
            )
        self._site_names = ["gripper0_right_grip_site", "gripper0_left_grip_site"]
        missing_sites = [name for name in self._site_names if name not in self._robosuite_env.sim.model.site_names]
        if missing_sites:
            raise ValueError(f"RoboCasa GR1 EEF sites are missing from the MuJoCo model: {missing_sites}")

        self._ik_solver = Gr1ArmIKSolver(
            model=self._robosuite_env.sim.model._model,
            data=self._robosuite_env.sim.data._data,
            site_names=self._site_names,
            robot_model=robot.robot_model.mujoco_model,
            robot_joint_names=self._right_joint_names + self._left_joint_names,
            waist_joint_names=list(controllers["torso"].joint_names),
            input_type="absolute",
            input_ref_frame="world",
            input_rotation_repr="quat_wxyz",
            solve_freq=float(self._robosuite_env.control_freq),
            posture_weights={},
            hand_pos_cost=1.0,
            hand_ori_cost=0.5,
            initial_qpos_as_posture_target=True,
        )

    def reset(self, **kwargs: Any):
        observation, info = super().reset(**kwargs)
        # hard_reset replaces both MjModel and MjData, including robot geometry.
        # Rebuild even on soft resets so the posture target belongs to this episode.
        self._build_ik_solver()
        self._action_index = 0
        self._chunk_camera_to_world = None
        self._world_eef_target = None
        return self._replace_joint_state_with_eef_pose(observation), info

    def step(self, action: dict[str, np.ndarray]):
        joint_action = self._eef_action_to_joint_action(action)
        composite = self._robosuite_env.robots[0].composite_controller
        run_controller = composite.run_controller
        last_waist = self._robosuite_env.sim.data.qpos[self._waist_qpos_indexes].copy()

        def run_with_waist_tracking(enabled_parts: dict[str, bool]) -> dict[str, np.ndarray]:
            nonlocal last_waist
            waist = self._robosuite_env.sim.data.qpos[self._waist_qpos_indexes]
            # BASIC normally holds arm joint goals for a whole policy step.
            # Re-solve before applying torques when the independently commanded
            # waist moves, using measured motion instead of assuming its command
            # has already been reached. Keep the chunk's world targets latched.
            if np.max(np.abs(waist - last_waist)) > 1e-5:
                qpos = self._solve_world_eef_target()
                composite.part_controllers["right"].set_goal(qpos[:7])
                composite.part_controllers["left"].set_goal(qpos[7:])
                last_waist = waist.copy()
            return run_controller(enabled_parts)

        # Scope the adapter to this step; restore it even if simulation fails.
        composite.run_controller = run_with_waist_tracking
        try:
            observation, reward, terminated, truncated, info = super().step(joint_action)
        finally:
            composite.run_controller = run_controller
        return self._replace_joint_state_with_eef_pose(observation), reward, terminated, truncated, info

    def _solve_world_eef_target(self) -> np.ndarray:
        """Solve the latched world target against the latest complete robot state."""
        if self._world_eef_target is None:
            raise RuntimeError("Cannot solve EEF joints before receiving a world pose target")
        qpos = np.asarray(self._ik_solver.solve(self._world_eef_target), dtype=np.float64)
        if qpos.shape != (14,) or not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"Mink IK returned invalid GR1 arm joints: shape={qpos.shape}")
        compensated = self._ik_solver.compensate_waist_velocity(qpos, self._arm_kd_over_kp)
        if not np.all(np.isfinite(compensated)):
            raise RuntimeError("GR1 waist velocity feedforward returned non-finite arm targets")
        return compensated

    def _camera_to_world(self) -> np.ndarray:
        sim = self._robosuite_env.sim
        camera_id = sim.model.camera_name2id(self.camera_name)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = sim.data.cam_xmat[camera_id].reshape(3, 3)
        transform[:3, 3] = sim.data.cam_xpos[camera_id]
        return transform @ self._CAMERA_AXIS_CORRECTION

    def _site_pose_in_camera(self, site_name: str, world_to_camera: np.ndarray) -> np.ndarray:
        sim = self._robosuite_env.sim
        site_id = sim.model.site_name2id(site_name)
        world_pose = np.eye(4, dtype=np.float64)
        world_pose[:3, :3] = sim.data.site_xmat[site_id].reshape(3, 3)
        world_pose[:3, 3] = sim.data.site_xpos[site_id]
        camera_pose = world_to_camera @ world_pose
        quaternion_xyzw = Rotation.from_matrix(camera_pose[:3, :3]).as_quat()
        return np.concatenate((camera_pose[:3, 3], quaternion_xyzw)).astype(np.float32, copy=False)

    def _replace_joint_state_with_eef_pose(self, observation: dict[str, Any]) -> dict[str, Any]:
        world_to_camera = np.linalg.inv(self._camera_to_world())
        converted = dict(observation)
        converted["state.right_arm"] = self._site_pose_in_camera(self._site_names[0], world_to_camera)
        converted["state.left_arm"] = self._site_pose_in_camera(self._site_names[1], world_to_camera)
        return converted

    @staticmethod
    def _camera_pose_to_world_action(pose: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"EEF action must be xyz+quaternion_xyzw with shape (7,), got {pose.shape}")
        if not np.all(np.isfinite(pose)):
            raise ValueError("EEF action contains non-finite values")
        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
        camera_pose[:3, 3] = pose[:3]
        world_pose = camera_to_world @ camera_pose
        quaternion_wxyz = np.roll(Rotation.from_matrix(world_pose[:3, :3]).as_quat(), 1)
        return np.concatenate((world_pose[:3, 3], quaternion_wxyz))

    def _eef_action_to_joint_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._action_index == 0 or self._chunk_camera_to_world is None:
            self._chunk_camera_to_world = self._camera_to_world()
        right_world = self._camera_pose_to_world_action(action["action.right_arm"], self._chunk_camera_to_world)
        left_world = self._camera_pose_to_world_action(action["action.left_arm"], self._chunk_camera_to_world)
        self._world_eef_target = np.concatenate((right_world, left_world))
        solved_qpos = self._solve_world_eef_target()

        converted = dict(action)
        converted["action.right_arm"] = solved_qpos[:7].astype(np.float32)
        converted["action.left_arm"] = solved_qpos[7:14].astype(np.float32)
        self._action_index = (self._action_index + 1) % self.n_action_steps
        if self._action_index == 0:
            self._chunk_camera_to_world = None
        return converted


class RobocasaGr1AlignedEefJointWrapper(RobocasaGr1EefPoseWrapper):
    """Expose aligned EEF references without replacing joint model state.

    Training stores EEF poses in a fixed frame whose origin is the GR1 base and
    whose axes point right/down/forward. Its camera-spatial relative labels use
    that same static pseudo-frame basis. Each observation therefore appends the
    absolute EEF references without replacing the model's joint state. Predicted
    arm targets return in the pseudo frame and are mapped through Mink IK; hand
    and waist commands pass through as joints unchanged.
    """

    _BASE_BODY_NAME = "robot0_base"

    def __init__(self, env: gym.Env, *, n_action_steps: int, camera_name: str = "egoview") -> None:
        super().__init__(env, n_action_steps=n_action_steps, camera_name=camera_name)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("Aligned RoboCasa EEF wrapper requires a Dict observation space")
        if self._BASE_BODY_NAME not in self._robosuite_env.sim.model.body_names:
            raise ValueError(f"MuJoCo model is missing the fixed GR1 base body {self._BASE_BODY_NAME!r}")

        observation_spaces = dict(env.observation_space.spaces)
        pose_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        observation_spaces["state.left_eef_pose"] = pose_space
        observation_spaces["state.right_eef_pose"] = pose_space
        self.observation_space = spaces.Dict(observation_spaces)

    def _pseudo_from_world(self) -> np.ndarray:
        """Match the offline converter's fixed ``T_pseudo_from_world`` exactly."""
        sim = self._robosuite_env.sim
        base_body_id = sim.model.body_name2id(self._BASE_BODY_NAME)
        world_from_base_rotation = sim.data.body_xmat[base_body_id].reshape(3, 3)
        world_from_base_position = sim.data.body_xpos[base_body_id]
        pseudo_from_world = np.eye(4, dtype=np.float64)
        pseudo_from_world[:3, :3] = PSEUDO_FROM_BASE_ROTATION @ world_from_base_rotation.T
        pseudo_from_world[:3, 3] = -pseudo_from_world[:3, :3] @ world_from_base_position
        return pseudo_from_world

    def _replace_joint_state_with_eef_pose(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Preserve the 29-D joint observation and append deployment-only geometry."""
        pseudo_from_world = self._pseudo_from_world()
        converted = dict(observation)
        converted["state.right_eef_pose"] = self._site_pose_in_camera(self._site_names[0], pseudo_from_world)
        converted["state.left_eef_pose"] = self._site_pose_in_camera(self._site_names[1], pseudo_from_world)
        return converted

    def _eef_action_to_joint_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Interpret only arm fields as pseudo-frame EEF poses; retain other joints."""
        if self._action_index == 0 or self._chunk_camera_to_world is None:
            self._chunk_camera_to_world = np.linalg.inv(self._pseudo_from_world())
        right_world = self._camera_pose_to_world_action(action["action.right_arm"], self._chunk_camera_to_world)
        left_world = self._camera_pose_to_world_action(action["action.left_arm"], self._chunk_camera_to_world)
        self._world_eef_target = np.concatenate((right_world, left_world))
        solved_qpos = self._solve_world_eef_target()

        converted = dict(action)
        converted["action.right_arm"] = solved_qpos[:7].astype(np.float32)
        converted["action.left_arm"] = solved_qpos[7:14].astype(np.float32)
        self._action_index = (self._action_index + 1) % self.n_action_steps
        if self._action_index == 0:
            self._chunk_camera_to_world = None
        return converted


__all__ = ["RobocasaGr1AlignedEefJointWrapper", "RobocasaGr1EefPoseWrapper"]
