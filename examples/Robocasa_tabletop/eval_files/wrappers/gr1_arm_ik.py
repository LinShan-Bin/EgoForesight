# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Use measured posture as the local reference for arm-only EEF IK.
"""Arm-only Mink IK for RoboCasa's independently controlled hands and waist."""

from __future__ import annotations

from typing import Any

import mink
import mujoco
import numpy as np
import qpsolvers
from robosuite.examples.third_party_controller.mink_controller import IKSolverMink
from scipy.spatial.transform import Rotation


class Gr1ArmIKSolver(IKSolverMink):
    """Solve absolute world-frame grip poses without moving unexecuted joints.

    RoboSuite's robot_joint_names controls state copying and output selection,
    but its default solver still optimizes the whole robot. Here the QP has
    only arm variables; the full measured configuration supplies FK/Jacobians.
    """

    def __init__(self, *, robot_joint_names: list[str], waist_joint_names: list[str], **kwargs: Any) -> None:
        super().__init__(robot_joint_names=robot_joint_names, **kwargs)
        if (self.input_type, self.input_ref_frame, self.input_rotation_repr) != ("absolute", "world", "quat_wxyz"):
            raise ValueError("GR1 arm IK requires absolute world-frame xyz+quaternion_wxyz targets")
        joints = [self.robot_model.joint(name) for name in robot_joint_names]
        if len(joints) != 14 or len(set(robot_joint_names)) != 14:
            raise ValueError("GR1 arm IK requires 14 distinct arm joints")
        if any(int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE) for joint in joints):
            raise ValueError("GR1 arm IK supports only scalar hinge joints")
        self._arm_dofs = np.array([joint.dofadr[0] for joint in joints], dtype=np.int64)
        self._arm_ranges = np.array([joint.range.copy() for joint in joints])
        self._waist_dofs = np.array([self.robot_model.joint(name).dofadr[0] for name in waist_joint_names])
        self._full_waist_dofs = np.array([self.full_model.joint(name).dofadr[0] for name in waist_joint_names])
        self._configuration_limit = mink.ConfigurationLimit(self.robot_model)

    def compensate_waist_velocity(self, arm_qpos: np.ndarray, kd_over_kp: np.ndarray) -> np.ndarray:
        """Add measured-waist velocity feedforward to absolute arm PD setpoints.

        A moving waist requires J_arm * dq_arm = -J_waist * dq_waist to hold
        the world pose. The arm PD accepts position only, so a (kd/kp)*dq_arm
        setpoint offset supplies the missing desired-velocity term. This uses
        the observed waist velocity, not its as-yet-unreached position command.
        """
        waist_velocity = self.full_model_data.qvel[self._full_waist_dofs]
        jacobians = []
        for name in self.site_names:
            position = np.zeros((3, self.robot_model.nv))
            rotation = np.zeros_like(position)
            mujoco.mj_jacSite(
                self.robot_model, self.configuration.data, position, rotation, self.robot_model.site(name).id
            )
            jacobians.append(np.vstack((position, 0.5 * rotation)))
        jacobian = np.vstack(jacobians)
        arm_jacobian = jacobian[:, self._arm_dofs]
        induced_velocity = jacobian[:, self._waist_dofs] @ waist_velocity
        # Damped least squares bounds amplification near singular postures.
        velocity = np.linalg.solve(arm_jacobian.T @ arm_jacobian + 1e-4 * np.eye(14), -arm_jacobian.T @ induced_velocity)
        return np.clip(arm_qpos + kd_over_kp * velocity, self._arm_ranges[:, 0], self._arm_ranges[:, 1])

    def update_robot_states(self) -> None:
        """Copy all measured joints, including the independently actuated waist."""
        self.configuration.model.body("robot0_base").pos = self.full_model.body("robot0_base").pos
        self.configuration.model.body("robot0_base").quat = self.full_model.body("robot0_base").quat
        self.configuration.update(self.full_model_data.qpos[self.all_robot_qpos_indexes_in_full_model])

    def solve(self, input_action: np.ndarray) -> np.ndarray:
        """Return arm joint targets after one constrained differential IK step."""
        action = np.asarray(input_action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f"GR1 arm IK expects two finite xyz+quaternion_wxyz poses, got {action.shape}")
        self.update_robot_states()
        # A fixed episode-initial posture is a competing soft objective, not a
        # null-space task: it moves even an EEF that already matches its target.
        self.set_posture_target(self.configuration.q.copy())
        for task, pose in zip(self.hand_tasks, action.reshape(2, 7), strict=True):
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = pose[:3]
            target[:3, :3] = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
            task.set_target(mink.SE3.from_matrix(target))

        dt = 1.0 / self.solve_freq
        problem = mink.build_ik(self.configuration, self.tasks, dt, 1e-5, limits=[self._configuration_limit])
        # The configuration limit contains independent joint-limit rows. Keep
        # every arm row, but not frozen hand/waist rows: contact can move those
        # joints slightly outside bounds, and arm IK cannot correct them.
        inequality, bound = None, None
        if problem.G is not None:
            projected = problem.G[:, self._arm_dofs]
            keep = np.any(np.abs(projected) > 1e-12, axis=1)
            if keep.any():
                inequality, bound = projected[keep], problem.h[keep]
        reduced = qpsolvers.Problem(
            problem.P[np.ix_(self._arm_dofs, self._arm_dofs)],
            problem.q[self._arm_dofs],
            inequality,
            bound,
        )
        result = qpsolvers.solve_problem(reduced, solver=self.solver)
        if result.x is None or result.x.shape != (14,) or not np.isfinite(result.x).all():
            raise RuntimeError("GR1 arm IK failed to find a finite joint-limit-constrained solution")
        velocity = np.zeros(self.robot_model.nv, dtype=np.float64)
        velocity[self._arm_dofs] = result.x / dt
        self.configuration.integrate_inplace(velocity, dt)
        return self.configuration.q[self.controlled_robot_qpos_indexes].copy()
