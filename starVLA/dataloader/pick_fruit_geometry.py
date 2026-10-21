# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Explicit R1 URDF FK and invertible OpenCV camera-spatial EEF labels.

"""Geometry for measured-joint references; no action targets are used as inputs."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def arm_fk(urdf: str | Path, joints: np.ndarray, side: str, tip: str) -> np.ndarray:
    """Return torso_link4-from-tip xyz/rotvec for seven measured arm joints."""
    root = ET.parse(urdf).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain = []
    link = tip
    while link != "torso_link4":
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    transforms = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
    used = []
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        fixed[:3, 3] = xyz
        transforms = transforms @ fixed
        if joint.get("type") != "fixed":
            index = [f"{side}_arm_joint{i}" for i in range(1, 8)].index(joint.get("name"))
            used.append(index)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            moving = np.broadcast_to(np.eye(4), transforms.shape).copy()
            moving[:, :3, :3] = Rotation.from_rotvec(joints[:, index, None] * axis).as_matrix()
            transforms = transforms @ moving
    assert sorted(used) == list(range(7)), used
    return np.concatenate((transforms[:, :3, 3], Rotation.from_matrix(transforms[:, :3, :3]).as_rotvec()), -1)


def camera_pose(pose: np.ndarray, camera_from_torso: np.ndarray) -> np.ndarray:
    """Transform absolute torso xyz/rotvec poses into OpenCV optical coordinates."""
    rotation = Rotation.from_matrix(camera_from_torso[:3, :3])
    return np.concatenate(
        (
            rotation.apply(pose[..., :3]) + camera_from_torso[:3, 3],
            (rotation * Rotation.from_rotvec(pose[..., 3:])).as_rotvec(),
        ),
        -1,
    )


def relative_pose(actions: np.ndarray, reference: np.ndarray, camera_from_torso: np.ndarray) -> np.ndarray:
    """Compute camera spatial dp and log(R_camera_action R_camera_reference.T)."""
    rotation = Rotation.from_matrix(camera_from_torso[:3, :3])
    return np.concatenate(
        (
            rotation.apply(actions[..., :3] - reference[..., :3]),
            (
                rotation
                * Rotation.from_rotvec(actions[..., 3:])
                * Rotation.from_rotvec(reference[..., 3:]).inv()
                * rotation.inv()
            ).as_rotvec(),
        ),
        -1,
    ).astype(np.float32)


def absolute_pose(relative: np.ndarray, reference: np.ndarray, camera_from_torso: np.ndarray) -> np.ndarray:
    """Recover torso absolute targets from spatial deltas and current measured FK."""
    rotation = Rotation.from_matrix(camera_from_torso[:3, :3])
    return np.concatenate(
        (
            rotation.inv().apply(relative[..., :3]) + reference[..., :3],
            (
                rotation.inv()
                * Rotation.from_rotvec(relative[..., 3:])
                * rotation
                * Rotation.from_rotvec(reference[..., 3:])
            ).as_rotvec(),
        ),
        -1,
    ).astype(np.float32)
