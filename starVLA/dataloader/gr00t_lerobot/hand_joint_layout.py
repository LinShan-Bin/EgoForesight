# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Align RoboCasa hand controls to the canonical Sharpa hand slots.

"""Finger-level alignment without reordering the pretrained 22-DoF Sharpa targets."""

# Fixed checkpoint-compatible feature order, left-aligned inside each 25-D hand-joint block.
SHARPA_HAND_JOINT_NAMES = (
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
)

# FourierHands.format_action expands [0,0,1,1,2,2,3,3,4,4,5] in XML actuator order:
# pinky, ring, middle, index, thumb pitch/distal, thumb yaw. Each coupled flexion
# remains ONE supervised control, represented by that finger's MCP flexion slot.
# Thumb yaw uses CMC abduction/adduction. This aligns fingers/control roles, not
# robot-specific kinematics or units; normalization stays source-specific.
ROBOCASA_HAND_SHARPA_TARGETS = (
    "pinky_MCP_FE",
    "ring_MCP_FE",
    "middle_MCP_FE",
    "index_MCP_FE",
    "thumb_MCP_FE",
    "thumb_CMC_AA",
)
ROBOCASA_HAND_SHARPA_INDICES = tuple(SHARPA_HAND_JOINT_NAMES.index(name) for name in ROBOCASA_HAND_SHARPA_TARGETS)
