# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Align native Sharpa hardware joints with canonical hand supervision.

"""Sharpa Wave hardware and canonical 22-joint conventions.

Hardware order: thumb5, index4, middle4, ring4, pinky5.
Canonical order: index4, middle4, pinky5, ring4, thumb5.
Both use radians and identical joint axes: permutation only, no sign/zero change.
The 22 values are left-aligned in each 25-D hand block; the last three are masked.
"""

import numpy as np

SHARPA_HARDWARE_JOINT_NAMES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)

HARDWARE_TO_CANONICAL_SHARPA_INDICES = (
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    17,
    18,
    19,
    20,
    21,
    13,
    14,
    15,
    16,
    0,
    1,
    2,
    3,
    4,
)


def sharpa_hardware_to_canonical(values: np.ndarray) -> np.ndarray:
    """Reorder a hand vector or per-joint statistics to the canonical 22-DoF order."""
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[-1] != 22:
        raise ValueError(f"Expected 22 Sharpa hardware joints on the last axis, got {values.shape}")
    return values[..., list(HARDWARE_TO_CANONICAL_SHARPA_INDICES)]
