# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Define merged canonical metadata families for conditioning.

DATASET_FAMILY_BY_ROBOT_TYPE = {
    "fourier_gr1_arms_waist": "unknown",
    "fourier_gr1_arms_waist_absolute": "unknown",
    "my_robotwin": "robotwin",
    "robocasa_gr1_aug_aligned": "unknown",
    "robocasa_gr1_eef": "unknown",
    "robotwin": "unknown",
    "robotwin_aug_aligned": "robotwin",
}
ROBOT_FAMILY_BY_ROBOT_TYPE = {
    "fourier_gr1_arms_waist": "unknown",
    "fourier_gr1_arms_waist_absolute": "unknown",
    "my_robotwin": "aloha",
    "robocasa_gr1_aug_aligned": "unknown",
    "robocasa_gr1_eef": "unknown",
    "robotwin": "unknown",
    "robotwin_aug_aligned": "aloha",
}
CANONICAL_ROBOT_TYPE_ALIASES = {
    "fourier_gr1_arms_waist": "fourier_gr1_arms_waist",
    "fourier_gr1_arms_waist_absolute": "fourier_gr1_arms_waist_absolute",
    "my_robotwin": "my_robotwin",
    "robocasa_gr1_aug_aligned": "robocasa_gr1_aug_aligned",
    "robocasa_gr1_eef": "robocasa_gr1_eef",
    "robotwin": "robotwin",
    "robotwin_aug_aligned": "my_robotwin",
}


def get_dataset_family(robot_type: str | None) -> str:
    return DATASET_FAMILY_BY_ROBOT_TYPE.get(str(robot_type), "unknown")


def get_robot_family(robot_type: str | None) -> str:
    return ROBOT_FAMILY_BY_ROBOT_TYPE.get(str(robot_type), "unknown")


def get_canonical_robot_type(robot_type: str | None) -> str:
    """Return the stable metadata identity shared by a schema and its augmentation."""
    value = "unknown" if robot_type in (None, "") else str(robot_type)
    return CANONICAL_ROBOT_TYPE_ALIASES.get(value, value)
