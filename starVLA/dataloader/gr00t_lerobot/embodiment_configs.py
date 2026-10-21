# EgoForesight modifications Copyright 2026 Chubin Zhang.

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

MODALITY_CONFIGS = {
    "gr1": {
        "state": ModalityConfig(
            delta_indices=[0], modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ],
        ),
    },
    "robocasa_gr1_aug_aligned": {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "left_eef_pose",
                "right_eef_pose",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "left_eef_pose",
                "right_eef_pose",
                "left_finger_positions",
                "right_finger_positions",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE if key.endswith("eef_pose") else ActionRepresentation.ABSOLUTE,
                    type=ActionType.EEF if key.endswith("eef_pose") else ActionType.NON_EEF,
                    format=ActionFormat.XYZ_ROTVEC if key.endswith("eef_pose") else ActionFormat.DEFAULT,
                )
                for key in (
                    "left_arm",
                    "right_arm",
                    "left_hand",
                    "right_hand",
                    "waist",
                    "left_eef_pose",
                    "right_eef_pose",
                    "left_finger_positions",
                    "right_finger_positions",
                )
            ],
        ),
    },
    "robocasa_gr1_eef": {
        "state": ModalityConfig(
            delta_indices=[0], modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ],
        ),
    },
}


def _mk_ee_leftright_config(
    action_delta_indices, left_key, right_key, gripper_left_key="left_gripper", gripper_right_key="right_gripper"
):
    """Build a MODALITY_CONFIGS entry for a 2-arm EE embodiment where the
    parquet has 6D left_key / right_key columns already in xyz+rotvec form."""
    return {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[left_key, gripper_left_key, right_key, gripper_right_key],
        ),
        "action": ModalityConfig(
            delta_indices=action_delta_indices,
            modality_keys=[left_key, gripper_left_key, right_key, gripper_right_key],
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ],
        ),
    }


_H30 = list(range(30))


MODALITY_CONFIGS["robotwin"] = _mk_ee_leftright_config(_H30, "left_arm", "right_arm")


def register_modality_config(config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT):
    assert embodiment_tag.value not in MODALITY_CONFIGS, f"Embodiment tag {embodiment_tag} already registered"
    MODALITY_CONFIGS[embodiment_tag.value] = config


_ROBOTWIN_AUG_KEYS = (
    "left_arm_joint",
    "left_gripper",
    "right_arm_joint",
    "right_gripper",
    "left_eef_pose",
    "right_eef_pose",
)


MODALITY_CONFIGS["robotwin_aug_aligned"] = {
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(_ROBOTWIN_AUG_KEYS)),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=list(_ROBOTWIN_AUG_KEYS),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE if key.endswith("eef_pose") else ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF if key.endswith("eef_pose") else ActionType.NON_EEF,
                format=ActionFormat.XYZ_ROTVEC if key.endswith("eef_pose") else ActionFormat.DEFAULT,
            )
            for key in _ROBOTWIN_AUG_KEYS
        ],
    ),
}
