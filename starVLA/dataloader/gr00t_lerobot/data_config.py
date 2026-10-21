# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EgoForesight modifications Copyright 2026 Chubin Zhang.

from abc import ABC, abstractmethod

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.canonical_action import (
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    get_canonical_action_dim,
    get_canonical_state_dim,
    get_relative_eef_frame,
    is_action_alignment_enabled,
    is_state_relative_eef_enabled,
)
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_configs import MODALITY_CONFIGS
from starVLA.dataloader.gr00t_lerobot.hand_joint_layout import ROBOCASA_HAND_SHARPA_INDICES
from starVLA.dataloader.gr00t_lerobot.relative_action_stats.types import ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.eef_align import EefPoseAlignTransform, EefPoseNormalizeTransform
from starVLA.dataloader.gr00t_lerobot.transform.gr00ttransform import GR00TTransform
from starVLA.dataloader.gr00t_lerobot.transform.rotvec_continuity import RotvecContinuityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    RelativeActionTransform,
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoResizePad,
    VideoToNumpy,
    VideoToTensor,
)


def _q99_schema(*keys: str) -> dict[str, str]:
    return {key: "q99" for key in keys}


def _symmetric_q99_schema(*field_names: str) -> dict[str, str]:
    return _q99_schema(*(f"{role}.{field}" for role in ("state", "action") for field in field_names))


_NORMALIZATION_MODE_MAP: dict[str, dict[str, str]] = {
    "RoboTwinAugAlignedDataConfig": _symmetric_q99_schema(
        "left_arm_joint",
        "left_gripper",
        "right_arm_joint",
        "right_gripper",
        "left_eef_pose",
        "right_eef_pose",
    ),
    "RoboTwinDataConfig": {
        "state.left_arm": "q99",
        "state.left_gripper": "q99",
        "state.right_arm": "q99",
        "state.right_gripper": "q99",
        "action.left_arm": "q99",
        "action.left_gripper": "q99",
        "action.right_arm": "q99",
        "action.right_gripper": "q99",
    },
    "FourierGr1ArmsWaistDataConfig": _symmetric_q99_schema("left_arm", "left_hand", "right_arm", "right_hand", "waist"),
    "FourierGr1ArmsWaistAbsoluteDataConfig": {
        **_q99_schema(
            "state.left_arm",
            "state.left_hand",
            "state.right_arm",
            "state.right_hand",
            "state.waist",
        ),
        **{
            key: "min_max"
            for key in (
                "action.left_arm",
                "action.left_hand",
                "action.right_arm",
                "action.right_hand",
                "action.waist",
            )
        },
    },
    "RobocasaGr1EefDataConfig": _symmetric_q99_schema("left_arm", "left_hand", "right_arm", "right_hand", "waist"),
    "RobocasaGr1AugAlignedDataConfig": {
        **_symmetric_q99_schema(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "left_eef_pose",
            "right_eef_pose",
            "left_finger_positions",
            "right_finger_positions",
        ),
        **{f"action.{key}": "min_max" for key in ("left_arm", "right_arm", "left_hand", "right_hand", "waist")},
    },
}


def _norm_modes_for_class(class_name: str, keys: list[str]) -> dict[str, str]:
    """Return an explicit per-field normalization schema for one DataConfig."""
    if class_name not in _NORMALIZATION_MODE_MAP:
        raise KeyError(
            f"Normalization schema is not registered for DataConfig {class_name!r}; "
            "add every state/action field to _NORMALIZATION_MODE_MAP"
        )
    table = _NORMALIZATION_MODE_MAP[class_name]
    missing_keys = [key for key in keys if key not in table]
    if missing_keys:
        raise KeyError(
            f"Normalization schema for DataConfig {class_name!r} is missing fields {missing_keys}; "
            f"registered fields={sorted(table)}"
        )
    invalid_modes = {key: table[key] for key in keys if table[key] not in {"q99", "min_max"}}
    if invalid_modes:
        raise ValueError(f"Normalization schema for DataConfig {class_name!r} contains invalid modes {invalid_modes}")
    return {key: table[key] for key in keys}


def _maybe_relative_eef_transform(action_canonical_slots):
    """Build [RelativeActionTransform, eef_keys] for EEF slots when the
    global state-relative EEF flag is on.

    Returns
    -------
    (transforms, eef_keys)
        transforms : list  -- [RelativeActionTransform] or [] when off / no EEF slot.
        eef_keys   : list  -- action.* keys whose slot is LEFT/RIGHT_EEF_POSE.
                              Empty when off / no EEF slot. Callers append these
                              to the downstream StateActionTransform's
                              relative_action_keys so normalization uses the
                              precomputed relative_action statistics.
    """
    if not is_state_relative_eef_enabled() or not action_canonical_slots:
        return [], []
    eef_keys = [k for k, slot in action_canonical_slots.items() if slot in ("LEFT_EEF_POSE", "RIGHT_EEF_POSE")]
    if not eef_keys:
        return [], []
    transform = RelativeActionTransform(
        apply_to=eef_keys,
        relative_action_info=[{"key": k, "type": ActionType.EEF, "format": ActionFormat.XYZ_ROTVEC} for k in eef_keys],
        reference_index=0,
        relative_eef_frame=get_relative_eef_frame(),
        in_place=True,
    )
    return [transform], eef_keys


def _state_action_relative_block(
    *,
    state_keys: list[str],
    action_keys: list[str],
    class_name: str,
    action_horizon: int,
    action_canonical_slots: dict[str, str] | None,
    eef_align_merges: list[tuple[str, str | None, str]] | None = None,
    eef_align_src_rotation: dict[str, str] | None = None,
) -> list:
    """Unified state/action -> [EefPoseAlign] -> [Relative] -> normalize block.

    Order is critical when the state-relative EEF flag is ON:
      1) StateActionToTensor(state + action)  -- both to tensors
      2) EefPoseAlignTransform (if given)      -- physical-unit merged EE pose
      3) RelativeActionTransform (if flag on)  -- SE(3) T_ref^{-1} @ T_k
      4) StateActionTransform(state normalize) -- ref pose stays raw above
      5) StateActionTransform(action normalize, relative_action_keys=[...])

    When the flag is OFF or there are no EEF slots, this collapses to the
    legacy behavior: normalize before EEF align (bit-identical to the code
    every EEF DataConfig used before this refactor).
    """
    rel_eef_transforms, rel_eef_keys = _maybe_relative_eef_transform(action_canonical_slots)
    # eef_align = (
    #     [EefPoseAlignTransform(merges=eef_align_merges, src_rotation=eef_align_src_rotation or {})]
    #     if eef_align_merges and is_action_alignment_enabled()
    #     else []
    # )

    if rel_eef_keys:
        # ON path: align first (physical units), then relative, then
        # rotvec-continuity on the RELATIVE chunk (physical radians, so the
        # quaternion double-cover check works), then normalize.
        #
        # Why continuity must run AFTER relative: RelativeActionTransform's
        # T_ref^{-1} @ T_k computation ends with Rotation.from_matrix(...
        # ).as_rotvec() which re-folds the rotvec magnitude into [0, pi].
        # Any continuity fix upstream is undone here. Fixing AFTER relative
        # is the only placement that survives.
        #
        # Why continuity must run BEFORE normalization: the transform uses
        # scipy.Rotation.from_rotvec on the input; once values are squashed
        # into [-1, 1] the double-cover check would treat every frame as a
        # tiny rotation (< pi shell) and never trigger.
        return [
            StateActionToTensor(apply_to=state_keys + action_keys),
            # *eef_align,
            *rel_eef_transforms,
            # Include state-side EEF keys too. Under the current
            # observation_indices=[0] convention state chunk length is 1, so
            # unwrap_rotvec_sequence returns identity; the extra keys become a
            # no-op with zero cost, and defensively cover any future
            # state-history extension where continuity would actually matter.
            RotvecContinuityTransform(
                keys=rel_eef_keys + [k.replace("action.", "state.", 1) for k in rel_eef_keys],
                time_axis=-2,
            ),
            StateActionTransform(
                apply_to=state_keys,
                normalization_modes=_norm_modes_for_class(class_name, state_keys),
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes=_norm_modes_for_class(class_name, action_keys),
                relative_action_keys=rel_eef_keys,
                action_horizon=action_horizon,
            ),
        ]
    # OFF path: RESTORE legacy order  normalize -> align.
    # Rationale:
    #   * EefPoseAlignTransform performs an IN-PLACE quat(7D) -> rotvec(6D)
    #     conversion on state/action keys. dataset_statistics.json stores
    #     min/max/q01/q99 for the ORIGINAL 7D quat. If normalize runs AFTER
    #     align, it broadcasts 7-length stats vectors against 6-length data
    #     and crashes with `mask[7] vs tensor[..., 6]`.
    #   * Normalizing quat(7D) against quat-shaped stats is dimensionally
    #     correct even though the network eventually sees rotvec -- xyz part
    #     is unaffected and the quat 4D range is bounded to [-1,1] anyway.
    #
    # Continuity is intentionally NOT inserted here -- the correct place
    # for the OFF-path pi-shell fix is after canonical scatter (rotvec is
    # unambiguously the last-3-of-6 slot at that point) with an
    # unnorm-aware variant. Follow-up issue.
    # After align (which produced 6D xyz+rotvec keys), normalize those merged
    # keys to [-1, 1] so downstream ConcatTransform receives values in the
    # same range as every other normalized key in the canonical layout.
    # See EefPoseNormalizeTransform docstring for the xyz/rotvec bounds
    # convention. Skipped when there is no align step (no merged keys to
    # normalize).
    # off_eef_bounds = {}
    # if action_canonical_slots and eef_align:
    #     for k, slot in action_canonical_slots.items():
    #         if slot in ("LEFT_EEF_POSE", "RIGHT_EEF_POSE"):
    #             # Default workspace envelope: xyz +/- 2 m covers both
    #             # humanoid dual-arm and single-arm setups.
    #             off_eef_bounds[k] = {"min": [-2.0, -2.0, -2.0], "max": [2.0, 2.0, 2.0]}
    #             # State-side twin
    #             sk = k.replace("action.", "state.", 1)
    #             off_eef_bounds[sk] = {"min": [-2.0, -2.0, -2.0], "max": [2.0, 2.0, 2.0]}
    # off_eef_normalize = (
    #     [EefPoseNormalizeTransform(apply_to=list(off_eef_bounds.keys()), xyz_bounds=off_eef_bounds)]
    #     if off_eef_bounds
    #     else []
    # )

    return [
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes=_norm_modes_for_class(class_name, state_keys),
        ),
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes=_norm_modes_for_class(class_name, action_keys),
        ),
        # *eef_align,
        # *off_eef_normalize,
    ]


def _standard_video_transforms(video_keys: list[str]) -> list:
    transforms = []
    for key in video_keys:
        transforms.extend(
            [
                VideoToTensor(apply_to=[key]),
                VideoResizePad(apply_to=[key], size=224, interpolation="linear", fill_value=0.0),
                VideoColorJitter(
                    apply_to=[key],
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=[key]),
            ]
        )
    return transforms


def _rotvec_eef_transform(
    config,
    *,
    state_canonical_slots: dict[str, str],
    action_canonical_slots: dict[str, str],
) -> ModalityTransform:
    transforms = _standard_video_transforms(config.video_keys)
    normalization_class_name = getattr(config, "normalization_class_name", type(config).__name__)
    transforms.extend(
        _state_action_relative_block(
            state_keys=config.state_keys,
            action_keys=config.action_keys,
            class_name=normalization_class_name,
            action_horizon=len(config.action_indices),
            action_canonical_slots=action_canonical_slots,
        )
    )
    transforms.extend(
        _concat_and_gr00t(
            video_keys=config.video_keys,
            state_keys=config.state_keys,
            action_keys=config.action_keys,
            state_canonical_slots=state_canonical_slots,
            action_canonical_slots=action_canonical_slots,
            state_canonical_slot_indices=getattr(config, "state_canonical_slot_indices", None),
            action_canonical_slot_indices=getattr(config, "action_canonical_slot_indices", None),
            state_horizon=len(config.observation_indices),
            action_horizon=len(config.action_indices),
        )
    )
    return ComposedModalityTransform(transforms=transforms)


def _concat_and_gr00t(
    *,
    video_keys: list[str],
    state_keys: list[str],
    action_keys: list[str],
    state_canonical_slots: dict[str, str] | None,
    action_canonical_slots: dict[str, str] | None,
    state_horizon: int,
    action_horizon: int,
    state_canonical_slot_indices: dict[str, list[int]] | None = None,
    action_canonical_slot_indices: dict[str, list[int]] | None = None,
    state_eef_rotation_source: dict[str, str] | None = None,
    action_eef_rotation_source: dict[str, str] | None = None,
    legacy_max_state_dim: int = 64,
    legacy_max_action_dim: int = 32,
) -> list:
    """Build the (ConcatTransform, GR00TTransform) tail of a DataConfig pipeline.

    When canonical alignment is enabled (``is_action_alignment_enabled()``),
    routes the modality through canonical scatter using the supplied slot maps;
    otherwise falls back to the dataset's legacy concat path with the same
    max_*_dim it had before this feature existed.
    """
    if is_action_alignment_enabled():
        return [
            ConcatTransform(
                video_concat_order=video_keys,
                state_canonical_slots=state_canonical_slots,
                action_canonical_slots=action_canonical_slots,
                state_canonical_slot_indices=state_canonical_slot_indices or {},
                action_canonical_slot_indices=action_canonical_slot_indices or {},
                state_eef_rotation_source=state_eef_rotation_source or {},
                action_eef_rotation_source=action_eef_rotation_source or {},
            ),
            # NB: pi-shell rotvec continuity for the shared rotvec EE path is
            # handled per-key BEFORE normalization in the state-relative ON
            # path, while values are still in physical radians. Placing it
            # here (after normalize + scatter) was a no-op: from_rotvec on
            # values already squashed into [-1, 1] treats every frame as a
            # tiny (< pi) rotation and the double-cover check never fires.
            GR00TTransform(
                state_horizon=state_horizon,
                action_horizon=action_horizon,
                max_state_dim=get_canonical_state_dim(),
                max_action_dim=get_canonical_action_dim(),
            ),
        ]
    return [
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        GR00TTransform(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=legacy_max_state_dim,
            max_action_dim=legacy_max_action_dim,
        ),
    ]


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(30))

    def __init__(self):
        self.set_relative_action_keys()
        super().__init__()

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def set_relative_action_keys(self):
        action_config = MODALITY_CONFIGS["gr1"]["action"]
        if action_config.action_configs is None:
            raise ValueError("Action configs are not set")
        relative_action_info = [
            {
                "key": f"action.{key}",  # 添加 "action." 前缀以匹配 apply_to 格式
                "type": cfg.type,
                "format": cfg.format,
            }
            for key, cfg in zip(action_config.modality_keys, action_config.action_configs)
            if cfg.rep == ActionRepresentation.RELATIVE
        ]
        self.relative_action_keys = [info["key"] for info in relative_action_info]
        self.relative_action_info = relative_action_info
        # self.relative_action_keys = []
        # self.relative_action_info = []

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoResizePad(apply_to=self.video_keys, size=224, interpolation="linear", fill_value=0.0),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            RelativeActionTransform(
                apply_to=self.relative_action_keys,
                relative_action_info=self.relative_action_info,
                reference_index=0,
                in_place=True,
            ),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=_norm_modes_for_class(type(self).__name__, self.state_keys),
            ),
            # action transforms
            # StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=_norm_modes_for_class(type(self).__name__, self.action_keys),
                relative_action_keys=self.relative_action_keys,
                action_horizon=len(self.action_indices),  # 与 data_config 中的 action chunk 长度一致
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistAbsoluteDataConfig(FourierGr1ArmsWaistDataConfig):
    """RoboCasa GR1 absolute joint commands, matching the LDA-1B target convention.

    Unlike ``FourierGr1ArmsWaistDataConfig``, this variant deliberately omits
    ``RelativeActionTransform``. The 29 active dimensions are scattered into
    physical joint/dex/waist canonical slots; all remaining dimensions stay
    masked. State remains a normalized conditioning signal rather than an
    action target.
    """

    action_indices = list(range(16))
    state_canonical_slot_indices = {
        f"state.{side}_hand": list(ROBOCASA_HAND_SHARPA_INDICES) for side in ("left", "right")
    }
    action_canonical_slot_indices = {
        f"action.{side}_hand": list(ROBOCASA_HAND_SHARPA_INDICES) for side in ("left", "right")
    }
    state_canonical_slots = {
        "state.left_arm": "LEFT_ARM_JOINT",
        "state.right_arm": "RIGHT_ARM_JOINT",
        "state.left_hand": "LEFT_DEX_JOINT",
        "state.right_hand": "RIGHT_DEX_JOINT",
        "state.waist": "WAIST_POSE",
    }
    action_canonical_slots = {
        "action.left_arm": "LEFT_ARM_JOINT",
        "action.right_arm": "RIGHT_ARM_JOINT",
        "action.left_hand": "LEFT_DEX_JOINT",
        "action.right_hand": "RIGHT_DEX_JOINT",
        "action.waist": "WAIST_POSE",
    }

    def __init__(self):
        # Do not call the parent constructor: it registers the arm fields as
        # relative actions from the legacy GR1 modality definition.
        self.relative_action_keys = []
        self.relative_action_info = []

    def transform(self) -> ModalityTransform:
        transforms = _standard_video_transforms(self.video_keys)
        transforms.extend(
            [
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=_norm_modes_for_class(type(self).__name__, self.state_keys),
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=_norm_modes_for_class(type(self).__name__, self.action_keys),
                ),
            ]
        )
        transforms.extend(
            _concat_and_gr00t(
                video_keys=self.video_keys,
                state_keys=self.state_keys,
                action_keys=self.action_keys,
                state_canonical_slots=self.state_canonical_slots,
                action_canonical_slots=self.action_canonical_slots,
                state_canonical_slot_indices=self.state_canonical_slot_indices,
                action_canonical_slot_indices=self.action_canonical_slot_indices,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
            )
        )
        return ComposedModalityTransform(transforms=transforms)


class RobocasaGr1EefDataConfig(FourierGr1ArmsWaistDataConfig):
    """RoboCasa GR1 poses converted offline to camera-frame xyz + rotvec."""

    state_canonical_slot_indices = {
        f"state.{side}_hand": list(ROBOCASA_HAND_SHARPA_INDICES) for side in ("left", "right")
    }
    action_canonical_slot_indices = {
        f"action.{side}_hand": list(ROBOCASA_HAND_SHARPA_INDICES) for side in ("left", "right")
    }
    state_canonical_slots = {
        "state.left_arm": "LEFT_EEF_POSE",
        "state.right_arm": "RIGHT_EEF_POSE",
        "state.left_hand": "LEFT_DEX_JOINT",
        "state.right_hand": "RIGHT_DEX_JOINT",
        "state.waist": "WAIST_POSE",
    }
    action_canonical_slots = {
        "action.left_arm": "LEFT_EEF_POSE",
        "action.right_arm": "RIGHT_EEF_POSE",
        "action.left_hand": "LEFT_DEX_JOINT",
        "action.right_hand": "RIGHT_DEX_JOINT",
        "action.waist": "WAIST_POSE",
    }

    def __init__(self):
        # Relative-action semantics come from MODALITY_CONFIGS through the
        # dataset metadata/statistics path. Runtime uses the canonical EEF
        # slots below, so the legacy GR1 NON_EEF key list must not be reused.
        self.relative_action_keys = []
        self.relative_action_info = []

    def transform(self) -> ModalityTransform:
        return _rotvec_eef_transform(
            self,
            state_canonical_slots=self.state_canonical_slots,
            action_canonical_slots=self.action_canonical_slots,
        )


class RobocasaGr1AugAlignedDataConfig(FourierGr1ArmsWaistAbsoluteDataConfig):
    """71-D action supervision with only 29 joint dimensions exposed as model state.

    Absolute joint commands coexist with spatial-relative grip-site EEF poses
    and absolute geometric fingertip positions. Both use a fixed base-origin
    pseudo-camera frame. Raw state EEFs are loaded ONLY to construct relative
    action labels; omitting them from state_canonical_slots prevents a pose
    dependency at model inference. The converted dataset retains full state.
    """

    relative_eef_with_absolute_joint = True
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.left_eef_pose",
        "state.right_eef_pose",
    ]
    action_keys = [key.replace("state.", "action.") for key in state_keys] + [
        "action.left_finger_positions",
        "action.right_finger_positions",
    ]
    state_canonical_slots = dict(FourierGr1ArmsWaistAbsoluteDataConfig.state_canonical_slots)
    action_canonical_slots = {
        **FourierGr1ArmsWaistAbsoluteDataConfig.action_canonical_slots,
        "action.left_eef_pose": "LEFT_EEF_POSE",
        "action.right_eef_pose": "RIGHT_EEF_POSE",
        "action.left_finger_positions": "LEFT_DEX_HAND",
        "action.right_finger_positions": "RIGHT_DEX_HAND",
    }

    def transform(self) -> ModalityTransform:
        """Derive relative EEF labels before normalization and scatter only joint state."""
        return _rotvec_eef_transform(
            self,
            state_canonical_slots=self.state_canonical_slots,
            action_canonical_slots=self.action_canonical_slots,
        )


class RoboTwinDataConfig(BaseDataConfig):
    video_keys = ["video.image_high", "video.image_left_wrist", "video.image_right_wrist"]
    state_keys = [
        "state.right_arm",
        "state.right_gripper",
        "state.left_arm",
        "state.left_gripper",
    ]
    action_keys = [
        "action.right_arm",
        "action.right_gripper",
        "action.left_arm",
        "action.left_gripper",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(30))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        state_canonical_slots = {
            "state.right_arm": "RIGHT_EEF_POSE",
            "state.right_gripper": "RIGHT_GRIPPER",
            "state.left_arm": "LEFT_EEF_POSE",
            "state.left_gripper": "LEFT_GRIPPER",
        }
        action_canonical_slots = {
            "action.right_arm": "RIGHT_EEF_POSE",
            "action.right_gripper": "RIGHT_GRIPPER",
            "action.left_arm": "LEFT_EEF_POSE",
            "action.left_gripper": "LEFT_GRIPPER",
        }
        transforms = [
            # video transforms - 为每个视角分别应用，避免不同尺寸的 concatenation 错误
            # 第一个视角：image_high
            VideoToTensor(apply_to=[self.video_keys[0]]),
            # VideoCrop(apply_to=[self.video_keys[0]], scale=0.95),
            # VideoResize(apply_to=[self.video_keys[0]], height=224, width=224, interpolation="linear"),
            VideoResizePad(apply_to=[self.video_keys[0]], size=224, interpolation="linear", fill_value=0.0),
            VideoColorJitter(
                apply_to=[self.video_keys[0]],
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=[self.video_keys[0]]),
            # 第二个视角：image_left_wrist
            VideoToTensor(apply_to=[self.video_keys[1]]),
            # VideoCrop(apply_to=[self.video_keys[1]], scale=0.95),
            # VideoResize(apply_to=[self.video_keys[1]], height=224, width=224, interpolation="linear"),
            VideoResizePad(apply_to=[self.video_keys[1]], size=224, interpolation="linear", fill_value=0.0),
            VideoColorJitter(
                apply_to=[self.video_keys[1]],
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=[self.video_keys[1]]),
            # 第三个视角：image_right_wrist
            VideoToTensor(apply_to=[self.video_keys[2]]),
            # VideoCrop(apply_to=[self.video_keys[2]], scale=0.95),
            # VideoResize(apply_to=[self.video_keys[2]], height=224, width=224, interpolation="linear"),
            VideoResizePad(apply_to=[self.video_keys[2]], size=224, interpolation="linear", fill_value=0.0),
            VideoColorJitter(
                apply_to=[self.video_keys[2]],
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=[self.video_keys[2]]),
        ]
        transforms.extend(
            _state_action_relative_block(
                state_keys=self.state_keys,
                action_keys=self.action_keys,
                class_name=type(self).__name__,
                action_horizon=len(self.action_indices),
                action_canonical_slots=action_canonical_slots,
            )
        )
        transforms.extend(
            _concat_and_gr00t(
                video_keys=self.video_keys,
                state_keys=self.state_keys,
                action_keys=self.action_keys,
                state_canonical_slots=state_canonical_slots,
                action_canonical_slots=action_canonical_slots,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
            )
        )
        return ComposedModalityTransform(transforms=transforms)


class RoboTwinAugAlignedDataConfig(RoboTwinDataConfig):
    """Absolute arm/gripper state and joint plus camera-spatial EEF supervision.

    Load reference EEFs only to derive labels; canonical state excludes poses.
    """

    relative_eef_with_absolute_joint = True
    state_keys = [
        "state.left_arm_joint",
        "state.left_gripper",
        "state.right_arm_joint",
        "state.right_gripper",
        "state.left_eef_pose",
        "state.right_eef_pose",
    ]
    action_keys = [key.replace("state.", "action.") for key in state_keys]
    state_canonical_slots = {
        "state.left_arm_joint": "LEFT_ARM_JOINT",
        "state.right_arm_joint": "RIGHT_ARM_JOINT",
        "state.left_gripper": "LEFT_GRIPPER",
        "state.right_gripper": "RIGHT_GRIPPER",
    }
    action_canonical_slots = {
        **{key.replace("state.", "action."): slot for key, slot in state_canonical_slots.items()},
        "action.left_eef_pose": "LEFT_EEF_POSE",
        "action.right_eef_pose": "RIGHT_EEF_POSE",
    }

    def transform(self) -> ModalityTransform:
        """Normalize absolute joints and relative EEFs before canonical scatter."""
        return _rotvec_eef_transform(
            self,
            state_canonical_slots=self.state_canonical_slots,
            action_canonical_slots=self.action_canonical_slots,
        )


ROBOT_TYPE_CONFIG_MAP = {
    "my_robotwin": RoboTwinDataConfig(),
    "robotwin_aug_aligned": RoboTwinAugAlignedDataConfig(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_waist_absolute": FourierGr1ArmsWaistAbsoluteDataConfig(),
    "robocasa_gr1_eef": RobocasaGr1EefDataConfig(),
    "robocasa_gr1_aug_aligned": RobocasaGr1AugAlignedDataConfig(),
}
