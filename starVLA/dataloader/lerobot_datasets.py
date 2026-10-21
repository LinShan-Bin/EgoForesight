# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import copy
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.base_supervision import get_base_supervision_policy
from starVLA.dataloader.gr00t_lerobot.canonical_action import (
    get_relative_eef_frame,
    is_action_alignment_enabled,
    is_state_relative_eef_enabled,
    set_relative_eef_frame,
    set_state_relative_eef_enabled,
)
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.quality_index import get_dataset_quality_filter


def collate_fn(batch):
    return batch


VALID_CONTROL_MODES = {"joint", "ee"}
VALID_EEF_FRAMES = {"camera", "robot_base"}
VALID_LANGUAGE_KEY_MODES = {"all", "task_only"}


def _get_dataset_override(mapping: dict, data_name: str) -> str | None:
    for key, value in mapping.items():
        key = str(key)
        if data_name == key or data_name.endswith(key) or Path(data_name).name == key:
            return str(value)
    return None


_VIDEO_BACKEND_ROBOT_TYPE_ALIASES = {}


def _select_video_backend(effective_data_cfg: dict, data_config, data_name: str, robot_type: str) -> str:
    video_backend = str(effective_data_cfg.get("video_backend", getattr(data_config, "video_backend", "decord")))
    robot_overrides = effective_data_cfg.get("video_backend_by_robot_type", {}) or {}
    dataset_overrides = effective_data_cfg.get("video_backend_by_dataset", {}) or {}
    override_robot_type = robot_type
    if override_robot_type not in robot_overrides:
        override_robot_type = _VIDEO_BACKEND_ROBOT_TYPE_ALIASES.get(robot_type, robot_type)
    if override_robot_type in robot_overrides:
        video_backend = str(robot_overrides[override_robot_type])
    dataset_backend = _get_dataset_override(dataset_overrides, data_name)
    if dataset_backend is not None:
        video_backend = dataset_backend
    return video_backend


def _relative_action_enabled_for_dataset(
    data_cfg: dict,
    control_mode: str | None,
    robot_type: str | None = None,
) -> bool:
    """Apply optional control-mode and robot-type scopes to relative actions."""
    requested = data_cfg.get("enable_relative_action", False) not in (False, "False", "false", None, "", 0)
    if not requested:
        return False

    excluded_robot_types = data_cfg.get("relative_action_excluded_robot_types", ()) or ()
    if isinstance(excluded_robot_types, str):
        excluded_robot_types = [excluded_robot_types]
    if robot_type is not None and robot_type in {str(value) for value in excluded_robot_types}:
        return False

    configured_modes = data_cfg.get("relative_action_control_modes", None)
    if configured_modes in (None, False, "False", ""):
        return True
    if isinstance(configured_modes, str):
        configured_modes = [configured_modes]
    enabled_modes = {str(mode) for mode in configured_modes}
    invalid_modes = enabled_modes - VALID_CONTROL_MODES
    if invalid_modes:
        raise ValueError(
            "relative_action_control_modes must contain only "
            f"{sorted(VALID_CONTROL_MODES)}, got {sorted(invalid_modes)}"
        )
    return control_mode in enabled_modes


def _quality_index_enabled_for_dataset(data_cfg: dict, robot_type: str) -> bool:
    """Allow selected robot families to bypass a globally enabled sample-quality index."""
    enabled = data_cfg.get("dataset_quality_index_enabled", False) not in (
        False,
        "False",
        "false",
        None,
        "",
        0,
    )
    if not enabled:
        return False
    excluded_robot_types = data_cfg.get("dataset_quality_index_excluded_robot_types", []) or []
    if isinstance(excluded_robot_types, str):
        excluded_robot_types = [excluded_robot_types]
    return str(robot_type) not in {str(value) for value in excluded_robot_types}


def _apply_language_key_mode(
    modality_config: dict[str, ModalityConfig], language_key_mode: str | None
) -> dict[str, ModalityConfig]:
    """Optionally omit subtask annotations before any parquet/metadata lookup occurs."""
    mode = "all" if language_key_mode in (None, "") else str(language_key_mode)
    if mode not in VALID_LANGUAGE_KEY_MODES:
        raise ValueError(f"language_key_mode must be one of {sorted(VALID_LANGUAGE_KEY_MODES)}, got {mode!r}")
    if mode == "all" or "language" not in modality_config:
        return modality_config

    language_config = modality_config["language"]
    task_keys = [key for key in language_config.modality_keys if "subtask" not in key.lower()]
    if not task_keys:
        raise ValueError(
            "language_key_mode='task_only' requires at least one non-subtask language key, "
            f"got {language_config.modality_keys}"
        )

    filtered_config = dict(modality_config)
    filtered_config["language"] = ModalityConfig(
        delta_indices=list(language_config.delta_indices),
        modality_keys=task_keys,
    )
    return filtered_config


def _enable_base_supervision_fields(data_config):
    """Copy-on-write activation of optional base state/action modality keys."""
    base_action_key = getattr(data_config, "base_action_key", None)
    if not base_action_key:
        raise ValueError(f"{type(data_config).__name__} has a base-supervision policy but no base_action_key")

    configured = copy.copy(data_config)
    configured.action_keys = list(configured.action_keys)
    if base_action_key not in configured.action_keys:
        configured.action_keys.append(base_action_key)

    base_state_key = getattr(configured, "base_state_key", None)
    if base_state_key:
        configured.state_keys = list(configured.state_keys)
        if base_state_key not in configured.state_keys:
            configured.state_keys.append(base_state_key)
    return configured


def _parse_mixture_item(
    item: tuple,
) -> tuple[str, float, str, int, int, str | None, str | None, str | None, int]:
    """Parse a mixture entry tuple by STRICT positional schema.

    Canonical (full) 9-field schema:
       index 0: data_name              (str)
       index 1: weight                 (float)
       index 2: robot_type             (str, key into ROBOT_TYPE_CONFIG_MAP)
       index 3: base_step_stride       (int, >=1)
       index 4: temporal_downsample_stride (int, >=1)
       index 5: control_mode           (str | None)
       index 6: language_embodiment    (str | None)
       index 7: eef_frame              (str | None)
       index 8: action_index_shift     (int >=0)

    Shorter tuples (3..8) are accepted for backwards compatibility; any
    omitted trailing field defaults as follows:
      base_step_stride=1, temporal_downsample_stride=1, control_mode=None,
      language_embodiment=None, eef_frame=None, action_index_shift=0.

    Position is now AUTHORITATIVE. To add a new field, append index 9 to
    every entry that wants to set it, and extend this parser with one more
    branch -- type-sensing sentinels are NOT used. None-valued fields are
    skipped at language-prompt assembly time so they never leak into the
    instruction text.
    """
    # Defaults (filled in by short-tuple fall-through)
    base_step_stride = 1
    temporal_downsample_stride = 1
    control_mode = None
    language_embodiment = None
    eef_frame = None
    action_index_shift = 0

    n = len(item)
    if n < 3 or n > 9:
        raise ValueError(f"Invalid mixture item length {n}: {item}")

    # Strict positional unpack
    d_name = item[0]
    d_weight = item[1]
    robot_type = item[2]
    if n >= 4:
        base_step_stride = item[3]
    if n >= 5:
        temporal_downsample_stride = item[4]
    if n >= 6:
        control_mode = item[5]
    if n >= 7:
        language_embodiment = item[6]
    if n >= 8:
        eef_frame = item[7]
    if n >= 9:
        action_index_shift = item[8]

    if control_mode is not None:
        control_mode = str(control_mode).lower()
        if control_mode not in VALID_CONTROL_MODES:
            raise ValueError(
                f"Invalid control_mode '{control_mode}' in mixture item {item}. "
                f"Expected one of {sorted(VALID_CONTROL_MODES)}."
            )

    if language_embodiment is not None:
        language_embodiment = str(language_embodiment).lower()

    if eef_frame is not None:
        eef_frame = str(eef_frame).lower().replace(" ", "_")
        if eef_frame not in VALID_EEF_FRAMES:
            raise ValueError(
                f"Invalid eef_frame '{eef_frame}' in mixture item {item}. "
                f"Expected one of {sorted(VALID_EEF_FRAMES)}."
            )

    action_index_shift = max(0, int(action_index_shift))

    return (
        d_name,
        float(d_weight),
        str(robot_type),
        max(1, int(temporal_downsample_stride)),
        max(1, int(base_step_stride)),
        control_mode,
        language_embodiment,
        eef_frame,
        action_index_shift,
    )


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    temporal_downsample_stride: int = 1,
    base_step_stride: int = 1,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    control_mode: str | None = None,
    language_embodiment: str | None = None,
    eef_frame: str | None = None,
    action_index_shift: int = 0,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """

    # Build a per-dataset cfg copy so robot-type-specific overrides do not mutate the global cfg.
    if data_cfg is None:
        effective_data_cfg = {}
    elif OmegaConf.is_config(data_cfg):
        effective_data_cfg = OmegaConf.to_container(data_cfg, resolve=False)
    else:
        effective_data_cfg = dict(data_cfg)

    dataset_overrides = effective_data_cfg.get("dataset_config_by_robot_type", {}) or {}
    effective_data_cfg.update(copy.deepcopy(dataset_overrides.get(robot_type, {})))

    effective_data_cfg["dataset_quality_index_enabled"] = _quality_index_enabled_for_dataset(
        effective_data_cfg,
        robot_type,
    )
    effective_data_cfg["temporal_downsample_stride"] = max(1, int(temporal_downsample_stride))
    effective_data_cfg["base_step_stride"] = max(1, int(base_step_stride))
    effective_data_cfg["action_index_shift"] = max(0, int(action_index_shift))

    base_supervision_policy = get_base_supervision_policy(data_name)
    if base_supervision_policy is not None and not is_action_alignment_enabled():
        print(
            f"Base supervision for {Path(data_name).name} requires enable_action_alignment=True; "
            "leaving the legacy action layout unchanged"
        )
        base_supervision_policy = None
    effective_data_cfg["base_supervision_policy"] = base_supervision_policy

    random_future_range = effective_data_cfg.get("image_pair_random_future_delta_range", None)
    max_random_future_delta = None
    if random_future_range not in (None, False, "False"):
        random_future_range = list(random_future_range)
        if len(random_future_range) != 2:
            raise ValueError(
                "image_pair_random_future_delta_range must contain exactly two values, " f"got {random_future_range}"
            )
        max_random_future_delta = max(int(random_future_range[0]), int(random_future_range[1]))

    configured_action_horizon = effective_data_cfg.get("action_horizon", None)
    action_horizon = int(configured_action_horizon) if configured_action_horizon is not None else None
    if max_random_future_delta is not None:
        action_horizon = max(int(action_horizon or 0), max_random_future_delta)
    if action_horizon is not None:
        action_horizon = max(1, int(action_horizon))
        effective_data_cfg["action_horizon"] = action_horizon

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    is_raw_native = bool(getattr(data_config, "raw_native", False))
    if base_supervision_policy is not None:
        data_config = _enable_base_supervision_fields(data_config)
    if action_horizon is not None:
        if base_supervision_policy is None:
            data_config = copy.copy(data_config)
        data_config.action_indices = list(range(action_horizon))
    modality_config = data_config.modality_config()
    modality_config = _apply_language_key_mode(
        modality_config,
        effective_data_cfg.get("language_key_mode", "all"),
    )
    enable_relative_action = _relative_action_enabled_for_dataset(
        effective_data_cfg,
        control_mode,
        robot_type,
    )
    global_state_relative_eef = is_state_relative_eef_enabled()
    global_relative_eef_frame = get_relative_eef_frame()
    dataset_state_relative_eef = global_state_relative_eef and enable_relative_action
    effective_data_cfg["enable_state_relative_eef"] = dataset_state_relative_eef
    dataset_relative_eef_frame = str(effective_data_cfg.get("relative_eef_frame", global_relative_eef_frame))
    if (control_mode == "ee" or bool(getattr(data_config, "relative_eef_with_absolute_joint", False))) and str(
        effective_data_cfg.get("control_mode_prompt_format", "legacy")
    ) == "ee_representation_v1":
        effective_data_cfg["action_representation"] = "relative" if dataset_state_relative_eef else "absolute"
    set_state_relative_eef_enabled(dataset_state_relative_eef)
    set_relative_eef_frame(dataset_relative_eef_frame)
    try:
        transforms = data_config.transform()
    finally:
        set_state_relative_eef_enabled(global_state_relative_eef)
        set_relative_eef_frame(global_relative_eef_frame)
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, "
            f"using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = _select_video_backend(effective_data_cfg, data_config, data_name, robot_type)
    quality_filter = None
    quality_filter_config = effective_data_cfg.get("quality_filter_config", None)
    if not is_raw_native and quality_filter_config not in (None, False, "False", ""):
        quality_filter = get_dataset_quality_filter(quality_filter_config, data_name)

    # Per-dataset sampling knobs come from mixture items:
    # - temporal_downsample_stride: scales per-modality delta indices
    # - base_step_stride: subsamples candidate base indices to shorten an epoch
    dataset_class = LeRobotSingleDataset
    dataset_kwargs = {}

    return dataset_class(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        video_backend_kwargs=copy.deepcopy(
            effective_data_cfg.get("video_backend_kwargs", getattr(data_config, "video_backend_kwargs", None)) or {}
        ),
        delete_pause_frame=delete_pause_frame,
        data_cfg=effective_data_cfg,
        enable_relative_action=enable_relative_action,
        control_mode=control_mode,
        language_embodiment=language_embodiment,
        eef_frame=eef_frame,
        robot_type=robot_type,
        end_effector_type=effective_data_cfg.get("end_effector_type", None),
        quality_filter=quality_filter,
        **dataset_kwargs,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    dataset_sampling_alpha = data_cfg.get("dataset_sampling_alpha", None)
    if dataset_sampling_alpha is False or dataset_sampling_alpha in (None, "", "False"):
        dataset_sampling_alpha = None
    else:
        dataset_sampling_alpha = float(dataset_sampling_alpha)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for item in mixture_spec:
        (
            d_name,
            d_weight,
            robot_type,
            temporal_downsample_stride,
            base_step_stride,
            control_mode,
            language_embodiment,
            eef_frame,
            action_index_shift,
        ) = _parse_mixture_item(item)
        dataset_key = (
            d_name,
            robot_type,
            temporal_downsample_stride,
            base_step_stride,
            control_mode,
            language_embodiment,
            eef_frame,
            action_index_shift,
        )
        if dataset_key in included_datasets:
            duplicate_item = (
                d_name,
                d_weight,
                robot_type,
                temporal_downsample_stride,
                base_step_stride,
                control_mode,
                language_embodiment,
                eef_frame,
                action_index_shift,
            )
            print("Skipping Duplicate Dataset: " f"`{duplicate_item}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append(
            (
                d_name,
                d_weight,
                robot_type,
                temporal_downsample_stride,
                base_step_stride,
                control_mode,
                language_embodiment,
                eef_frame,
                action_index_shift,
            )
        )

    dataset_mixture = []
    for (
        d_name,
        d_weight,
        robot_type,
        temporal_downsample_stride,
        base_step_stride,
        control_mode,
        language_embodiment,
        eef_frame,
        action_index_shift,
    ) in filtered_mixture_spec:
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            temporal_downsample_stride=temporal_downsample_stride,
            base_step_stride=base_step_stride,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            control_mode=control_mode,
            language_embodiment=language_embodiment,
            eef_frame=eef_frame,
            action_index_shift=action_index_shift,
        )
        dataset_mixture.append((dataset, d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        dataset_sampling_alpha=dataset_sampling_alpha,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
