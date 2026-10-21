# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Route canonical metadata and validate resume normalizers before overwriting run statistics.

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.canonical_action import (
    set_action_alignment_enabled,
    set_canonical_dims,
    set_relative_eef_frame,
    set_state_relative_eef_enabled,
)

logger = get_logger(__name__)


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _ee_relative_actions_enabled(data_cfg) -> bool:
    requested = data_cfg.get("enable_relative_action", False) not in (
        False,
        "False",
        "false",
        None,
        "",
        0,
    )
    if not requested:
        return False
    configured_modes = data_cfg.get("relative_action_control_modes", None)
    if configured_modes in (None, False, "False", ""):
        return True
    if isinstance(configured_modes, str):
        configured_modes = [configured_modes]
    return "ee" in {str(mode) for mode in configured_modes}


def _freeze_gc_before_fork():
    """Move every currently-tracked Python object into the GC frozen set so
    DataLoader workers (forked from this process) never write refcounts back
    onto the metadata pages they inherit.

    Background: with `num_workers>0` PyTorch fork()s the main process. After
    fork, every refcount increment on an inherited object dirties the
    underlying 4 KB page and triggers Copy-on-Write. With ~70 LeRobotSingleDataset
    instances each holding multi-hundred-MB `_all_steps` int64 arrays plus
    pandas DataFrame metadata, worker RSS climbs steadily and the OOM killer
    eventually sends SIGKILL — exactly the symptom we hit at ~50k steps with
    large multi-dataset runs.

    Calling gc.freeze() here is the Instagram trick: subsequent GC scans skip
    these objects entirely, so refcount churn no longer dirties their pages.
    """
    gc.collect()
    if hasattr(gc, "freeze"):
        gc.freeze()


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def build_dataloader(cfg, dataset_py="lerobot_datasets"):

    if dataset_py == "r1pro_merged_dataset":
        from starVLA.dataloader.r1pro_merged_dataset import build_merged_dataloader

        return build_merged_dataloader(cfg)

    if dataset_py == "pick_fruit_dataset":
        from starVLA.dataloader.pick_fruit_dataset import build_pick_fruit_dataloader

        return build_pick_fruit_dataloader(cfg)

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data

        action_model_cfg = cfg.framework.get("action_model", {}) if hasattr(cfg, "framework") else {}
        action_space_cfg = action_model_cfg.get("action_space", {}) if action_model_cfg is not None else {}
        action_representation = action_space_cfg.get("representation", None)
        prompt_format = str(vla_dataset_cfg.get("control_mode_prompt_format", "legacy"))
        if prompt_format == "ee_representation_v1":
            if action_representation not in ("relative", "absolute"):
                raise ValueError(
                    "control_mode_prompt_format='ee_representation_v1' requires "
                    "framework.action_model.action_space.representation to be 'relative' or 'absolute'"
                )
            vla_dataset_cfg.action_representation = str(action_representation)
        elif prompt_format != "legacy":
            raise ValueError(
                f"Unsupported control_mode_prompt_format={prompt_format!r}; "
                "expected 'legacy' or 'ee_representation_v1'"
            )

        # Toggle canonical action/state alignment on/off process-wide
        # BEFORE building any DataConfig instance, because DataConfig.transform()
        # reads this flag to decide whether to install canonical_slots and the
        # EefPoseAlignTransform. Default False keeps legacy 32-D behavior.
        enable_alignment = bool(vla_dataset_cfg.get("enable_action_alignment", False))
        set_action_alignment_enabled(enable_alignment)

        # Toggle state-relative EEF action pipeline. When ON, EEF-slot action
        # keys are expressed relative to the current-state EEF pose (SE(3)
        # T_ref^{-1} * T_k) and normalized against precomputed relative_action
        # statistics. Requires enable_relative_action=True to bring those
        # stats along; otherwise the normalizer would fall back to absolute
        # stats and produce nonsense scales.
        enable_state_rel_eef = bool(vla_dataset_cfg.get("enable_state_relative_eef", False))
        if prompt_format == "ee_representation_v1":
            expected_relative = action_representation == "relative"
            ee_relative_actions = _ee_relative_actions_enabled(vla_dataset_cfg)
            if enable_state_rel_eef != expected_relative or ee_relative_actions != expected_relative:
                raise ValueError(
                    "EE action prompt semantics disagree with the dataloader transform: "
                    f"representation={action_representation!r}, "
                    f"enable_state_relative_eef={enable_state_rel_eef}, "
                    f"ee_relative_actions={ee_relative_actions}"
                )
        if enable_state_rel_eef:
            assert enable_alignment, (
                "enable_state_relative_eef=True requires enable_action_alignment=True "
                "because the state-relative EEF pipeline is installed through canonical EEF slots."
            )
            assert bool(vla_dataset_cfg.get("enable_relative_action", False)), (
                "enable_state_relative_eef=True requires enable_relative_action=True "
                "so relative_action statistics get generated."
            )
        set_state_relative_eef_enabled(enable_state_rel_eef)
        relative_eef_frame = str(vla_dataset_cfg.get("relative_eef_frame", "eef"))
        set_relative_eef_frame(relative_eef_frame)
        enable_camera_motion_compensation = bool(vla_dataset_cfg.get("enable_camera_motion_compensation", True))

        # Sync canonical action/state dim with the hyperparam-driven model head
        # so dataloader output and model input stay shape-matched. If the user
        # does not set --framework.action_model.action_dim / state_dim, the
        # static CANONICAL_*_DIM defaults (130/130) are preserved.
        head_action_dim = None
        head_state_dim = None
        if hasattr(cfg, "framework"):
            if cfg.framework.get("action_model", None) is not None:
                head_action_dim = cfg.framework.action_model.get("action_dim", None)
                head_state_dim = cfg.framework.action_model.get("state_dim", None)
            elif cfg.framework.get("latent_action", None) is not None:
                head_action_dim = cfg.framework.latent_action.get("action_dim", None)
                head_state_dim = cfg.framework.latent_action.get("state_dim", None)
        set_canonical_dims(action_dim=head_action_dim, state_dim=head_state_dim)
        if dist.is_initialized() and dist.get_rank() == 0:
            if head_action_dim or head_state_dim:
                logger.info(
                    f"canonical dims overridden by hyperparam: action_dim={head_action_dim}, state_dim={head_state_dim}"
                )
        if dist.is_initialized() and dist.get_rank() == 0:
            compensation_scope = (
                "active for camera-frame relative EEF" if enable_state_rel_eef else "inactive without state-relative EEF"
            )
            logger.info(
                "action_alignment="
                f"{'ON (canonical 130-D/effective head dim)' if enable_alignment else 'OFF (legacy 32-D)'}"
            )
            logger.info(
                "camera_motion_compensation="
                f"{'ON' if enable_camera_motion_compensation else 'OFF'} "
                f"({compensation_scope}); relative_eef_frame={relative_eef_frame}"
            )

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        expected_statistics_path = vla_dataset_cfg.get("expected_statistics_path", None)
        if expected_statistics_path not in (None, False, "", "False", "false"):
            from starVLA.dataloader.gr00t_lerobot.resume_statistics import validate_resume_statistics

            validate_resume_statistics(vla_dataset, expected_statistics_path)

        num_workers = int(cfg.datasets.vla_data.get("num_workers", 8))
        if num_workers > 0:
            prefetch_factor = int(cfg.datasets.vla_data.get("prefetch_factor", 1))
            if prefetch_factor <= 0:
                raise ValueError(f"prefetch_factor must be positive, got {prefetch_factor}")
            persistent_workers = cfg.datasets.vla_data.get("persistent_workers", False) not in (
                False,
                "False",
                None,
            )
        else:
            # PyTorch requires both options to be disabled in single-process loading mode.
            prefetch_factor = None
            persistent_workers = False
        print(
            f"vla_num_workers: {num_workers}, prefetch_factor: {prefetch_factor}, "
            f"persistent_workers: {persistent_workers}"
        )
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            # shuffle=True
        )
        if _is_main_process():

            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        _freeze_gc_before_fork()
        return vla_train_dataloader
    raise ValueError(f"Unsupported dataset_py={dataset_py!r}")
