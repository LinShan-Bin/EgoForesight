#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Precompute train-consistent spatial EEF statistics for aligned RoboTwin.
"""Compute the existing versioned relative-action cache over every converted episode."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_OUTPUT = "data/robotwin/aligned/aloha_agilex-clean_50_single_task-joint-aug"


def compute(task: Path, horizon: int) -> str:
    """Populate exactly the cache signature used by the aligned training recipe."""
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    from starVLA.dataloader.gr00t_lerobot.relative_action_stats.relative_stats import generate_rel_stats

    if not (task / "meta/joint_augmentation.json").is_file():
        raise ValueError(f"Incomplete converted task: {task}")
    generate_rel_stats(
        task,
        EmbodimentTag.ROBOTWIN_AUG_ALIGNED,
        max_episodes=-1,
        temporal_downsample_stride=1,
        action_index_shift=0,
        action_delta_indices=list(range(horizon)),
        camera_motion_compensation=True,
        relative_eef_frame="camera_spatial",
        static_camera_frame=True,
    )
    return task.name


def main() -> None:
    """Process independent task caches in parallel without video decoding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()
    if args.horizon < 1 or args.workers < 1:
        parser.error("horizon and workers must be positive")
    tasks = (
        [args.root / task for task in args.tasks]
        if args.tasks
        else sorted(p for p in args.root.iterdir() if (p / "meta/joint_augmentation.json").is_file())
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compute, task, args.horizon) for task in tasks]
        for future in as_completed(futures):
            print(f"Relative statistics ready: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
