# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications Copyright 2026 Chubin Zhang.
"""Official RoboTwin HDF5 -> aligned EEF -> optional mirrors -> joint+EEF repositories.

Never calls a VLM, prunes demonstrations, or changes official instructions implicitly.
Each finalized task is a resumable unit. Incomplete outputs are rejected, not deleted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.robotwin_aligned.build_lerobot import OUTPUT_ROOT, RAW_ROOT, SETTING_OUTPUTS
from scripts.robotwin_aligned.common import ROBOTWIN_ALIGNED_PIPELINE_VERSION, VIDEO_KEYS


def repository_is_complete(repo: Path, expected: int) -> bool:
    """Check the completed manifest and all episode/video files before resuming."""
    try:
        info = json.loads((repo / "meta/info.json").read_text())
        return (
            info["robotwin_aligned_pipeline_version"] == ROBOTWIN_ALIGNED_PIPELINE_VERSION
            and info["total_episodes"] == expected
            and info["total_frames"] > 0
            and len(list((repo / "data").glob("chunk-*/*.parquet"))) == expected
            and len(list((repo / "videos").glob("chunk-*/*/*.mp4"))) == expected * len(VIDEO_KEYS)
        )
    except (OSError, KeyError, ValueError):
        return False


def run(module: str, *arguments: str) -> None:
    """Use the active environment and propagate failures; no background services."""
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    """Build all requested official tasks with native geometry and explicit mirroring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--settings", nargs="+", choices=SETTING_OUTPUTS, default=list(SETTING_OUTPUTS))
    parser.add_argument("--tasks", nargs="+")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--mirror-prompt-dir", type=Path, help="<setting>/<task>.json per-episode prompt maps")
    policy.add_argument("--swap-direction-words", action="store_true", help="Opt in to geometric/word-only mirrors")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--horizon", type=int, default=48)
    args = parser.parse_args()
    tasks = args.tasks or sorted(p.name for p in args.raw_root.iterdir() if p.is_dir())
    if not tasks or args.horizon < 1 or args.workers < 1:
        parser.error("Need nonempty tasks and positive horizon/workers")
    for setting in args.settings:
        source_root = args.output_root / SETTING_OUTPUTS[setting]
        for task in tasks:
            raw = args.raw_root / task / f"aloha-agilex_{setting}" / "data"
            count = len(list(raw.glob("episode*.hdf5")))
            if not count:
                raise FileNotFoundError(f"No official raw episodes: {raw}")
            repo = source_root / task
            if not repository_is_complete(repo, count):
                if repo.exists():
                    raise FileExistsError(f"Inspect incomplete or differently sized output before rebuilding: {repo}")
                run(
                    "scripts.robotwin_aligned.build_lerobot",
                    "--raw-root",
                    args.raw_root,
                    "--output-root",
                    args.output_root,
                    "--task",
                    task,
                    "--setting",
                    setting,
                )
            if args.mirror_prompt_dir or args.swap_direction_words:
                mirror = repo.with_name(task + "-mirrored")
                if not repository_is_complete(mirror, count):
                    if mirror.exists():
                        raise FileExistsError(f"Inspect incomplete mirror output: {mirror}")
                    policy_args = (
                        ["--prompt-map", args.mirror_prompt_dir / setting / f"{task}.json"]
                        if args.mirror_prompt_dir
                        else ["--swap-direction-words"]
                    )
                    run("scripts.robotwin_aligned.mirror", "--source", repo, "--output", mirror, *policy_args)
        output = source_root.with_name(source_root.name + "-joint-aug")
        run(
            "scripts.robotwin_aligned.augment_joints",
            "--source-root",
            source_root,
            "--output-root",
            output,
            "--raw-root",
            args.raw_root,
            "--raw-setting",
            setting,
            "--workers",
            args.workers,
            "--variants",
            "both" if args.mirror_prompt_dir or args.swap_direction_words else "original",
            "--tasks",
            *tasks,
        )
        if not args.skip_stats:
            from scripts.robotwin_aligned.precompute_joint_stats import compute

            suffixes = ("", "-mirrored") if args.mirror_prompt_dir or args.swap_direction_words else ("",)
            for task in tasks:
                for suffix in suffixes:
                    compute(output / (task + suffix), args.horizon)


if __name__ == "__main__":
    main()
