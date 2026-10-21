# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Convert raw RoboTwin episodes to camera-calibrated LeRobot repositories.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from scripts.robotwin_aligned.common import (
    ROBOTWIN_HEAD_W2C,
    VIDEO_KEYS,
    Episode,
    LeRobotWriter,
    StaticThresholds,
    compose_state,
    decode_jpeg,
    pose7_wxyz_to_pose6,
    select_non_static_frames,
)

RAW_ROOT = Path("data/robotwin/raw")
OUTPUT_ROOT = Path("data/robotwin/aligned")
SETTING_OUTPUTS = {
    "clean_50": "aloha_agilex-clean_50_single_task",
    "randomized_500": "aloha_agilex-randomized_500_single_task",
}
RAW_CAMERA_KEYS = {
    VIDEO_KEYS[0]: "head_camera",
    VIDEO_KEYS[1]: "left_camera",
    VIDEO_KEYS[2]: "right_camera",
}


def _episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected episode filename: {path}")
    return int(match.group(1))


def _initial_prompt(task_dir: Path, episode_number: int) -> str:
    path = task_dir / f"instructions/episode{episode_number}.json"
    with path.open() as handle:
        instructions = json.load(handle)
    candidates = instructions.get("seen") or instructions.get("unseen") or []
    if not candidates:
        raise ValueError(f"no prompt candidates in {path}")
    return str(candidates[episode_number % len(candidates)]).strip()


def convert_episode(path: Path, prompt: str, thresholds: StaticThresholds) -> tuple[Episode, dict[str, int]]:
    with h5py.File(path, "r") as source:
        left_pose7 = source["endpose/left_endpose"][:]
        right_pose7 = source["endpose/right_endpose"][:]
        left_gripper = source["endpose/left_gripper"][:]
        right_gripper = source["endpose/right_gripper"][:]
        frame_count = len(left_pose7)
        frames = {
            output_key: [decode_jpeg(value) for value in source[f"observation/{raw_key}/rgb"][:]]
            for output_key, raw_key in RAW_CAMERA_KEYS.items()
        }
    lengths = {frame_count, len(right_pose7), len(left_gripper), len(right_gripper)}
    lengths.update(len(value) for value in frames.values())
    if len(lengths) != 1:
        raise ValueError(f"inconsistent source lengths in {path}: {sorted(lengths)}")

    keep = select_non_static_frames(
        left_pose7,
        right_pose7,
        left_gripper,
        right_gripper,
        frames,
        thresholds,
    )
    if len(keep) < 2:
        raise ValueError(f"episode has fewer than two retained frames after filtering: {path}")
    state_indices = np.asarray(keep[:-1], dtype=np.int64)
    action_indices = np.asarray(keep[1:], dtype=np.int64)
    left_pose6 = pose7_wxyz_to_pose6(left_pose7)
    right_pose6 = pose7_wxyz_to_pose6(right_pose7)
    state = compose_state(
        left_pose6[state_indices],
        left_gripper[state_indices],
        right_pose6[state_indices],
        right_gripper[state_indices],
    )
    action = compose_state(
        left_pose6[action_indices],
        left_gripper[action_indices],
        right_pose6[action_indices],
        right_gripper[action_indices],
    )
    output_frames = {key: [frames[key][index] for index in state_indices] for key in VIDEO_KEYS}
    return (
        Episode(state=state, action=action, w2c=ROBOTWIN_HEAD_W2C.copy(), frames=output_frames, prompt=prompt),
        {"source": frame_count, "retained": len(keep), "written": len(state)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--setting", choices=sorted(SETTING_OUTPUTS), required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--translation-threshold", type=float, default=StaticThresholds.translation_m)
    parser.add_argument("--rotation-threshold", type=float, default=StaticThresholds.rotation_rad)
    parser.add_argument("--gripper-threshold", type=float, default=StaticThresholds.gripper)
    parser.add_argument("--image-threshold", type=float, default=StaticThresholds.image_mean_abs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = args.raw_root / args.task / f"aloha-agilex_{args.setting}"
    paths = sorted((task_dir / "data").glob("episode*.hdf5"), key=_episode_number)
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"no episodes found under {task_dir}")
    output = args.output_root / SETTING_OUTPUTS[args.setting] / args.task
    thresholds = StaticThresholds(
        translation_m=args.translation_threshold,
        rotation_rad=args.rotation_threshold,
        gripper=args.gripper_threshold,
        image_mean_abs=args.image_threshold,
    )
    writer = LeRobotWriter(output, overwrite=args.overwrite)
    totals = {"source": 0, "retained": 0, "written": 0}
    for path in tqdm(paths, desc=f"build {args.task}/{args.setting}"):
        episode_number = _episode_number(path)
        episode, counts = convert_episode(path, _initial_prompt(task_dir, episode_number), thresholds)
        writer.add_episode(episode)
        for key, value in counts.items():
            totals[key] += value
    writer.finalize()
    (output / "meta/source_episodes.json").write_text(
        json.dumps({str(index): _episode_number(path) for index, path in enumerate(paths)}, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(paths),
                **totals,
                "interior_frames_skipped": totals["source"] - totals["retained"],
                "w2c": ROBOTWIN_HEAD_W2C.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
