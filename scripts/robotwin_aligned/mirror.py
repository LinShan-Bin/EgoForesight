# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications Copyright 2026 Chubin Zhang.
"""Deterministic geometric reflection; no model-based relabeling or hidden prompt service."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from scripts.robotwin_aligned.common import (
    CHUNK_SIZE,
    VIDEO_KEYS,
    Episode,
    LeRobotWriter,
    decode_video,
    iter_jsonl,
    mirror_state_action,
)


def reflect_direction_words(prompt: str) -> str:
    """Swap explicit English left/right words, preserving case (not semantic adjudication)."""

    def replace(match: re.Match) -> str:
        old = match.group(0)
        new = {"left": "right", "right": "left"}[old.lower()]
        return new.upper() if old.isupper() else new.capitalize() if old[0].isupper() else new

    return re.sub(r"\b(left|right)\b", replace, prompt, flags=re.IGNORECASE)


def mirror_repository(
    source: Path, output: Path, prompt_map: dict[str, str] | None, swap_words: bool = False, overwrite: bool = False
) -> None:
    """Swap wrists/arms and reflect pixels/poses using the original geometric implementation."""
    if source.resolve() == output.resolve():
        raise ValueError("Mirroring must never overwrite the source")
    if (prompt_map is None) == (not swap_words):
        raise ValueError("Choose an explicit per-episode prompt map OR acknowledge direction-word reflection")
    tasks = {int(row["task_index"]): row["task"] for row in iter_jsonl(source / "meta/tasks.jsonl")}
    episodes = list(iter_jsonl(source / "meta/episodes.jsonl"))
    if prompt_map is not None and set(prompt_map) != {str(row["episode_index"]) for row in episodes}:
        raise ValueError("Prompt map keys must exactly cover every dense source episode index")
    writer = LeRobotWriter(output, overwrite=overwrite)
    for row in tqdm(episodes, desc="mirror"):
        index = int(row["episode_index"])
        chunk = index // CHUNK_SIZE
        table = pq.read_table(source / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet")
        original_prompt = tasks[int(table["task_index"][0].as_py())]
        prompt = str(prompt_map[str(index)]) if prompt_map is not None else reflect_direction_words(original_prompt)
        if not prompt.strip():
            raise ValueError(f"Empty mirror prompt for episode {index}")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        w2c_rows = np.asarray(table["camera_params.head_w2c"].to_pylist(), dtype=np.float32)
        if not np.all(w2c_rows == w2c_rows[0]):
            raise ValueError("RoboTwin reflection requires the fixed head-camera calibration")
        w2c = w2c_rows[0].reshape(4, 4)
        frames = {
            key: decode_video(source / f"videos/chunk-{chunk:03d}/{key}/episode_{index:06d}.mp4") for key in VIDEO_KEYS
        }
        reflected = {
            key: [image[:, ::-1] for image in frames[other]]
            for key, other in zip(VIDEO_KEYS, (VIDEO_KEYS[0], VIDEO_KEYS[2], VIDEO_KEYS[1]))
        }
        writer.add_episode(
            Episode(
                mirror_state_action(state, w2c), mirror_state_action(action, w2c), w2c.copy(), reflected, prompt.strip()
            )
        )
    writer.finalize()
    for name in ("source_episodes.json", "audit_exclusions.json"):
        if (source / "meta" / name).is_file():
            shutil.copy2(source / "meta" / name, output / "meta" / name)
    (output / "meta/mirror.json").write_text(
        json.dumps(
            {
                "method": "original_geometric_reflection",
                "prompt_policy": "provided" if prompt_map else "direction_words",
                "limitation": "No semantic corrections: inspect asymmetric objects and all mirrored instructions.",
            },
            indent=2,
        )
    )


def main() -> None:
    """Reflect an existing aligned task repository after explicit prompt-policy selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--prompt-map", type=Path, help='JSON {"0": "mirrored instruction", ...}')
    policy.add_argument("--swap-direction-words", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prompts = json.loads(args.prompt_map.read_text()) if args.prompt_map else None
    mirror_repository(args.source, args.output, prompts, args.swap_direction_words, args.overwrite)


if __name__ == "__main__":
    main()
