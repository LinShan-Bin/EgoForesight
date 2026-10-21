# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Train native R1Pro recordings with complete global updates.

"""Reuse calibrated camera-relative caches with commanded joint supervision."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from starVLA.dataloader.pick_fruit_dataset import STATE_SLOTS, collate_examples, load_pick_fruit_dataset


class FullUpdateSampler(Sampler):
    """End epochs only after a complete global batch, avoiding short accumulation updates."""

    def __init__(self, length: int, global_batch: int, shuffle: bool, seed: int) -> None:
        self.length, self.size = length, length // global_batch * global_batch
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        if self.size == 0:
            raise ValueError("Training dataset is smaller than one complete global batch")

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Shuffle reproducibly each epoch on all ranks before Accelerate shards batches."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(self.length, generator=generator).tolist()[: self.size])
        return iter(range(self.size))


def build_merged_dataloader(cfg: object) -> DataLoader:
    """Preserve canonical relative EEF + absolute joint supervision and joint-only state."""
    data = cfg.datasets.vla_data
    robot = load_pick_fruit_dataset(data.cache_dir, int(data.action_horizon), "train")
    if robot.manifest.get("arm_joint_source") != "emitted_ik":
        raise ValueError("Joint-supervised R1Pro training requires commanded arm labels")
    jitter = data.get("image_pair_color_jitter", {})
    if jitter.get("enabled", False):
        from torchvision.transforms import ColorJitter

        robot.color_jitter = ColorJitter(
            **{k: float(jitter.get(k, 0)) for k in ("brightness", "contrast", "saturation", "hue")}
        )
    dataset = robot
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        statistics = {}
        statistics[robot.manifest["dataset_id"]] = robot.statistics
        (output / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2))
        (output / "dataset_manifest.json").write_text(json.dumps(robot.manifest, indent=2))
    workers = int(data.get("num_workers", 4))
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    global_batch = int(data.per_device_batch_size) * world * int(cfg.trainer.gradient_accumulation_steps)
    return DataLoader(
        dataset,
        batch_size=int(data.per_device_batch_size),
        sampler=FullUpdateSampler(len(dataset), global_batch, True, int(cfg.seed)),
        drop_last=True,
        collate_fn=collate_examples,
        num_workers=workers,
        persistent_workers=False,
        prefetch_factor=int(data.get("prefetch_factor", 2)) if workers else None,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )
