# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Restore an inference MoT module from its paired training EMA teacher.
"""EMA sidecars contain the MoT module, not a complete inference policy."""

from pathlib import Path


def teacher_path(checkpoint: Path) -> Path:
    """Match the trainer's EMA sidecar naming convention."""
    if checkpoint.name == "pytorch_model.pt":
        return checkpoint.with_name("ema_teachers.pt")
    suffix = "_pytorch_model.pt"
    stem = checkpoint.name.removesuffix(suffix) if checkpoint.name.endswith(suffix) else checkpoint.name
    return checkpoint.with_name(stem + "_ema_teachers.pt")


def read_teacher(checkpoint: Path, teacher: str) -> tuple[dict, dict]:
    """Read and validate one explicitly selected teacher on CPU."""
    import torch

    if teacher not in {"foresight_forcing"}:
        raise ValueError(f"Unknown EMA teacher: {teacher}")
    path = teacher_path(checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("format_version") != 1:
        raise ValueError(f"Unsupported EMA format: {path}")
    if teacher not in state or not isinstance(state[teacher].get("module"), dict):
        raise ValueError(f"Checkpoint has no {teacher} EMA teacher: {path}")
    entry = state[teacher]
    return entry["module"], dict(teacher=teacher, checkpoint=str(path), decay=float(entry["decay"]))


def apply_teacher(model: object, checkpoint: Path, teacher: str) -> dict:
    """Replace only mot_model, retaining all online backbones and normalization stats."""
    import torch

    state, metadata = read_teacher(checkpoint, teacher)
    module = getattr(model, "mot_model", None)
    if module is None:
        raise ValueError("EMA inference requires a model with mot_model")
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"EMA keys differ: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"EMA tensor shape mismatch: {key}")
    module.load_state_dict(state, strict=True)
    return {**metadata, "tensors": len(state)}
