# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Export a complete inference policy using a MoT EMA teacher sidecar.
"""Replace the online MoT module with its matching EMA teacher, preserving other backbones."""

import argparse
import json
from pathlib import Path

import torch


def export_policy(teacher_path: Path) -> Path:
    """Validate keys/shapes and atomically export an EMA policy beside its source."""
    suffix = "_ema_teachers.pt"
    if not teacher_path.name.endswith(suffix):
        raise ValueError(f"Expected *{suffix}, got {teacher_path}")
    stem = teacher_path.name.removesuffix(suffix)
    online_path = teacher_path.with_name(f"{stem}_pytorch_model.pt")
    output_path = teacher_path.with_name(f"eval_{stem}_foresight_ema_policy.pt")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path}")
    online = torch.load(online_path, map_location="cpu", weights_only=True, mmap=True)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True, mmap=True)
    if teacher.get("format_version") != 1:
        raise ValueError(f"Unsupported EMA format: {teacher.get('format_version')}")
    ema = teacher["foresight_forcing"]["module"]
    expected = {k.removeprefix("mot_model.") for k in online if k.startswith("mot_model.")}
    if set(ema) != expected:
        raise ValueError(f"EMA key mismatch: missing={expected - set(ema)}, extra={set(ema) - expected}")
    for key, value in ema.items():
        target = online[f"mot_model.{key}"]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"EMA tensor mismatch: {key}")
        online[f"mot_model.{key}"] = value.to(dtype=target.dtype)
    temporary = output_path.with_suffix(".pt.tmp")
    try:
        torch.save(online, temporary)
        saved = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if set(saved) != set(online):
            raise ValueError("Exported policy key mismatch")
        for key in expected:
            if not torch.equal(saved[f"mot_model.{key}"], online[f"mot_model.{key}"]):
                raise ValueError(f"Exported EMA tensor differs: {key}")
        temporary.rename(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = dict(
        teacher=str(teacher_path.resolve()),
        online=str(online_path.resolve()),
        output=str(output_path.resolve()),
        replaced_tensors=len(expected),
        total_tensors=len(online),
        decay=teacher["foresight_forcing"]["decay"],
    )
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    return output_path


def main() -> None:
    """Export each supplied sidecar into a separately named inference checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teachers", nargs="+", type=Path)
    args = parser.parse_args()
    for teacher in args.teachers:
        export_policy(teacher)


if __name__ == "__main__":
    main()
