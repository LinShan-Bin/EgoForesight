# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Load the separately licensed DINOv3 encoder without racing on a shared cache.

import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_dino_encoder(
    dino_model_type: str,
    dino_model_variant: str,
    dino_model_path: str | None = None,
    dino_weights_path: str | None = None,
) -> nn.Module:
    """Load DINO backbones from an external, separately licensed checkout."""

    if dino_model_type == "dinov3":
        if not dino_model_path:
            raise ValueError("dino_model_path is required for dinov3 local loading")
        repo_path = str(Path(dino_model_path).expanduser().resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from dinov3.hub import backbones as dinov3_backbones

        if not hasattr(dinov3_backbones, dino_model_variant):
            raise ValueError(f"Unsupported DINOv3 backbone variant: {dino_model_variant}")
        factory = getattr(dinov3_backbones, dino_model_variant)
        if dino_weights_path:
            weights_path = Path(dino_weights_path).expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(f"DINOv3 weights do not exist: {weights_path}")
            # Passing a local path to the upstream DINOv3 hub helper still
            # routes it through torch.hub.load_state_dict_from_url(), copying
            # the checkpoint into TORCH_HOME. Multiple distributed ranks can
            # race while populating that copy. Build the architecture without
            # pretrained weights and load the existing local file directly.
            encoder = factory(pretrained=False, weights=str(weights_path))
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
            encoder.load_state_dict(state_dict, strict=True)
        else:
            encoder = factory()
    else:
        raise ValueError(f"Unsupported DINO model type: {dino_model_type}")
    encoder.eval()
    return encoder
