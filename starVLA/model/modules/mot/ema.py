# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Retain FP32 EMA teaching and representation loss utilities for MoT.

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.mot.flow_matching import (
    dual_timesteps,
    masked_mean,
    sample_beta_timesteps,
    sample_timesteps,
)


class EMATeacher:
    """
    Standalone exponential-moving-average teacher for training-only use.

    The shadow parameters and floating-point buffers are always kept in FP32 so
    high-decay updates do not disappear when the student trains in BF16. Teacher
    forwards still honor the caller's autocast context.

    This helper intentionally does not inherit from ``nn.Module``. Keeping it on
    a trainer (or even assigning it as a plain attribute of a deployment module)
    therefore does not register teacher weights in the student's ``state_dict``.
    Call ``state_dict`` explicitly only for training resume.
    """

    def __init__(
        self,
        student: nn.Module,
        decay: float = 0.9999,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(student, nn.Module):
            raise TypeError(f"student must be an nn.Module, got {type(student)!r}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must lie in [0, 1), got {decay}")
        self.decay = float(decay)
        self.module = copy.deepcopy(student)
        if device is not None:
            self.module.to(device=device)
        self.module.float()
        self.module.eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Update parameters and floating buffers from the current student."""

        teacher_parameters = dict(self.module.named_parameters())
        student_parameters = dict(student.named_parameters())
        if teacher_parameters.keys() != student_parameters.keys():
            raise ValueError("student parameter structure does not match the EMA teacher")
        for name, teacher_parameter in teacher_parameters.items():
            student_parameter = student_parameters[name].detach().to(device=teacher_parameter.device)
            if teacher_parameter.is_floating_point():
                if teacher_parameter.dtype != torch.float32:
                    raise RuntimeError(
                        "EMA teacher floating-point parameters must remain FP32, "
                        f"but {name!r} has dtype {teacher_parameter.dtype}"
                    )
                teacher_parameter.lerp_(
                    student_parameter.to(dtype=torch.float32),
                    1.0 - self.decay,
                )
            elif teacher_parameter.is_complex():
                teacher_parameter.lerp_(
                    student_parameter.to(dtype=teacher_parameter.dtype),
                    1.0 - self.decay,
                )
            else:
                teacher_parameter.copy_(student_parameter)

        teacher_buffers = dict(self.module.named_buffers())
        student_buffers = dict(student.named_buffers())
        if teacher_buffers.keys() != student_buffers.keys():
            raise ValueError("student buffer structure does not match the EMA teacher")
        for name, teacher_buffer in teacher_buffers.items():
            student_buffer = student_buffers[name].detach().to(device=teacher_buffer.device)
            if teacher_buffer.is_floating_point():
                if teacher_buffer.dtype != torch.float32:
                    raise RuntimeError(
                        "EMA teacher floating-point buffers must remain FP32, "
                        f"but {name!r} has dtype {teacher_buffer.dtype}"
                    )
                teacher_buffer.lerp_(
                    student_buffer.to(dtype=torch.float32),
                    1.0 - self.decay,
                )
            elif teacher_buffer.is_complex():
                teacher_buffer.lerp_(
                    student_buffer.to(dtype=teacher_buffer.dtype),
                    1.0 - self.decay,
                )
            else:
                teacher_buffer.copy_(student_buffer)
        self.module.eval()

    @torch.inference_mode()
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run an inference-only EMA teacher forward."""

        self.module.eval()
        return self.module(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> "EMATeacher":
        """Move the teacher while preserving its FP32 shadow-weight invariant."""

        requested_dtype = kwargs.get("dtype")
        if requested_dtype not in (None, torch.float32):
            raise ValueError(f"EMA teacher dtype is fixed to torch.float32, got {requested_dtype}")
        for argument in args:
            if isinstance(argument, torch.dtype) and argument != torch.float32:
                raise ValueError(f"EMA teacher dtype is fixed to torch.float32, got {argument}")
            if torch.is_tensor(argument) and argument.dtype != torch.float32:
                raise ValueError("EMA teacher cannot follow a non-FP32 reference tensor; " f"got dtype {argument.dtype}")

        self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return teacher state explicitly for a training-resume checkpoint."""

        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        """Restore an explicitly saved teacher state."""

        incompatible = self.module.load_state_dict(state_dict, strict=strict)
        self.module.float()
        self.module.eval()
        self.module.requires_grad_(False)
        return incompatible


def cosine_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    feature_dim: int = -1,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Align student and detached EMA-teacher representations with cosine distance.

    ``1 - cosine`` is used instead of ``-cosine`` so identical features have
    zero loss; the two forms differ only by a constant and have identical
    gradients. A token mask can restrict alignment to selected stream tokens.
    """

    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student and teacher features must have identical shapes, got "
            f"{tuple(student_features.shape)} and {tuple(teacher_features.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    student = student_features.float()
    # EMA forwards run under inference_mode. Clone once so autograd can safely
    # save the fixed target for the student's cosine-similarity backward pass.
    teacher = teacher_features.detach().clone().float()
    similarity = F.cosine_similarity(student, teacher, dim=feature_dim, eps=eps)
    return masked_mean(1.0 - similarity, mask)
