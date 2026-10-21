# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Fuse overlapping continuous action chunks on an explicit action clock.
"""Model-independent temporal fusion for continuous actions in a common frame."""

from collections import deque

import numpy as np


class TemporalEnsembler:
    """Keep raw [H, D] predictions and fuse candidates for the same action time.

    Each instance belongs to one environment. Callers own the execution clock
    and must reset history whenever that environment resets. This linear reducer
    is intended for joint targets, not quaternion or moving-frame pose vectors.
    """

    def __init__(self, mode: str = "exponential", half_life_steps: float = 4.0) -> None:
        if mode not in {"latest", "uniform", "exponential"}:
            raise ValueError(f"Unsupported temporal ensemble mode: {mode!r}")
        if not np.isfinite(half_life_steps) or half_life_steps <= 0:
            raise ValueError(f"half_life_steps must be finite and positive, got {half_life_steps}")
        self.mode = mode
        self.half_life_steps = half_life_steps
        self._history: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        """Remove all predictions from the previous episode."""
        self._history.clear()

    def discard_before(self, step: int) -> None:
        """Remove chunks that cannot contribute at or after step."""
        self._history = deque((s, a) for s, a in self._history if s + len(a) > step)

    def add(self, actions: np.ndarray, *, start_step: int) -> None:
        """Store a finite, nonempty chunk at a strictly increasing start time."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or min(actions.shape) == 0 or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite nonempty [H,D] actions, got shape={actions.shape}")
        if start_step < 0 or (self._history and start_step <= self._history[-1][0]):
            raise ValueError(f"Chunk start_step must be nonnegative and increasing, got {start_step}")
        if self._history and actions.shape[1] != self._history[-1][1].shape[1]:
            raise ValueError("Action dimension changed without resetting temporal history")
        self.discard_before(start_step)
        self._history.append((start_step, actions.copy()))

    def fuse(self, *, start_step: int, num_steps: int) -> np.ndarray:
        """Fuse a covered interval without advancing time or modifying history."""
        if start_step < 0 or num_steps <= 0:
            raise ValueError("start_step must be nonnegative and num_steps must be positive")
        result = []
        for step in range(start_step, start_step + num_steps):
            candidates = [(s, a[step - s]) for s, a in self._history if 0 <= step - s < len(a)]
            if not candidates:
                raise ValueError(f"No action prediction covers step {step}")
            if self.mode == "latest":
                result.append(candidates[-1][1])
                continue
            starts = np.asarray([s for s, _ in candidates], dtype=np.float64)
            weights = np.ones(len(candidates), dtype=np.float64)
            if self.mode == "exponential":
                weights = np.exp2((starts - starts[-1]) / self.half_life_steps)
            weights /= weights.sum()
            result.append(np.sum(np.stack([a for _, a in candidates]) * weights[:, None], axis=0))
        return np.asarray(result, dtype=np.float32)


class TemporalPoseEnsembler(TemporalEnsembler):
    """Fuse world-frame dual-arm xyz/wxyz/gripper targets without averaging quaternion components."""

    def add(self, actions: np.ndarray, *, start_step: int) -> None:
        if np.asarray(actions).ndim != 2 or np.asarray(actions).shape[1] != 16:
            raise ValueError("Expected dual-arm world-frame actions with 16 values")
        super().add(actions, start_step=start_step)

    def fuse(self, *, start_step: int, num_steps: int) -> np.ndarray:
        """Average positions/grippers linearly and rotations with a weighted quaternion mean."""
        if start_step < 0 or num_steps <= 0:
            raise ValueError("start_step must be nonnegative and num_steps must be positive")
        result = []
        for step in range(start_step, start_step + num_steps):
            candidates = [(s, a[step - s]) for s, a in self._history if 0 <= step - s < len(a)]
            if not candidates:
                raise ValueError(f"No action prediction covers step {step}")
            if self.mode == "latest":
                result.append(candidates[-1][1])
                continue
            starts = np.asarray([s for s, _ in candidates], dtype=np.float64)
            weights = np.ones(len(candidates), dtype=np.float64)
            if self.mode == "exponential":
                weights = np.exp2((starts - starts[-1]) / self.half_life_steps)
            weights /= weights.sum()
            values = np.stack([action for _, action in candidates]).astype(np.float64)
            fused = np.average(values, axis=0, weights=weights)
            for offset in (0, 8):
                quaternions = values[:, offset + 3 : offset + 7]
                covariance = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)
                _, eigenvectors = np.linalg.eigh(covariance)
                quaternion = eigenvectors[:, -1]
                if np.dot(quaternion, quaternions[-1]) < 0:
                    quaternion = -quaternion
                fused[offset + 3 : offset + 7] = quaternion
            result.append(fused)
        return np.asarray(result, dtype=np.float32)
