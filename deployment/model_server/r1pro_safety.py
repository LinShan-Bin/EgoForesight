# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Inspect physical R1Pro targets and gate publication by amplitude and operator input.

"""Physical action rejection and a cancellable terminal preview; never clip targets."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.policy_inference import ProtocolError
from starVLA.dataloader.pick_fruit_geometry import arm_fk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionLimits:
    """Physical limits with roughly 20-34% margin above observed training maxima.

    Anchor limits compare every row to this request's original measured state.
    Step limits also include measured state -> row zero, then every adjacent row.
    These are dataset-based anomaly limits, not collision/workspace certification.
    """

    eef_anchor_m: float = 0.20
    eef_step_m: float = 0.06
    eef_anchor_deg: float = 100.0
    eef_step_deg: float = 30.0
    arm_anchor_rad: float = 1.40
    arm_step_rad: float = 0.13
    hand_anchor_rad: float = 1.20
    hand_step_rad: float = 0.90

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class TerminalConfirmation:
    """Pause for ordinary stdin input without requiring a TTY."""

    async def wait(self, label: str) -> None:
        """Accept Enter; the caller checks the deadline again after input returns."""
        try:
            input(f"SAFETY PREVIEW {label}: press Enter to publish this action: ")
        except (EOFError, OSError) as exc:
            raise ProtocolError(
                f"operator_confirmation_failed: input() failed ({type(exc).__name__}); action discarded",
                "inference_failed",
            ) from exc


class AsyncTerminalConfirmation:
    """Read Enter on POSIX without blocking observation, reset or disconnect handling."""

    async def wait(self, label: str) -> None:
        """Remove the reader on cancellation; a retired prompt cannot consume new input."""
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        fd = None
        registered = False

        def readable() -> None:
            if ready.done():
                return
            try:
                value = os.read(fd, 1)
                if not value:
                    raise EOFError("stdin closed")
                if value == b"\n" and not ready.done():
                    ready.set_result(None)
            except (EOFError, OSError) as exc:
                if not ready.done():
                    ready.set_exception(exc)

        try:
            fd = sys.stdin.fileno()
            print(f"SAFETY PREVIEW {label}: press Enter to publish this action: ", end="", flush=True)
            loop.add_reader(fd, readable)
            registered = True
            await ready
        except (EOFError, OSError, ValueError, AttributeError, NotImplementedError) as exc:
            raise ProtocolError(
                "operator_confirmation_failed: async stdin unavailable; action discarded", "inference_failed"
            ) from exc
        finally:
            if registered:
                loop.remove_reader(fd)


class R1ProSafetyGate:
    """Print each physical chunk, with independent amplitude and first-publication gates."""

    def __init__(
        self,
        urdf: Path,
        control_mode: str,
        limits: MotionLimits,
        safety_mode: bool = False,
        confirmation: Any = None,
        amplitude_guard: bool = True,
    ) -> None:
        if control_mode not in {"eef", "joint"}:
            raise ValueError("Expected eef or joint control")
        self.urdf, self.control_mode, self.limits = urdf, control_mode, limits
        self.safety_mode = safety_mode
        self.amplitude_guard = amplitude_guard
        self.confirmation = confirmation
        if safety_mode and confirmation is None:
            self.confirmation = TerminalConfirmation()

    def inspect(self, channels: dict[str, np.ndarray], actions: dict[str, np.ndarray]) -> dict:
        """Compute EEF diagnostics and both displacement measures from unnormalized measurements."""
        eef_state, eef_action, metrics = {}, {}, []

        def record(channel: str, name: str, values: np.ndarray) -> None:
            flat = np.asarray(values).reshape(-1)
            row = int(np.argmax(flat))
            metrics.append(
                {
                    "channel": channel,
                    "metric": name,
                    "row": row,
                    "value": float(flat[row]),
                    "limit": getattr(self.limits, name),
                }
            )

        for side in ("left", "right"):
            reference = arm_fk(self.urdf, np.asarray(channels[f"arm_{side}"])[None], side, f"{side}_gripper_link")[0]
            ref_rotation = Rotation.from_rotvec(reference[3:])
            eef_state[side] = np.r_[reference[:3], ref_rotation.as_quat()]
            if self.control_mode == "eef":
                pose = np.asarray(actions[f"eef_{side}"])
                rotation = Rotation.from_quat(pose[:, 3:])
            else:
                fk = arm_fk(self.urdf, np.asarray(actions[f"arm_{side}"]), side, f"{side}_gripper_link")
                rotation = Rotation.from_rotvec(fk[:, 3:])
                pose = np.c_[fk[:, :3], rotation.as_quat()]
            eef_action[side] = pose
            previous_rotation = Rotation.from_quat(np.vstack([ref_rotation.as_quat(), rotation.as_quat()[:-1]]))
            record(side, "eef_anchor_m", np.linalg.norm(pose[:, :3] - reference[:3], axis=-1))
            record(
                side, "eef_step_m", np.linalg.norm(np.diff(pose[:, :3], axis=0, prepend=reference[None, :3]), axis=-1)
            )
            record(side, "eef_anchor_deg", np.rad2deg((rotation * ref_rotation.inv()).magnitude()))
            record(side, "eef_step_deg", np.rad2deg((rotation * previous_rotation.inv()).magnitude()))
            for kind in ("arm", "hand") if self.control_mode == "joint" else ("hand",):
                key = f"{kind}_{side}"
                target, state = np.asarray(actions[key]), np.asarray(channels[key])
                record(key, f"{kind}_anchor_rad", np.max(np.abs(target - state), axis=-1))
                record(key, f"{kind}_step_rad", np.max(np.abs(np.diff(target, axis=0, prepend=state[None])), axis=-1))
        return {"eef_state": eef_state, "eef_action": eef_action, "metrics": metrics}

    def print_and_check(self, channels: dict, actions: dict, label: str) -> None:
        """Print the original physical chunk and reject any violation, including unused suffixes."""
        report = self.inspect(channels, actions)
        LOGGER.info(
            "R1PRO PREVIEW %s mode=%s frame=torso_link4 pose=[x,y,z,qx,qy,qz,qw] position=m joints=rad",
            label,
            self.control_mode,
        )
        LOGGER.info(
            "EEF state = FK(current measured arm joints), not controller target: %s",
            json.dumps({side: value.tolist() for side, value in report["eef_state"].items()}),
        )
        LOGGER.info(
            "Measured joint state (wire order): %s",
            json.dumps({key: value.tolist() for key, value in channels.items() if key.startswith(("arm_", "hand_"))}),
        )
        if self.control_mode == "joint":
            LOGGER.info("EEF action below is FK(predicted arm joint targets); wire actions remain joint targets")
        for row in range(len(next(iter(actions.values())))):
            LOGGER.info(
                "action[%02d] EEF=%s wire=%s",
                row,
                json.dumps({side: value[row].tolist() for side, value in report["eef_action"].items()}),
                json.dumps({key: value[row].tolist() for key, value in actions.items()}),
            )
        LOGGER.info("Action amplitude maxima: %s", json.dumps(report["metrics"]))
        violations = [
            item for item in report["metrics"] if not math.isfinite(item["value"]) or item["value"] > item["limit"]
        ]
        if violations and self.amplitude_guard:
            detail = "; ".join(
                f"{v['channel']} {v['metric']} row={v['row']} value={v['value']:.6g} limit={v['limit']:.6g}"
                for v in violations
            )
            raise ProtocolError(f"action_limit_exceeded: entire chunk rejected; {detail}", "inference_failed")

    async def review(
        self, channels: dict, actions: dict, label: str, first: bool, cancelled: asyncio.Event, deadline_ns: int
    ) -> bool:
        """Keep the first result unpublished until fresh input; timeout/cancel never grants approval."""
        if cancelled.is_set():
            return False
        self.print_and_check(channels, actions, label)
        if not self.safety_mode or not first:
            return not cancelled.is_set()
        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            raise ProtocolError("operator_confirmation_timeout: inference budget exhausted", "deadline_exceeded")
        approval = asyncio.create_task(self.confirmation.wait(label))
        revoked = asyncio.create_task(cancelled.wait())
        try:
            done, _ = await asyncio.wait({approval, revoked}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if cancelled.is_set():
                LOGGER.warning("SAFETY PREVIEW revoked; action discarded: %s", label)
                return False
            if approval not in done or time.monotonic_ns() >= deadline_ns:
                raise ProtocolError(
                    "operator_confirmation_timeout: action discarded; submit a fresh observation", "deadline_exceeded"
                )
            approval.result()  # EOF or a failed terminal read is not confirmation.
            return True
        finally:
            for pending in (approval, revoked):
                pending.cancel()
            await asyncio.gather(approval, revoked, return_exceptions=True)
