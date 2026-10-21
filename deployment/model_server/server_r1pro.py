# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Serve R1Pro targets through separate synchronous and asynchronous 1.0 protocols.

"""Run with python -m deployment.model_server.server_r1pro --help."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from deployment.model_server.r1pro_policy import R1ProPolicy, load_adapter
from deployment.model_server.r1pro_safety import AsyncTerminalConfirmation, MotionLimits, R1ProSafetyGate
from deployment.model_server.tools.async_policy_inference import AsyncPolicyInferenceServer, validate_contract
from deployment.model_server.tools.ema_policy import apply_teacher, read_teacher
from deployment.model_server.tools.policy_inference import PolicyInferenceServer
from deployment.model_server.tools.r1pro_compile import compile_policy


def main() -> None:
    """Validate assets, load the selected online/EMA policy, warm up, then accept clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--ema-teacher",
        choices=("online", "foresight_forcing"),
        default="foresight_forcing",
        help="Use online weights or the matching step's foresight EMA sidecar (default: foresight_forcing)",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Serve policy.inference.async 1.0 (one sender/receiver pair), not the sync protocol",
    )
    parser.add_argument(
        "--async-max-horizon",
        type=int,
        default=100,
        help="Async horizon cap, additionally limited by the saved model horizon",
    )
    parser.add_argument("--control-mode", choices=("eef", "joint"), default="eef")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accelerate MoT forward and check exact eager equivalence before listening (default: disabled)",
    )
    parser.add_argument(
        "--compile-vlm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also capture the Qwen3.5 vision tower and decoder with all selected MoT3 features",
    )
    parser.add_argument("--compile-max-graphs", type=int, default=2)
    parser.add_argument("--max-session-requests", type=int, default=10_000)
    parser.add_argument("--session-lifetime", type=float, default=3600)
    parser.add_argument(
        "--safety-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for terminal input before the first action in every session/reset (default: disabled)",
    )
    parser.add_argument(
        "--amplitude-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject the complete action chunk when any physical amplitude limit is exceeded (default: enabled)",
    )
    for name, default in asdict(MotionLimits()).items():
        parser.add_argument("--max-" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument(
        "--validate-only", action="store_true", help="Check saved assets and print contract without loading weights"
    )
    args = parser.parse_args()
    if args.flow_steps < 1 or args.warmup_runs < 1 or not 1 <= args.port <= 65535:
        parser.error("flow-steps/warmup-runs must be positive and port must be 1..65535")
    if args.compile_vlm and not args.compile_model:
        parser.error("--compile-vlm requires --compile")
    if args.compile_max_graphs < 1:
        parser.error("compile-max-graphs must be positive")
    if not 1 <= args.async_max_horizon <= 100:
        parser.error("async-max-horizon must be 1..100")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    adapter, config, provenance = load_adapter(
        args.run_dir, args.checkpoint, args.calibration, args.urdf, args.control_mode
    )
    if args.async_mode:
        _, model_horizon = validate_contract(adapter.contract)
        provenance["async_transport"] = {
            "protocol": "policy.inference.async",
            "version": [1, 0],
            "action_dim": 58,
            "max_horizon": min(args.async_max_horizon, model_horizon),
            "state_order": ["arm_left", "arm_right", "hand_left", "hand_right"],
            "action_order": (
                [f"{'arm' if args.control_mode == 'joint' else 'eef'}_{side}" for side in ("left", "right")]
                + ["hand_left", "hand_right"]
            ),
        }
    provenance["compilation"] = {
        "enabled": args.compile_model,
        "status": "requested_not_run" if args.compile_model else "disabled",
    }
    provenance["flow_steps"] = args.flow_steps
    provenance["ema_teacher"] = args.ema_teacher
    if args.ema_teacher != "online":
        # Check the paired file/teacher before loading backbones; validate-only checks it too.
        teacher_state, metadata = read_teacher(args.checkpoint.resolve(), args.ema_teacher)
        provenance["ema"] = {**metadata, "tensors": len(teacher_state)}
        del teacher_state
    limits = MotionLimits(**{name: getattr(args, "max_" + name) for name in asdict(MotionLimits())})
    provenance["action_amplitude_limits"] = asdict(limits)
    provenance["amplitude_guard"] = args.amplitude_guard
    provenance["operator_preview"] = "first_action_per_session_and_reset" if args.safety_mode else "disabled"
    if args.validate_only:
        print(json.dumps({"contract": adapter.contract, "provenance": provenance}, indent=2))
        return

    guard = R1ProSafetyGate(
        args.urdf,
        args.control_mode,
        limits,
        args.safety_mode,
        confirmation=AsyncTerminalConfirmation() if args.async_mode and args.safety_mode else None,
        amplitude_guard=args.amplitude_guard,
    )
    logging.info("Action amplitude gate: %s", args.amplitude_guard)
    logging.info("Action rejection limits (physical units): %s", json.dumps(asdict(limits)))
    logging.info("Operator safety preview: %s", args.safety_mode)

    import torch

    from starVLA.model.framework import build_framework

    logging.info("Loading R1Pro %s policy from %s", args.control_mode, args.checkpoint)
    # Use the exact training config and strict student weights,
    # model-owned autocast, raw normalized_actions. No generic deployment adapter/unnormalizer.
    model = build_framework(config)
    weights = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    if args.ema_teacher != "online":
        provenance["ema"] = apply_teacher(model, args.checkpoint.resolve(), args.ema_teacher)
        logging.info("Loaded EMA inference weights: %s", json.dumps(provenance["ema"]))
    else:
        logging.info("Using online/student inference weights: %s", args.checkpoint)
    model = model.to(args.device).eval()
    policy = R1ProPolicy(model, adapter, args.flow_steps)
    if args.compile_model:
        provenance["compilation"] = compile_policy(
            policy,
            max_graphs=args.compile_max_graphs,
            compile_vlm=args.compile_vlm,
        )
        logging.info("Verified compiled inference: %s", json.dumps(provenance["compilation"]))
    logging.info("Warming up model and physical action decoder")
    policy.warmup(args.warmup_runs)
    server_options = {
        "max_requests": args.max_session_requests,
        "session_lifetime": args.session_lifetime,
        "publication_guard": guard,
    }
    if args.async_mode:
        server = AsyncPolicyInferenceServer(policy, max_horizon=args.async_max_horizon, **server_options)
    else:
        server = PolicyInferenceServer(policy, provenance, **server_options)
    asyncio.run(server.run(args.host, args.port))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()
