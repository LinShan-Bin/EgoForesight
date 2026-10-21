# EgoForesight modifications Copyright 2026 Chubin Zhang.
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modification: Add model-configured action spaces and configurable dynamic batching.

import argparse
import logging
import math
import os
import random
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deployment.model_server.tools.ema_policy import apply_teacher
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def set_seed(seed: int) -> None:
    """Fix all possible random seeds for reproducibility.

    Args:
        seed: Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def override_inference_steps(vla: Any, num_inference_timesteps: int) -> list[str]:
    """Override inference steps across supported policy layouts."""
    if num_inference_timesteps <= 0:
        raise ValueError("num_inference_timesteps must be positive")

    overridden = []
    if hasattr(vla, "inference_steps"):
        vla.inference_steps = num_inference_timesteps
        overridden.append("inference_steps")

    action_model = getattr(vla, "action_model", None)
    if action_model is not None and hasattr(action_model, "num_inference_timesteps"):
        action_model.num_inference_timesteps = num_inference_timesteps
        if hasattr(action_model, "config"):
            action_model.config.num_inference_timesteps = num_inference_timesteps
        overridden.append("action_model.num_inference_timesteps")

    if not overridden:
        raise AttributeError("Loaded policy exposes neither inference_steps nor action_model.num_inference_timesteps")
    return overridden


def override_inference_visual_schedule(vla: Any, visual_timestep_alpha: float) -> None:
    """Set a custom vision/action timestep relationship for inference."""

    value = float(visual_timestep_alpha)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"inference_visual_timestep_alpha must be finite and positive, got {value}")

    visited: set[int] = set()
    node = vla
    while node is not None and id(node) not in visited:
        visited.add(id(node))
        node.inference_visual_timestep_alpha = value
        node = getattr(node, "policy", None)


def main(args) -> None:
    # Set random seeds for reproducibility
    set_seed(args.seed)

    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    ema_metadata = None
    if args.ema_teacher != "online":
        ema_metadata = apply_teacher(vla, Path(args.ckpt_path), args.ema_teacher)
        logging.info("Loaded EMA inference weights: %s", ema_metadata)

    cfg_scale = getattr(args, "cfg_scale", 1.0)
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("cfg_scale must be finite and non-negative")
    if cfg_scale != 1.0:
        if not getattr(vla, "cfg_context_dropout_enabled", False) or getattr(vla, "cfg_context_dropout_prob", 0) <= 0:
            raise ValueError("CFG requires a policy trained with context dropout")
        vla.inference_cfg_scale = cfg_scale
        logging.info("Using classifier-free guidance scale %s", cfg_scale)

    if args.num_inference_timesteps is not None:
        overridden = override_inference_steps(vla, args.num_inference_timesteps)
        logging.info(
            "Overriding action diffusion inference steps to %d via %s",
            args.num_inference_timesteps,
            ", ".join(overridden),
        )
    if args.use_bf16:  # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    vla = vla.build_inference_policy()
    if args.inference_visual_timestep_alpha is not None:
        override_inference_visual_schedule(vla, args.inference_visual_timestep_alpha)
        logging.info(
            "Setting visual timestep exponent to %.6f",
            args.inference_visual_timestep_alpha,
        )
    action_space_metadata = getattr(vla, "action_space_metadata", None)
    if action_space_metadata is None:
        logging.info("Model does not declare a deployment action-space adapter")
    else:
        logging.info("Loaded model action-space adapter: %s", action_space_metadata)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        batch_log_interval=args.batch_log_interval,
        batch_trace_path=args.batch_trace_path,
        metadata={
            "env": "egofore_benchmarks",
            "ema": ema_metadata,
            "action_space": action_space_metadata,
            "max_batch_size": args.max_batch_size,
            "batch_wait_ms": args.batch_wait_ms,
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--ema_teacher", choices=["online", "foresight_forcing"], default="online")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use SSH forwarding for remote clients")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum samples per dynamic batch; 1 preserves serial inference.",
    )
    parser.add_argument(
        "--batch_wait_ms",
        type=float,
        default=0.0,
        help="Maximum time to collect compatible requests after the first request arrives.",
    )
    parser.add_argument(
        "--batch_log_interval",
        type=int,
        default=100,
        help="Log dynamic-batch statistics every N executed batches.",
    )
    parser.add_argument(
        "--batch_trace_path",
        type=str,
        default=None,
        help="Optional JSONL path for request-arrival and batch-dispatch tracing.",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale; 1 disables guidance.")
    parser.add_argument(
        "--num_inference_timesteps",
        type=int,
        default=None,
        help="Override the checkpoint action diffusion/flow-matching inference steps.",
    )
    parser.add_argument(
        "--inference_visual_timestep_alpha",
        type=float,
        default=None,
        help="Set inference visual timestep schedule as t_visual = t_action^alpha. "
        "alpha=1 keeps the legacy t_visual=t_action path.",
    )
    parser.add_argument(
        "--idle_timeout",
        type=int,
        default=1800,
        help="Idle timeout in seconds, -1 means never close",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
