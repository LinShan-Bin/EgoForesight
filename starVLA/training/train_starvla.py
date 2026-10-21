# EgoForesight modifications Copyright 2026 Chubin Zhang.
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modification: Default MoT3 training to block compilation with explicit opt-out.
# ruff: noqa: E402


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import gc
import json
import math
import os
import re
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Tuple


def configure_rank_local_triton_cache(environ: MutableMapping[str, str] | None = None) -> str:
    """Choose a job- and rank-local Triton cache before importing FLA kernels."""
    env = os.environ if environ is None else environ
    configured_cache = env.get("TRITON_CACHE_DIR", "").strip()
    if configured_cache:
        return configured_cache

    elastic_run_id = env.get("TORCHELASTIC_RUN_ID", "").strip()
    if not elastic_run_id or elastic_run_id.lower() == "none":
        master_addr = env.get("MASTER_ADDR", "localhost")
        master_port = env.get("MASTER_PORT", "single_process")
        elastic_run_id = f"{master_addr}_{master_port}"

    rank = env.get("RANK", env.get("LOCAL_RANK", "0"))

    def safe_path_component(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return sanitized or "unknown"

    default_cache_root = f"/tmp/starvla_triton_cache_uid_{os.getuid()}"
    cache_root = env.get("STARVLA_TRITON_CACHE_ROOT", default_cache_root)
    cache_dir = Path(cache_root) / f"job_{safe_path_component(elastic_run_id)}" / f"rank_{safe_path_component(rank)}"
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)


# FLA constructs Triton autotuners while the model modules are imported below.
# Isolating caches here prevents concurrent ranks from racing on group metadata.
configure_rank_local_triton_cache()

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from starVLA.model.framework import build_framework
from starVLA.model.modules.mot.open_loop_eval import MoTRoboTwinOpenLoopEvaluator
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.muon_runtime import (
    configure_deepspeed_async_norm_mask,
    configure_deepspeed_global_gradient_clipping,
    get_deepspeed_muon_aux_adamw_clip_metrics,
    install_deepspeed_muon_aux_adamw_clipper,
)
from starVLA.training.trainer_utils.optimizers import build_optimizer, validate_muon_runtime

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`

logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    optimizer = build_optimizer(model=model, cfg=cfg)

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            parameter_count = sum(parameter.numel() for parameter in group["params"])
            logger.info(
                f"LR Group {group['name']}: optimizer={group.get('optimizer_type', 'adamw')}, "
                f"lr={group['lr']}, tensors={len(group['params'])}, parameters={parameter_count:,}"
            )

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


def build_accelerator(cfg, deepspeed_plugin=None) -> Accelerator:
    """Build an Accelerator whose fixed-step scheduler is advanced explicitly by the trainer."""
    if deepspeed_plugin is None:
        deepspeed_plugin = DeepSpeedPlugin()
    optimizer_config = cfg.trainer.get("optimizer", {})
    uses_deepspeed_muon = (
        str(optimizer_config.get("name", "AdamW")).lower() == "muon"
        and str(optimizer_config.get("backend", "deepspeed")).lower() == "deepspeed"
    )
    configured_global_clip = cfg.trainer.get("gradient_clipping", None)
    active_global_clip = 0.0 if uses_deepspeed_muon else configured_global_clip
    configure_deepspeed_global_gradient_clipping(deepspeed_plugin, active_global_clip)
    if uses_deepspeed_muon:
        # Keep the effective run config aligned with the engine. Auxiliary
        # AdamW clipping is configured independently on the optimizer.
        cfg.trainer["gradient_clipping"] = 0.0
    return Accelerator(
        gradient_accumulation_steps=int(cfg.trainer.get("gradient_accumulation_steps", 1)),
        deepspeed_plugin=deepspeed_plugin,
        # The scheduler is parameterized in global optimizer steps. Accelerate's
        # default wrapper advances it once per process when split_batches=False,
        # shortening both warmup and decay by world_size.
        step_scheduler_with_optimizer=False,
    )


def configure_muon_optimizer_runtime(model: torch.nn.Module, cfg: Any) -> dict[str, Any] | None:
    """Validate Muon, configure norm masking, and install auxiliary-AdamW clipping."""
    runtime = validate_muon_runtime(model, cfg)
    if runtime is None or runtime["backend"] != "deepspeed":
        return runtime

    runtime["async_norm_mask"] = configure_deepspeed_async_norm_mask(
        model, enabled=cfg.trainer.optimizer.get("async_norm_mask", True)
    )
    aux_max_norm = float(cfg.trainer.optimizer.get("aux_gradient_clipping", 1.0))
    if aux_max_norm <= 0.0:
        runtime["aux_adamw_clipping"] = {"installed": False, "max_norm": aux_max_norm}
        return runtime

    runtime["aux_adamw_clipping"] = install_deepspeed_muon_aux_adamw_clipper(
        model,
        max_norm=aux_max_norm,
    )
    return runtime


def gradient_norm_metrics(
    global_grad_norm: torch.Tensor | float | None,
    max_grad_norm: float | None,
) -> dict[str, float]:
    """Convert a global clipping-input norm into stable logging metrics."""
    if global_grad_norm is None:
        return {}
    if torch.is_tensor(global_grad_norm):
        if global_grad_norm.numel() != 1:
            raise ValueError(f"Expected a scalar global gradient norm, got shape={tuple(global_grad_norm.shape)}")
        grad_norm = float(global_grad_norm.detach().float().cpu())
    else:
        grad_norm = float(global_grad_norm)

    invalid_norm = not math.isfinite(grad_norm) or grad_norm < 0.0
    metrics = {
        "clip_input_norm/pre_clip_global": grad_norm,
        "clip_input_norm/invalid": float(invalid_norm),
    }
    clipping_enabled = max_grad_norm is not None and float(max_grad_norm) > 0.0
    if clipping_enabled:
        metrics["grad_clip/max_norm"] = float(max_grad_norm)
    if invalid_norm:
        metrics["grad_clip/coefficient"] = float("nan")
        metrics["grad_clip/was_clipped"] = 0.0
        return metrics
    if not clipping_enabled:
        metrics["grad_clip/coefficient"] = 1.0
        metrics["grad_clip/was_clipped"] = 0.0
        return metrics

    clip_threshold = float(max_grad_norm)
    clip_coefficient = min(1.0, clip_threshold / (grad_norm + 1.0e-6))
    clipping_triggered = float(clip_coefficient < 1.0)
    metrics["grad_clip/coefficient"] = clip_coefficient
    metrics["grad_clip/was_clipped"] = clipping_triggered
    return metrics


COMPACT_LOSS_METRIC_KEYS = frozenset(
    {
        "loss",
        "total_loss",
        "action_loss",
        "raw_action_flow_loss",
        "reliability_weighted_action_flow_loss",
        "foresight_action_rep_loss",
        "vision_flow_loss",
        "diffusion_action_loss",
        "flowmatching_loss",
        "vlm_loss",
        "vlm_fast_loss",
        "fast_token_loss",
    }
)
COMPACT_TIMING_METRIC_KEYS = frozenset(
    {
        "vla_data_time",
        "vlm_data_time",
        "data_time",
        "model_time",
    }
)


def compact_training_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep raw loss components, the backward objective, pre-clip norm, and all LRs."""

    return {
        key: value
        for key, value in metrics.items()
        if key in COMPACT_LOSS_METRIC_KEYS
        or key in COMPACT_TIMING_METRIC_KEYS
        or key == "clip_input_norm/pre_clip_global"
        or key == "learning_rate"
        or key.startswith("learning_rate/")
    }


def compute_max_train_steps(cfg, accelerator, vla_train_dataloader, max_train_epochs: int) -> Tuple[int, int, int]:
    """compute max_train_steps from epochs and data size"""
    per_device_batch_size = cfg.datasets.vla_data.per_device_batch_size
    total_batch_size = per_device_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    dataset_size = None
    if hasattr(vla_train_dataloader, "dataset") and vla_train_dataloader.dataset is not None:
        try:
            dataset_size = len(vla_train_dataloader.dataset)
        except TypeError:
            dataset_size = None
    if dataset_size is None:
        dataset_size = len(vla_train_dataloader) * per_device_batch_size
    steps_per_epoch = math.ceil(dataset_size / total_batch_size)
    max_train_steps = steps_per_epoch * max_train_epochs
    return max_train_steps, dataset_size, total_batch_size


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.writer = None
        self.open_loop_evaluator = None
        self._pending_ema_model_checkpoint: str | None = None
        self._resume_training_state = False
        self._resume_scheduler_state = False

    def _get_torch_compile_modules(self) -> set[str]:
        framework = getattr(self.config, "framework", None)
        default_modules = "mot_blocks" if getattr(framework, "name", None) == "Qwen35MoT3" else "action"
        modules = getattr(self.config.trainer, "torch_compile_modules", default_modules)
        if isinstance(modules, str):
            return {module.strip() for module in modules.split(",") if module.strip()}
        return set(modules)

    def _maybe_compile_model_modules(self) -> None:
        framework = getattr(self.config, "framework", None)
        default_enabled = getattr(framework, "name", None) == "Qwen35MoT3"
        if not getattr(self.config.trainer, "use_torch_compile", default_enabled):
            return

        compile_mode = getattr(self.config.trainer, "torch_compile_mode", "default")
        compile_modules = self._get_torch_compile_modules()
        unwrapped = self.accelerator.unwrap_model(self.model)

        if "mot" in compile_modules or "mot_blocks" in compile_modules:
            from types import MethodType

            from starVLA.model.modules.mot.modeling_mot3 import MoT3Block

            mot_model = getattr(unwrapped, "mot_model", None)
            blocks = getattr(mot_model, "blocks", None)
            if blocks is None or not all(isinstance(block, MoT3Block) for block in blocks):
                raise ValueError("torch_compile_modules=mot_blocks requires MoT3 blocks")
            # Compile the unbound method: module names/checkpoint keys stay intact,
            # and a later deepcopy binds each method to the copied block (e.g. EMA).
            compiled_forward = torch.compile(MoT3Block.forward, mode=compile_mode, fullgraph=True, dynamic=True)
            for block in blocks:
                block.forward = MethodType(compiled_forward, block)
            logger.info(f"Compiled {len(blocks)} MoT3 block forwards with mode={compile_mode}.")

        if "qwen" in compile_modules or "qwen_vl" in compile_modules:
            qwen_vl_interface = getattr(unwrapped, "qwen_vl_interface", None)
            qwen_model = getattr(qwen_vl_interface, "model", None)
            if qwen_model is None:
                logger.warning("torch.compile requested for qwen, but qwen_vl_interface.model was not found.")
            else:
                logger.info(f"Applying torch.compile to qwen_vl_interface.model with mode={compile_mode}...")
                qwen_vl_interface.model = torch.compile(
                    qwen_model,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
                logger.info("Compiled qwen_vl_interface.model with torch.compile.")

        if "action" in compile_modules or "dit" in compile_modules:
            action_model = getattr(unwrapped, "action_model", None)
            if action_model is None:
                logger.warning("torch.compile requested for action, but no action_model module was found.")
                return

            dit_model = getattr(action_model, "model", None)
            if dit_model is None:
                logger.warning("torch.compile requested for action, but action_model.model was not found.")
                return

            logger.info(f"Applying torch.compile to action_model.model with mode={compile_mode}...")
            action_model.model = torch.compile(
                dit_model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )
            logger.info("Compiled action_model.model with torch.compile.")

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing()  # TODO merge with load pretrained weights

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.lr_scheduler, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.vla_train_dataloader,
        )
        if self._resume_training_state:
            self._load_latest_optimizer_state()
        muon_runtime = configure_muon_optimizer_runtime(self.model, self.config)
        if muon_runtime is not None and self.accelerator.is_main_process:
            logger.info(f"Validated Muon runtime: {muon_runtime}")
        self._restore_pending_training_ema_state()
        self._maybe_compile_model_modules()

        self._init_wandb()
        self._init_open_loop_evaluator()

    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")

            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()

            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps

            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize TensorBoard"""
        if self.accelerator.is_main_process:
            log_dir = os.path.join(self.config.output_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard writer initialized at {log_dir}")

    def _init_open_loop_evaluator(self) -> None:
        open_loop_config = self.config.trainer.get("open_loop", None)
        enabled = (
            open_loop_config is not None
            and open_loop_config.get("enabled", True) not in (False, "False", "false", None, 0)
            and int(self.config.trainer.get("open_loop_interval", 0)) > 0
        )
        if not enabled:
            return

        initialization_error = [None]
        if self.accelerator.is_main_process:
            try:
                deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
                zero_stage = int(getattr(deepspeed_plugin, "zero_stage", 0) or 0)
                if zero_stage > 2:
                    raise RuntimeError(
                        "Main-process-only open-loop evaluation requires replicated model parameters "
                        f"(DeepSpeed ZeRO-2 or lower), got ZeRO-{zero_stage}"
                    )
                training_mixture = getattr(self.vla_train_dataloader, "dataset", None)
                if training_mixture is None:
                    raise TypeError("Training dataloader does not expose its LeRobot mixture dataset")
                self.open_loop_evaluator = MoTRoboTwinOpenLoopEvaluator(
                    open_loop_config=open_loop_config,
                    data_config=self.config.datasets.vla_data,
                    training_mixture=training_mixture,
                    chunk_size=int(self.config.framework.action_model.chunk_size),
                    output_dir=self.config.output_dir,
                )
                logger.info("Initialized fixed clean/randomized RoboTwin open-loop evaluator")
            except Exception as exc:
                initialization_error[0] = f"{type(exc).__name__}: {exc}"

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(initialization_error, src=0)
        if initialization_error[0] is not None:
            raise RuntimeError(f"Failed to initialize RoboTwin open-loop evaluation: {initialization_error[0]}")
        self.accelerator.wait_for_everyone()

    def _is_open_loop_step(self) -> bool:
        interval = int(self.config.trainer.get("open_loop_interval", 0))
        return interval > 0 and self.completed_steps > 0 and self.completed_steps % interval == 0

    def _log_open_loop_to_tensorboard(
        self,
        metrics: dict[str, float],
        artifacts: dict[str, Path],
    ) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"open_loop/{key}", value, self.completed_steps)
        for key, image_path in artifacts.items():
            if not image_path.exists():
                continue
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    image_array = np.asarray(image.convert("RGB"))
                self.writer.add_image(
                    f"open_loop/{key}",
                    image_array,
                    self.completed_steps,
                    dataformats="HWC",
                )
            except Exception as exc:
                logger.warning(f"Failed to log open-loop image {image_path}: {exc}")
        self.writer.flush()

    def _run_open_loop(self) -> None:
        """Run the fixed episodes on rank zero, then restore synchronized training."""
        self.accelerator.wait_for_everyone()
        evaluation_error = [None]
        if self.accelerator.is_main_process:
            if self.open_loop_evaluator is None:
                evaluation_error[0] = "Open-loop evaluator was not initialized on the main process"
            else:
                inference_model = self.accelerator.unwrap_model(self.model)
                was_training = inference_model.training
                inference_model.eval()
                try:
                    with torch.inference_mode():
                        metrics, artifacts = self.open_loop_evaluator.run(
                            inference_model,
                            self.completed_steps,
                        )
                    self._log_open_loop_to_tensorboard(metrics, artifacts)
                    logger.info(
                        "Open-loop step %d metrics: %s",
                        self.completed_steps,
                        metrics,
                    )
                except Exception as exc:
                    evaluation_error[0] = f"{type(exc).__name__}: {exc}"
                finally:
                    inference_model.train(was_training)
                    self._release_train_memory()

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(evaluation_error, src=0)
        self.accelerator.wait_for_everyone()
        if evaluation_error[0] is not None:
            open_loop_config = self.config.trainer.get("open_loop", {})
            fail_on_error = open_loop_config.get("fail_on_error", True) not in (
                False,
                "False",
                "false",
                None,
                0,
            )
            message = f"RoboTwin open-loop evaluation failed: {evaluation_error[0]}"
            if fail_on_error:
                raise RuntimeError(message)
            if self.accelerator.is_main_process:
                logger.warning(message)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._resume_training_state = False
        self._resume_scheduler_state = False

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            pinned_checkpoint = getattr(self.config.trainer, "resume_checkpoint", None)
            if pinned_checkpoint:
                # A launcher may pin the latest complete model/optimizer/EMA
                # bundle. Do not rescan and race a newly published model file.
                checkpoint_path = Path(pinned_checkpoint).expanduser().resolve()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Pinned resume checkpoint does not exist: {checkpoint_path}")
                checkpoint_match = re.fullmatch(r"steps_(\d+)_pytorch_model\.pt", checkpoint_path.name)
                if checkpoint_match is None:
                    raise ValueError(
                        "trainer.resume_checkpoint requires a checkpoint named steps_<N>_pytorch_model.pt, "
                        f"got {checkpoint_path}"
                    )
                if checkpoint_path.parent != Path(self.checkpoint_dir).resolve():
                    raise ValueError(
                        "trainer.resume_checkpoint must belong to this run's checkpoint directory "
                        f"{Path(self.checkpoint_dir).resolve()}, got {checkpoint_path}"
                    )
                resume_from_checkpoint = str(checkpoint_path)
                self.completed_steps = int(checkpoint_match.group(1))
            else:
                resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)

            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self._pending_ema_model_checkpoint = self.resume_from_checkpoint
                self._resume_training_state = True
                self._resume_scheduler_state = True
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            configured_resume_step = getattr(self.config.trainer, "resume_step", None)
            if configured_resume_step is not None:
                resume_step = int(configured_resume_step)
                if resume_step < 0:
                    raise ValueError(f"trainer.resume_step must be non-negative, got {resume_step}")
                checkpoint_match = re.search(r"steps_(\d+)_pytorch_model\.pt$", str(pretrained_checkpoint))
                if checkpoint_match is None:
                    raise ValueError(
                        "trainer.resume_step requires a checkpoint named steps_<N>_pytorch_model.pt, "
                        f"got {pretrained_checkpoint}"
                    )
                checkpoint_step = int(checkpoint_match.group(1))
                if checkpoint_step != resume_step:
                    raise ValueError(
                        f"trainer.resume_step={resume_step} does not match checkpoint step {checkpoint_step}: "
                        f"{pretrained_checkpoint}"
                    )
                self.completed_steps = resume_step
                self._pending_ema_model_checkpoint = str(pretrained_checkpoint)
                self._resume_scheduler_state = True
                logger.warning(
                    "Branching from model/EMA checkpoint at step %d with a fresh optimizer; "
                    "the LR scheduler and dataloaders will resume at that step.",
                    resume_step,
                )
            else:
                self.completed_steps = 0
                if getattr(self.config.trainer, "restore_pretrained_ema", False):
                    sidecar = self._ema_checkpoint_path(pretrained_checkpoint)
                    if not sidecar.is_file():
                        raise FileNotFoundError(f"Requested pretrained EMA teacher is missing: {sidecar}")
                    self._pending_ema_model_checkpoint = str(pretrained_checkpoint)
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    @staticmethod
    def _ema_checkpoint_path(model_checkpoint_path: str | Path) -> Path:
        """Derive the non-model EMA sidecar path without affecting checkpoint discovery."""

        model_path = Path(model_checkpoint_path)
        if model_path.name == "pytorch_model.pt":
            return model_path.with_name("ema_teachers.pt")
        suffix = "_pytorch_model.pt"
        if model_path.name.endswith(suffix):
            return model_path.with_name(f"{model_path.name.removesuffix(suffix)}_ema_teachers.pt")
        return model_path.with_name(f"{model_path.name}_ema_teachers.pt")

    def _restore_pending_training_ema_state(self) -> None:
        """Restore EMA only after Accelerator has prepared the student dtype/device."""

        if self._pending_ema_model_checkpoint is None:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        expects_state = getattr(unwrapped_model, "expects_training_ema_state", None)
        load_state = getattr(unwrapped_model, "load_training_ema_state_dict", None)
        if not callable(load_state):
            self._pending_ema_model_checkpoint = None
            return

        sidecar_path = self._ema_checkpoint_path(self._pending_ema_model_checkpoint)
        if not sidecar_path.is_file():
            if callable(expects_state) and expects_state():
                logger.warning(
                    "EMA sidecar is missing for resumed checkpoint %s; "
                    "initializing the teacher lazily from the restored student",
                    self._pending_ema_model_checkpoint,
                )
            self._pending_ema_model_checkpoint = None
            return
        try:
            ema_state = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        except TypeError:
            ema_state = torch.load(sidecar_path, map_location="cpu")
        loaded = load_state(ema_state)
        logger.info("Restored EMA teacher state %s from %s", loaded, sidecar_path)
        self._pending_ema_model_checkpoint = None

    def _save_training_ema_state(self, model_checkpoint_path: str | Path) -> None:
        """Save plain EMA weights beside a model checkpoint, never in optimizer state."""

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict_fn = getattr(unwrapped_model, "training_ema_state_dict", None)
        if not callable(state_dict_fn):
            return
        ema_state = state_dict_fn()
        if len(ema_state) <= 1:
            expects_state = getattr(unwrapped_model, "expects_training_ema_state", None)
            if callable(expects_state) and expects_state():
                logger.warning("Configured EMA teacher was not initialized at checkpoint time")
            return
        sidecar_path = self._ema_checkpoint_path(model_checkpoint_path)
        torch.save(ema_state, sidecar_path)
        logger.info("EMA teacher state saved at %s", sidecar_path)

    @staticmethod
    def _release_train_memory():
        """Best-effort release of cached CPU/GPU memory before memory-heavy ops (e.g. checkpoint)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_checkpoint_step(self) -> bool:
        save_interval = int(self.config.trainer.save_interval)
        return self.completed_steps > 0 and self.completed_steps % save_interval == 0

    def _save_checkpoint(self):
        """save current training state"""
        self._release_train_memory()

        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"
            torch.save(state_dict, model_checkpoint_path)
            del state_dict
            self._save_training_ema_state(model_checkpoint_path)
            self._release_train_memory()

            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(
                    output_dir / "config.yaml",
                    use_original_values=False,
                )
                logger.info("✅ Configuration files saved")

        self._save_latest_optimizer_state()
        self.accelerator.wait_for_everyone()

    def _learning_rate_metrics(self) -> dict[str, float]:
        """Summarize shape-specific optimizer LRs without logging hundreds of Muon groups."""
        metrics = {"learning_rate": float(self.lr_scheduler.get_last_lr()[0])}
        buckets: dict[tuple[str, str], list[float]] = {}
        for group in getattr(self.optimizer, "param_groups", ()):
            group_name = str(group.get("name", "base"))
            module_name = group_name.split("/", 1)[0]
            optimizer_type = str(group.get("optimizer_type", "adamw"))
            buckets.setdefault((module_name, optimizer_type), []).append(float(group["lr"]))

        for (module_name, optimizer_type), learning_rates in sorted(buckets.items()):
            metric_prefix = f"learning_rate/{module_name}/{optimizer_type}"
            if min(learning_rates) == max(learning_rates):
                metrics[metric_prefix] = learning_rates[0]
            else:
                metrics[f"{metric_prefix}_min"] = min(learning_rates)
                metrics[f"{metric_prefix}_max"] = max(learning_rates)
        return metrics

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics.update(self._learning_rate_metrics())
                logging_config = self.config.trainer.get("logging", {}) or {}
                compact_logging = bool(logging_config.get("compact", False))
                if compact_logging:
                    metrics = compact_training_metrics(metrics)
                else:
                    metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to TensorBoard
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    @staticmethod
    def _resume_data_position(
        completed_steps: int,
        gradient_accumulation_steps: int,
        batches_per_epoch: int,
        eval_interval: int = 0,
        save_interval: int = 0,
    ) -> tuple[int, int, int, int]:
        """Map completed optimizer steps to a deterministic local dataloader position."""
        if completed_steps < 0:
            raise ValueError(f"completed_steps must be non-negative, got {completed_steps}")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive, " f"got {gradient_accumulation_steps}")
        if batches_per_epoch <= 0:
            raise ValueError(f"batches_per_epoch must be positive, got {batches_per_epoch}")

        eval_batches = 0
        if eval_interval > 0:
            eval_batches = completed_steps // eval_interval
            if save_interval > 0:
                # The training loop deliberately skips evaluation whenever an
                # eval step is also a model-checkpoint step.
                eval_batches -= completed_steps // math.lcm(eval_interval, save_interval)

        consumed_batches = completed_steps * gradient_accumulation_steps + eval_batches
        epoch, batch_offset = divmod(consumed_batches, batches_per_epoch)
        return consumed_batches, eval_batches, epoch, batch_offset

    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch: int) -> None:
        """Synchronize both the distributed sampler and index-deterministic dataset epoch."""
        set_dataloader_epoch = getattr(dataloader, "set_epoch", None)
        if callable(set_dataloader_epoch):
            set_dataloader_epoch(epoch)
        else:
            sampler = getattr(dataloader, "sampler", None)
            set_sampler_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_sampler_epoch):
                set_sampler_epoch(epoch)

        # Accelerate's DataLoaderShard sets either the sampler or dataset epoch.
        # The LeRobot mixture itself also keys deterministic sampling by epoch, so
        # set it explicitly when a distributed sampler is present.
        dataset = getattr(dataloader, "dataset", None)
        set_dataset_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_dataset_epoch):
            set_dataset_epoch(epoch)

    def _create_data_iterators(self):
        """Create data iterators, efficiently restoring the deterministic sample position."""
        batches_per_epoch = len(self.vla_train_dataloader)
        consumed_batches, eval_batches, resume_epoch, batch_offset = self._resume_data_position(
            completed_steps=int(self.completed_steps),
            gradient_accumulation_steps=int(self.accelerator.gradient_accumulation_steps),
            batches_per_epoch=batches_per_epoch,
            eval_interval=int(self.config.trainer.get("eval_interval", 0)),
            save_interval=int(self.config.trainer.get("save_interval", 0)),
        )
        self.vla_epoch_count = resume_epoch
        self._set_dataloader_epoch(self.vla_train_dataloader, resume_epoch)

        if batch_offset:
            resume_dataloader = self.accelerator.skip_first_batches(
                self.vla_train_dataloader,
                num_batches=batch_offset,
            )
            self._set_dataloader_epoch(resume_dataloader, resume_epoch)
            self.vla_iter = iter(resume_dataloader)
        else:
            self.vla_iter = iter(self.vla_train_dataloader)

        if self.completed_steps > 0 and self.accelerator.is_main_process:
            logger.info(
                "Resumed deterministic dataloader position: "
                f"consumed_local_batches={consumed_batches}, epoch={resume_epoch}, "
                f"batch_offset={batch_offset}/{batches_per_epoch}, eval_batches={eval_batches}"
            )

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count = getattr(self, "vla_epoch_count", 0) + 1
            self._set_dataloader_epoch(self.vla_train_dataloader, self.vla_epoch_count)
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=False,
            ncols=80,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if not self._last_optimizer_step_performed:
                continue

            # update progress after a real optimizer update
            self.completed_steps += 1
            progress_bar.update(1)

            if self.accelerator.is_local_main_process:
                postfix = {
                    "data_times": f"{t_end_data - t_start_data:.3f}",
                    "model_times": f"{t_end_model - t_start_model:.3f}",
                }
                if self.completed_steps % self.config.trainer.logging_frequency == 0:
                    if "vlm_fast_loss" in step_metrics and "diffusion_action_loss" in step_metrics:
                        postfix["total"] = f"{step_metrics['action_loss']:.4f}"
                        postfix["fast"] = f"{step_metrics['vlm_fast_loss']:.4f}"
                        postfix["diff"] = f"{step_metrics['diffusion_action_loss']:.4f}"
                    else:
                        postfix["loss"] = f"{step_metrics.get('action_dit_loss', step_metrics['action_loss']):.4f}"
                progress_bar.set_postfix(postfix)

            is_checkpoint_step = self._is_checkpoint_step()

            # evaluate model (skip on checkpoint steps to avoid eval + save memory spike)
            if self.completed_steps % self.config.trainer.eval_interval == 0 and not is_checkpoint_step:
                step_metrics = self.eval_action_model(step_metrics)
            elif (
                is_checkpoint_step
                and self.completed_steps % self.config.trainer.eval_interval == 0
                and self.accelerator.is_main_process
            ):
                logger.info(
                    "Skipping eval at step %d (checkpoint save step; reducing peak memory).",
                    self.completed_steps,
                )

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if is_checkpoint_step:
                self._save_checkpoint()

            if self._is_open_loop_step():
                self._run_open_loop()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        batch_examples = self._get_next_batch()
        examples = []
        for example in batch_examples:
            if "action" not in example:
                continue
            action_valid = np.asarray(example.get("action_valid", True))
            if action_valid.size != 1:
                raise ValueError(
                    "Action evaluation expects one action_valid value per sample, " f"got shape {action_valid.shape}"
                )
            action_mask = example.get("action_mask")
            if not bool(action_valid.reshape(-1)[0]) or (
                action_mask is not None and not np.asarray(action_mask, dtype=bool).any()
            ):
                continue
            examples.append(example)

        if not examples:
            if self.accelerator.is_main_process:
                logger.warning("Skipping action evaluation because this batch contains no action-supervised samples.")
            del batch_examples
            self.accelerator.wait_for_everyone()
            return step_metrics

        score = 0.0
        actions = [example["action"] for example in examples]  # label
        # Extract action_mask if it exists
        action_masks = [example.get("action_mask", None) for example in examples]
        has_action_mask = any(mask is not None for mask in action_masks)
        horizons = {np.asarray(action).shape[0] for action in actions}
        if len(horizons) > 1:
            horizon = max(horizons)
            padded_actions, padded_masks = [], []
            for action, mask in zip(actions, action_masks):
                action = np.asarray(action)
                padded = np.zeros((horizon, action.shape[1]), dtype=action.dtype)
                padded[: len(action)] = action
                valid = np.zeros_like(padded, dtype=bool)
                valid[: len(action)] = True if mask is None else np.asarray(mask, dtype=bool)
                padded_actions.append(padded)
                padded_masks.append(valid)
            actions, action_masks = padded_actions, padded_masks
            has_action_mask = True
        if has_action_mask:
            action_masks = np.array(action_masks)
        else:
            action_masks = None

        # Predict actions using the model
        inference_model = self.accelerator.unwrap_model(self.model)
        output_dict = inference_model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray

            if action_masks is not None:
                # Apply action_mask: only compute score for valid actions (mask=1)
                num_pots = np.sum(action_masks)
                # Compute the metric score with mask applied
                score = TrainerUtils.euclidean_distance(normalized_actions * action_masks, actions * action_masks)
            else:
                # B, Chunk, dim = actions.shape
                num_pots = np.prod(actions.shape)
                # Compute the metric score
                score = TrainerUtils.euclidean_distance(normalized_actions, actions)

            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        del batch_examples, examples
        self.accelerator.wait_for_everyone()
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    @staticmethod
    def _extract_step_metrics(output_dict: dict) -> dict:
        """从 model.forward 返回值中提取可记录的标量 loss。"""
        metrics = {}
        action_loss = output_dict.get("action_loss", output_dict.get("loss"))
        if action_loss is None:
            raise KeyError("Model output must contain 'loss' or backward-compatible 'action_loss'")
        metrics["action_loss"] = action_loss.item()
        if "loss" in output_dict:
            metrics["loss"] = output_dict["loss"].item()
        for key, value in output_dict.items():
            if key in metrics:
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[key] = value.item()

        if "diffusion_action_loss" in output_dict:
            metrics["diffusion_action_loss"] = output_dict["diffusion_action_loss"].item()
            metrics["action_dit_loss"] = metrics["diffusion_action_loss"]
        else:
            metrics["action_dit_loss"] = metrics["action_loss"]

        if "vlm_fast_loss" in output_dict:
            metrics["vlm_fast_loss"] = output_dict["vlm_fast_loss"].item()

        # 兼容 QwenFastPerceiver 等框架的命名
        if "fast_token_loss" in output_dict:
            metrics["vlm_fast_loss"] = output_dict["fast_token_loss"].item()
        if "flowmatching_loss" in output_dict:
            metrics["diffusion_action_loss"] = output_dict["flowmatching_loss"].item()
            metrics["action_dit_loss"] = metrics["diffusion_action_loss"]

        # Foresight Forcing optimizes the reliability-weighted action loss, but
        # keeps the unweighted DiT loss comparable across training runs.
        if "raw_action_flow_loss" in metrics:
            metrics["action_dit_loss"] = metrics["raw_action_flow_loss"]
        if "reliability_weighted_action_flow_loss" in metrics:
            metrics["weighted_action_dit_loss"] = metrics["reliability_weighted_action_flow_loss"]

        return metrics

    def _active_gradient_clipping_threshold(self) -> float | None:
        """Return the clipping threshold that actually owns the optimizer update."""
        deepspeed_wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        deepspeed_engine = getattr(deepspeed_wrapper, "engine", None)
        gradient_clipping = getattr(deepspeed_engine, "gradient_clipping", None)
        if callable(gradient_clipping):
            threshold = float(gradient_clipping())
            return threshold if threshold > 0.0 else None

        configured_threshold = self.config.trainer.gradient_clipping
        if configured_threshold is None:
            return None
        threshold = float(configured_threshold)
        return threshold if threshold > 0.0 else None

    def _uses_deepspeed_optimizer(self) -> bool:
        return getattr(self.accelerator, "deepspeed_engine_wrapped", None) is not None

    def _is_gradient_logging_step(self) -> bool:
        logging_frequency = max(1, int(getattr(self.config.trainer, "logging_frequency", 1)))
        return (int(self.completed_steps) + 1) % logging_frequency == 0

    def _collect_gradient_metrics(self) -> dict[str, float]:
        """Clip ordinary gradients or read DeepSpeed's cached pre-clip optimizer-input norm.

        For hybrid Muon, DeepSpeed's value mixes post-Newton-Schulz Muon updates
        with raw auxiliary-Adam gradients; it is not a raw backward-gradient norm.
        """
        if not self.accelerator.sync_gradients:
            return {}
        clip_threshold = self._active_gradient_clipping_threshold()
        should_log = self._is_gradient_logging_step()
        if not should_log and (self._uses_deepspeed_optimizer() or clip_threshold is None):
            return {}
        if self._uses_deepspeed_optimizer():
            deepspeed_wrapper = self.accelerator.deepspeed_engine_wrapped
            get_global_grad_norm = getattr(deepspeed_wrapper, "get_global_grad_norm", None)
            if not callable(get_global_grad_norm):
                deepspeed_engine = getattr(deepspeed_wrapper, "engine", None)
                get_global_grad_norm = getattr(deepspeed_engine, "get_global_grad_norm", None)
            global_grad_norm = get_global_grad_norm() if callable(get_global_grad_norm) else None
        else:
            probe_threshold = clip_threshold if clip_threshold is not None else math.inf
            global_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), probe_threshold)
        if not should_log:
            return {}
        metrics = gradient_norm_metrics(global_grad_norm, clip_threshold)
        if self._uses_deepspeed_optimizer():
            deepspeed_wrapper = self.accelerator.deepspeed_engine_wrapped
            deepspeed_engine = getattr(deepspeed_wrapper, "engine", None)
            if deepspeed_engine is not None:
                metrics.update(get_deepspeed_muon_aux_adamw_clip_metrics(deepspeed_engine))
        return metrics

    def _step_lr_scheduler_after_optimizer(self) -> bool:
        """Advance the scheduler exactly once after a successful synchronized optimizer update."""
        optimizer_step_was_skipped = bool(getattr(self.accelerator, "optimizer_step_was_skipped", False))
        self._last_optimizer_step_performed = self.accelerator.sync_gradients and not optimizer_step_was_skipped
        if self._last_optimizer_step_performed:
            self.lr_scheduler.step()
        return self._last_optimizer_step_performed

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                total_loss = output_dict.get("loss", output_dict.get("action_loss"))
                if total_loss is None:
                    raise KeyError("Model output must contain 'loss' or backward-compatible 'action_loss'")

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            gradient_metrics = {}
            if not self._uses_deepspeed_optimizer():
                gradient_metrics = self._collect_gradient_metrics()

            # optimizer step
            self.optimizer.step()
            if self._step_lr_scheduler_after_optimizer():
                if self._uses_deepspeed_optimizer():
                    # Accelerate 1.14 performs the DeepSpeed step from backward();
                    # older integrations do it from optimizer.step(). Reading here
                    # avoids a previous-step cache wherever the norm is exposed.
                    gradient_metrics = self._collect_gradient_metrics()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                update_teacher = getattr(unwrapped_model, "update_self_flow_teacher", None)
                if callable(update_teacher):
                    update_teacher()
            # AcceleratedOptimizer.zero_grad() is intentionally a no-op on
            # non-sync micro-steps and clears accumulated gradients here only
            # after the synchronized update.
            self.optimizer.zero_grad()

        metrics = self._extract_step_metrics(output_dict)
        if self._last_optimizer_step_performed:
            metrics.update(gradient_metrics)
        return metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        should_save_final = self.config.trainer.get("save_final", True)
        if should_save_final and self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            model_checkpoint_path = os.path.join(final_checkpoint, "pytorch_model.pt")
            torch.save(state_dict, model_checkpoint_path)
            self._save_training_ema_state(model_checkpoint_path)
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if should_save_final:
            self._save_latest_optimizer_state()

        # close TensorBoard writer
        if self.accelerator.is_main_process:
            self.writer.close()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)

    accelerator = build_accelerator(cfg, DeepSpeedPlugin())
    accelerator.print(accelerator.state)

    logger.info("VLA Training :: Warming Up")
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    # compute max_train_steps from max_train_epochs if provided
    max_train_epochs = getattr(cfg.trainer, "max_train_epochs", None)
    if max_train_epochs is not None:
        cfg.trainer.max_train_steps, dataset_size, total_batch_size = compute_max_train_steps(
            cfg=cfg,
            accelerator=accelerator,
            vla_train_dataloader=vla_train_dataloader,
            max_train_epochs=max_train_epochs,
        )
        if accelerator.is_main_process:
            logger.info(
                f"Computed max_train_steps={cfg.trainer.max_train_steps} "
                f"from max_train_epochs={max_train_epochs}, "
                f"dataset_size={dataset_size}, total_batch_size={total_batch_size}"
            )

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


def cli(argv: list[str] | None = None) -> None:
    """Load a training config and CLI overrides, including access-tracked checkpoint configs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args(argv)

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Access-tracked checkpoint configs can omit this optional, pre-tracking flag.
    if cfg.get("is_debug", False) and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("127.0.0.1", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)


if __name__ == "__main__":
    cli()
