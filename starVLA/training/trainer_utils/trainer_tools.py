# EgoForesight modifications Copyright 2026 Chubin Zhang.
"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

import json
import os
import re
from typing import Tuple

import numpy as np
import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)

_OPTIMIZER_POLICY_ALIASES = {
    "action_rep_projector": "mot_model",
    "vision_rep_projector": "mot_model",
}


def resolve_optimizer_policy_group(group_name: str) -> str:
    """Resolve modules that intentionally inherit another module's optimizer policy."""

    top_level_group = str(group_name).split("/", 1)[0].split(".", 1)[0]
    return _OPTIMIZER_POLICY_ALIASES.get(top_level_group, top_level_group)


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters
            params = [
                p
                for p in module.parameters()
                if p.requires_grad and id(p) not in frozen_params and id(p) not in used_params
            ]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError as error:
            raise ValueError(f"Module path `{module_name}` from trainer.learning_rate does not exist") from error

    # Representation projectors are registered at framework root for checkpoint
    # compatibility, but semantically belong to the MoT action model. Unless a
    # projector has an explicit LR entry, inherit the mot_model policy so CLI
    # overrides remain synchronized.
    for module_name, policy_group in _OPTIMIZER_POLICY_ALIASES.items():
        if module_name in lr_cfg:
            continue
        module = getattr(model, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        params = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in frozen_params and id(parameter) not in used_params
        ]
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": lr_cfg.get(policy_group, base_lr),
                    "name": module_name,
                }
            )
            used_params.update(id(parameter) for parameter in params)

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in used_params and id(p) not in frozen_params
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from PIL import Image
from torchvision.ops import box_iou


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#" * 30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if dist.get_rank == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        def _log_incompatible_keys(scope_name, incompatible_keys):
            if dist.get_rank() != 0:
                return
            missing_keys = list(incompatible_keys.missing_keys)
            unexpected_keys = list(incompatible_keys.unexpected_keys)
            if missing_keys:
                print(f"⚠️ [{scope_name}] missing_keys ({len(missing_keys)}): {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ [{scope_name}] unexpected_keys ({len(unexpected_keys)}): {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                print(f"✅ [{scope_name}] no missing_keys or unexpected_keys")

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        incompatible_keys = module.load_state_dict(sub_state_dict, strict=True)
                        _log_incompatible_keys(path, incompatible_keys)
                        if dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                incompatible_keys = model.load_state_dict(checkpoint, strict=False)
                _log_incompatible_keys("<full_model>", incompatible_keys)
                if dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    def _get_deepspeed_engine(self):
        """Return the DeepSpeed engine when Accelerate wrapped the model with one."""
        model = getattr(self, "model", None)
        candidates = [model]
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and model is not None:
            try:
                candidates.append(accelerator.unwrap_model(model))
            except Exception:
                pass

        for candidate in candidates:
            if self._is_deepspeed_engine(candidate):
                return candidate
        return None

    @staticmethod
    def _is_deepspeed_engine(candidate) -> bool:
        if candidate is None or not callable(getattr(candidate, "save_checkpoint", None)):
            return False
        candidate_type = type(candidate)
        return candidate_type.__name__ == "DeepSpeedEngine" or candidate_type.__module__.startswith("deepspeed.")

    def _write_optimizer_state_metadata(self, optimizer_state_dir: str, state_format: str) -> None:
        metadata = {
            "steps": int(getattr(self, "completed_steps", 0)),
            "format": state_format,
            "world_size": int(getattr(getattr(self, "accelerator", None), "num_processes", 1)),
        }
        with open(os.path.join(optimizer_state_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _save_latest_optimizer_state(self) -> None:
        """Save a step-indexed optimizer state, including DeepSpeed ZeRO shards."""
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return

        optimizer_state_dir = os.path.join(self.checkpoint_dir, "latest_optimizer_state")
        deepspeed_engine = self._get_deepspeed_engine()
        if deepspeed_engine is not None:
            self._save_latest_deepspeed_optimizer_state(deepspeed_engine, optimizer_state_dir)
        else:
            self._save_latest_torch_optimizer_state(optimizer_state_dir)

    def _save_latest_deepspeed_optimizer_state(self, deepspeed_engine, optimizer_state_dir: str) -> None:
        accelerator = self.accelerator
        accelerator.wait_for_everyone()

        tag = f"step_{int(getattr(self, 'completed_steps', 0))}"
        client_state = {
            "steps": int(getattr(self, "completed_steps", 0)),
            "checkpoint_type": "latest_optimizer_state",
        }
        deepspeed_engine.save_checkpoint(
            optimizer_state_dir,
            tag=tag,
            client_state=client_state,
            save_latest=True,
        )
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            self._write_optimizer_state_metadata(os.path.join(optimizer_state_dir, tag), "deepspeed")
            self._write_optimizer_state_metadata(optimizer_state_dir, "deepspeed")
            logger.info(f"DeepSpeed optimizer state saved at {os.path.join(optimizer_state_dir, tag)}")
        accelerator.wait_for_everyone()

    def _save_latest_torch_optimizer_state(self, optimizer_state_dir: str) -> None:
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tag = f"step_{int(getattr(self, 'completed_steps', 0))}"
            step_state_dir = os.path.join(optimizer_state_dir, tag)
            os.makedirs(step_state_dir, exist_ok=True)
            optimizer_state_path = os.path.join(step_state_dir, "optimizer_state.pt")
            tmp_state_path = optimizer_state_path + ".tmp"
            scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
            torch.save(
                {
                    "steps": int(getattr(self, "completed_steps", 0)),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                },
                tmp_state_path,
            )
            os.replace(tmp_state_path, optimizer_state_path)
            self._write_optimizer_state_metadata(step_state_dir, "torch")
            self._write_optimizer_state_metadata(optimizer_state_dir, "torch")
            latest_path = os.path.join(optimizer_state_dir, "latest")
            latest_tmp_path = latest_path + ".tmp"
            with open(latest_tmp_path, "w", encoding="utf-8") as f:
                f.write(tag)
            os.replace(latest_tmp_path, latest_path)
            logger.info(f"Optimizer state saved at {optimizer_state_path}")
        accelerator.wait_for_everyone()

    def _load_latest_optimizer_state(self) -> None:
        """Restore optimizer and scheduler state matching the selected model checkpoint."""
        optimizer_state_dir = os.path.join(self.checkpoint_dir, "latest_optimizer_state")
        model_step = int(getattr(self, "completed_steps", 0))
        step_state_dir = os.path.join(optimizer_state_dir, f"step_{model_step}")
        step_metadata_path = os.path.join(step_state_dir, "metadata.json")
        metadata_path = (
            step_metadata_path
            if os.path.isfile(step_metadata_path)
            else os.path.join(optimizer_state_dir, "metadata.json")
        )
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "A model checkpoint was selected for resume, but its optimizer metadata does not exist: "
                f"{metadata_path}"
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        optimizer_step = int(metadata.get("steps", -1))
        if optimizer_step != model_step:
            raise RuntimeError(
                f"Model checkpoint is at step {model_step}, " f"but latest optimizer state is at step {optimizer_step}."
            )

        state_format = metadata.get("format")
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        if state_format == "deepspeed":
            self._load_latest_deepspeed_optimizer_state(
                optimizer_state_dir=optimizer_state_dir,
                metadata=metadata,
                expected_step=model_step,
            )
        elif state_format == "torch":
            self._load_latest_torch_optimizer_state(
                optimizer_state_dir=(step_state_dir if os.path.isfile(step_metadata_path) else optimizer_state_dir),
                expected_step=model_step,
            )
        else:
            raise RuntimeError(f"Unsupported optimizer state format in {metadata_path}: {state_format}")
        accelerator.wait_for_everyone()
        logger.info(f"Resumed optimizer and scheduler state at step {model_step}")

    def _load_latest_deepspeed_optimizer_state(
        self,
        optimizer_state_dir: str,
        metadata: dict,
        expected_step: int,
    ) -> None:
        deepspeed_engine = self._get_deepspeed_engine()
        if deepspeed_engine is None:
            raise RuntimeError(
                "Checkpoint contains DeepSpeed optimizer state, but the current run is not using DeepSpeed."
            )

        saved_world_size = int(metadata.get("world_size", self.accelerator.num_processes))
        current_world_size = int(self.accelerator.num_processes)
        if saved_world_size != current_world_size:
            raise RuntimeError(
                "DeepSpeed ZeRO optimizer state cannot be safely resumed with a different world size: "
                f"checkpoint={saved_world_size}, current={current_world_size}."
            )

        # The standalone model checkpoint has already been loaded. DeepSpeed would
        # otherwise reload its model copy and index every frozen parameter in the
        # old frozen-param fragments. Temporarily marking frozen parameters as
        # trainable avoids that legacy path; the no-op module loader preserves the
        # already-restored model while optimizer and scheduler state are loaded.
        frozen_parameters = [
            parameter for parameter in deepspeed_engine.module.parameters() if not parameter.requires_grad
        ]
        try:
            for parameter in frozen_parameters:
                parameter.requires_grad_(True)
            tag = f"step_{expected_step}"
            load_path, client_state = deepspeed_engine.load_checkpoint(
                optimizer_state_dir,
                tag=tag if os.path.isdir(os.path.join(optimizer_state_dir, tag)) else None,
                load_module_strict=False,
                custom_load_fn=lambda src, dst: None,
            )
        finally:
            for parameter in frozen_parameters:
                parameter.requires_grad_(False)

        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed optimizer state from {optimizer_state_dir}")
        restored_step = int(client_state.get("steps", -1))
        if restored_step != expected_step:
            raise RuntimeError(
                f"DeepSpeed client state is at step {restored_step}, "
                f"but the model checkpoint is at step {expected_step}."
            )

    def _load_latest_torch_optimizer_state(self, optimizer_state_dir: str, expected_step: int) -> None:
        optimizer_state_path = os.path.join(optimizer_state_dir, "optimizer_state.pt")
        if not os.path.isfile(optimizer_state_path):
            raise FileNotFoundError(f"Latest optimizer state does not exist: {optimizer_state_path}")

        state = torch.load(optimizer_state_path, map_location="cpu", weights_only=True)
        restored_step = int(state.get("steps", -1))
        if restored_step != expected_step:
            raise RuntimeError(
                f"Torch optimizer state is at step {restored_step}, "
                f"but the model checkpoint is at step {expected_step}."
            )
        if "optimizer" not in state:
            raise RuntimeError(f"Torch optimizer state is missing the optimizer payload: {optimizer_state_path}")
        self.optimizer.load_state_dict(state["optimizer"])

        scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
        scheduler_state = state.get("lr_scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            # Compatibility with older lightweight checkpoints that saved only
            # optimizer state. Transformer schedulers are deterministic by step.
            scheduler.step(expected_step)

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # 获取所有符合命名规则，确保只匹配以 .pt 结尾的文件
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if re.match(r"steps_(\d+)_pytorch_model\.pt$", f)  # 添加 $ 确保以 .pt 结尾
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # 确保是文件
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # 提取步数并排序
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_pytorch_model\.pt", ckpt).group(1))) for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # 按步数排序，获取最新的 checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0
