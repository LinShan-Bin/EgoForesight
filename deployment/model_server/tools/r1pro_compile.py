# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Opt-in, parity-checked MoT/VLM acceleration without changing weights or action semantics.

"""Opt-in native CUDA Graph replay with strict eager-parity startup checks.

MoT3 uses whole-flow capture with request-local conditioning reuse and persistent
BF16 Linear weight copies; other MoT models use per-step capture. Optionally
capture the Qwen vision tower and decoder, including selected intermediate
features. CPU preprocessing, DINO and multimodal orchestration remain eager.
Every tensor input is refreshed on every invocation; returned tensors own their
storage. Use one policy worker. Graphs replay original, not Inductor, kernels.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def _map_tensors(value: Any, function: Callable) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: _map_tensors(getattr(value, field.name), function) for field in dataclasses.fields(value)},
        )
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported CUDA Graph value: {type(value).__name__}")


def _signature(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, dict):
        return ("dict", tuple((key, _signature(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_signature(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported CUDA Graph input: {type(value).__name__}")


def _tensors(value: Any) -> list[torch.Tensor]:
    result = []
    _map_tensors(value, result.append)
    return result


def _check_tensor_outputs(actual: Any, expected: Any) -> None:
    left, right = _tensors(actual), _tensors(expected)
    if not left or len(left) != len(right):
        raise RuntimeError("CUDA Graph changed the output structure")
    for index, (observed, reference) in enumerate(zip(left, right, strict=True)):
        if (
            observed.shape != reference.shape
            or not torch.isfinite(observed).all()
            or not torch.equal(observed, reference)
        ):
            raise RuntimeError(f"CUDA Graph failed exact eager equivalence at output {index}")


class NativeMoTGraph:
    """Bounded, shape-keyed capture of the original kernels with owned inputs/outputs.

    A dedicated stream serializes capture, input copies, replay and output copies
    across the startup thread and the async worker. Each new shape is checked
    against eager execution before use; failures propagate rather than fall back.
    """

    def __init__(self, forward: Callable, *, max_graphs: int = 2, name: str = "MoT") -> None:
        if max_graphs < 1:
            raise ValueError("compile-max-graphs must be positive")
        self.forward = forward
        self.name = name
        self.max_graphs = max_graphs
        self.cache: OrderedDict = OrderedDict()
        self.lock = RLock()
        self.stream = None
        self.captures = 0
        self.replays = 0

    @torch.inference_mode()
    def __call__(self, **kwargs: Any) -> Any:
        tensors = _tensors(kwargs)
        if not tensors or any(tensor.device.type != "cuda" or tensor.requires_grad for tensor in tensors):
            raise ValueError("Native CUDA Graph requires CUDA inference tensors")
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All graph inputs must use one CUDA device")
        key = (_signature(kwargs), torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        with self.lock, torch.cuda.device(device):
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=device)
            if self.stream.device != device:
                raise ValueError("Cannot move a captured policy to another device")
            caller = torch.cuda.current_stream(device)
            self.stream.wait_stream(caller)
            try:
                with torch.cuda.stream(self.stream):
                    if key not in self.cache:
                        self._capture(key, kwargs, device)
                    static, graph, output = self.cache[key]
                    self.cache.move_to_end(key)
                    for target, source in zip(_tensors(static), tensors, strict=True):
                        target.copy_(source)
                        source.record_stream(self.stream)
                    graph.replay()
                    result = _map_tensors(output, torch.clone)
                    self.replays += 1
                for tensor in _tensors(result):
                    tensor.record_stream(caller)
                return result
            finally:
                caller.wait_stream(self.stream)

    def _capture(self, key: tuple, kwargs: dict, device: torch.device) -> None:
        if len(self.cache) >= self.max_graphs:
            torch.cuda.synchronize(device)
            self.cache.popitem(last=False)
        static = _map_tensors(kwargs, torch.clone)
        # Never capture pointers into autocast's temporary weight cache: the
        # enclosing predict_action context clears it before the next request.
        autocast = dict(
            device_type="cuda",
            enabled=torch.is_autocast_enabled("cuda"),
            dtype=torch.get_autocast_dtype("cuda"),
            cache_enabled=False,
        )
        with torch.autocast(**autocast):
            for _ in range(3):
                expected = self.forward(**static)
        expected = _map_tensors(expected, torch.clone)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream), torch.autocast(**autocast):
            output = self.forward(**static)
        graph.replay()
        _check_tensor_outputs(output, expected)
        self.cache[key] = (static, graph, output)
        self.captures += 1
        LOGGER.info(
            "Captured %s CUDA Graph #%d; %d/%d shapes retained",
            self.name,
            self.captures,
            len(self.cache),
            self.max_graphs,
        )


class BF16LinearCache:
    """Own inference-only BF16 Linear weight copies without modifying state_dict.

    FP32 master parameters and normalization weights remain untouched. Copies use
    the same rounding as BF16 autocast, but outlive request-local autocast caches.
    Install after checkpoint/EMA loading; mutation requires rebuilding the policy.
    """

    @torch.inference_mode()
    def __init__(self, model: torch.nn.Module) -> None:
        self.entries: list[dict[str, Any]] = []
        self.bytes = 0
        try:
            for module in model.modules():
                if type(module) is not torch.nn.Linear or module.weight.dtype != torch.float32:
                    continue
                if module.training or module.weight.device.type != "cuda":
                    raise ValueError("BF16 Linear caching requires eval-mode CUDA modules")
                weight = module.weight.to(torch.bfloat16)
                bias = None if module.bias is None else module.bias.to(torch.bfloat16)
                entry = dict(
                    module=module,
                    original=module.forward,
                    had_forward="forward" in module.__dict__,
                    parameters=(module.weight, module.bias),
                    versions=tuple(None if value is None else value._version for value in (module.weight, module.bias)),
                    copies=(weight, bias),
                )
                self.entries.append(entry)
                self.bytes += sum(value.numel() * value.element_size() for value in (weight, bias) if value is not None)
                module.forward = self._forward(entry)
        except BaseException:
            self.restore()
            raise

    @staticmethod
    def _forward(entry: dict[str, Any]) -> Callable:
        def forward(input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - match nn.Linear.forward
            if entry["module"].training or torch.is_grad_enabled():
                raise RuntimeError("Cached BF16 Linear weights support inference only")
            if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
                return torch.nn.functional.linear(input, *entry["copies"])
            return entry["original"](input)

        return forward

    def validate(self) -> None:
        """Reject stale copies before replay, where Python Linear guards do not run."""
        for entry in self.entries:
            module = entry["module"]
            for current, original, version in zip(
                (module.weight, module.bias), entry["parameters"], entry["versions"], strict=True
            ):
                if module.training or current is not original or (current is not None and current._version != version):
                    raise RuntimeError("MoT weights changed after compilation; reload and recompile the policy")

    def restore(self) -> None:
        """Restore original forwards on failed startup or explicit benchmark teardown."""
        for entry in self.entries:
            if entry["had_forward"]:
                entry["module"].forward = entry["original"]
            else:
                entry["module"].__dict__.pop("forward", None)
        self.entries.clear()


class NativeFlowGraph:
    """Capture the complete original flow loop with fresh externally sampled noise.

    Only the four tensor batch fields read by MoT2/3 integration are retained;
    images, text, metadata and CPU arrays never enter a CUDA Graph. Context and
    intermediate velocities stay on the graph stream throughout all flow steps.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_graphs: int = 2,
        linear_cache: BF16LinearCache | None = None,
    ) -> None:
        self.model = model
        self.linear_cache = linear_cache
        self.integrate = model._integrate_flow
        self.graph = NativeMoTGraph(self._forward, max_graphs=max_graphs, name="complete flow loop")

    def _forward(self, **kwargs: Any) -> Any:
        with self.model._autocast_context():
            for mask_key, kwargs_key in (
                ("context_mask", "inference_mot_kwargs"),
                ("unconditional_mask", "unconditional_kwargs"),
            ):
                if kwargs[kwargs_key] is None:
                    continue
                mot_kwargs = dict(kwargs[kwargs_key])
                mot_kwargs["inference_cache"] = self.model.mot_model.prepare_inference_cache(
                    context=kwargs["context"],
                    current_vision=kwargs["current_vision"],
                    context_mask=kwargs[mask_key],
                    meta_tokens=mot_kwargs.get("meta_tokens"),
                    meta_mask=mot_kwargs.get("meta_mask"),
                )
                kwargs[kwargs_key] = mot_kwargs
        return self.integrate(**kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        if self.model.training:
            raise RuntimeError("Whole-flow CUDA Graph supports inference only")
        if self.linear_cache is not None:
            self.linear_cache.validate()
        kwargs["batch"] = {
            key: value
            for key, value in kwargs["batch"].items()
            if key in {"state", "state_mask", "action_mask", "_flow_action_schema_mask"}
        }
        return self.graph(**kwargs)


class VisionVLMGraph:
    """Capture the Qwen vision tower after preparing shape-only positional inputs."""

    def __init__(self, visual: Any, *, max_graphs: int = 2) -> None:
        self.visual = visual
        self.original = visual.forward
        self.graph = NativeMoTGraph(self._forward, max_graphs=max_graphs, name="VLM vision tower")
        self.metadata: OrderedDict = OrderedDict()
        self.max_graphs = max_graphs

    def _forward(self, *, grid_key: tuple, **kwargs: Any) -> Any:
        del grid_key
        return self.original(**kwargs)

    @torch.inference_mode()
    def __call__(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> Any:
        from transformers.vision_utils import (
            get_vision_bilinear_indices_and_weights,
            get_vision_cu_seqlens,
            get_vision_position_ids,
        )

        key = (tuple(map(tuple, grid_thw.cpu().tolist())), str(grid_thw.device))
        if key not in self.metadata:
            indices, weights = get_vision_bilinear_indices_and_weights(
                grid_thw,
                num_grid_per_side=self.visual.num_grid_per_side,
                spatial_merge_size=self.visual.spatial_merge_size,
            )
            self.metadata[key] = {
                "bilinear_indices": indices,
                "bilinear_weights": weights,
                "position_ids": get_vision_position_ids(grid_thw, self.visual.spatial_merge_size),
                "cu_seqlens": get_vision_cu_seqlens(grid_thw),
            }
            if len(self.metadata) > self.max_graphs:
                self.metadata.popitem(last=False)
        self.metadata.move_to_end(key)
        previous_captures = self.graph.captures
        output = self.graph(
            hidden_states=hidden_states, grid_thw=grid_thw, grid_key=key, **{**kwargs, **self.metadata[key]}
        )
        if self.graph.captures != previous_captures:
            _check_tensor_outputs(output, self.original(hidden_states, grid_thw, **kwargs))
        return output


class SelectedVLMGraph:
    """Capture the vision tower and decoder with all selected intermediate features.

    The multimodal base still refreshes image embeddings and RoPE positions on
    every observation. Vision and decoder calls are temporarily intercepted; no
    modules/weights are replaced. Hooks run during capture, not on replay, and
    the graph explicitly returns every requested intermediate feature.
    """

    def __init__(self, interface: Any, hidden_layers: tuple[int, ...], *, max_graphs: int = 2) -> None:
        self.interface = interface
        self.base, self.language = interface._context_language_model()
        self.hidden_layers = tuple(hidden_layers)
        self.original = self.language.forward
        self.visual = VisionVLMGraph(self.base.visual, max_graphs=max_graphs)
        self.lock = RLock()
        self.graph = NativeMoTGraph(self._forward_with_states, max_graphs=max_graphs, name="VLM decoder")
        self.fast_path = interface.require_gdn_fast_path()

    def _forward_with_states(self, **kwargs: Any) -> tuple:
        captured, hooks = {}, []
        layers = self.language.layers
        for index in sorted(set(self.hidden_layers) - {len(layers)}):

            def capture(_module: Any, args: tuple, call_kwargs: dict, index: int = index) -> None:
                captured[index] = call_kwargs.get("hidden_states", args[0] if args else None)

            hooks.append(layers[index].register_forward_pre_hook(capture, with_kwargs=True))
        try:
            output = self.original(**kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if len(layers) in self.hidden_layers:
            captured[len(layers)] = output.last_hidden_state
        selected = tuple(captured[index] for index in self.hidden_layers)
        if any(not isinstance(value, torch.Tensor) for value in selected):
            raise RuntimeError("VLM graph did not observe all selected hidden states")
        return output, selected

    @torch.inference_mode()
    def __call__(self, requested_layers: tuple, kwargs: dict) -> tuple:
        if tuple(requested_layers) != self.hidden_layers:
            raise ValueError("VLM graph requested hidden layers differ from the serving configuration")
        if self.base.training or self.language.training:
            raise ValueError("VLM CUDA Graph is inference-only")
        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("labels", None)
        layer_kwargs.update(output_hidden_states=False, return_dict=True, use_cache=False)
        selected = None
        calls = 0

        def forward(*args: Any, **call_kwargs: Any) -> Any:
            nonlocal selected, calls
            calls += 1
            if calls != 1 or call_kwargs.get("past_key_values") is not None or call_kwargs.get("use_cache"):
                raise RuntimeError("VLM graph requires exactly one stateless decoder call")
            (_, normalized), _ = self.interface._normalize_native_cuda_graph_call(args, call_kwargs)
            previous_captures = self.graph.captures
            output, selected = self.graph(**normalized)
            if self.graph.captures != previous_captures:
                # Prove replacing a batch-one all-ones mask with None is exact,
                # including all eight features, for every newly captured shape.
                reference = self._forward_with_states(**call_kwargs)
                _check_tensor_outputs((output, selected), reference)
            return output

        with self.lock, self.interface._autocast_context(torch.bfloat16):
            had_forward = "forward" in self.language.__dict__
            previous_forward = self.language.forward
            self.language.forward = forward
            had_visual_forward = "forward" in self.base.visual.__dict__
            previous_visual_forward = self.base.visual.forward
            self.base.visual.forward = self.visual
            try:
                self.base(**layer_kwargs)
            finally:
                if had_visual_forward:
                    self.base.visual.forward = previous_visual_forward
                else:
                    del self.base.visual.forward
                if had_forward:
                    self.language.forward = previous_forward
                else:
                    del self.language.forward
        if calls != 1 or selected is None:
            raise RuntimeError("Multimodal VLM did not call the captured language decoder")
        return selected


def _check_policy_outputs(actual: dict, expected: dict) -> dict:
    differences = {}
    if actual.keys() != expected.keys():
        raise RuntimeError("Compiled inference changed policy output fields")
    for name in expected:
        observed, reference = np.asarray(actual[name]), np.asarray(expected[name])
        if observed.shape != reference.shape or not np.isfinite(observed).all():
            raise RuntimeError(f"Compiled inference returned invalid {name}")
        difference = float(np.max(np.abs(observed.astype(np.float64) - reference.astype(np.float64))))
        differences[name] = difference
        if not np.array_equal(observed, reference):
            raise RuntimeError(f"Compiled inference failed exact eager equivalence: {name} max_abs={difference:.9g}")
    return differences


def compile_policy(
    policy: Any,
    *,
    max_graphs: int = 2,
    compile_vlm: bool = False,
) -> dict:
    """Install acceleration after EMA/device/eval setup; fail before server bind on drift.

    These synthetic startup checks guard implementation parity; they do not
    establish real-robot accuracy or physical safety.
    """
    if max_graphs < 1:
        raise ValueError("compile-max-graphs must be positive")
    model = policy.model
    whole_flow = type(model).__name__ == "Qwen35MoT3"
    mot = getattr(model, "mot_model", None)
    if mot is None or model.training or mot.training:
        raise ValueError("Compilation requires an eval-mode model with mot_model")
    device = next(mot.parameters()).device
    if device.type != "cuda":
        raise ValueError("R1Pro compiled inference requires CUDA")
    state_keys = tuple(model.state_dict())
    original = mot.forward
    original_flow = getattr(model, "_integrate_flow", None)
    had_flow = "_integrate_flow" in model.__dict__
    interface = getattr(model, "qwen_vl_interface", None)
    if compile_vlm and (interface is None or not hasattr(model, "vlm_feature_layers")):
        raise ValueError("VLM compilation requires the Qwen3.5 selected-layer interface")
    if compile_vlm and (
        getattr(interface, "context_encoder_compiled", False)
        or getattr(interface, "_selected_context_cuda_graph", None) is not None
    ):
        raise ValueError("VLM already has an installed compiler")
    had_forward = "forward" in mot.__dict__
    started = time.perf_counter()
    rng = np.random.default_rng(42)
    probes = []
    for index, prompt in enumerate(("Pick fruit.", "Please pick up the red fruit and put it into the basket.")):
        channels = {
            key: (
                rng.integers(0, 256, size=desc["shape"], dtype=np.uint8)
                if desc["quantity"] == "rgb"
                else np.full(desc["shape"], 0.01 * index, dtype=np.float32)
            )
            for key, desc in policy.contract["observations"].items()
        }
        probes.append((*policy.adapter.inputs(channels, prompt), 42 + index))
    linear_cache = None
    try:
        with torch.random.fork_rng(devices=[device.index]):
            references = [
                model.predict_action([example], num_flow_steps=policy.flow_steps, seed=seed)
                for example, _, seed in probes
            ]
            if whole_flow:
                if isinstance(original, NativeMoTGraph) or isinstance(original_flow, NativeFlowGraph):
                    raise ValueError("Whole-flow capture requires unwrapped MoT and flow callables")
                linear_cache = BF16LinearCache(mot)
                model._integrate_flow = NativeFlowGraph(model, max_graphs=max_graphs, linear_cache=linear_cache)
            else:
                mot.forward = NativeMoTGraph(original, max_graphs=max_graphs)
            if compile_vlm:
                interface._selected_context_cuda_graph = SelectedVLMGraph(
                    interface, tuple(model.vlm_feature_layers), max_graphs=max_graphs
                )
            checks = []
            # Repeat the first shape after the other prompt to exercise refreshed graph inputs.
            for index in (0, 1, 0):
                example, measured, seed = probes[index]
                result = model.predict_action([example], num_flow_steps=policy.flow_steps, seed=seed)
                checks.append(_check_policy_outputs(result, references[index]))
                _check_policy_outputs(
                    policy.adapter.outputs(result["normalized_actions"], measured),
                    policy.adapter.outputs(references[index]["normalized_actions"], measured),
                )
        if tuple(model.state_dict()) != state_keys:
            raise RuntimeError("Compilation changed model state_dict keys")
    except BaseException:
        if linear_cache is not None:
            linear_cache.restore()
        if whole_flow:
            if had_flow:
                model._integrate_flow = original_flow
            else:
                model.__dict__.pop("_integrate_flow", None)
        if compile_vlm:
            interface.__dict__.pop("_selected_context_cuda_graph", None)
        if had_forward:
            mot.forward = original
        else:
            mot.__dict__.pop("forward", None)
        raise
    torch.cuda.synchronize(device)
    return {
        "enabled": True,
        "status": "verified_synthetic_exact",
        "backend": "native-cuda-graph",
        "scope": "complete flow loop" if whole_flow else "mot_model.forward",
        "vlm_compiled": compile_vlm,
        "vlm_backend": "native-cuda-graph" if compile_vlm else None,
        "vlm_scope": "vision tower and language decoder with all selected hidden states" if compile_vlm else None,
        "vlm_vision_tower_compiled": compile_vlm,
        "vlm_hidden_layers": list(model.vlm_feature_layers) if compile_vlm else [],
        "dino_compiled": False,
        "outer_flow_loop_compiled": whole_flow,
        "request_local_conditioning_cache": whole_flow,
        "bf16_linear_weight_cache": (
            None
            if linear_cache is None
            else {
                "modules": len(linear_cache.entries),
                "bytes": linear_cache.bytes,
                "master_parameters_changed": False,
            }
        ),
        "flow_steps": policy.flow_steps,
        "options": {"warmup": 3, "max_graphs": max_graphs},
        "setup_and_validation_seconds": time.perf_counter() - started,
        "parity_checks_max_abs": checks,
    }
