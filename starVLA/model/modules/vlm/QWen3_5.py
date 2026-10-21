# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Add Qwen3.5 context encoding, selected-layer capture, and fixed-shape CUDA Graph inference.

import copy
import inspect
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

try:
    import transformers
except ImportError:  # Keep the factory importable so initialization can report the missing dependency clearly.
    transformers = None

try:
    from accelerate.logging import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

fallback_logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
_MODEL_CLASS_CANDIDATES = (
    "Qwen3_5ForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
)
_MISSING_FORWARD = object()


def _log_warning(message: str, *args: Any) -> None:
    """Log before or after Accelerate state initialization."""
    try:
        logger.warning(message, *args)
    except RuntimeError:
        fallback_logger.warning(message, *args)


def _config_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a key from dictionaries, OmegaConf nodes, or attribute-style configuration objects."""
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)

    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (AttributeError, TypeError):
            pass
    return getattr(container, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_transformers_components(transformers_module: Any = None) -> tuple[Any, Any, str]:
    """Resolve the native Qwen3.5 class first and then supported Transformers auto-model fallbacks."""
    module = transformers if transformers_module is None else transformers_module
    checked_classes = ", ".join(_MODEL_CLASS_CANDIDATES)
    if module is None:
        raise ImportError(
            "Qwen3.5 VLM requires Transformers with AutoProcessor and a multimodal conditional-generation "
            f"loader. Checked model classes: {checked_classes}."
        )

    processor_class = getattr(module, "AutoProcessor", None)
    if processor_class is None:
        version = getattr(module, "__version__", "unknown")
        raise ImportError(f"Transformers {version} does not expose AutoProcessor, which is required by Qwen3.5 VLM.")

    for class_name in _MODEL_CLASS_CANDIDATES:
        model_class = getattr(module, class_name, None)
        if model_class is not None:
            if class_name != _MODEL_CLASS_CANDIDATES[0]:
                version = getattr(module, "__version__", "unknown")
                _log_warning(
                    "Transformers %s does not expose %s; loading Qwen3.5 through the compatible %s fallback.",
                    version,
                    _MODEL_CLASS_CANDIDATES[0],
                    class_name,
                )
            return model_class, processor_class, class_name

    version = getattr(module, "__version__", "unknown")
    raise ImportError(
        f"Transformers {version} cannot load Qwen3.5 multimodal checkpoints. Checked model classes: "
        f"{checked_classes}. Upgrade Transformers to a version with Qwen3.5 multimodal support."
    )


def _infer_hidden_size(model: nn.Module) -> int:
    """Infer the language hidden size from common Hugging Face multimodal config layouts."""
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)

    if hidden_size is None:
        get_text_config = getattr(model_config, "get_text_config", None)
        if callable(get_text_config):
            hidden_size = getattr(get_text_config(), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError(
            "Could not infer Qwen3.5 hidden_size from model.config.text_config.hidden_size or "
            "model.config.hidden_size."
        )
    return int(hidden_size)


def _last_hidden_state(output: Any) -> torch.Tensor:
    """Extract the final hidden-state tensor from common Transformers model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        hidden_state = output.get("last_hidden_state")
    else:
        hidden_state = getattr(output, "last_hidden_state", None)
    if hidden_state is None and isinstance(output, (tuple, list)) and output:
        hidden_state = output[0]
    if not isinstance(hidden_state, torch.Tensor):
        raise RuntimeError(f"Qwen language model returned no last_hidden_state tensor: {type(output).__name__}")
    return hidden_state


def _replace_last_hidden_state(output: Any, hidden_state: torch.Tensor) -> Any:
    """Rebuild a language-model output while replacing only its final hidden state."""
    if isinstance(output, torch.Tensor):
        return hidden_state
    if is_dataclass(output) and not isinstance(output, type):
        return replace(output, last_hidden_state=hidden_state)
    if isinstance(output, dict):
        result = type(output)(output)
        result["last_hidden_state"] = hidden_state
        return result
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("Qwen language model returned an empty tuple")
        return (hidden_state, *output[1:])
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Qwen language model returned an empty list")
        return [hidden_state, *output[1:]]
    if hasattr(output, "last_hidden_state"):
        result = copy.copy(output)
        result.last_hidden_state = hidden_state
        return result
    raise RuntimeError(f"Cannot rebuild Qwen language-model output type {type(output).__name__}")


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return metadata that must remain fixed for a native CUDA Graph input."""
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
    )


def _tensor_metadata(tensor: torch.Tensor, *, leaf_index: int | None = None) -> dict[str, Any]:
    """Return JSON-serializable tensor metadata for benchmark reports."""
    metadata = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "layout": str(tensor.layout).removeprefix("torch."),
        "requires_grad": bool(tensor.requires_grad),
    }
    if leaf_index is not None:
        metadata["pytree_leaf_index"] = leaf_index
    return metadata


def _static_value_matches(expected: Any, actual: Any) -> bool:
    """Compare graph-static Python leaves without invoking arbitrary tensor-like equality."""
    scalar_types = (type(None), bool, int, float, str, bytes, torch.dtype, torch.device)
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, scalar_types):
        return bool(expected == actual)
    return expected is actual


def _exact_tensor_report(reference: torch.Tensor, candidate: torch.Tensor, reference_name: str) -> dict[str, Any]:
    """Compute an atol=rtol=0 correctness report and retain useful diagnostics."""
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Native CUDA Graph changed {reference_name} shape from "
            f"{tuple(reference.shape)} to {tuple(candidate.shape)}"
        )
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    absolute_error = (candidate_float - reference_float).abs()
    error = candidate_float - reference_float
    reference_flat = reference_float.flatten().double()
    candidate_flat = candidate_float.flatten().double()
    denominator = reference_flat.norm() * candidate_flat.norm()
    exact_match = bool(torch.equal(reference, candidate))
    cosine_similarity = (
        float(torch.dot(reference_flat, candidate_flat) / denominator)
        if float(denominator) > 0.0
        else float(exact_match)
    )
    return {
        "passed": exact_match,
        "exact_match": exact_match,
        "atol": 0.0,
        "rtol": 0.0,
        "max_abs": float(absolute_error.max()),
        "mean_abs": float(absolute_error.mean()),
        "rmse": float(torch.sqrt(torch.mean(error.square()))),
        "cosine_similarity": cosine_similarity,
        "reference": reference_name,
    }


def _collect_output_tensors(output: Any) -> list[torch.Tensor]:
    """Collect tensor fields from a Transformers output without traversing arbitrary model objects."""
    if isinstance(output, torch.Tensor):
        return [output]
    if is_dataclass(output) and not isinstance(output, type):
        values = [getattr(output, field.name) for field in fields(output)]
    elif isinstance(output, Mapping):
        values = list(output.values())
    elif isinstance(output, (tuple, list)):
        values = list(output)
    elif hasattr(output, "__dict__"):
        values = list(vars(output).values())
    else:
        return []

    tensors = []
    for value in values:
        if value is not None:
            tensors.extend(_collect_output_tensors(value))
    return tensors


def _require_hidden_state_only_output(output: Any) -> torch.Tensor:
    """Reject output templates that would otherwise retain stale tensors across graph replays."""
    hidden_state = _last_hidden_state(output)
    additional_tensors = [tensor for tensor in _collect_output_tensors(output) if tensor is not hidden_state]
    if additional_tensors:
        metadata = [_tensor_metadata(tensor) for tensor in additional_tensors]
        raise RuntimeError(
            "native_cuda_graph only supports language-model outputs whose sole tensor is "
            f"last_hidden_state; found additional tensor outputs: {metadata}"
        )
    return hidden_state


def _module_storage_signature(module: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Lock parameter/buffer object, storage, metadata, and mutation version for graph lifetime."""
    signature = []
    for kind, named_tensors in (
        ("parameter", module.named_parameters(recurse=True)),
        ("buffer", module.named_buffers(recurse=True)),
    ):
        for name, tensor in named_tensors:
            signature.append(
                (
                    kind,
                    name,
                    id(tensor),
                    tensor.data_ptr(),
                    _tensor_signature(tensor),
                    _tensor_mutation_version(tensor),
                )
            )
    return tuple(signature)


def _tensor_mutation_version(tensor: torch.Tensor) -> int | None:
    """Read the mutation version when the tensor type tracks one."""
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


class _TensorOnlyLanguageModel(nn.Module):
    """Expose only tensor leaves while retaining one fixed original language-model call tree."""

    def __init__(self, language_model: nn.Module, call_tree: Any) -> None:
        super().__init__()
        self.language_model = language_model
        leaves, self.tree_spec = tree_flatten(call_tree)
        self.tensor_indices = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        tensor_index_set = set(self.tensor_indices)
        self.static_values = tuple(None if index in tensor_index_set else leaf for index, leaf in enumerate(leaves))

    def sample_tensors(self, call_tree: Any) -> tuple[torch.Tensor, ...]:
        """Extract tensor leaves after checking the complete pytree structure."""
        leaves, tree_spec = tree_flatten(call_tree)
        if tree_spec != self.tree_spec:
            raise RuntimeError("Qwen language-model call structure changed before CUDA Graph capture")
        return tuple(leaves[index] for index in self.tensor_indices)

    def call_original(self, *tensor_inputs: torch.Tensor) -> Any:
        """Rebuild and invoke the exact original language-model call."""
        if len(tensor_inputs) != len(self.tensor_indices):
            raise RuntimeError(f"Expected {len(self.tensor_indices)} Qwen tensor inputs, got {len(tensor_inputs)}")
        leaves = list(self.static_values)
        for index, tensor in zip(self.tensor_indices, tensor_inputs, strict=True):
            leaves[index] = tensor
        args, kwargs = tree_unflatten(leaves, self.tree_spec)
        return self.language_model(*args, **kwargs)

    def forward(self, *tensor_inputs: torch.Tensor) -> torch.Tensor:
        return _last_hidden_state(self.call_original(*tensor_inputs))


@dataclass
class _NativeCudaGraphCapture:
    graph: Any
    static_inputs: tuple[torch.Tensor, ...]
    static_output: torch.Tensor
    capture_cold_ms: float
    replay_stream_id: int | None


class _NativeCudaGraphLanguageProxy:
    """Replay one fixed-shape graph while preserving the original Transformers output contract."""

    def __init__(
        self,
        adapter: _TensorOnlyLanguageModel,
        call_tree: Any,
        capture: _NativeCudaGraphCapture,
        output_template: Any,
        attention_mask_signature: tuple[Any, ...],
        language_model: nn.Module,
    ) -> None:
        leaves, tree_spec = tree_flatten(call_tree)
        if tree_spec != adapter.tree_spec:
            raise RuntimeError("Native CUDA Graph proxy received a mismatched initial Qwen call structure")
        self.tree_spec = adapter.tree_spec
        self.tensor_indices = adapter.tensor_indices
        tensor_index_set = set(self.tensor_indices)
        self.static_values = tuple(None if index in tensor_index_set else leaf for index, leaf in enumerate(leaves))
        self.tensor_signatures = tuple(_tensor_signature(leaves[index]) for index in self.tensor_indices)
        self.graph = capture.graph
        self.static_inputs = capture.static_inputs
        self.static_output = capture.static_output
        self.output_template = output_template
        self.attention_mask_signature = attention_mask_signature
        self.language_model = language_model
        self.module_storage_signature = _module_storage_signature(language_model)
        self.replay_stream_id = capture.replay_stream_id
        self.replay_lock = threading.Lock()

    def _normalize_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if args:
            raise RuntimeError("Native CUDA Graph Qwen replay only accepts the captured keyword-only call")
        attention_mask = kwargs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            raise RuntimeError("Native CUDA Graph Qwen replay requires an all-ones attention_mask tensor")
        if _tensor_signature(attention_mask) != self.attention_mask_signature:
            raise RuntimeError(
                "Native CUDA Graph Qwen attention_mask metadata changed from "
                f"{self.attention_mask_signature} to {_tensor_signature(attention_mask)}"
            )
        if not bool(torch.all(attention_mask == 1).item()):
            raise RuntimeError("Native CUDA Graph Qwen replay only accepts an all-ones attention_mask")
        normalized_kwargs = dict(kwargs)
        normalized_kwargs["attention_mask"] = None
        return args, normalized_kwargs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Validate the fixed call, copy tensor values, replay, and rebuild the HF output."""
        if self.language_model.training:
            raise RuntimeError("Native CUDA Graph Qwen replay requires language_model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("Native CUDA Graph Qwen replay requires torch.no_grad() or torch.inference_mode()")
        if _module_storage_signature(self.language_model) != self.module_storage_signature:
            raise RuntimeError(
                "Native CUDA Graph Qwen parameter/buffer storage or mutation version changed after capture"
            )
        if self.replay_stream_id is not None:
            current_stream_id = int(torch.cuda.current_stream(self.static_output.device).cuda_stream)
            if current_stream_id != self.replay_stream_id:
                raise RuntimeError("Native CUDA Graph Qwen replay must run on the same CUDA stream used for capture")

        with self.replay_lock:
            leaves, tree_spec = tree_flatten(self._normalize_call(args, kwargs))
            if tree_spec != self.tree_spec:
                raise RuntimeError("Native CUDA Graph Qwen call structure changed after capture")
            tensor_index_set = set(self.tensor_indices)
            for index, (expected, actual) in enumerate(zip(self.static_values, leaves, strict=True)):
                if index not in tensor_index_set and not _static_value_matches(expected, actual):
                    raise RuntimeError(
                        f"Native CUDA Graph Qwen non-tensor leaf {index} changed from " f"{expected!r} to {actual!r}"
                    )
            for static_input, expected_signature, index in zip(
                self.static_inputs,
                self.tensor_signatures,
                self.tensor_indices,
                strict=True,
            ):
                runtime_input = leaves[index]
                if not isinstance(runtime_input, torch.Tensor):
                    raise RuntimeError(
                        f"Native CUDA Graph Qwen tensor leaf {index} changed to {type(runtime_input).__name__}"
                    )
                if _tensor_signature(runtime_input) != expected_signature:
                    raise RuntimeError(
                        f"Native CUDA Graph Qwen tensor leaf {index} metadata changed from "
                        f"{expected_signature} to {_tensor_signature(runtime_input)}"
                    )
                static_input.copy_(runtime_input)
            self.graph.replay()
            owned_hidden_state = self.static_output.clone()
        return _replace_last_hidden_state(self.output_template, owned_hidden_state)


@dataclass
class _NativeCudaGraphState:
    language_model: nn.Module
    raw_forward: Any
    proxy: _NativeCudaGraphLanguageProxy
    capture: _NativeCudaGraphCapture


class _QWen3_5_VL_Interface(nn.Module):
    """Uniform StarVLA interface around a Qwen3.5 multimodal conditional-generation model."""

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        """Load Qwen3.5 and apply the configured ``freeze`` or ``full`` VLM tuning policy."""
        super().__init__()

        framework_config = _config_get(config, "framework", {})
        qwenvl_config = _config_get(framework_config, "qwenvl", {})
        model_id = _config_get(qwenvl_config, "base_vlm", "Qwen/Qwen3.5-4B")
        tuning_mode = str(_config_get(qwenvl_config, "tuning_mode", "freeze")).lower()
        if tuning_mode not in {"freeze", "full"}:
            raise ValueError(f"Unsupported qwenvl.tuning_mode={tuning_mode!r}; expected one of: 'freeze', 'full'.")

        model_class, processor_class, loader_name = _resolve_transformers_components()
        attn_implementation = _config_get(qwenvl_config, "attn_implementation", "flash_attention_2")
        model_kwargs = {
            "attn_implementation": attn_implementation,
            "dtype": torch.bfloat16,
        }
        model_kwargs.update(kwargs)
        try:
            model = model_class.from_pretrained(model_id, **model_kwargs)
        except ValueError as error:
            if "qwen3_5" in str(error).lower() or "qwen3.5" in str(error).lower():
                version = getattr(transformers, "__version__", "unknown")
                raise ImportError(
                    f"Transformers {version} cannot deserialize Qwen3.5. Install Transformers >=5.5.0 "
                    "(the existing StarVLA/GR00T environment pins 4.51.3, so use a dedicated environment)."
                ) from error
            raise
        processor = processor_class.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.model_loader_name = loader_name
        self.tuning_mode = tuning_mode
        self.hidden_size = _infer_hidden_size(model)
        self.context_encoder_compiled = False
        self.context_encoder_compiled_regions = 0
        self.context_encoder_compile_strategy = None
        self.context_encoder_capture_report = None
        self._native_cuda_graph_state: _NativeCudaGraphState | None = None
        from starVLA.model.modules.vlm.qwen35_compute import (
            enable_image_layout_optimization,
            supports_image_layout_optimization,
        )

        base_model = getattr(self.model, "model", None)
        self.optimize_image_layout = _as_bool(
            _config_get(qwenvl_config, "optimize_image_layout", supports_image_layout_optimization(base_model))
        )
        if self.optimize_image_layout:
            enable_image_layout_optimization(self.model.model)

        # Existing StarVLA frameworks access this compatibility attribute directly.
        self.model.config.hidden_size = self.hidden_size

        require_gdn_fast_path = _as_bool(_config_get(qwenvl_config, "require_gdn_fast_path", False))
        if require_gdn_fast_path:
            self.require_gdn_fast_path()

        train_vlm = tuning_mode == "full"
        for parameter in self.model.parameters():
            parameter.requires_grad_(train_vlm)
        if not train_vlm:
            self.model.eval()
        self.gradient_checkpointing = train_vlm and _as_bool(_config_get(qwenvl_config, "gradient_checkpointing", True))
        if train_vlm:
            # KV caching is incompatible with checkpointed teacher-forcing training and wastes memory otherwise.
            self.model.config.use_cache = False
            text_config = getattr(self.model.config, "text_config", None)
            if text_config is not None:
                text_config.use_cache = False
        if self.gradient_checkpointing:
            enable_gradient_checkpointing = getattr(self.model, "gradient_checkpointing_enable", None)
            if enable_gradient_checkpointing is None:
                _log_warning(
                    "Qwen3.5 full tuning requested gradient checkpointing, but %s does not expose "
                    "gradient_checkpointing_enable().",
                    type(self.model).__name__,
                )
            else:
                try:
                    enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    # Transformers releases before the kwargs API still support the no-argument form.
                    enable_gradient_checkpointing()

    def train(self, mode: bool = True) -> "_QWen3_5_VL_Interface":
        """Keep a frozen Qwen backbone in evaluation mode."""
        super().train(mode)
        if self.tuning_mode == "freeze":
            self.model.eval()
        return self

    def compile_context_encoder(
        self,
        *,
        strategy: str = "regional_mlp",
        backend: str = "inductor",
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False,
        emulate_precision_casts: bool = True,
        example_inputs: Optional[Mapping[str, Any]] = None,
        capture_warmup: int = 3,
    ) -> nn.Module:
        """Compile the full fast-kernel text model or its repeated FFN regions."""
        base_model, language_model = self._context_language_model()
        if self.context_encoder_compiled:
            return language_model
        if strategy == "native_cuda_graph":
            if example_inputs is None:
                raise ValueError("native_cuda_graph requires fixed batch-one example_inputs")
            return self.capture_context_encoder_cuda_graph(example_inputs, warmup=capture_warmup)
        compile_function = getattr(torch, "compile", None)
        if not callable(compile_function):
            raise RuntimeError("This PyTorch build does not expose torch.compile")
        supported_strategies = {
            "native_cuda_graph",
            "whole_language_model",
            "regional_decoder_layers",
            "regional_full_attention_layers",
            "regional_mlp",
        }
        if strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported Qwen3.5 compile strategy {strategy!r}; " f"expected one of {sorted(supported_strategies)}."
            )
        compile_kwargs = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        if emulate_precision_casts:
            if backend != "inductor":
                raise ValueError(
                    "emulate_precision_casts is an Inductor option; either use backend='inductor' "
                    "or disable eager BF16 precision emulation"
                )
            if mode != "default":
                raise ValueError(
                    "PyTorch cannot combine compile mode presets with Inductor options; "
                    "use mode='default' when emulate_precision_casts is enabled"
                )
            compile_kwargs["options"] = {"emulate_precision_casts": True}
        else:
            compile_kwargs["mode"] = mode

        if strategy == "whole_language_model":
            self.require_gdn_fast_path()
            base_model.language_model = compile_function(language_model, **compile_kwargs)
            compiled_regions = 1
            compiled_language_model = base_model.language_model
        elif strategy == "regional_decoder_layers":
            layers = getattr(language_model, "layers", None)
            if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
                raise RuntimeError(
                    "Qwen3.5 language_model.layers must be a non-empty ModuleList "
                    "for decoder-layer regional compilation"
                )
            for layer_index, layer in enumerate(layers):
                layers[layer_index] = compile_function(layer, **compile_kwargs)
            compiled_regions = len(layers)
            compiled_language_model = language_model
        elif strategy == "regional_full_attention_layers":
            layers = getattr(language_model, "layers", None)
            if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
                raise RuntimeError(
                    "Qwen3.5 language_model.layers must be a non-empty ModuleList "
                    "for full-attention regional compilation"
                )
            layer_types = self._configured_decoder_layer_types(language_model, require_complete=True)
            full_attention_indices = [
                layer_index for layer_index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
            ]
            if not full_attention_indices:
                raise RuntimeError("Qwen3.5 config contains no full_attention decoder layers to compile")
            for layer_index in full_attention_indices:
                layers[layer_index] = compile_function(layers[layer_index], **compile_kwargs)
            compiled_regions = len(full_attention_indices)
            compiled_language_model = language_model
        else:
            layers = getattr(language_model, "layers", ())
            compiled_regions = 0
            for layer in layers:
                mlp = getattr(layer, "mlp", None)
                if not isinstance(mlp, nn.Module):
                    raise RuntimeError("A Qwen3.5 decoder layer does not expose its MLP for regional compilation")
                layer.mlp = compile_function(mlp, **compile_kwargs)
                compiled_regions += 1
            if compiled_regions == 0:
                raise RuntimeError("Qwen3.5 language_model.layers is empty; no MLP regions were compiled")
            compiled_language_model = language_model

        self.context_encoder_compiled = True
        self.context_encoder_compiled_regions = compiled_regions
        self.context_encoder_compile_strategy = strategy
        self.context_encoder_capture_report = None
        return compiled_language_model

    def capture_context_encoder_cuda_graph(
        self,
        example_inputs: Mapping[str, Any],
        *,
        warmup: int = 3,
    ) -> nn.Module:
        """
        Capture the original Qwen language-model forward as one fixed batch-one native CUDA Graph.

        The graph is inference-only and intentionally fails closed when tensor metadata, the complete
        Python call structure, or any non-tensor flag differs from the captured call.
        """
        _, language_model = self._context_language_model()
        if self._native_cuda_graph_state is not None:
            return language_model
        if self.context_encoder_compiled:
            raise RuntimeError(
                "native_cuda_graph must capture the original Qwen language model before any other compile strategy"
            )
        if warmup < 1:
            raise ValueError(f"native_cuda_graph requires at least one warmup iteration, got {warmup}")
        if language_model.training or self.model.training:
            raise RuntimeError("native_cuda_graph requires the Qwen model to be in eval mode")
        if not isinstance(example_inputs, Mapping):
            raise TypeError("native_cuda_graph example_inputs must be a mapping")

        self.require_gdn_fast_path()
        state_dict_keys = tuple(self.model.state_dict().keys())
        input_embeddings = getattr(language_model, "embed_tokens", None)
        raw_forward = language_model.__dict__.get("forward", _MISSING_FORWARD)
        captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record_language_model_call(
            _module: nn.Module,
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> None:
            captured_calls.append((tuple(call_args), dict(call_kwargs)))

        hook = language_model.register_forward_pre_hook(record_language_model_call, with_kwargs=True)
        try:
            eager_context = self._run_fixed_context(example_inputs).detach().clone()
        finally:
            hook.remove()

        if len(captured_calls) != 1:
            raise RuntimeError(
                "native_cuda_graph requires exactly one language_model call per encode_context, "
                f"observed {len(captured_calls)}"
            )

        call_args, call_kwargs = captured_calls[0]
        try:
            normalized_call, attention_mask = self._normalize_native_cuda_graph_call(call_args, call_kwargs)
            adapter = _TensorOnlyLanguageModel(language_model, normalized_call).eval()
            sample_tensors = adapter.sample_tensors(normalized_call)
            if not sample_tensors:
                raise RuntimeError("native_cuda_graph found no tensor inputs in the Qwen language-model call")
            capture = self._capture_native_cuda_graph(adapter, sample_tensors, warmup)

            with self.trace_gdn_fast_path() as post_capture_original_gdn_trace:
                direct_output_template = self._run_tensor_only_language_model(adapter, sample_tensors)
            self._require_complete_gdn_trace(post_capture_original_gdn_trace)
            direct_reference = _require_hidden_state_only_output(direct_output_template).detach().clone()

            for static_input, sample_tensor in zip(capture.static_inputs, sample_tensors, strict=True):
                static_input.copy_(sample_tensor)
            capture.graph.replay()
            self._synchronize_cuda_graph_device(capture.static_output.device)
            direct_graph_output = capture.static_output.detach().clone()
            direct_correctness = _exact_tensor_report(
                direct_reference,
                direct_graph_output,
                "fast_eager_original_language_model",
            )
            if not direct_correctness["passed"]:
                raise RuntimeError(
                    "native_cuda_graph failed exact direct language-model equivalence: " f"{direct_correctness}"
                )

            proxy = _NativeCudaGraphLanguageProxy(
                adapter=adapter,
                call_tree=normalized_call,
                capture=capture,
                output_template=direct_output_template,
                attention_mask_signature=_tensor_signature(attention_mask),
                language_model=language_model,
            )
            language_model.forward = proxy.forward
            graphed_context = self._run_fixed_context(example_inputs).detach().clone()
            context_correctness = _exact_tensor_report(
                eager_context,
                graphed_context,
                "fast_eager_full_multimodal_context",
            )
            if not context_correctness["passed"]:
                raise RuntimeError("native_cuda_graph failed exact full-context equivalence: " f"{context_correctness}")
            if tuple(self.model.state_dict().keys()) != state_dict_keys:
                raise RuntimeError("native_cuda_graph changed Qwen state_dict keys")
            if self._context_language_model()[1] is not language_model:
                raise RuntimeError("native_cuda_graph replaced the original Qwen language_model module")
            if getattr(language_model, "embed_tokens", None) is not input_embeddings:
                raise RuntimeError("native_cuda_graph replaced the original Qwen input embeddings")

            tensor_leaves, _ = tree_flatten(normalized_call)
            static_input_metadata = [
                _tensor_metadata(tensor_leaves[index], leaf_index=index) for index in adapter.tensor_indices
            ]
            tensor_index_set = set(adapter.tensor_indices)
            static_non_tensor_inputs = [
                {
                    "pytree_leaf_index": index,
                    "type": type(value).__name__,
                    "value": (
                        repr(value) if isinstance(value, (type(None), bool, int, float, str)) else "identity-locked"
                    ),
                }
                for index, value in enumerate(tensor_leaves)
                if index not in tensor_index_set
            ]
            capture_report = {
                "strategy": "native_cuda_graph",
                "capture_cold_ms": capture.capture_cold_ms,
                "capture_warmup_iterations": warmup,
                "batch_size": 1,
                "sequence_length": int(attention_mask.shape[-1]),
                "fixed_shape": True,
                "single_inflight": True,
                "serialized_replay": True,
                "same_cuda_stream_required": True,
                "owned_output_clone": True,
                "static_input_count": len(capture.static_inputs),
                "static_inputs": static_input_metadata,
                "static_non_tensor_leaf_count": len(static_non_tensor_inputs),
                "static_non_tensor_inputs": static_non_tensor_inputs,
                "attention_mask": {
                    "normalized_to_none": True,
                    "required_value": "all_ones",
                    "metadata": _tensor_metadata(attention_mask),
                },
                "post_capture_original_gdn_trace": dict(post_capture_original_gdn_trace),
                "direct_language_model_correctness": direct_correctness,
                "full_context_correctness": context_correctness,
                "language_model_identity_preserved": True,
                "input_embedding_identity_preserved": True,
                "state_dict_keys_preserved": True,
            }
            self._native_cuda_graph_state = _NativeCudaGraphState(
                language_model=language_model,
                raw_forward=raw_forward,
                proxy=proxy,
                capture=capture,
            )
            self.context_encoder_compiled = True
            self.context_encoder_compiled_regions = 1
            self.context_encoder_compile_strategy = "native_cuda_graph"
            self.context_encoder_capture_report = capture_report
            return language_model
        except BaseException:
            self._restore_language_model_forward(language_model, raw_forward)
            self._native_cuda_graph_state = None
            self.context_encoder_compiled = False
            self.context_encoder_compiled_regions = 0
            self.context_encoder_compile_strategy = None
            self.context_encoder_capture_report = None
            raise

    def restore_native_cuda_graph_context_encoder(self) -> bool:
        """Restore the exact pre-capture language-model forward descriptor or raw instance attribute."""
        state = self._native_cuda_graph_state
        if state is None:
            return False
        self._restore_language_model_forward(state.language_model, state.raw_forward)
        self._native_cuda_graph_state = None
        self.context_encoder_compiled = False
        self.context_encoder_compiled_regions = 0
        self.context_encoder_compile_strategy = None
        self.context_encoder_capture_report = None
        return True

    @staticmethod
    def _restore_language_model_forward(language_model: nn.Module, raw_forward: Any) -> None:
        if raw_forward is _MISSING_FORWARD:
            language_model.__dict__.pop("forward", None)
        else:
            language_model.forward = raw_forward

    @staticmethod
    def _normalize_native_cuda_graph_call(
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> tuple[Any, torch.Tensor]:
        if call_args:
            raise RuntimeError("native_cuda_graph requires Qwen language_model.forward to use keyword arguments")
        inputs_embeds = call_kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            raise RuntimeError("native_cuda_graph requires rank-three Qwen inputs_embeds")
        if inputs_embeds.shape[0] != 1:
            raise RuntimeError(
                f"native_cuda_graph only supports batch size 1, got inputs_embeds={tuple(inputs_embeds.shape)}"
            )
        attention_mask = call_kwargs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise RuntimeError("native_cuda_graph requires a rank-two attention_mask tensor")
        if tuple(attention_mask.shape) != (1, inputs_embeds.shape[1]):
            raise RuntimeError(
                "native_cuda_graph attention_mask must be [1, sequence_length], got "
                f"{tuple(attention_mask.shape)} for inputs_embeds={tuple(inputs_embeds.shape)}"
            )
        if not bool(torch.all(attention_mask == 1).item()):
            raise RuntimeError("native_cuda_graph only supports an all-ones attention_mask")
        position_ids = call_kwargs.get("position_ids")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.ndim != 3
            or position_ids.shape[1] != 1
            or position_ids.shape[-1] != inputs_embeds.shape[1]
        ):
            raise RuntimeError("native_cuda_graph requires Qwen position_ids shaped [axes, 1, sequence_length]")

        normalized_kwargs = dict(call_kwargs)
        normalized_kwargs["attention_mask"] = None
        return (call_args, normalized_kwargs), attention_mask

    def _capture_native_cuda_graph(
        self,
        adapter: _TensorOnlyLanguageModel,
        sample_tensors: tuple[torch.Tensor, ...],
        warmup: int,
    ) -> _NativeCudaGraphCapture:
        """Warm up and capture the tensor-only original language model on a side stream."""
        if not torch.cuda.is_available():
            raise RuntimeError("native_cuda_graph requires CUDA")
        device = sample_tensors[0].device
        if device.type != "cuda" or any(tensor.device != device for tensor in sample_tensors):
            raise RuntimeError("native_cuda_graph requires every Qwen tensor input on one CUDA device")
        if any(tensor.requires_grad for tensor in sample_tensors):
            raise RuntimeError("native_cuda_graph only supports inference tensors without gradients")
        static_inputs = tuple(tensor.detach().clone(memory_format=torch.preserve_format) for tensor in sample_tensors)

        self._synchronize_cuda_graph_device(device)
        started = time.perf_counter()
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.cuda.stream(side_stream),
            torch.inference_mode(),
            self._autocast_context(torch.bfloat16),
        ):
            for _ in range(warmup):
                adapter(*static_inputs)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        self._synchronize_cuda_graph_device(device)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph),
            torch.inference_mode(),
            self._autocast_context(torch.bfloat16),
        ):
            static_output = adapter(*static_inputs)
        self._synchronize_cuda_graph_device(device)
        capture_cold_ms = (time.perf_counter() - started) * 1000.0
        replay_stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        return _NativeCudaGraphCapture(
            graph=graph,
            static_inputs=static_inputs,
            static_output=static_output,
            capture_cold_ms=capture_cold_ms,
            replay_stream_id=replay_stream_id,
        )

    def _run_fixed_context(self, example_inputs: Mapping[str, Any]) -> torch.Tensor:
        with torch.inference_mode(), self._autocast_context(torch.bfloat16):
            context = self.encode_context(**dict(example_inputs))
        if not isinstance(context, torch.Tensor):
            raise RuntimeError(f"Qwen encode_context returned {type(context).__name__}, expected Tensor")
        return context

    def _run_tensor_only_language_model(
        self,
        adapter: _TensorOnlyLanguageModel,
        sample_tensors: tuple[torch.Tensor, ...],
    ) -> Any:
        with torch.inference_mode(), self._autocast_context(torch.bfloat16):
            return adapter.call_original(*sample_tensors)

    @staticmethod
    def _synchronize_cuda_graph_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _require_complete_gdn_trace(counters: Mapping[str, int]) -> None:
        expected_calls = int(counters["expected_layers"])
        call_keys = ("causal_conv1d_calls", "fla_delta_rule_calls", "fla_fused_norm_calls")
        if expected_calls <= 0 or any(int(counters[key]) != expected_calls for key in call_keys):
            raise RuntimeError(f"native_cuda_graph post-capture GDN fast-kernel trace failed: {dict(counters)}")

    def _context_language_model(self) -> tuple[nn.Module, nn.Module]:
        """Return the multimodal base model and its text transformer."""
        base_model = getattr(self.model, "model", None)
        if not isinstance(base_model, nn.Module):
            raise RuntimeError("Qwen3.5 context compilation requires model.model to expose the multimodal base model")
        language_model = getattr(base_model, "language_model", None)
        if not isinstance(language_model, nn.Module):
            raise RuntimeError(
                "Qwen3.5 context compilation requires model.model.language_model to expose the text transformer"
            )
        return base_model, language_model

    @staticmethod
    def _original_module(module: nn.Module) -> nn.Module:
        """Unwrap a torch.compile OptimizedModule for structural inspection."""
        original_module = getattr(module, "_orig_mod", None)
        return original_module if isinstance(original_module, nn.Module) else module

    def _configured_decoder_layer_types(
        self,
        language_model: nn.Module,
        *,
        require_complete: bool = False,
    ) -> tuple[str, ...]:
        """Read the per-layer hybrid attention layout from supported Qwen3.5 config locations."""
        language_model = self._original_module(language_model)
        language_config = getattr(language_model, "config", None)
        configured_layer_types = getattr(language_config, "layer_types", None)
        if configured_layer_types is None:
            text_config = getattr(getattr(self.model, "config", None), "text_config", None)
            configured_layer_types = getattr(text_config, "layer_types", None)
        layer_types = tuple(configured_layer_types or ())

        if require_complete:
            layers = getattr(language_model, "layers", ())
            if len(layer_types) != len(layers):
                raise RuntimeError(
                    "Qwen3.5 hybrid compilation requires one config.layer_types entry "
                    f"per decoder layer, got {len(layer_types)} types for {len(layers)} layers"
                )
            unsupported_types = sorted(set(layer_types) - {"linear_attention", "full_attention"})
            if unsupported_types:
                raise RuntimeError(f"Qwen3.5 config contains unsupported decoder layer types: {unsupported_types}")
        return layer_types

    def _linear_attention_modules(self) -> list[nn.Module]:
        """Collect the Qwen3.5 GatedDeltaNet modules from the text transformer."""
        _, language_model = self._context_language_model()
        language_model = self._original_module(language_model)
        configured_layer_types = self._configured_decoder_layer_types(language_model)

        modules = []
        for layer_index, layer in enumerate(getattr(language_model, "layers", ())):
            layer_type = getattr(layer, "layer_type", None)
            if layer_type is None and layer_index < len(configured_layer_types):
                layer_type = configured_layer_types[layer_index]
            linear_attention = getattr(layer, "linear_attn", None)
            if layer_type != "linear_attention" and not isinstance(linear_attention, nn.Module):
                continue
            if not isinstance(linear_attention, nn.Module):
                raise RuntimeError(
                    f"Qwen3.5 config marks decoder layer {layer_index} as linear_attention, "
                    "but the layer does not expose linear_attn"
                )
            modules.append(linear_attention)
        return modules

    def gdn_fast_path_status(self) -> dict[str, Any]:
        """Inspect whether every GatedDeltaNet layer is wired to the optional CUDA/Triton kernels."""
        linear_attention_modules = self._linear_attention_modules()
        causal_modules = [
            getattr(getattr(module, "causal_conv1d_fn", None), "__module__", "") for module in linear_attention_modules
        ]
        delta_rule_modules = [
            getattr(getattr(module, "chunk_gated_delta_rule", None), "__module__", "")
            for module in linear_attention_modules
        ]
        norm_modules = [type(getattr(module, "norm", None)).__module__ for module in linear_attention_modules]
        causal_layers = sum(module_name.startswith("causal_conv1d.") for module_name in causal_modules)
        delta_rule_layers = sum(module_name.startswith("fla.ops.") for module_name in delta_rule_modules)
        fused_norm_layers = sum(module_name.startswith("fla.modules.") for module_name in norm_modules)
        layer_count = len(linear_attention_modules)
        available = layer_count > 0 and causal_layers == delta_rule_layers == fused_norm_layers == layer_count
        return {
            "available": available,
            "linear_attention_layers": layer_count,
            "causal_conv1d_layers": causal_layers,
            "fla_delta_rule_layers": delta_rule_layers,
            "fla_fused_norm_layers": fused_norm_layers,
            "causal_conv1d_modules": sorted(set(causal_modules)),
            "fla_delta_rule_modules": sorted(set(delta_rule_modules)),
            "fla_norm_modules": sorted(set(norm_modules)),
        }

    def require_gdn_fast_path(self) -> dict[str, Any]:
        """Reject the silent Transformers PyTorch fallback for any Qwen3.5 GDN layer."""
        status = self.gdn_fast_path_status()
        if not status["available"]:
            raise RuntimeError(
                "Qwen3.5 GatedDeltaNet fast path is required, but at least one layer is using the slow "
                f"PyTorch fallback: {status}. Install fla-core, flash-linear-attention, and causal-conv1d "
                "before importing Transformers."
            )
        return status

    @contextmanager
    def trace_gdn_fast_path(self) -> Iterator[dict[str, int]]:
        """Count one real invocation of each GDN fast kernel without retaining probe wrappers."""
        status = self.require_gdn_fast_path()
        counters = {
            "expected_layers": int(status["linear_attention_layers"]),
            "causal_conv1d_calls": 0,
            "fla_delta_rule_calls": 0,
            "fla_fused_norm_calls": 0,
        }
        originals = []
        hooks = []

        for linear_attention in self._linear_attention_modules():
            causal_conv1d = linear_attention.causal_conv1d_fn
            delta_rule = linear_attention.chunk_gated_delta_rule
            originals.append((linear_attention, causal_conv1d, delta_rule))

            def traced_causal_conv1d(*args: Any, _function: Any = causal_conv1d, **kwargs: Any) -> Any:
                counters["causal_conv1d_calls"] += 1
                return _function(*args, **kwargs)

            def traced_delta_rule(*args: Any, _function: Any = delta_rule, **kwargs: Any) -> Any:
                counters["fla_delta_rule_calls"] += 1
                return _function(*args, **kwargs)

            linear_attention.causal_conv1d_fn = traced_causal_conv1d
            linear_attention.chunk_gated_delta_rule = traced_delta_rule

            norm = getattr(linear_attention, "norm", None)
            if isinstance(norm, nn.Module):
                hooks.append(
                    norm.register_forward_hook(
                        lambda _module, _inputs, _output: counters.__setitem__(
                            "fla_fused_norm_calls", counters["fla_fused_norm_calls"] + 1
                        )
                    )
                )

        try:
            yield counters
        finally:
            for linear_attention, causal_conv1d, delta_rule in originals:
                linear_attention.causal_conv1d_fn = causal_conv1d
                linear_attention.chunk_gated_delta_rule = delta_rule
            for hook in hooks:
                hook.remove()

    def _autocast_context(self, dtype: torch.dtype) -> Any:
        if torch.cuda.is_available():
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        first_parameter = next(self.model.parameters(), None)
        return first_parameter.device if first_parameter is not None else torch.device("cpu")

    def _move_to_model_device(self, batch_inputs: Any) -> Any:
        device = self._model_device()
        move = getattr(batch_inputs, "to", None)
        if callable(move):
            return move(device)
        if isinstance(batch_inputs, Mapping):
            return {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch_inputs.items()
            }
        raise TypeError("Qwen3.5 processor output must be a mapping or expose a .to(device) method.")

    def _instruction_prompt(self, instruction: str) -> str:
        datasets_config = _config_get(self.config, "datasets", {})
        vla_data_config = _config_get(datasets_config, "vla_data", {})
        cot_prompt = _config_get(vla_data_config, "CoT_prompt", None)
        return str(cot_prompt).replace("{instruction}", instruction) if cot_prompt else instruction

    @staticmethod
    def _as_image_list(sample_images: Any) -> Sequence[Any]:
        if sample_images is None:
            return ()
        if isinstance(sample_images, (list, tuple)):
            if any(image is None for image in sample_images):
                raise ValueError("A sample's image list cannot mix image values with None")
            return sample_images
        return [sample_images]

    def forward(self, **kwargs: Any) -> Any:
        """Delegate a training or encoding forward pass to the Qwen3.5 backbone."""
        with self._autocast_context(torch.bfloat16):
            return self.model(**kwargs)

    def generate(self, **kwargs: Any) -> Any:
        """Delegate autoregressive generation to the Qwen3.5 backbone."""
        # Generation changes the sequence layout between decoding steps.
        kwargs.pop("_starvla_image_layout", None)
        with self._autocast_context(torch.float16):
            return self.model.generate(**kwargs)

    def encode_context(
        self,
        *,
        hidden_layers: Sequence[int] | None = None,
        selected_hidden_states_only: bool = False,
        required_decoder_layer_type: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Encode multimodal tokens from the final state or selected intermediate layers.

        ``hidden_layers=None`` deliberately retains the low-memory legacy path,
        which requests only ``last_hidden_state`` from the multimodal base
        model. Layer indices follow Hugging Face ``hidden_states`` semantics:
        zero is the embedding output and positive values are transformer-layer
        outputs.
        """
        if hidden_layers is not None:
            requested_layers = tuple(int(layer) for layer in hidden_layers)
            if not requested_layers:
                raise ValueError("hidden_layers must contain at least one layer index")
            if any(layer < 0 for layer in requested_layers):
                raise ValueError(f"hidden_layers must be non-negative, got {requested_layers}")
            if selected_hidden_states_only:
                return self._encode_selected_hidden_states(
                    requested_layers,
                    kwargs,
                    required_decoder_layer_type=required_decoder_layer_type,
                )

            layer_kwargs = dict(kwargs)
            layer_kwargs.pop("labels", None)
            layer_kwargs["output_hidden_states"] = True
            layer_kwargs["return_dict"] = True
            layer_kwargs["use_cache"] = False
            base_model = getattr(self.model, "model", None)
            with self._autocast_context(torch.bfloat16):
                outputs = base_model(**layer_kwargs) if isinstance(base_model, nn.Module) else self.model(**layer_kwargs)
            hidden_states = (
                outputs.get("hidden_states") if isinstance(outputs, Mapping) else getattr(outputs, "hidden_states", None)
            )
            if hidden_states is None:
                raise RuntimeError(
                    "Qwen3.5 base model did not return hidden_states for layerwise MoT. "
                    "Use vlm_feature_layers=-1 for final-state conditioning or a Transformers "
                    "Qwen3.5 implementation that supports output_hidden_states=True."
                )
            layer_count = len(hidden_states)
            invalid = [layer for layer in requested_layers if layer >= layer_count]
            if invalid:
                raise ValueError(
                    f"Requested Qwen hidden layers {invalid}, but the model returned {layer_count} "
                    f"states (valid indices: 0..{layer_count - 1})"
                )
            return tuple(hidden_states[layer] for layer in requested_layers)

        # Keep this branch unchanged in spirit: the default does not retain
        # intermediate VLM activations.
        base_model = getattr(self.model, "model", None)
        if isinstance(base_model, nn.Module):
            base_kwargs = dict(kwargs)
            base_kwargs.pop("labels", None)
            base_kwargs["output_hidden_states"] = False
            base_kwargs["return_dict"] = True
            base_kwargs["use_cache"] = False
            with self._autocast_context(torch.bfloat16):
                outputs = base_model(**base_kwargs)
            last_hidden_state = (
                outputs.get("last_hidden_state")
                if isinstance(outputs, Mapping)
                else getattr(outputs, "last_hidden_state", None)
            )
            if last_hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
                last_hidden_state = outputs[0]
            if last_hidden_state is not None:
                return last_hidden_state

        # Compatibility fallback for auto-model wrappers that do not expose
        # their multimodal base model. This path may retain all layer states.
        forward_kwargs = dict(kwargs)
        forward_kwargs.setdefault("output_hidden_states", True)
        forward_kwargs.setdefault("return_dict", True)
        # Avoid materializing vocabulary logits for every multimodal token:
        # MoT only consumes hidden states from this conditional-LM forward.
        forward_parameters = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in forward_parameters:
            forward_kwargs.setdefault("logits_to_keep", 1)
        elif "num_logits_to_keep" in forward_parameters:
            forward_kwargs.setdefault("num_logits_to_keep", 1)
        outputs = self.forward(**forward_kwargs)

        hidden_states = (
            outputs.get("hidden_states") if isinstance(outputs, Mapping) else getattr(outputs, "hidden_states", None)
        )
        if hidden_states is not None and len(hidden_states) > 0:
            return hidden_states[-1]

        last_hidden_state = (
            outputs.get("last_hidden_state")
            if isinstance(outputs, Mapping)
            else getattr(outputs, "last_hidden_state", None)
        )
        if last_hidden_state is not None:
            return last_hidden_state
        raise RuntimeError(
            "Qwen3.5 forward output did not include hidden states. Ensure the selected Transformers loader "
            "supports output_hidden_states=True."
        )

    def _encode_selected_hidden_states(
        self,
        requested_layers: tuple[int, ...],
        kwargs: Mapping[str, Any],
        *,
        required_decoder_layer_type: str | None,
    ) -> tuple[torch.Tensor, ...]:
        """Capture only requested decoder states instead of retaining the full hidden-state tuple."""

        if self.gradient_checkpointing:
            raise RuntimeError(
                "selected_hidden_states_only is incompatible with Qwen gradient checkpointing; "
                "use the normal output_hidden_states path or disable Qwen checkpointing"
            )
        if self.context_encoder_compiled:
            raise RuntimeError(
                "selected_hidden_states_only requires the eager Qwen decoder so temporary hooks "
                "can capture exact layer outputs"
            )

        base_model, language_model = self._context_language_model()
        language_model = self._original_module(language_model)
        layers = getattr(language_model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise RuntimeError(
                "selected_hidden_states_only requires model.model.language_model.layers " "to be a non-empty ModuleList"
            )
        layer_count = len(layers)
        invalid = [layer for layer in requested_layers if layer > layer_count]
        if invalid:
            raise ValueError(
                f"Requested Qwen hidden layers {invalid}, but the model has {layer_count} decoder "
                f"layers (valid hidden-state indices: 0..{layer_count})"
            )
        if required_decoder_layer_type is not None:
            self.validate_hidden_layers(
                requested_layers,
                required_decoder_layer_type=required_decoder_layer_type,
            )

        selected_graph = getattr(self, "_selected_context_cuda_graph", None)
        if selected_graph is not None:
            return selected_graph(requested_layers, kwargs)

        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for hidden_index in sorted(set(requested_layers) - {layer_count}):

            def capture_layer_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
                *,
                index: int = hidden_index,
            ) -> None:
                hidden_state = call_kwargs.get("hidden_states")
                if hidden_state is None and args:
                    hidden_state = args[0]
                if not isinstance(hidden_state, torch.Tensor):
                    raise RuntimeError("Qwen decoder layer received no tensor hidden_states input")
                captured[index] = hidden_state

            hooks.append(
                layers[hidden_index].register_forward_pre_hook(
                    capture_layer_input,
                    with_kwargs=True,
                )
            )

        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("labels", None)
        layer_kwargs["output_hidden_states"] = False
        layer_kwargs["return_dict"] = True
        layer_kwargs["use_cache"] = False
        try:
            with self._autocast_context(torch.bfloat16):
                outputs = base_model(**layer_kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        if layer_count in requested_layers:
            captured[layer_count] = _last_hidden_state(outputs)
        missing = [layer for layer in requested_layers if layer not in captured]
        if missing:
            raise RuntimeError(
                "Qwen selected-layer hooks did not observe requested hidden states "
                f"{missing}; the decoder forward structure may have changed"
            )
        return tuple(captured[layer] for layer in requested_layers)

    def forward_with_selected_hidden_states(
        self,
        *,
        hidden_layers: Sequence[int] | None,
        required_decoder_layer_type: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, tuple[torch.Tensor, ...]]:
        """Run one supervised CausalLM forward while capturing MoT context layers.

        The returned states cover the complete supervised sequence. Callers must
        apply their prompt-only context mask before exposing them to another
        model. Capturing decoder inputs and the multimodal base-model output
        avoids ``output_hidden_states=True`` and its full-layer memory cost.
        """

        if self.gradient_checkpointing:
            raise RuntimeError(
                "joint supervised context capture is incompatible with Qwen gradient checkpointing; "
                "disable framework.qwenvl.gradient_checkpointing"
            )
        if self.context_encoder_compiled:
            raise RuntimeError(
                "joint supervised context capture requires the eager Qwen decoder so temporary hooks "
                "can capture exact layer outputs"
            )

        base_model, language_model = self._context_language_model()
        language_model = self._original_module(language_model)
        layers = getattr(language_model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise RuntimeError(
                "joint supervised context capture requires model.model.language_model.layers "
                "to be a non-empty ModuleList"
            )
        layer_count = len(layers)
        requested_layers = (layer_count,) if hidden_layers is None else tuple(int(layer) for layer in hidden_layers)
        if not requested_layers:
            raise ValueError("hidden_layers must contain at least one layer index")
        self.validate_hidden_layers(
            requested_layers,
            required_decoder_layer_type=required_decoder_layer_type,
        )

        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for hidden_index in sorted(set(requested_layers) - {layer_count}):

            def capture_layer_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
                *,
                index: int = hidden_index,
            ) -> None:
                hidden_state = call_kwargs.get("hidden_states")
                if hidden_state is None and args:
                    hidden_state = args[0]
                if not isinstance(hidden_state, torch.Tensor):
                    raise RuntimeError("Qwen decoder layer received no tensor hidden_states input")
                captured[index] = hidden_state

            hooks.append(
                layers[hidden_index].register_forward_pre_hook(
                    capture_layer_input,
                    with_kwargs=True,
                )
            )

        if layer_count in requested_layers:

            def capture_final_base_state(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
            ) -> None:
                captured[layer_count] = _last_hidden_state(output)

            # The multimodal base output is post-final-norm and therefore has
            # exactly the semantics of hidden_states[layer_count].
            hooks.append(base_model.register_forward_hook(capture_final_base_state))

        model_kwargs = dict(kwargs)
        model_kwargs.pop("prompt_attention_mask", None)
        model_kwargs["output_hidden_states"] = False
        model_kwargs["return_dict"] = True
        model_kwargs["use_cache"] = False
        try:
            outputs = self.forward(**model_kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        missing = [layer for layer in requested_layers if layer not in captured]
        if missing:
            raise RuntimeError(
                "Qwen joint supervised forward did not expose requested hidden states "
                f"{missing}; the decoder forward structure may have changed"
            )
        return outputs, tuple(captured[layer] for layer in requested_layers)

    def validate_hidden_layers(
        self,
        hidden_layers: Sequence[int],
        *,
        required_decoder_layer_type: str | None = None,
    ) -> tuple[int, ...]:
        """Validate HF hidden-state indices against the loaded decoder's actual layer types."""

        requested_layers = tuple(int(layer) for layer in hidden_layers)
        _, language_model = self._context_language_model()
        language_model = self._original_module(language_model)
        layers = getattr(language_model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise RuntimeError("Qwen language_model.layers must be a non-empty ModuleList")
        layer_count = len(layers)
        invalid = [layer for layer in requested_layers if layer < 0 or layer > layer_count]
        if invalid:
            raise ValueError(f"Requested Qwen hidden layers {invalid}, but valid indices are 0..{layer_count}")
        if required_decoder_layer_type is None:
            return requested_layers

        layer_types = self._configured_decoder_layer_types(
            language_model,
            require_complete=True,
        )
        invalid_type = []
        for hidden_index in requested_layers:
            if hidden_index == 0:
                invalid_type.append((hidden_index, "embedding"))
                continue
            decoder_type = layer_types[hidden_index - 1]
            if decoder_type != required_decoder_layer_type:
                invalid_type.append((hidden_index, decoder_type))
        if invalid_type:
            raise ValueError(
                "Requested Qwen hidden states must come from "
                f"{required_decoder_layer_type!r} decoder layers; got {invalid_type}"
            )
        return requested_layers

    def build_qwenvl_inputs(
        self,
        images: Sequence[Any],
        instructions: Optional[Sequence[str]] = None,
        solutions: Optional[Sequence[str]] = None,
        conversations: Optional[Sequence[Sequence[Mapping[str, Any]]]] = None,
        apply_instruction_prompt: bool = True,
        return_prompt_attention_mask: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Build text or multimodal chat inputs with assistant-only solution labels."""
        if conversations is not None and (instructions is not None or solutions is not None):
            raise ValueError("Pass either conversations or instructions/solutions, not both.")
        if conversations is None and instructions is None:
            raise ValueError("Either conversations or instructions must be provided.")
        sample_count = len(conversations) if conversations is not None else len(instructions)
        if len(images) != sample_count:
            raise ValueError("Images and instructions must have the same batch length.")
        if solutions is not None and len(solutions) != sample_count:
            raise ValueError("Solutions and instructions must have the same batch length.")

        messages = []
        for sample_index, sample_images in enumerate(images):
            image_content = [{"type": "image", "image": image} for image in self._as_image_list(sample_images)]
            if conversations is not None:
                sample_messages = []
                images_pending = image_content
                for turn in conversations[sample_index]:
                    role = str(turn.get("role", "")).strip().lower()
                    if role not in {"user", "assistant"}:
                        raise ValueError(f"Unsupported instruction-tuning conversation role: {role!r}")
                    text = str(turn.get("content", "")).strip()
                    if not text:
                        raise ValueError("Instruction-tuning conversation turns must not be empty.")
                    content = []
                    if role == "user" and images_pending:
                        content.extend(images_pending)
                        images_pending = []
                    content.append({"type": "text", "text": text})
                    sample_messages.append({"role": role, "content": content})
                if not any(turn["role"] == "assistant" for turn in sample_messages):
                    raise ValueError("Instruction-tuning conversations require at least one assistant turn.")
            else:
                instruction = instructions[sample_index]
                prompt = self._instruction_prompt(instruction) if apply_instruction_prompt else str(instruction)
                content = [*image_content, {"type": "text", "text": prompt}]
                sample_messages = [{"role": "user", "content": content}]
                if solutions is not None:
                    sample_messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": solutions[sample_index]}],
                        }
                    )
            messages.append(sample_messages)

        template_kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        processor_call_kwargs = {"padding": True}
        processor_call_kwargs.update(kwargs.pop("processor_kwargs", {}))
        processor_call_kwargs.update(kwargs)
        template_kwargs["processor_kwargs"] = processor_call_kwargs
        # Transformers 4.51 tokenizers consume these kwargs at the top level,
        # while newer multimodal processors read the nested processor_kwargs.
        template_kwargs.update(processor_call_kwargs)
        supervised_conversations = conversations is not None
        template_kwargs["add_generation_prompt"] = solutions is None and not supervised_conversations
        batch_inputs = self.processor.apply_chat_template(messages, **template_kwargs)

        if supervised_conversations:
            batch_inputs["labels"] = self._build_chat_assistant_labels(batch_inputs)
        elif solutions is not None:
            prompt_messages = [sample_messages[:1] for sample_messages in messages]
            prompt_kwargs = dict(template_kwargs)
            prompt_kwargs["add_generation_prompt"] = True
            prompt_inputs = self.processor.apply_chat_template(prompt_messages, **prompt_kwargs)
            prompt_attention_mask = self._build_solution_prompt_mask(batch_inputs, prompt_inputs)
            batch_inputs["labels"] = self._build_solution_labels(
                batch_inputs,
                prompt_attention_mask,
            )
            if return_prompt_attention_mask:
                batch_inputs["prompt_attention_mask"] = prompt_attention_mask
        elif return_prompt_attention_mask:
            raise ValueError("return_prompt_attention_mask=True requires supervised solutions")

        layout = None
        if getattr(self, "optimize_image_layout", False):
            from starVLA.model.modules.vlm.qwen35_compute import LAYOUT_KEY, prepare_image_layout

            layout = prepare_image_layout(self.model.model, batch_inputs, self._model_device())
        batch_inputs = self._move_to_model_device(batch_inputs)
        if layout is not None:
            batch_inputs[LAYOUT_KEY] = layout
        return batch_inputs

    def _build_chat_assistant_labels(self, batch_inputs: Mapping[str, Any]) -> torch.Tensor:
        """Supervise Qwen chat assistant content while masking roles, user turns, and padding."""
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Multi-turn label construction requires attention_mask.")

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        marker_to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(marker_to_id) or not callable(encode):
            raise TypeError("Multi-turn instruction tuning requires a tokenizer with Qwen chat marker support.")
        im_start_id = int(marker_to_id("<|im_start|>"))
        im_end_id = int(marker_to_id("<|im_end|>"))
        assistant_role_ids = [int(token_id) for token_id in encode("assistant\n", add_special_tokens=False)]
        if im_start_id == im_end_id or not assistant_role_ids:
            raise ValueError("Qwen tokenizer returned invalid chat marker IDs for multi-turn instruction tuning.")

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for sample_index in range(input_ids.shape[0]):
            positions = torch.nonzero(attention_mask[sample_index], as_tuple=False).flatten()
            tokens = input_ids[sample_index, positions].tolist()
            cursor = 0
            while cursor < len(tokens):
                if tokens[cursor] != im_start_id or tokens[cursor + 1 : cursor + 1 + len(assistant_role_ids)] != (
                    assistant_role_ids
                ):
                    cursor += 1
                    continue
                span_start = cursor + 1 + len(assistant_role_ids)
                span_end = span_start
                while span_end < len(tokens) and tokens[span_end] != im_end_id:
                    span_end += 1
                if span_end < len(tokens):
                    span_end += 1
                supervised_positions = positions[span_start:span_end]
                labels[sample_index, supervised_positions] = input_ids[sample_index, supervised_positions]
                cursor = span_end
        return labels

    @staticmethod
    def _build_solution_prompt_mask(
        batch_inputs: Mapping[str, Any],
        prompt_inputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """Keep the exact causal prefix shared by prompt-only and supervised chats."""

        input_ids = batch_inputs["input_ids"]
        prompt_input_ids = prompt_inputs.get("input_ids")
        attention_mask = batch_inputs.get("attention_mask")
        prompt_attention_mask = prompt_inputs.get("attention_mask")
        if attention_mask is None or prompt_attention_mask is None or prompt_input_ids is None:
            raise ValueError(
                "Qwen3.5 prompt-mask construction requires input_ids and attention_mask from both "
                "full and prompt tokenization."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"Full attention mask shape {tuple(attention_mask.shape)} does not match input IDs "
                f"{tuple(input_ids.shape)}"
            )
        if prompt_attention_mask.ndim != 2 or prompt_attention_mask.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "Prompt attention mask must be [B,S_prompt] with the same batch size as the full chat, "
                f"got {tuple(prompt_attention_mask.shape)} for full inputs {tuple(input_ids.shape)}"
            )
        if prompt_input_ids.shape != prompt_attention_mask.shape:
            raise ValueError(
                f"Prompt input IDs shape {tuple(prompt_input_ids.shape)} does not match prompt attention mask "
                f"{tuple(prompt_attention_mask.shape)}"
            )

        context_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for batch_index in range(input_ids.shape[0]):
            non_padding_positions = torch.nonzero(attention_mask[batch_index], as_tuple=False).flatten()
            prompt_positions = torch.nonzero(
                prompt_attention_mask[batch_index],
                as_tuple=False,
            ).flatten()
            comparison_length = min(non_padding_positions.numel(), prompt_positions.numel())
            full_prefix_positions = non_padding_positions[:comparison_length]
            prompt_prefix_positions = prompt_positions[:comparison_length]
            matching_tokens = input_ids[batch_index, full_prefix_positions].eq(
                prompt_input_ids[batch_index, prompt_prefix_positions]
            )
            mismatches = torch.nonzero(~matching_tokens, as_tuple=False).flatten()
            shared_prefix_length = int(mismatches[0].item()) if mismatches.numel() else comparison_length
            if shared_prefix_length <= 0:
                raise ValueError(
                    "Qwen3.5 full supervised chat shares no causal prefix with its generation "
                    f"prompt at batch index {batch_index}"
                )
            full_prefix_positions = full_prefix_positions[:shared_prefix_length]
            context_mask[batch_index, full_prefix_positions] = True
        return context_mask

    @staticmethod
    def _build_solution_labels(
        batch_inputs: Mapping[str, Any],
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask padding and prompt tokens without relying on any action-token vocabulary range."""

        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Qwen3.5 VLM label construction requires attention_mask")
        if prompt_attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"Prompt attention mask shape {tuple(prompt_attention_mask.shape)} does not match input IDs "
                f"{tuple(input_ids.shape)}"
            )

        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX
        labels[prompt_attention_mask.bool()] = IGNORE_INDEX
        return labels


# Backward-compatible spelling for callers that omit the separator between "3" and "5".
_QWen35_VL_Interface = _QWen3_5_VL_Interface
