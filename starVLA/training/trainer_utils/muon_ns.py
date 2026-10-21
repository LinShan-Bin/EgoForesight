# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Normalize Muon matrices in FP32 before either backend's half-precision NS iteration.

"""Optimized Newton-Schulz kernels that preserve DeepSpeed Muon semantics."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from starVLA.training.trainer_utils.muon_fused import install_zero_fused_context, orthogonalize_fused_update

_DEEPSPEED_NS_COEFFICIENT = (3.4445, -4.7750, 2.0315)
_SUPPORTED_NS_KERNELS = {"deepspeed", "hybrid", "quack"}
_HYBRID_MIN_GRAM_DIMENSION = 2048
_HYBRID_MIN_ASPECT_RATIO = 2.0
_PATCHED_MODULE_NAMES = (
    "deepspeed.runtime.zero.muon.original_muon",
    "deepspeed.runtime.zero.stage_1_and_2",
    "deepspeed.runtime.zero.stage3",
)


def _load_quack_symmetric_gemm() -> Callable[..., Any]:
    original_custom_op = torch.library.custom_op

    def _torch28_compatible_custom_op(*args: Any, **kwargs: Any) -> Any:
        """Teach PyTorch 2.8's schema inference to accept CUTLASS enum defaults."""

        # Quack 0.6 also calls custom_op(name, function, ...), while older
        # versions use it as a decorator. Normalize defaults in both forms.
        function = kwargs.pop("fn", None)
        if len(args) > 1:
            function, args = args[1], (args[0], *args[2:])
        decorator = original_custom_op(*args, **kwargs)

        def apply(function: Callable[..., Any]) -> Any:
            defaults = function.__defaults__
            if defaults is not None:
                function.__defaults__ = tuple(
                    default.value if isinstance(default, Enum) and isinstance(default.value, int) else default
                    for default in defaults
                )
            keyword_defaults = function.__kwdefaults__
            if keyword_defaults is not None:
                function.__kwdefaults__ = {
                    key: default.value if isinstance(default, Enum) and isinstance(default.value, int) else default
                    for key, default in keyword_defaults.items()
                }
            return decorator(function)

        return apply(function) if function is not None else apply

    try:
        # Quack 0.4 still resolves two annotation-only types through
        # ``cute.core``. Newer CUTLASS DSL releases promoted them publicly.
        # Alias them before importing Quack; no kernel behavior is changed.
        import cutlass.cute as cute

        for type_name in ("ThrCopy", "ThrMma"):
            if not hasattr(cute.core, type_name) and hasattr(cute, type_name):
                setattr(cute.core, type_name, getattr(cute, type_name))
        # Quack 0.3/0.4 uses CUTLASS enum objects as custom-op defaults.
        # PyTorch 2.9 accepts them, while 2.8's schema parser requires the
        # underlying integer. Limit the compatibility wrapper to module import.
        torch.library.custom_op = _torch28_compatible_custom_op
        try:
            from quack.gemm_interface import gemm_symmetric
        finally:
            torch.library.custom_op = original_custom_op
    except Exception as error:
        raise ImportError(
            "Muon ns_kernel=quack requires a compatible quack-kernels installation for the active "
            "PyTorch/CUDA environment."
        ) from error
    return gemm_symmetric


def _quack_symmetric_mm(
    left: Tensor,
    output: Tensor,
    *,
    right: Tensor | None = None,
    residual: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    diagonal: float = 0.0,
    gemm_symmetric: Callable[..., Any],
) -> None:
    """Compute a symmetric ``left @ right.T`` product through Quack."""
    right = left if right is None else right
    gemm_symmetric(
        left,
        right.T,
        C=residual,
        out=output,
        alpha=alpha,
        beta=beta,
    )
    if diagonal:
        output.diagonal().add_(diagonal)


@torch.no_grad()
@torch.compile
def deepspeed_gram_newton_schulz(
    gradient: Tensor,
    *,
    steps: int = 5,
    eps: float = 1.0e-7,
) -> Tensor:
    """Preserve DeepSpeed 0.19.3's Gram recurrence with FP32 input normalization.

    Casting unnormalized BF16 gradients/momentum to FP16 can round small
    entries to zero. Normalize each original matrix before narrowing its
    exponent range, just as the Quack path does. Keep compilation, the square
    fallback, coefficients, and iteration-two restart from DeepSpeed.
    """
    if gradient.ndim != 2:
        raise ValueError(f"Gram NS expects a 2D matrix, got shape={tuple(gradient.shape)}")
    if steps <= 0:
        raise ValueError(f"Newton-Schulz steps must be positive, got {steps}")

    original_dtype = gradient.dtype
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    # FP32 reductions must precede the FP16 cast, including for fused maps.
    x = (x / (x.norm() + eps)).to(torch.float16)
    coefficient_a, coefficient_b, coefficient_c = _DEEPSPEED_NS_COEFFICIENT

    if x.shape[0] == x.shape[1]:
        for _ in range(steps):
            gram = x @ x.T
            polynomial = coefficient_b * gram + coefficient_c * gram @ gram
            x = coefficient_a * x + polynomial @ x
    else:
        gram = x @ x.T
        factor = None
        for iteration in range(steps):
            if iteration == 2:
                x = factor @ x
                gram = x @ x.T
                factor = None
            polynomial = coefficient_b * gram + coefficient_c * gram @ gram
            if factor is None:
                factor = polynomial.clone()
                factor.diagonal().add_(coefficient_a)
            else:
                factor = coefficient_a * factor + polynomial @ factor
            if iteration < steps - 1 and iteration + 1 != 2:
                gram_polynomial = coefficient_a * gram + polynomial @ gram
                gram = coefficient_a * gram_polynomial + polynomial @ gram_polynomial
        x = factor @ x

    if transposed:
        x = x.T
    return x.to(original_dtype)


@torch.no_grad()
def quack_gram_newton_schulz(
    gradient: Tensor,
    *,
    steps: int = 5,
    eps: float = 1.0e-7,
    gemm_symmetric: Callable[..., Any] | None = None,
) -> Tensor:
    """Run DeepSpeed's Gram NS recurrence using Quack symmetric GEMMs.

    The polynomial coefficient, fp16 iteration, orientation rule, and restart
    at iteration two match DeepSpeed 0.19.2. Quack changes only how products
    known to be symmetric are evaluated.
    """
    if gradient.ndim != 2:
        raise ValueError(f"Quack Gram NS expects a 2D matrix, got shape={tuple(gradient.shape)}")
    if steps <= 0:
        raise ValueError(f"Newton-Schulz steps must be positive, got {steps}")
    if not gradient.is_cuda:
        raise ValueError("Quack Gram NS requires a CUDA tensor")
    if gemm_symmetric is None:
        gemm_symmetric = _load_quack_symmetric_gemm()

    original_dtype = gradient.dtype
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = (x / (x.norm() + eps)).half().contiguous()

    size = x.shape[0]
    gram = torch.empty(size, size, dtype=x.dtype, device=x.device)
    _quack_symmetric_mm(x, gram, gemm_symmetric=gemm_symmetric)

    factor: Tensor | None = None
    coefficient_a, coefficient_b, coefficient_c = _DEEPSPEED_NS_COEFFICIENT
    polynomial = torch.empty_like(gram)
    factor_buffers = (torch.empty_like(gram), torch.empty_like(gram))
    factor_buffer_index = 0
    gram_polynomial = torch.empty_like(gram)
    next_gram = torch.empty_like(gram)

    for iteration in range(steps):
        if iteration == 2:
            if factor is None:
                raise RuntimeError("DeepSpeed-compatible Gram NS restart requires a prior factor")
            x = factor @ x
            _quack_symmetric_mm(x, gram, gemm_symmetric=gemm_symmetric)
            factor = None

        _quack_symmetric_mm(
            gram,
            polynomial,
            residual=gram,
            alpha=coefficient_c,
            beta=coefficient_b,
            gemm_symmetric=gemm_symmetric,
        )
        if factor is None:
            factor_buffers[factor_buffer_index].copy_(polynomial)
            factor_buffers[factor_buffer_index].diagonal().add_(coefficient_a)
            factor = factor_buffers[factor_buffer_index]
        else:
            next_factor_index = 1 - factor_buffer_index
            # Z and Q are analytically commuting symmetric polynomials, but
            # half-precision rounding does not preserve exact symmetry here.
            # DMuon deliberately keeps this one operation on cuBLAS.
            torch.addmm(
                factor,
                polynomial,
                factor,
                alpha=1.0,
                beta=coefficient_a,
                out=factor_buffers[next_factor_index],
            )
            factor_buffer_index = next_factor_index
            factor = factor_buffers[factor_buffer_index]

        if iteration < steps - 1 and iteration + 1 != 2:
            _quack_symmetric_mm(
                gram,
                gram_polynomial,
                right=polynomial,
                residual=gram,
                alpha=1.0,
                beta=coefficient_a,
                gemm_symmetric=gemm_symmetric,
            )
            _quack_symmetric_mm(
                polynomial,
                next_gram,
                right=gram_polynomial,
                residual=gram_polynomial,
                alpha=1.0,
                beta=coefficient_a,
                gemm_symmetric=gemm_symmetric,
            )
            gram, next_gram = next_gram, gram

    if factor is None:
        raise RuntimeError("DeepSpeed-compatible Gram NS did not produce a final factor")
    x = factor @ x
    if transposed:
        x = x.T
    return x.to(original_dtype)


@torch.no_grad()
def _prepare_muon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    beta: float,
    nesterov: bool,
) -> tuple[Tensor, torch.dtype]:
    """Apply momentum while tolerating ZeRO-2's FP32-state/BF16-grad split."""
    original_dtype = gradient.dtype
    working_gradient = gradient.to(momentum_buffer.dtype)
    momentum_buffer.lerp_(working_gradient, 1.0 - beta)
    update = working_gradient.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(len(update), -1)
    if update.ndim != 2:
        raise ValueError(f"Muon requires a matrix update, got shape={tuple(update.shape)}")
    return update, original_dtype


def _finish_muon_update(update: Tensor, original_dtype: torch.dtype) -> Tensor:
    return update.to(original_dtype) if update.dtype != original_dtype else update


@torch.no_grad()
def deepspeed_muon_update_compatible(
    gradient: Tensor,
    momentum_buffer: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    ns_method: str = "gram",
    is_expert_group: bool = False,
) -> Tensor:
    """DeepSpeed Muon math with a ZeRO-2 mixed-state dtype compatibility fix."""
    if ns_method != "gram":
        raise ValueError(f"Muon ns_kernel=deepspeed only supports ns_method='gram', got {ns_method!r}")
    update, original_dtype = _prepare_muon_update(gradient, momentum_buffer, beta, nesterov)
    update = orthogonalize_fused_update(
        gradient, update, lambda matrix: deepspeed_gram_newton_schulz(matrix, steps=ns_steps)
    )
    return _finish_muon_update(update, original_dtype)


@torch.no_grad()
def quack_muon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    ns_method: str = "gram",
    is_expert_group: bool = False,
) -> Tensor:
    """Drop-in replacement for DeepSpeed's ``muon_update``."""
    if ns_method != "gram":
        raise ValueError(f"Muon ns_kernel=quack only supports ns_method='gram', got {ns_method!r}")

    update, original_dtype = _prepare_muon_update(gradient, momentum_buffer, beta, nesterov)
    update = orthogonalize_fused_update(
        gradient, update, lambda matrix: quack_gram_newton_schulz(matrix, steps=ns_steps)
    )
    return _finish_muon_update(update, original_dtype)


@torch.no_grad()
def hybrid_muon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    ns_method: str = "gram",
    is_expert_group: bool = False,
) -> Tensor:
    """Use Quack only for the rectangular, large-Gram regime where it wins."""
    if ns_method != "gram":
        raise ValueError(f"Muon ns_kernel=hybrid only supports ns_method='gram', got {ns_method!r}")

    update, original_dtype = _prepare_muon_update(gradient, momentum_buffer, beta, nesterov)

    def orthogonalize(matrix: Tensor) -> Tensor:
        short_dimension = min(matrix.shape)
        aspect_ratio = max(matrix.shape) / short_dimension
        use_quack = short_dimension >= _HYBRID_MIN_GRAM_DIMENSION and aspect_ratio >= _HYBRID_MIN_ASPECT_RATIO
        if use_quack:
            return quack_gram_newton_schulz(matrix, steps=ns_steps)
        return deepspeed_gram_newton_schulz(matrix, steps=ns_steps)

    update = orthogonalize_fused_update(gradient, update, orthogonalize)
    return _finish_muon_update(update, original_dtype)


def configure_deepspeed_muon_ns_kernel(ns_kernel: str, *, require_zero: bool = True) -> dict[str, Any]:
    """Install or restore the process-wide DeepSpeed Muon NS implementation."""
    ns_kernel = ns_kernel.lower()
    if ns_kernel not in _SUPPORTED_NS_KERNELS:
        raise ValueError(f"Unsupported Muon ns_kernel={ns_kernel!r}; expected one of {sorted(_SUPPORTED_NS_KERNELS)}")
    if ns_kernel in {"hybrid", "quack"}:
        _load_quack_symmetric_gemm()

    patched_modules: list[str] = []
    for module_name in _PATCHED_MODULE_NAMES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        current = getattr(module, "muon_update", None)
        if current is None:
            continue
        original = getattr(module, "_egofore_original_muon_update", None)
        if original is None:
            original = current
            module._egofore_original_muon_update = original
        replacements = {
            "deepspeed": deepspeed_muon_update_compatible,
            "hybrid": hybrid_muon_update,
            "quack": quack_muon_update,
        }
        module.muon_update = replacements[ns_kernel]
        zero_class = getattr(module, "DeepSpeedZeroOptimizer", None)
        if zero_class is not None:
            install_zero_fused_context(zero_class)
        patched_modules.append(module_name)

    required_module = "deepspeed.runtime.zero.stage_1_and_2"
    if require_zero and required_module not in patched_modules:
        raise RuntimeError(
            "Could not patch DeepSpeed ZeRO-2's pre-flatten muon_update call; " f"patched modules were {patched_modules}"
        )
    return {
        "name": ns_kernel,
        "patched_modules": tuple(patched_modules),
        "coefficient": _DEEPSPEED_NS_COEFFICIENT,
        "steps": 5,
        "restart_iteration": 2,
        "hybrid_min_gram_dimension": _HYBRID_MIN_GRAM_DIMENSION,
        "hybrid_min_aspect_ratio": _HYBRID_MIN_ASPECT_RATIO,
    }
