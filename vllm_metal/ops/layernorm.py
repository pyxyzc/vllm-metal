# SPDX-License-Identifier: Apache-2.0
"""Layer normalization operations for Metal backend."""

import torch

# Try to import Metal kernels
# NOTE: We only check importability here, not Metal availability.
# Metal contexts cannot survive fork(), so we defer initialization to first use.
try:
    import vllm_metal_rust

    _METAL_IMPORTABLE = hasattr(vllm_metal_rust, "metal_rms_norm")
except ImportError:
    _METAL_IMPORTABLE = False

# Lazy initialization state
_metal_initialized = False
_metal_available = False


def _ensure_metal_initialized():
    """Initialize Metal lazily in the worker process."""
    global _metal_initialized, _metal_available
    if _metal_initialized:
        return _metal_available
    _metal_initialized = True
    if _METAL_IMPORTABLE:
        try:
            # This will initialize the Metal context
            _metal_available = vllm_metal_rust.is_metal_available()
        except Exception:
            _metal_available = False
    return _metal_available


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Root Mean Square Layer Normalization.

    Computes: out = (input / sqrt(mean(input^2) + epsilon)) * weight

    This is the normalization used in LLaMA, Mistral, and other models.
    Uses Metal kernel when available, falls back to PyTorch otherwise.

    Args:
        out: Output tensor
        input: Input tensor
        weight: Scale weight tensor
        epsilon: Small constant for numerical stability
    """
    # Try Metal kernel first
    if _ensure_metal_initialized() and input.device.type == "cpu":
        try:
            # Metal kernels work on CPU tensors (unified memory)
            vllm_metal_rust.metal_rms_norm(out, input, weight, epsilon)
            return
        except Exception:
            pass  # Fall back to PyTorch

    # PyTorch fallback
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + epsilon)
    out.copy_(input_normalized * weight)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Fused residual addition and RMS normalization.

    Computes:
        input = input + residual
        input = rms_norm(input, weight, epsilon)

    This modifies input in-place for both the residual addition
    and the normalization. Uses Metal kernel when available.

    Args:
        input: Input tensor (modified in-place)
        residual: Residual tensor to add
        weight: Scale weight tensor
        epsilon: Small constant for numerical stability
    """
    # Try Metal kernel first
    if _ensure_metal_initialized() and input.device.type == "cpu":
        try:
            # Create output buffer (same as input for in-place)
            output = input.clone()
            vllm_metal_rust.metal_fused_add_rms_norm(
                input, residual, weight, output, epsilon
            )
            input.copy_(output)
            return
        except Exception:
            pass  # Fall back to PyTorch

    # PyTorch fallback - add residual in-place
    input.add_(residual)

    # Compute RMS norm in-place
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input.mul_(torch.rsqrt(variance + epsilon))
    input.mul_(weight)


def layer_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> None:
    """Standard Layer Normalization.

    Computes: out = (input - mean) / sqrt(var + epsilon) * weight + bias

    Args:
        out: Output tensor
        input: Input tensor
        weight: Scale weight tensor
        bias: Bias tensor
        epsilon: Small constant for numerical stability
    """
    normalized_shape = input.shape[-1:]
    result = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, epsilon
    )
    out.copy_(result)


def fused_add_layer_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> None:
    """Fused residual addition and layer normalization.

    Args:
        input: Input tensor (modified in-place)
        residual: Residual tensor to add
        weight: Scale weight tensor
        bias: Bias tensor
        epsilon: Small constant for numerical stability
    """
    # Add residual in-place
    input.add_(residual)

    # Apply layer norm
    normalized_shape = input.shape[-1:]
    result = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, epsilon
    )
    input.copy_(result)
