# SPDX-License-Identifier: Apache-2.0
"""
vLLM Metal Backend - Hardware plugin for Apple Silicon

This module provides Metal backend support for vLLM,
enabling high-performance LLM inference on Apple Silicon devices.
"""

import os

# Metal requires V2 model runner which properly handles prefill_token_ids
# in the scheduler. This must be set BEFORE vllm.envs is imported anywhere.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

# Metal/Rust contexts cannot survive fork(), so use spawn multiprocessing.
# This must be set before any multiprocessing is used.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

__version__ = "0.1.0"


def register() -> str:
    """Register the Metal platform with vLLM.

    Returns:
        The fully qualified class name of the MetalPlatform.
    """
    return "vllm_metal.platform.MetalPlatform"


def register_ops() -> None:
    """Register Metal-specific operations with vLLM."""
    from vllm_metal.ops import register_metal_ops

    register_metal_ops()
