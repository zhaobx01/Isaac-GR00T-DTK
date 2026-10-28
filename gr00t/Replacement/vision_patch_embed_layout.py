"""Switchable channels_last_3d layout for the Qwen3-VL vision patch-embed Conv3d."""

from __future__ import annotations

import logging
import os

import torch

ENV_VAR = "GR00T_VISION_PATCH_EMBED_CHANNELS_LAST"
_HOOK_MARK = "_gr00t_channels_last_3d_hook"

logger = logging.getLogger(__name__)


def resolve_enabled(environ=None) -> bool:
    """Return the switch state. Unset means enabled, which is the historical behavior."""
    value = (os.environ if environ is None else environ).get(ENV_VAR, "1")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV_VAR} must be 0 or 1")
    return value == "1"


def _to_channels_last_3d(_module, inputs):
    # Skip while tracing: the legacy TorchScript ONNX exporter cannot lower a
    # memory_format on .contiguous() ("onnx memory_format support is not
    # implemented"). This is a runtime-only perf hint, so the plain contiguous
    # graph is numerically equivalent for export.
    if not inputs or torch.jit.is_tracing():
        return inputs
    x = inputs[0]
    if isinstance(x, torch.Tensor) and x.dim() == 5:
        return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
    return inputs


def apply(visual: torch.nn.Module, environ=None) -> int:
    """Convert every Conv3d under ``visual`` and hook its 5D input; return the count.

    The switch is read once, here. Turning it off requires a new process: weights that
    were already converted are not restored by a later call.
    """
    if not resolve_enabled(environ):
        logger.debug("%s=0; vision patch-embed Conv3d keeps its default layout.", ENV_VAR)
        return 0
    patched = 0
    for module in visual.modules():
        if isinstance(module, torch.nn.Conv3d) and not getattr(module, _HOOK_MARK, False):
            module.to(memory_format=torch.channels_last_3d)
            module.register_forward_pre_hook(_to_channels_last_3d)
            setattr(module, _HOOK_MARK, True)
            patched += 1
    if patched:
        logger.debug(
            "Applied channels_last_3d to %d vision patch-embed Conv3d module(s) "
            "(torch>=2.9 cuDNN Conv3d perf workaround).",
            patched,
        )
    return patched
