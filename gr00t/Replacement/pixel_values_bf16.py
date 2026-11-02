# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 pre-cast of ``pixel_values`` inside the DataLoader worker.

Under DeepSpeed, ``Trainer._prepare_input`` runs ``tensor.to(device, dtype=bf16)`` on every
floating tensor. For a float32 ``pixel_values`` that converts on the CPU in the main thread and
produces a new, non-pinned tensor, so the pinned buffer made by the DataLoader pin thread is never
the H2D source. Casting in the worker means the pin thread pins the BF16 tensor, the later
``.to(device, dtype=bf16)`` converts nothing, and H2D reads pinned memory.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping

import torch

ENV_VAR = "GR00T_PIXEL_VALUES_BF16"

logger = logging.getLogger(__name__)


def resolve_enabled(environ=None) -> bool:
    """Return the switch state. Unset means disabled."""
    value = (os.environ if environ is None else environ).get(ENV_VAR, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV_VAR} must be 0 or 1")
    return value == "1"


def validate_training_config(config, deepspeed_config) -> None:
    """Reject setups where HF would not cast floats to BF16 itself.

    ``Trainer._prepare_input`` only casts when DeepSpeed is enabled, so with DDP, a single GPU,
    fp16 or fp32 an early BF16 cast would change the numeric path instead of moving it.
    """
    training = config.training
    if deepspeed_config is None:
        raise ValueError(
            f"{ENV_VAR}=1 requires DeepSpeed (multi-GPU without --use-ddp); Transformers only "
            "casts floating inputs to BF16 itself when DeepSpeed is enabled"
        )
    if not training.bf16 or training.fp16:
        raise ValueError(f"{ENV_VAR}=1 requires bf16=True and fp16=False")
    if training.dataloader_num_workers == 0:
        logger.warning(
            "%s=1 with dataloader_num_workers=0: the cast still runs in the main process, "
            "so this brings no benefit.",
            ENV_VAR,
        )


class Bf16PixelValuesCollator:
    """Wrap a collator and cast ``output["inputs"]["pixel_values"]`` to BF16.

    A module-level class so it can be pickled for ``spawn``/``forkserver`` DataLoaders. Any other
    attribute (for example ``.processor``) is forwarded to the wrapped collator.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        # Only called when normal lookup fails. Guard ``inner`` itself so that copy/pickle on a
        # half-built instance does not recurse.
        if name == "inner" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __call__(self, features):
        output = self.inner(features)
        inputs = output.get("inputs") if isinstance(output, MutableMapping) else None
        if not isinstance(inputs, MutableMapping):
            raise ValueError(f"{ENV_VAR}=1: collator output has no 'inputs' mapping")
        pixel_values = inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"{ENV_VAR}=1: inputs['pixel_values'] is missing or not a tensor")
        if pixel_values.device.type != "cpu":
            raise ValueError(
                f"{ENV_VAR}=1: inputs['pixel_values'] must be a CPU tensor, "
                f"got {pixel_values.device}"
            )
        if pixel_values.dtype == torch.float32:
            inputs["pixel_values"] = pixel_values.to(torch.bfloat16)
        elif pixel_values.dtype != torch.bfloat16:
            raise ValueError(
                f"{ENV_VAR}=1: inputs['pixel_values'] must be float32 or bfloat16, "
                f"got {pixel_values.dtype}"
            )
        return output


def install(collator, config, deepspeed_config, environ=None):
    """Return ``collator`` unchanged when disabled, otherwise a validated BF16 wrapper."""
    if not resolve_enabled(environ):
        return collator
    validate_training_config(config, deepspeed_config)
    if isinstance(collator, Bf16PixelValuesCollator):
        return collator
    return Bf16PixelValuesCollator(collator)
