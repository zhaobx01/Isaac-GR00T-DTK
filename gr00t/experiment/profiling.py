# SPDX-License-Identifier: Apache-2.0
"""Configurable profiler window, independent of training execution."""

import torch


def make_profiler_schedule(config):
    return torch.profiler.schedule(
        skip_first=config.profiler_skip_first,
        wait=config.profiler_wait,
        warmup=config.profiler_warmup,
        active=config.profiler_active,
        repeat=config.profiler_repeat,
    )
