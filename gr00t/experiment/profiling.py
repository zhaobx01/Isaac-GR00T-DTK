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


class StopAfterTrace:
    """Stop a profiler immediately after its first successful trace export."""

    def __init__(self, profiler, on_stopped=None):
        self.profiler = profiler
        self.on_stopped = on_stopped
        self.active = False
        self.trace_exported = False

    def __enter__(self):
        self.profiler.start()
        self.active = True
        return self

    def mark_trace_exported(self):
        self.trace_exported = True

    def step(self):
        if not self.active:
            return
        self.profiler.step()
        if self.trace_exported:
            self.close()
            if self.on_stopped is not None:
                self.on_stopped()

    def close(self):
        if self.active:
            self.active = False
            self.profiler.stop()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def baseline_optimizer_evidence_callback(trainer):
    """Check and record the unwrapped baseline optimizer after engine setup."""
    import json
    from pathlib import Path
    from transformers import TrainerCallback

    class BaselineOptimizerEvidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            optimizer = getattr(getattr(trainer, "model_wrapped", None), "optimizer", trainer.optimizer)
            seen = set()
            while hasattr(optimizer, "optimizer"):
                if id(optimizer) in seen:
                    raise RuntimeError("Cycle in optimizer wrappers")
                seen.add(id(optimizer))
                optimizer = optimizer.optimizer
            if type(optimizer) is not torch.optim.AdamW:
                raise RuntimeError(f"Expected baseline torch.optim.AdamW, got {type(optimizer)}")
            record = {
                "rank": args.process_index,
                "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "parameter_groups": [
                    {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
                    for group in optimizer.param_groups
                ],
            }
            out = Path(args.output_dir) / "optimizer"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"rank-{args.process_index:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
            return control

    return BaselineOptimizerEvidence()
