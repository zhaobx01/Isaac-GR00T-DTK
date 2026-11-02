# SPDX-License-Identifier: Apache-2.0
"""Opt-in Apex AdamW selection using the existing Transformers optimizer factory."""
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path


def resolve_optimizer(original, environ=None):
    """Preserve the configured optimizer unless the switch explicitly enables Apex."""
    environ = os.environ if environ is None else environ
    value = environ.get("GR00T_APEX_FUSED_ADAM", "0")
    if value not in {"0", "1"}:
        raise ValueError("GR00T_APEX_FUSED_ADAM must be 0 or 1")
    if value == "0":
        return original
    if original not in {"adamw_torch", "adamw_torch_fused", "adamw_apex_fused"}:
        raise ValueError("Apex FusedAdam can only replace an AdamW optimizer")
    return "adamw_apex_fused"


def validate_deepspeed_config(optimizer, config):
    """Keep the HF optimizer factory authoritative and reject CPU/NVMe offload."""
    if optimizer != "adamw_apex_fused" or config is None:
        return
    if "optimizer" in config:
        raise ValueError("Apex FusedAdam conflicts with a DeepSpeed optimizer definition")
    zero = config.get("zero_optimization", {})
    offload = zero.get("offload_optimizer") or {}
    if zero.get("cpu_offload") or offload.get("device", "none") != "none":
        raise ValueError("Apex FusedAdam requires GPU execution; optimizer offload is unsupported")


def optimizer_evidence_callback(trainer):
    """Fail on unavailable Apex kernels or optimizer substitution; record actual settings."""
    try:
        fused_adam = importlib.import_module("apex.optimizers").FusedAdam
        applier = importlib.import_module("apex.multi_tensor_apply").multi_tensor_applier
        kernels = importlib.import_module("amp_C")
        if not applier.available or not callable(getattr(kernels, "multi_tensor_adam", None)):
            raise RuntimeError("Apex multi-tensor extension is unavailable")
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError("Apex FusedAdam requires a compatible Apex compiled extension") from exc

    from transformers import TrainerCallback

    class OptimizerEvidenceCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            wrapped = getattr(trainer, "model_wrapped", None)
            optimizer = getattr(wrapped, "optimizer", trainer.optimizer)
            seen = set()
            while hasattr(optimizer, "optimizer"):
                if id(optimizer) in seen:
                    raise RuntimeError("Cycle in Apex FusedAdam optimizer wrappers")
                seen.add(id(optimizer))
                optimizer = optimizer.optimizer
            if type(optimizer) is not fused_adam:
                raise RuntimeError(f"Expected Apex FusedAdam, got {type(optimizer)}")
            if optimizer.adam_w_mode != 1:
                raise RuntimeError("Apex FusedAdam must run in AdamW mode")
            try:
                apex_version = version("apex")
            except PackageNotFoundError:
                apex_version = "unknown"
            actual = {
                "rank": args.process_index,
                "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "apex_version": apex_version,
                "adam_w_mode": optimizer.adam_w_mode,
                "parameter_groups": [
                    {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
                    for group in optimizer.param_groups
                ],
            }
            output = Path(args.output_dir) / "optimizer"
            output.mkdir(parents=True, exist_ok=True)
            (output / f"rank-{args.process_index:02d}.json").write_text(
                json.dumps(actual, indent=2) + "\n"
            )
            logging.info("Apex FusedAdam enabled: %s", actual)
            return control

    return OptimizerEvidenceCallback()
