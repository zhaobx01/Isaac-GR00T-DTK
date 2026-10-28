"""Select an opt-in training extension without importing it on the default path."""
import os


def resolve_compile_variant(environ=None):
    value = (os.environ if environ is None else environ).get("GR00T_COMPILE_VARIANT", "off")
    if value not in {"off", "vl", "dit-model", "dit-model-vl"}:
        raise ValueError("GR00T_COMPILE_VARIANT must be off, vl, dit-model or dit-model-vl")
    return value


def resolve_legacy_compile_request(training, compile_variant):
    """Select restored CLI scopes without installing two sets of wrappers."""
    request = {
        "blocks": training.compile_dit_blocks,
        "whole": training.compile_dit_model,
        "text_layers": training.compile_text_layers,
        "text_forwards": training.compile_text_layer_forwards,
        "vl_blocks": training.compile_vl_self_attention_blocks,
        "mode": training.compile_mode,
    }
    if not any(request[key] for key in request if key != "mode"):
        return None
    if compile_variant != "off":
        raise ValueError(
            "Legacy compile CLI flags cannot be combined with GR00T_COMPILE_VARIANT; "
            "set GR00T_COMPILE_VARIANT=off or remove the CLI compile flags"
        )
    if request["blocks"] and request["whole"]:
        raise ValueError("--compile-dit-blocks and --compile-dit-model are mutually exclusive")
    if request["text_layers"] and request["text_forwards"]:
        raise ValueError(
            "--compile-text-layers and --compile-text-layer-forwards are mutually exclusive"
        )
    return request


def resolve_trainer_class(config):
    from gr00t.experiment.trainer import Gr00tTrainer

    value = os.environ.get("GR00T_ZERO2_ALLGATHER_PIPELINE", "0")
    if value not in {"0", "1"}:
        raise ValueError("GR00T_ZERO2_ALLGATHER_PIPELINE must be 0 or 1")
    if value == "0":
        return Gr00tTrainer
    from gr00t.Replacement.zero2_allgather import PipelineTrainer, validate_training_config

    validate_training_config(config)
    return PipelineTrainer


def resolve_zero2_comm_switches(environ=None) -> tuple[bool, bool]:
    """Return (overlap, reduce_scatter); both are off unless the variable is exactly "1"."""
    environ = os.environ if environ is None else environ
    switches = []
    for name in ("GR00T_ZERO2_OVERLAP", "GR00T_ZERO2_REDUCE_SCATTER"):
        value = environ.get(name, "0")
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be 0 or 1")
        switches.append(value == "1")
    return switches[0], switches[1]


def resolve_zero2_pack_timing(environ=None) -> bool:
    """Packing-window CUDA Events add diagnostic overhead; on only when GR00T_ZERO2_PACK_TIMING=1."""
    environ = os.environ if environ is None else environ
    value = environ.get("GR00T_ZERO2_PACK_TIMING", "0")
    if value not in {"0", "1"}:
        raise ValueError("GR00T_ZERO2_PACK_TIMING must be 0 or 1")
    return value == "1"


def load_experiment_deepspeed(default):
    path = os.environ.get("GR00T_DEEPSPEED_CONFIG_PATH")
    if not path:
        return default
    import json
    from pathlib import Path
    config = json.loads(Path(path).read_text())
    if config.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("Experiment DeepSpeed configuration must use ZeRO-2")
    return config
