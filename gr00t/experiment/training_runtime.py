"""Select an opt-in training extension without importing it on the default path."""
import os


def resolve_compile_variant(environ=None):
    from gr00t.Replacement.model_compile import resolve_variant

    return resolve_variant(environ)


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
