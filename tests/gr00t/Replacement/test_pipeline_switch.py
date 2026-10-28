"""The experimental pipeline must be opt-in and isolated from normal training."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gr00t.configs.base_config import Config
from gr00t.experiment.training_runtime import resolve_trainer_class


@pytest.mark.parametrize("value", [None, "0"])
def test_disabled_does_not_import_pipeline(value):
    environment = dict(os.environ)
    environment.pop("GR00T_ZERO2_ALLGATHER_PIPELINE", None)
    if value is not None:
        environment["GR00T_ZERO2_ALLGATHER_PIPELINE"] = value
    result = subprocess.run([sys.executable, "-c", """
import sys
from gr00t.configs.base_config import Config
from gr00t.experiment.training_runtime import resolve_trainer_class
from gr00t.experiment.trainer import Gr00tTrainer
assert resolve_trainer_class(Config()) is Gr00tTrainer
assert 'gr00t.Replacement.zero2_allgather' not in sys.modules
assert 'create_optimizer' not in Gr00tTrainer.__dict__
"""], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", ["true", "", "2", "pipeline"])
def test_invalid_switch_is_rejected(monkeypatch, value):
    monkeypatch.setenv("GR00T_ZERO2_ALLGATHER_PIPELINE", value)
    with pytest.raises(ValueError, match="GR00T_ZERO2_ALLGATHER_PIPELINE"):
        resolve_trainer_class(Config())


def test_enabled_selects_specialized_trainer(monkeypatch):
    monkeypatch.setenv("GR00T_ZERO2_ALLGATHER_PIPELINE", "1")
    config = Config()
    config.training.num_gpus = 8
    config.training.save_strategy = "no"
    config.training.optim = "adamw_torch"
    from gr00t.Replacement.zero2_allgather import PipelineTrainer
    assert resolve_trainer_class(config) is PipelineTrainer


def test_enabled_rejects_unsupported_config(monkeypatch):
    monkeypatch.setenv("GR00T_ZERO2_ALLGATHER_PIPELINE", "1")
    config = Config()
    config.training.deepspeed_stage = 3
    with pytest.raises(ValueError, match="ZeRO-2"):
        resolve_trainer_class(config)


def test_old_allgather_and_diagnostic_fields_are_removed():
    from gr00t.configs.finetune_config import FinetuneConfig
    for cls in (type(Config().training), FinetuneConfig):
        fields = cls.__dataclass_fields__
        assert not any(key.startswith("zero2_allgather_")
                       or key in {"deepspeed_config_path", "diagnose_zero_comm"} for key in fields)


def test_existing_visual_compilation_and_runtime_tools_remain():
    root = Path(__file__).resolve().parents[3]
    backbone = (root / "gr00t/model/modules/qwen3_backbone.py").read_text()
    assert '_compile_qwen3_vision_block_classes(self.model.visual)' in backbone
    assert 'use_cache=False' not in backbone
    source = (root / "gr00t/experiment/experiment.py").read_text()
    assert 'NumericalMonitor' in source and 'StepMetricsCallback' in source
    assert 'make_profiler_schedule' in source


def test_specialized_trainer_regroups_once_before_engine_setup(monkeypatch):
    import torch
    from gr00t.Replacement.zero2_allgather import PipelineTrainer
    from gr00t.experiment.trainer import Gr00tTrainer

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = object.__new__(PipelineTrainer)
    trainer.model = model
    trainer.zero2_allgather_group_elements = 12
    trainer._pipeline_groups_prepared = False
    monkeypatch.setattr(Gr00tTrainer, "create_optimizer", lambda self: optimizer)
    assert trainer.create_optimizer() is optimizer
    plan = trainer.zero2_allgather_group_plan
    assert len(plan) == 2  # The oversized weight remains whole; bias is separate.
    assert [name for group in plan for name in group["parameter_names"]] == ["weight", "bias"]
    assert trainer.create_optimizer() is optimizer
    assert trainer.zero2_allgather_group_plan is plan


@pytest.mark.parametrize("raises", [False, True])
def test_specialized_trainer_always_restores_on_train_exit(monkeypatch, raises):
    from types import SimpleNamespace
    from gr00t.Replacement.zero2_allgather import PipelineTrainer
    from gr00t.experiment.trainer import Gr00tTrainer

    closed = []
    trainer = object.__new__(PipelineTrainer)
    trainer._pipeline_callback = SimpleNamespace(close=lambda: closed.append(True))

    def train(self, *args, **kwargs):
        if raises:
            raise RuntimeError("simulated training failure")
        return "result"

    monkeypatch.setattr(Gr00tTrainer, "train", train)
    if raises:
        with pytest.raises(RuntimeError, match="simulated"):
            trainer.train()
    else:
        assert trainer.train() == "result"
    assert closed == [True]
