# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the opt-in Apex selector and runtime evidence, not GPU kernels."""
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def extension():
    return importlib.import_module("gr00t.Replacement.fused_adam")


@pytest.mark.parametrize("original", ["adamw_torch", "adamw_torch_fused", "adafactor"])
@pytest.mark.parametrize("env", [{}, {"GR00T_APEX_FUSED_ADAM": "0"}])
def test_disabled_preserves_original_optimizer(extension, original, env):
    assert extension.resolve_optimizer(original, env) == original


def test_disabled_module_load_and_resolution_work_without_apex(extension, monkeypatch):
    for name in ("apex", "apex.optimizers", "amp_C"):
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location("fused_adam_without_apex", extension.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolve_optimizer("adamw_torch", {}) == "adamw_torch"


@pytest.mark.parametrize("value", ["", "true", "2", "-1"])
def test_invalid_switch_rejected(extension, value):
    with pytest.raises(ValueError, match="GR00T_APEX_FUSED_ADAM"):
        extension.resolve_optimizer("adamw_torch", {"GR00T_APEX_FUSED_ADAM": value})


def test_enabled_selects_apex_and_can_revert_in_new_resolution(extension, monkeypatch):
    monkeypatch.setenv("GR00T_APEX_FUSED_ADAM", "1")
    monkeypatch.delenv("GR00T_ZERO2_ALLGATHER_PIPELINE", raising=False)
    assert extension.resolve_optimizer("adamw_torch") == "adamw_apex_fused"
    monkeypatch.setenv("GR00T_APEX_FUSED_ADAM", "0")
    assert extension.resolve_optimizer("adamw_torch") == "adamw_torch"


def test_apex_can_be_selected_with_allgather_pipeline(extension):
    assert extension.resolve_optimizer("adamw_torch", {
        "GR00T_APEX_FUSED_ADAM": "1", "GR00T_ZERO2_ALLGATHER_PIPELINE": "1",
    }) == "adamw_apex_fused"


def test_apex_does_not_silently_replace_other_algorithms(extension):
    with pytest.raises(ValueError, match="AdamW"):
        extension.resolve_optimizer("adafactor", {"GR00T_APEX_FUSED_ADAM": "1"})


def test_deepspeed_default_accepted_without_mutation(extension):
    config = {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}}}
    extension.validate_deepspeed_config("adamw_apex_fused", config)
    assert config == {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}}}
    extension.validate_deepspeed_config("adamw_apex_fused", None)


@pytest.mark.parametrize("config", [
    {"optimizer": {"type": "Adam"}},
    {"zero_optimization": {"offload_optimizer": {"device": "cpu"}}},
    {"zero_optimization": {"cpu_offload": True}},
])
def test_apex_rejects_deepspeed_optimizer_override_and_offload(extension, config):
    with pytest.raises(ValueError, match="DeepSpeed|offload"):
        extension.validate_deepspeed_config("adamw_apex_fused", config)
    extension.validate_deepspeed_config("adamw_torch", config)


@pytest.fixture
def cpu_apex(monkeypatch):
    """Substitute only the GPU dependency so evidence can inspect a real CPU optimizer."""
    class CpuFusedAdam(torch.optim.AdamW):
        adam_w_mode = 1

    optimizers = ModuleType("apex.optimizers")
    optimizers.FusedAdam = CpuFusedAdam
    multi = ModuleType("apex.multi_tensor_apply")
    multi.multi_tensor_applier = SimpleNamespace(available=True)
    kernels = ModuleType("amp_C")
    kernels.multi_tensor_adam = lambda *args: None
    monkeypatch.setitem(sys.modules, "apex.optimizers", optimizers)
    monkeypatch.setitem(sys.modules, "apex.multi_tensor_apply", multi)
    monkeypatch.setitem(sys.modules, "amp_C", kernels)
    return CpuFusedAdam


@pytest.mark.parametrize("wrapped", [False, True])
def test_records_underlying_optimizer_and_preserves_parameter_groups(extension, cpu_apex, tmp_path, wrapped):
    model = torch.nn.Linear(2, 2)
    optimizer = cpu_apex([
        {"params": [model.weight], "weight_decay": 1e-5},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=1e-4, betas=(0.8, 0.95), eps=1e-6)
    outer = SimpleNamespace(optimizer=optimizer) if wrapped else optimizer
    trainer = SimpleNamespace(optimizer=outer)
    if wrapped:
        trainer.optimizer = torch.optim.AdamW(model.parameters())
        trainer.model_wrapped = SimpleNamespace(optimizer=outer)
    callback = extension.optimizer_evidence_callback(trainer)
    control = object()
    assert callback.on_train_begin(SimpleNamespace(output_dir=tmp_path, process_index=3), None, control) is control
    report = json.loads((tmp_path / "optimizer" / "rank-03.json").read_text())
    assert report["optimizer_class"].endswith("CpuFusedAdam")
    assert report["adam_w_mode"] == 1
    assert report["parameter_groups"] == [
        {"lr": 1e-4, "betas": [0.8, 0.95], "eps": 1e-6, "weight_decay": 1e-5},
        {"lr": 1e-4, "betas": [0.8, 0.95], "eps": 1e-6, "weight_decay": 0.0},
    ]
    assert len(optimizer.param_groups) == 2


@pytest.mark.parametrize("wrong_mode", [False, True])
def test_runtime_rejects_replacement_or_non_adamw_mode(extension, cpu_apex, tmp_path, wrong_mode):
    cls = cpu_apex if wrong_mode else torch.optim.AdamW
    optimizer = cls([torch.nn.Parameter(torch.ones(2))])
    if wrong_mode:
        optimizer.adam_w_mode = 0
    callback = extension.optimizer_evidence_callback(SimpleNamespace(optimizer=optimizer))
    with pytest.raises(RuntimeError, match="FusedAdam|AdamW"):
        callback.on_train_begin(SimpleNamespace(output_dir=tmp_path, process_index=0), None, None)
    assert not (tmp_path / "optimizer").exists()


@pytest.mark.parametrize("missing", ["apex.optimizers", "amp_C"])
def test_missing_apex_or_compiled_extension_fails_explicitly(extension, cpu_apex, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(RuntimeError, match="Apex|extension"):
        extension.optimizer_evidence_callback(SimpleNamespace())


def test_unavailable_multitensor_extension_fails_explicitly(extension, cpu_apex, monkeypatch):
    monkeypatch.setattr(sys.modules["apex.multi_tensor_apply"].multi_tensor_applier, "available", False)
    with pytest.raises(RuntimeError, match="extension"):
        extension.optimizer_evidence_callback(SimpleNamespace())


@pytest.mark.parametrize("enabled, expected", [("0", "adamw_torch"), ("1", "adamw_apex_fused")])
def test_experiment_resolves_optimizer_before_distributed_initialization(extension, monkeypatch, enabled, expected):
    from gr00t.configs.base_config import Config
    from gr00t.experiment import experiment

    config = Config()
    config.training.optim = "adamw_torch"
    monkeypatch.setenv("GR00T_APEX_FUSED_ADAM", enabled)
    monkeypatch.setenv("GR00T_ZERO2_ALLGATHER_PIPELINE", "0")
    monkeypatch.setenv("GR00T_COMPILE_VARIANT", "off")
    monkeypatch.setenv("GR00T_COMPILE_SEQUENCE_POLICY", "auto")
    monkeypatch.setenv("GR00T_COMPILE_LENGTH_DIAGNOSTICS", "0")
    monkeypatch.setenv("GR00T_ZERO2_OVERLAP", "0")
    monkeypatch.setenv("GR00T_ZERO2_REDUCE_SCATTER", "0")
    monkeypatch.setattr(experiment, "warn_configs", lambda config: None)
    monkeypatch.setattr(experiment, "check_resume_compatibility", lambda training: None)

    class BeforeDeviceSetup(Exception):
        pass

    def stop_before_device_setup():
        assert config.training.optim == expected
        raise BeforeDeviceSetup()

    monkeypatch.setattr(experiment, "_init_distributed_process_group", stop_before_device_setup)
    with pytest.raises(BeforeDeviceSetup):
        experiment.run(config)


def test_trainer_factory_retains_decay_groups_and_adam_hyperparameters(extension, cpu_apex, tmp_path):
    from transformers import TrainingArguments

    from gr00t.experiment.trainer import Gr00tTrainer

    model = torch.nn.Linear(2, 2)
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none",
        optim=extension.resolve_optimizer("adamw_torch", {"GR00T_APEX_FUSED_ADAM": "1"}),
        learning_rate=1e-4, weight_decay=1e-5, adam_beta1=0.8, adam_beta2=0.95, adam_epsilon=1e-6,
    )
    trainer = Gr00tTrainer(model=model, args=args)
    optimizer = trainer.create_optimizer()
    assert type(optimizer) is cpu_apex
    groups = {group["weight_decay"]: group for group in optimizer.param_groups}
    assert [id(p) for p in groups[1e-5]["params"]] == [id(model.weight)]
    assert [id(p) for p in groups[0.0]["params"]] == [id(model.bias)]
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-4
        assert group["betas"] == (0.8, 0.95)
        assert group["eps"] == 1e-6
