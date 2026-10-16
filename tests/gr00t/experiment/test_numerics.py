"""CPU regression coverage for numerical diagnostics and safe saving."""

import json
from datetime import timedelta
from pathlib import Path
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
from safetensors.torch import save_file
import torch
import torch.distributed as dist
from transformers import TrainingArguments

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.numerics import (
    NumericalError,
    NumericalMonitor,
    scan_checkpoint,
    witness_indices,
)
from gr00t.experiment.trainer import Gr00tTrainer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, inputs):
        prediction = self.linear(inputs["features"])
        return {"loss": (prediction - inputs["target"]).square().mean()}


@pytest.mark.parametrize("size", [0, 1, 1536, 50331648, 2**30])
def test_large_parameter_witness_indices_never_round_out_of_bounds(size):
    indices = witness_indices(size, "cpu")
    if size:
        assert indices[0] == 0
        assert indices[-1] == size - 1
        assert (indices < size).all()
    else:
        assert indices.numel() == 0


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"inputs": {"features": torch.tensor([0.5, 1.0]),
                           "target": torch.tensor([0.25]), "action_mask": torch.ones(1)}}


def collate(samples):
    return {"inputs": {key: torch.stack([sample["inputs"][key] for sample in samples])
                       for key in samples[0]["inputs"]}}


def test_scan_all_shards_and_reject_missing_index(tmp_path):
    save_file({"good": torch.ones(3)}, tmp_path / "one.safetensors")
    save_file({"bad": torch.tensor([float("nan"), float("inf")])}, tmp_path / "two.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({
        "weight_map": {"good": "one.safetensors", "bad": "two.safetensors"}
    }))
    report = scan_checkpoint(tmp_path)
    assert report["counts"]["tensors"] == 2
    assert report["counts"]["nan"] == report["counts"]["inf"] == 1
    assert not report["passed"]
    index.write_text(json.dumps({
        "weight_map": {"good": "one.safetensors", "missing": "one.safetensors"}
    }))
    with pytest.raises(ValueError, match="Missing indexed"):
        scan_checkpoint(tmp_path)


@pytest.mark.parametrize("stage", ["input", "forward", "gradient", "optimizer_state"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_detect_nonfinite_at_boundary(tmp_path, stage, nonfinite):
    monitor = NumericalMonitor(tmp_path)
    monitor.device = torch.device("cpu")
    monitor.observe(stage, torch.tensor([0.0, nonfinite]))
    with pytest.raises(NumericalError, match=stage):
        monitor.boundary(stage)
    record = json.loads((monitor.root / "rank-00.jsonl").read_text())
    failure = record["failures"][0]
    assert failure["nan"] + failure["inf"] == 1


def test_empty_action_mask_stops_before_forward(tmp_path):
    monitor = NumericalMonitor(tmp_path)
    monitor.device = torch.device("cpu")
    with pytest.raises(NumericalError, match="action_mask_empty"):
        monitor.prepare_batch({"action_mask": torch.zeros(2, 3)})


def test_real_cpu_training_updates_and_saves_only_when_explicit(tmp_path):
    torch.set_num_threads(1)
    model = ToyModel()
    before = model.linear.weight.detach().clone()
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                             max_steps=3, per_device_train_batch_size=2, save_strategy="no",
                             logging_steps=1, optim="adamw_torch", learning_rate=1e-3,
                             disable_tqdm=True, remove_unused_columns=False,
                             logging_nan_inf_filter=False)
    trainer = Gr00tTrainer(model=model, args=args, train_dataset=ToyDataset(),
                           data_collator=collate, numerics_monitor=monitor)
    trainer.train()
    assert not torch.equal(before, model.linear.weight)
    assert monitor.saw_update and monitor.saw_gradient
    assert not list(tmp_path.glob("checkpoint-*"))
    assert not list(tmp_path.glob("*.safetensors"))
    trainer.save_model()
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert scan_checkpoint(tmp_path)["passed"]
    assert not model.linear.weight._backward_hooks
    assert not list(tmp_path.rglob("*.pt"))


def test_training_failure_removes_monitor_hooks_and_does_not_save(tmp_path):
    model = ToyModel()
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    injection = model.linear.weight.register_hook(lambda grad: grad * float("nan"))
    before = model.linear.weight.detach().clone()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                             max_steps=2, per_device_train_batch_size=2, save_strategy="no",
                             optim="adamw_torch", disable_tqdm=True,
                             remove_unused_columns=False, logging_nan_inf_filter=False)
    trainer = Gr00tTrainer(model=model, args=args, train_dataset=ToyDataset(),
                           data_collator=collate, numerics_monitor=monitor)
    with pytest.raises(NumericalError, match="pre_update"):
        trainer.train()
    injection.remove()
    assert torch.equal(before, model.linear.weight)
    assert not model.linear.weight._backward_hooks
    assert not list(tmp_path.glob("checkpoint-*"))
    assert not list(tmp_path.glob("*.safetensors"))


def test_corrupt_gradient_rejected_before_optimizer_update(tmp_path):
    model = ToyModel()
    before = model.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    monitor.trainer = SimpleNamespace(deepspeed=None, optimizer=optimizer)
    monitor.on_train_begin(None, None, None)
    model.linear.weight.register_hook(lambda grad: grad * float("nan"))
    model({"features": torch.ones(2, 2), "target": torch.ones(2, 1)})["loss"].backward()
    with pytest.raises(NumericalError, match="pre_update"):
        optimizer.step()
    assert torch.equal(before, model.linear.weight)
    monitor.close()


def test_numerics_rejects_unsupported_zero_stage_before_model_loading():
    from gr00t.experiment.experiment import run

    config = get_default_config()
    config.training.check_numerics = True
    config.training.num_gpus = 8
    config.training.deepspeed_stage = 3
    with pytest.raises(ValueError, match="ZeRO-2"):
        run(config)


def test_corrupt_optimizer_state_is_rejected(tmp_path):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model({"features": torch.ones(2, 2), "target": torch.ones(2, 1)})["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    monitor.trainer = SimpleNamespace(deepspeed=None, optimizer=optimizer)
    monitor.on_train_begin(None, None, None)
    optimizer.state[model.linear.weight]["exp_avg"].fill_(float("nan"))
    model({"features": torch.ones(2, 2), "target": torch.ones(2, 1)})["loss"].backward()
    before = model.linear.weight.detach().clone()
    with pytest.raises(NumericalError, match="optimizer_state"):
        optimizer.step()
    assert torch.equal(before, model.linear.weight)
    monitor.close()


def test_forward_hooks_only_record_until_boundary(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    handle = model.register_forward_hook(monitor.forward_hook("test_module"))
    with torch.no_grad():
        model[0].weight.fill_(float("inf"))
    model(torch.ones(2, 2))
    with pytest.raises(NumericalError, match="forward.test_module"):
        monitor.boundary("forward")
    handle.remove()
    monitor.close()


def test_config_serializes_new_options():
    config = get_default_config()
    config.training.check_numerics = True
    config.training.save_final_model = False
    config.training.save_strategy = "no"
    result = OmegaConf.create(config.__dict__)
    restored = OmegaConf.create(OmegaConf.to_yaml(result))
    assert restored.training.check_numerics
    assert not restored.training.save_final_model
    assert restored.training.save_strategy == "no"


def test_deepspeed_negative_group_norms_rejected_before_update(tmp_path):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    zero = SimpleNamespace(optimizer=optimizer, averaged_gradients={},
                           single_partition_of_fp32_groups=list(model.parameters()),
                           get_grad_norm_direct=lambda *args: torch.tensor(-1.0))

    def step():
        zero.get_grad_norm_direct()
        zero.get_grad_norm_direct()
        optimizer.step()

    engine = SimpleNamespace(optimizer=zero, step=step)
    monitor = NumericalMonitor(tmp_path)
    monitor.attach(model)
    monitor.trainer = SimpleNamespace(deepspeed=engine)
    monitor.on_train_begin(None, None, None)
    model({"features": torch.ones(2, 2), "target": torch.ones(2, 1)})["loss"].backward()
    before = model.linear.weight.detach().clone()
    with pytest.raises(NumericalError, match="negative_gradient_group_norm"):
        engine.step()
    assert monitor.norms == [-1.0, -1.0]
    assert torch.equal(before, model.linear.weight)
    monitor.close()


def _rank_failure_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        monitor = NumericalMonitor(output)
        monitor.device = torch.device("cpu")
        monitor.observe("gradient", torch.tensor(float("nan") if rank == 1 else 1.0))
        with pytest.raises(NumericalError):
            monitor.boundary("backward")
        (Path(output) / f"rank-{rank}.exited").touch()
    finally:
        dist.destroy_process_group()


def test_two_ranks_fail_together_without_hanging(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    context = torch.multiprocessing.spawn(_rank_failure_worker,
                                         args=(str(tmp_path / "rendezvous"), str(tmp_path)),
                                         nprocs=2, join=False)
    deadline = time.monotonic() + 40
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Numerical failure did not terminate both ranks within 40 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert (tmp_path / "rank-0.exited").exists()
    assert (tmp_path / "rank-1.exited").exists()
