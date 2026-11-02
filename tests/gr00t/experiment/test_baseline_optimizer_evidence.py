# SPDX-License-Identifier: Apache-2.0
"""Observe the actual optimizer under the DeepSpeed wrapper without a GPU."""
import json
from types import SimpleNamespace

import pytest
import torch

from gr00t.experiment import profiling


def test_baseline_records_engine_optimizer_groups(tmp_path):
    assert hasattr(profiling, "baseline_optimizer_evidence_callback")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "weight_decay": 1e-5},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=1e-4)
    trainer = SimpleNamespace(optimizer=None, model_wrapped=SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer)))
    callback = profiling.baseline_optimizer_evidence_callback(trainer)
    callback.on_train_begin(SimpleNamespace(output_dir=tmp_path, process_index=2), None, None)
    record = json.loads((tmp_path / "optimizer/rank-02.json").read_text())
    assert record["optimizer_class"] == "torch.optim.adamw.AdamW"
    assert [g["weight_decay"] for g in record["parameter_groups"]] == [1e-5, 0.0]
    assert all(g["lr"] == 1e-4 for g in record["parameter_groups"])


def test_baseline_rejects_unexpected_optimizer(tmp_path):
    assert hasattr(profiling, "baseline_optimizer_evidence_callback")
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    trainer = SimpleNamespace(optimizer=optimizer, model_wrapped=None)
    callback = profiling.baseline_optimizer_evidence_callback(trainer)
    with pytest.raises(RuntimeError, match="AdamW"):
        callback.on_train_begin(SimpleNamespace(output_dir=tmp_path, process_index=0), None, None)
