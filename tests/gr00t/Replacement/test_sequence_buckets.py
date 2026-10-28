"""CPU contracts for the opt-in sequence bucket adapter."""

import json

import pytest
import torch

from gr00t.Replacement import sequence_buckets


def test_policy_defaults_off_and_requires_a_real_configuration(tmp_path):
    assert sequence_buckets.resolve_policy({}) == ("auto", None, False)
    with pytest.raises(ValueError, match="GR00T_COMPILE_BUCKET_CONFIG"):
        sequence_buckets.resolve_policy({"GR00T_COMPILE_SEQUENCE_POLICY": "dynamic-buckets"})
    with pytest.raises(ValueError, match="GR00T_COMPILE_LENGTH_DIAGNOSTICS"):
        sequence_buckets.resolve_policy({"GR00T_COMPILE_LENGTH_DIAGNOSTICS": "yes"})


def test_bucket_lookup_rejects_out_of_range_and_mismatched_structure(tmp_path):
    path = tmp_path / "buckets.json"
    path.write_text(json.dumps({"schema_version": 1, "buckets": [
        {"target": "vl", "batch": 4, "feature_dim": 16, "dtype": "torch.float32",
         "min_length": 3, "max_length": 8}
    ]}))
    buckets = sequence_buckets.load_buckets(path)
    assert sequence_buckets.select_bucket(buckets, "vl", torch.zeros(4, 5, 16))["max_length"] == 8
    with pytest.raises(ValueError, match="No sequence bucket"):
        sequence_buckets.select_bucket(buckets, "vl", torch.zeros(4, 9, 16))
    with pytest.raises(ValueError, match="No sequence bucket"):
        sequence_buckets.select_bucket(buckets, "vl", torch.zeros(5, 5, 16))


def test_dit_marks_only_matching_sequence_dimensions_and_rejects_mask_mismatch(monkeypatch):
    calls = []
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda tensor, dimension, **kw: calls.append((tensor, dimension, kw)))
    features = torch.zeros(4, 6, 16)
    mask = torch.ones(4, 6, dtype=torch.bool)
    buckets = [{"target": "dit", "batch": 4, "feature_dim": 16, "dtype": "torch.float32",
                "min_length": 3, "max_length": 8}]
    sequence_buckets.mark_arguments(buckets, "dit", (), {
        "encoder_hidden_states": features, "image_mask": mask, "backbone_attention_mask": mask,
        "encoder_attention_mask": mask,
    })
    assert {id(t) for t, _, _ in calls} == {id(features), id(mask)}
    assert all(dim == 1 and kw == {"min": 3, "max": 8} for _, dim, kw in calls)
    with pytest.raises(ValueError, match="sequence dimension"):
        sequence_buckets.mark_arguments(buckets, "dit", (), {
            "encoder_hidden_states": features, "image_mask": torch.ones(4, 7, dtype=torch.bool),
        })
    with pytest.raises(ValueError, match="mask"):
        sequence_buckets.mark_arguments(buckets, "dit", (), {
            "encoder_hidden_states": features, "image_mask": mask,
            "backbone_attention_mask": mask, "encoder_attention_mask": mask.float(),
        })


def test_adapter_restores_original_compiled_callable_after_failure():
    class Target:
        def __init__(self):
            self._compiled_call_impl = lambda *_a, **_kw: "original"

    target = Target()
    original = target._compiled_call_impl
    adapter = sequence_buckets.SequenceAdapter([("vl", target)], buckets=[])
    with pytest.raises(ValueError, match="No sequence bucket"):
        with adapter:
            target._compiled_call_impl(torch.zeros(4, 5, 16))
    assert target._compiled_call_impl is original


def test_same_bucket_reuses_compiled_graph_for_distinct_sequence_lengths():
    torch._dynamo.reset()
    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    class Target:
        def __init__(self):
            self._compiled_call_impl = torch.compile(lambda hidden_states: hidden_states.sin() + 1, backend=backend)

    target = Target()
    buckets = [{"target": "vl", "batch": 4, "feature_dim": 16, "dtype": "torch.float32",
                "min_length": 3, "max_length": 8}]
    with sequence_buckets.SequenceAdapter([("vl", target)], buckets):
        for length in (4, 7, 4):
            values = torch.randn(4, length, 16)
            torch.testing.assert_close(target._compiled_call_impl(values), values.sin() + 1)
    assert len(compiled_graphs) == 1


def test_opt_in_rejects_legacy_or_incomplete_compile_scopes():
    with pytest.raises(ValueError, match="legacy"):
        sequence_buckets.validate_activation("dynamic-buckets", False, "off", {"whole": True})
    with pytest.raises(ValueError, match="dit-model-vl"):
        sequence_buckets.validate_activation("dynamic-buckets", False, "vl", None)
    sequence_buckets.validate_activation("auto", False, "off", {"whole": True})


def test_diagnostics_record_original_batch_shape_and_exact_step(tmp_path):
    recorder = sequence_buckets.DiagnosticRecorder(tmp_path, rank=1, physical_device="2")
    recorder.step = 7
    batch = {"inputs": {"input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                        "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}}
    recorder.record_batch(batch)
    rows = [json.loads(line) for line in (tmp_path / "rank-01.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 7 and rows[0]["rank"] == 1
    assert rows[0]["physical_device"] == "2"
    assert rows[0]["valid_lengths"] == [2, 3]
    assert rows[0]["padded_length"] == 3
    assert len(rows[0]["input_sha256"]) == 64


def test_diagnostic_adapter_records_call_and_restores_without_marking(tmp_path, monkeypatch):
    class Target:
        _compiled_call_impl = staticmethod(lambda value: value + 1)

    def fail_mark(*args, **kwargs):
        pytest.fail("auto diagnostics must not mark tensors dynamic")

    monkeypatch.setattr(torch._dynamo, "mark_dynamic", fail_mark)
    target = Target()
    original = target._compiled_call_impl
    recorder = sequence_buckets.DiagnosticRecorder(tmp_path, rank=0, physical_device="1")
    recorder.step = 2
    with sequence_buckets.SequenceAdapter([("vl", target)], buckets=None, recorder=recorder):
        torch.testing.assert_close(target._compiled_call_impl(torch.ones(4, 5, 16)), torch.full((4, 5, 16), 2.0))
    assert target._compiled_call_impl is original
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [(x["stage"], x["step"]) for x in rows] == [("begin", 2), ("end", 2)]
    assert rows[0]["call_id"] == rows[1]["call_id"]


def test_diagnostics_distinguish_compilation_paths(tmp_path):
    class Target:
        _compiled_call_impl = staticmethod(lambda value: value)

    recorder = sequence_buckets.DiagnosticRecorder(tmp_path, rank=0, physical_device="1")
    recorder.step = 3
    target = Target()
    with sequence_buckets.SequenceAdapter([("vl", target, "vl.block.0")], None, recorder):
        target._compiled_call_impl(torch.ones(4, 5, 16))
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert all(row["target_path"] == "vl.block.0" for row in rows)


def test_trainer_instrumentation_records_the_batch_and_restores_method(tmp_path):
    class Trainer:
        def __init__(self):
            self.callbacks = []

        def _prepare_inputs(self, batch):
            return batch

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

    trainer = Trainer()
    original = trainer._prepare_inputs
    recorder = sequence_buckets.DiagnosticRecorder(tmp_path, rank=0, physical_device="1")
    attachment = sequence_buckets.TrainerDiagnostics(trainer, recorder)
    attachment.install()
    trainer.callbacks[0].on_step_begin(None, type("State", (), {"global_step": 6})(), None)
    batch = {"inputs": {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}}
    assert trainer._prepare_inputs(batch) is batch
    attachment.close()
    assert trainer._prepare_inputs == original and not trainer.callbacks
    assert json.loads(recorder.path.read_text())["step"] == 7
