from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gr00t.Replacement import zero2_comm
from gr00t.experiment.training_runtime import resolve_zero2_comm_switches


# ---------------------------------------------------------------- fakes

class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))


class FakeEvent:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream):
        self.log.append(("record", self.name, stream.name))


class FakeAccelerator:
    def __init__(self, log):
        self.log = log
        self.compute = FakeStream("compute", log)

    def current_stream(self):
        return self.compute

    @contextmanager
    def stream(self, stream):
        self.log.append(("stream", stream.name))
        yield

    def resolves_data_dependency(self):
        return False


class FakeBucket:
    def __init__(self, size=10):
        self.buffer = [torch.zeros(size), torch.zeros(size)]
        self.params, self.grads = [], []
        self.elements = 0
        self.index = 0
        self.has_moe_params = False

    def clear(self):
        self.params.clear()
        self.grads.clear()
        self.elements = 0
        self.index = 0
        self.has_moe_params = False


class FakeParam:
    def __init__(self, pid, values):
        self.pid = pid
        self.grad = values
        self.param_idx_in_group = pid

    def numel(self):
        return self.grad.numel()


class FakeOptimizer:
    overlap_comm = True
    contiguous_gradients = True
    reduce_scatter = True
    partition_gradients = True
    use_multi_rank_bucket_allreduce = True
    cpu_offload = False
    zenflow = False
    sequence_parallel_size = 1
    gradient_predivide_factor = 1.0
    reduce_bucket_size = 10
    dp_process_group = None

    def __init__(self, log, dtype=torch.bfloat16):
        self.log = log
        self.reduction_stream = FakeStream("reduction", log)
        self.ipg_buckets = {dtype: FakeBucket()}
        self.dtype = dtype
        self.params_already_reduced = {}
        self.extra_large_param_to_reduce = {}
        self.ready_for_gradients = False
        self.flushed_indices = []

    def get_gradient_for_reduction(self, param):
        return param.grad

    def get_param_comm_dtype(self, param):
        return self.dtype

    def get_param_id(self, param):
        return param.pid

    def report_ipg_memory_usage(self, *args):
        pass

    def reduce_ipg_grads(self):
        bucket = self.ipg_buckets[self.dtype]
        self.flushed_indices.append(bucket.index)
        self.log.append(("reduce_ipg_grads", bucket.index))
        bucket.clear()

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        raise AssertionError("original must not run")

    def average_tensor(self, tensor, communication_data_type):
        raise AssertionError("original must not run")

    def setup_buckets(self):
        self.log.append(("setup_buckets",))
        self.ready_for_gradients = True

    def allreduce_and_scatter(self, bucket, communication_data_type, numel_per_bucket=500000000,
                              log=None, divide=True, process_group=None):
        raise AssertionError("not used")


def make_installation(log, **kwargs):
    optimizer = FakeOptimizer(log)
    accelerator = FakeAccelerator(log)
    counter = iter(range(1000))
    installation = zero2_comm.Zero2CommInstallation(
        optimizer,
        accelerator=accelerator,
        event_factory=lambda: FakeEvent(f"e{next(counter)}", log),
        **kwargs,
    )
    return optimizer, installation


# ---------------------------------------------------------------- switches

@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, (False, False)),
        ({"GR00T_ZERO2_OVERLAP": "1"}, (True, False)),
        ({"GR00T_ZERO2_REDUCE_SCATTER": "1"}, (False, True)),
        ({"GR00T_ZERO2_OVERLAP": "1", "GR00T_ZERO2_REDUCE_SCATTER": "1"}, (True, True)),
        ({"GR00T_ZERO2_OVERLAP": "0", "GR00T_ZERO2_REDUCE_SCATTER": "0"}, (False, False)),
    ],
)
def test_switch_resolution(environ, expected):
    assert resolve_zero2_comm_switches(environ) == expected


@pytest.mark.parametrize("name", ["GR00T_ZERO2_OVERLAP", "GR00T_ZERO2_REDUCE_SCATTER"])
@pytest.mark.parametrize("value", ["true", "", "2", "on"])
def test_invalid_switch_value_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_zero2_comm_switches({name: value})


def test_default_path_does_not_import_the_replacement():
    code = """
import sys
from gr00t.experiment.training_runtime import resolve_zero2_comm_switches
assert resolve_zero2_comm_switches({}) == (False, False)
assert 'gr00t.Replacement.zero2_comm' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_config_validation():
    from gr00t.configs.base_config import Config

    config = Config()
    config.training.deepspeed_stage = 2
    config.training.use_ddp = False
    config.training.num_gpus = 8
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.gradient_accumulation_steps = 1
    zero2_comm.validate_training_config(config, True, True)

    config.training.gradient_accumulation_steps = 2
    with pytest.raises(ValueError, match="accumulation"):
        zero2_comm.validate_training_config(config, True, False)
    zero2_comm.validate_training_config(config, False, False)

    config.training.gradient_accumulation_steps = 1
    config.training.deepspeed_stage = 3
    with pytest.raises(ValueError, match="ZeRO-2"):
        zero2_comm.validate_training_config(config, False, True)


# ---------------------------------------------------------------- vendor pin

def test_pinned_hashes_match_the_installed_vendor_build():
    pytest.importorskip("deepspeed")
    from importlib.metadata import version

    if version("deepspeed") != zero2_comm._SUPPORTED_DEEPSPEED_VERSION:
        pytest.skip("pinned DTK DeepSpeed build is not installed")
    assert zero2_comm.validate_vendor_build(*zero2_comm.current_vendor_versions_and_hashes())


def test_vendor_mismatch_is_rejected():
    hashes = dict(zero2_comm._SUPPORTED_METHOD_HASHES)
    zero2_comm.validate_vendor_build(
        zero2_comm._SUPPORTED_DEEPSPEED_VERSION, zero2_comm._SUPPORTED_TORCH_VERSION, hashes
    )
    hashes["average_tensor"] = "0" * 64
    with pytest.raises(RuntimeError, match="pinned"):
        zero2_comm.validate_vendor_build(
            zero2_comm._SUPPORTED_DEEPSPEED_VERSION, zero2_comm._SUPPORTED_TORCH_VERSION, hashes
        )
    with pytest.raises(RuntimeError, match="pinned"):
        zero2_comm.validate_vendor_build(
            "0.0.0", zero2_comm._SUPPORTED_TORCH_VERSION, dict(zero2_comm._SUPPORTED_METHOD_HASHES)
        )


# ---------------------------------------------------------------- optimizer validation

def test_supported_optimizer_passes_validation():
    log = []
    zero2_comm.validate_optimizer(FakeOptimizer(log), FakeAccelerator(log), require_deepspeed_type=False)


def test_real_optimizer_class_is_required_by_default():
    log = []
    with pytest.raises(RuntimeError, match="DeepSpeedZeroOptimizer"):
        zero2_comm.validate_optimizer(FakeOptimizer(log), FakeAccelerator(log))


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("overlap_comm", False, "overlap_comm"),
        ("contiguous_gradients", False, "contiguous_gradients"),
        ("reduce_scatter", False, "reduce_scatter"),
        ("partition_gradients", False, "partition_gradients"),
        ("use_multi_rank_bucket_allreduce", False, "use_multi_rank_bucket_allreduce"),
        ("cpu_offload", True, "cpu_offload"),
        ("zenflow", True, "zenflow"),
        ("sequence_parallel_size", 2, "sequence"),
        ("gradient_predivide_factor", 2.0, "predivide"),
    ],
)
def test_unsupported_optimizer_settings_are_rejected(attribute, value, message):
    log = []
    optimizer = FakeOptimizer(log)
    setattr(optimizer, attribute, value)
    with pytest.raises(RuntimeError, match=message):
        zero2_comm.validate_optimizer(optimizer, FakeAccelerator(log), require_deepspeed_type=False)


def test_multiple_communication_dtypes_are_rejected():
    log = []
    optimizer = FakeOptimizer(log)
    optimizer.ipg_buckets[torch.float32] = FakeBucket()
    with pytest.raises(RuntimeError, match="one gradient communication dtype"):
        zero2_comm.validate_optimizer(optimizer, FakeAccelerator(log), require_deepspeed_type=False)


def test_synchronized_device_or_missing_reduction_stream_is_rejected():
    log = []
    accelerator = FakeAccelerator(log)
    accelerator.resolves_data_dependency = lambda: True
    with pytest.raises(RuntimeError, match="reduction stream"):
        zero2_comm.validate_optimizer(FakeOptimizer(log), accelerator, require_deepspeed_type=False)
    optimizer = FakeOptimizer(log)
    optimizer.reduction_stream = None
    with pytest.raises(RuntimeError, match="reduction stream"):
        zero2_comm.validate_optimizer(optimizer, FakeAccelerator(log), require_deepspeed_type=False)


def test_changed_method_signature_is_rejected():
    log = []

    class Changed(FakeOptimizer):
        def reduce_ipg_grads(self, comm_dtype=None):
            pass

    with pytest.raises(RuntimeError, match="reduce_ipg_grads signature"):
        zero2_comm.validate_optimizer(Changed(log), FakeAccelerator(log), require_deepspeed_type=False)


# ---------------------------------------------------------------- owner layout

def _layout_optimizer(log):
    optimizer = FakeOptimizer(log)
    p0, p1, p2 = FakeParam(0, torch.zeros(6)), FakeParam(1, torch.zeros(4)), FakeParam(2, torch.zeros(5))
    optimizer.bit16_groups = [[p0, p1, p2]]
    optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1], 2: [1, 2]}}
    optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0, 2: 0}, 2: {2: 2}}}
    bucket = optimizer.ipg_buckets[torch.bfloat16]
    bucket.params = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    return optimizer, bucket


def test_owner_slices_merge_adjacent_runs():
    optimizer, bucket = _layout_optimizer([])
    spec, total = zero2_comm.build_rank_and_offsets(optimizer, bucket)
    assert total == 15
    assert spec == [(0, 0, 3), (1, 3, 9), (2, 12, 3)]


def test_offsets_are_sorted_when_partition_ids_arrive_unordered():
    optimizer, bucket = _layout_optimizer([])
    optimizer.param_to_partition_ids[0][0] = [1, 0]
    spec, _ = zero2_comm.build_rank_and_offsets(optimizer, bucket)
    assert spec[0] == (0, 0, 3)


# ---------------------------------------------------------------- stream ordering

def _feed(optimizer, count, numel=6):
    params = [FakeParam(i, torch.full((numel,), float(i + 1))) for i in range(count)]
    for pid, param in enumerate(params):
        optimizer.params_already_reduced[pid] = False
        optimizer.reduce_independent_p_g_buckets_and_remove_grads(param, 0)
    return params


def test_overlap_alternates_halves_and_never_makes_compute_wait_for_reduction():
    log = []
    optimizer, installation = make_installation(log, overlap=True, reduce_scatter=False)
    params = _feed(optimizer, 5)
    assert optimizer.flushed_indices == [0, 1, 0, 1]
    assert [e for e in log if e[0] == "wait_stream"] == []
    records = [e for e in log if e[0] == "record"]
    waits = [e for e in log if e[0] == "wait_event"]
    assert [r[2] for r in records] == ["reduction"] * 4
    assert len(waits) == 3
    assert installation.scheduler.records == 4 and installation.scheduler.waits == 3
    # every wait on compute happens only after the event it waits for was recorded
    seen = set()
    for entry in log:
        if entry[0] == "record":
            seen.add(entry[1])
        if entry[0] == "wait_event":
            assert entry[2] in seen
    bucket = optimizer.ipg_buckets[torch.bfloat16]
    assert bucket.index == 0
    assert params[4].grad.data_ptr() == bucket.buffer[0].data_ptr()
    assert params[3].grad.data_ptr() == bucket.buffer[1].data_ptr()


def test_gradients_are_copied_into_the_active_half():
    log = []
    optimizer, _ = make_installation(log, overlap=True, reduce_scatter=False)
    bucket = optimizer.ipg_buckets[torch.bfloat16]
    first = FakeParam(0, torch.full((6,), 1.0))
    second = FakeParam(1, torch.full((6,), 2.0))
    optimizer.params_already_reduced.update({0: False, 1: False})
    optimizer.reduce_independent_p_g_buckets_and_remove_grads(first, 0)
    optimizer.reduce_independent_p_g_buckets_and_remove_grads(second, 0)
    assert bucket.buffer[0][:6].tolist() == [1.0] * 6
    assert bucket.buffer[1][:6].tolist() == [2.0] * 6
    assert bucket.elements == 6 and bucket.index == 1


def test_extra_large_parameter_is_reduced_from_its_own_gradient():
    log = []
    optimizer, _ = make_installation(log, overlap=True, reduce_scatter=False)
    big = FakeParam(0, torch.ones(25))
    optimizer.params_already_reduced[0] = False
    optimizer.reduce_independent_p_g_buckets_and_remove_grads(big, 0)
    assert optimizer.extra_large_param_to_reduce[torch.bfloat16] is big
    assert optimizer.ipg_buckets[torch.bfloat16].elements == 25


def test_setup_buckets_resets_events_only_for_a_fresh_backward():
    log = []
    optimizer, installation = make_installation(log, overlap=True, reduce_scatter=False)
    installation.scheduler.record_consumed((torch.bfloat16, 0), optimizer.reduction_stream)
    optimizer.ready_for_gradients = True
    optimizer.setup_buckets()
    assert installation.scheduler.wait_before_reuse((torch.bfloat16, 0)) is True
    installation.scheduler.record_consumed((torch.bfloat16, 0), optimizer.reduction_stream)
    optimizer.ready_for_gradients = False
    optimizer.setup_buckets()
    assert installation.scheduler.wait_before_reuse((torch.bfloat16, 0)) is False


class FakeTensor:
    def __init__(self, log, numel=15):
        self.log, self._numel = log, numel

    def numel(self):
        return self._numel

    def div_(self, value):
        self.log.append(("div_", value))

    def narrow(self, dim, offset, numel):
        return ("slice", offset, numel)

    def record_stream(self, stream):
        self.log.append(("record_stream", stream.name))


def _patch_distributed(monkeypatch, recorded):
    monkeypatch.setattr(
        zero2_comm,
        "dist",
        SimpleNamespace(
            get_world_size=lambda group=None: 4,
            get_rank=lambda group=None: 1,
            is_available=lambda: False,
            is_initialized=lambda: False,
        ),
    )

    def fake_reduce_by_owner(slices, owners, **kwargs):
        recorded.append((slices, owners, kwargs))
        return "reduce_scatter"

    monkeypatch.setattr(zero2_comm, "reduce_by_owner", fake_reduce_by_owner)


def test_overlap_makes_reduction_wait_for_compute_only(monkeypatch):
    log, recorded = [], []
    _patch_distributed(monkeypatch, recorded)
    optimizer, installation = make_installation(log, overlap=True, reduce_scatter=True)
    optimizer.bit16_groups, bucket = _layout_optimizer(log)[0].bit16_groups, None
    optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1], 2: [1, 2]}}
    optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0, 2: 0}, 2: {2: 2}}}
    optimizer.ipg_buckets[torch.bfloat16].params = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    optimizer.average_tensor(FakeTensor(log), torch.bfloat16)
    assert ("wait_stream", "reduction", "compute") in log
    assert ("wait_stream", "compute", "reduction") not in log
    assert ("record_stream", "reduction") in log
    slices, owners, kwargs = recorded[0]
    assert owners == [0, 1, 2]
    assert slices == [("slice", 0, 3), ("slice", 3, 9), ("slice", 12, 3)]
    assert kwargs["rank"] == 1 and kwargs["world_size"] == 4 and kwargs["dtype"] is torch.bfloat16
    assert ("div_", 4.0) in log
    assert installation.stats["reduce_scatter"] == 1


def test_reduce_scatter_alone_keeps_the_original_two_way_wait(monkeypatch):
    log, recorded = [], []
    _patch_distributed(monkeypatch, recorded)
    optimizer, _ = make_installation(log, overlap=False, reduce_scatter=True)
    optimizer.bit16_groups = _layout_optimizer(log)[0].bit16_groups
    optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1], 2: [1, 2]}}
    optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0, 2: 0}, 2: {2: 2}}}
    optimizer.ipg_buckets[torch.bfloat16].params = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    optimizer.average_tensor(FakeTensor(log), torch.bfloat16)
    assert ("wait_stream", "reduction", "compute") in log
    assert ("wait_stream", "compute", "reduction") in log
    assert not any(entry[0] == "record_stream" for entry in log)


def test_bucket_layout_mismatch_fails_loudly(monkeypatch):
    log = []
    _patch_distributed(monkeypatch, [])
    optimizer, _ = make_installation(log, overlap=True, reduce_scatter=True)
    optimizer.bit16_groups = _layout_optimizer(log)[0].bit16_groups
    optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1], 2: [1, 2]}}
    optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0, 2: 0}, 2: {2: 2}}}
    optimizer.ipg_buckets[torch.bfloat16].params = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    with pytest.raises(RuntimeError, match="covers 15 elements"):
        optimizer.average_tensor(FakeTensor(log, numel=14), torch.bfloat16)


# ---------------------------------------------------------------- install / restore

def test_installation_binds_only_selected_methods_and_restores():
    log = []
    optimizer, installation = make_installation(log, overlap=False, reduce_scatter=True)
    assert "average_tensor" in optimizer.__dict__
    assert "reduce_independent_p_g_buckets_and_remove_grads" not in optimizer.__dict__
    installation.close()
    assert "average_tensor" not in optimizer.__dict__
    installation.close()

    optimizer, installation = make_installation(log, overlap=True, reduce_scatter=False)
    for name in ("average_tensor", "reduce_independent_p_g_buckets_and_remove_grads", "setup_buckets"):
        assert name in optimizer.__dict__
    installation.close()
    assert not any(name in optimizer.__dict__ for name in
                   ("average_tensor", "reduce_independent_p_g_buckets_and_remove_grads", "setup_buckets"))


def test_installation_restores_a_previous_instance_override():
    log = []
    optimizer = FakeOptimizer(log)
    marker = MethodType(lambda self, tensor, dtype: "previous", optimizer)
    optimizer.average_tensor = marker
    installation = zero2_comm.Zero2CommInstallation(
        optimizer, overlap=False, reduce_scatter=True, accelerator=FakeAccelerator(log)
    )
    assert optimizer.average_tensor != marker
    installation.close()
    assert optimizer.average_tensor is marker


def test_installation_requires_a_switch():
    with pytest.raises(ValueError):
        zero2_comm.Zero2CommInstallation(FakeOptimizer([]), overlap=False, reduce_scatter=False,
                                         accelerator=FakeAccelerator([]))


def _patch_callback_environment(monkeypatch, log):
    monkeypatch.setattr(zero2_comm, "validate_vendor_build", lambda *args: True)
    monkeypatch.setattr(zero2_comm, "current_vendor_versions_and_hashes", lambda: (None, None, None))
    monkeypatch.setattr(zero2_comm, "get_accelerator", lambda: FakeAccelerator(log))
    original = zero2_comm.validate_optimizer
    monkeypatch.setattr(
        zero2_comm, "validate_optimizer",
        lambda optimizer, accelerator=None: original(optimizer, accelerator, require_deepspeed_type=False),
    )


def test_callback_installs_at_train_begin_and_restores_at_train_end(monkeypatch, tmp_path):
    log = []
    _patch_callback_environment(monkeypatch, log)
    optimizer = FakeOptimizer(log)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(optimizer=optimizer))
    callback = zero2_comm.Zero2CommCallback(trainer, True, True, output_dir=tmp_path)
    callback.on_train_begin(None, None, SimpleNamespace())
    assert "average_tensor" in optimizer.__dict__
    with pytest.raises(RuntimeError, match="twice"):
        callback.on_train_begin(None, None, SimpleNamespace())
    callback.on_train_end(None, None, SimpleNamespace())
    assert "average_tensor" not in optimizer.__dict__
    report = (tmp_path / "zero2_comm" / "runtime-rank-00.json").read_text()
    assert '"overlap": true' in report and '"reduce_scatter": true' in report
    assert '"comm"' in report and '"layout_digest"' in report


def test_close_is_idempotent_and_restores_after_an_aborted_run(monkeypatch, tmp_path):
    log = []
    _patch_callback_environment(monkeypatch, log)
    optimizer = FakeOptimizer(log)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(optimizer=optimizer))
    callback = zero2_comm.Zero2CommCallback(trainer, True, True, output_dir=tmp_path)
    callback.on_train_begin(None, None, SimpleNamespace())
    callback.close()
    callback.close()
    assert "average_tensor" not in optimizer.__dict__
    assert (tmp_path / "zero2_comm" / "runtime-rank-00.json").exists()


def test_layout_digest_changes_with_the_owner_layout():
    first, _ = _layout_optimizer([])
    second, _ = _layout_optimizer([])
    assert zero2_comm.layout_digest(first) == zero2_comm.layout_digest(second)
    second.grad_start_offset[0][1][0] = 2
    assert zero2_comm.layout_digest(first) != zero2_comm.layout_digest(second)
    third, _ = _layout_optimizer([])
    third.reduce_bucket_size = 99
    assert zero2_comm.layout_digest(first) != zero2_comm.layout_digest(third)


def test_ranks_with_different_layouts_all_fail_at_install(monkeypatch):
    log = []
    _patch_callback_environment(monkeypatch, log)
    optimizer = FakeOptimizer(log)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(optimizer=optimizer))

    def fake_gather(status):
        other = {"ok": True, "config": (True, True, "different")}
        return [status, other]

    monkeypatch.setattr(zero2_comm, "_gather_statuses", fake_gather)
    callback = zero2_comm.Zero2CommCallback(trainer, True, True)
    with pytest.raises(RuntimeError, match="disagree"):
        callback.on_train_begin(None, None, SimpleNamespace())
    assert "average_tensor" not in optimizer.__dict__


def test_switch_consistency_check_is_a_no_op_without_a_process_group():
    zero2_comm.verify_switch_consistency(False, False)


def test_callback_leaves_the_optimizer_untouched_when_validation_fails(monkeypatch):
    log = []
    _patch_callback_environment(monkeypatch, log)
    optimizer = FakeOptimizer(log)
    optimizer.cpu_offload = True
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(optimizer=optimizer))
    callback = zero2_comm.Zero2CommCallback(trainer, True, False)
    with pytest.raises(RuntimeError, match="cpu_offload"):
        callback.on_train_begin(None, None, SimpleNamespace())
    assert not any(name in optimizer.__dict__ for name in ("average_tensor", "setup_buckets"))


def test_callback_supports_transformers_lifecycle_events():
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.trainer_callback import CallbackHandler

    callback = zero2_comm.Zero2CommCallback(SimpleNamespace(), True, True)
    handler = CallbackHandler([callback], None, None, None, None)
    for event in vars(TrainerCallback):
        if event.startswith("on_") and event not in {"on_train_begin", "on_train_end"}:
            handler.call_event(event, SimpleNamespace(), TrainerState(), TrainerControl(), metrics={})


# ---------------------------------------------------------------- numerics on a real process group

def _numeric_worker(rank, world, port, mode):
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    deepspeed.init_distributed(dist_backend="gloo", dist_init_required=True)
    try:
        log = []
        optimizer = FakeOptimizer(log, dtype=torch.float32)
        optimizer.overlap_comm = False
        optimizer.flatten = torch._utils._flatten_dense_tensors
        optimizer.unflatten = torch._utils._unflatten_dense_tensors
        for name in ("allreduce_and_scatter", "allreduce_and_copy_with_multiple_ranks", "allreduce_bucket"):
            setattr(optimizer, name, MethodType(DeepSpeedZeroOptimizer.__dict__[name], optimizer))
        p0, p1 = FakeParam(0, torch.zeros(6)), FakeParam(1, torch.zeros(4))
        optimizer.bit16_groups = [[p0, p1]]
        optimizer.partition_size = [5]
        optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1]}}
        optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0}}}
        bucket = optimizer.ipg_buckets[torch.float32]
        bucket.params = [(0, 0, 0), (0, 1, 1)]
        bucket.elements = 10
        installation = None
        if mode == "native":
            installation = zero2_comm.Zero2CommInstallation(
                optimizer, overlap=False, reduce_scatter=True, accelerator=FakeAccelerator(log)
            )
        else:
            optimizer.average_tensor = MethodType(DeepSpeedZeroOptimizer.__dict__["average_tensor"], optimizer)
        base = torch.arange(1.0, 11.0)
        tensor = base * (rank + 1)
        optimizer.average_tensor(tensor, torch.float32)
        expected = base * (sum(range(1, world + 1)) / world)
        owned = slice(0, 3) if rank == 0 else slice(3, 10)
        assert torch.allclose(tensor[owned], expected[owned]), (rank, tensor.tolist())
        if installation is not None:
            assert installation.stats["reduce_scatter"] == 1, installation.stats
            installation.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["deepspeed_original", "native"])
def test_average_tensor_gives_owners_the_mean_on_a_real_process_group(mode):
    pytest.importorskip("deepspeed")
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_numeric_worker, args=(2, port, mode), nprocs=2)


def test_pack_timing_switch(monkeypatch):
    from gr00t.experiment.training_runtime import resolve_zero2_pack_timing

    assert resolve_zero2_pack_timing({}) is False
    assert resolve_zero2_pack_timing({"GR00T_ZERO2_PACK_TIMING": "1"}) is True
    with pytest.raises(ValueError):
        resolve_zero2_pack_timing({"GR00T_ZERO2_PACK_TIMING": "yes"})


@pytest.mark.parametrize("enabled", [False, True])
def test_average_tensor_forwards_the_pack_timing_flag(monkeypatch, enabled):
    log, recorded = [], []
    _patch_distributed(monkeypatch, recorded)
    optimizer, _ = make_installation(log, overlap=False, reduce_scatter=True, pack_timing=enabled)
    optimizer.bit16_groups = _layout_optimizer(log)[0].bit16_groups
    optimizer.param_to_partition_ids = {0: {0: [0, 1], 1: [1], 2: [1, 2]}}
    optimizer.grad_start_offset = {0: {0: {0: 0}, 1: {0: 3, 1: 0, 2: 0}, 2: {2: 2}}}
    optimizer.ipg_buckets[torch.bfloat16].params = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    optimizer.average_tensor(FakeTensor(log), torch.bfloat16)
    assert recorded[0][2]["time_packing"] is enabled


class _TimedEvent:
    def __init__(self, ms=1.5, error=None):
        self.ms, self.error = ms, error

    def synchronize(self):
        if self.error:
            raise self.error

    def elapsed_time(self, other):
        return other.ms


def test_read_pack_timing_waits_and_reports_the_failure_reason():
    ok = [(_TimedEvent(), _TimedEvent(1.5)), (_TimedEvent(), _TimedEvent(2.0))]
    assert zero2_comm._read_pack_timing(ok) == (3.5, None)
    assert zero2_comm._read_pack_timing([]) == (0.0, None)
    broken = [(_TimedEvent(), _TimedEvent(error=RuntimeError("device lost")))]
    assert zero2_comm._read_pack_timing(broken) == (None, "RuntimeError: device lost")


@pytest.mark.parametrize("enabled", [False, True])
def test_report_only_has_pack_ms_when_timing_is_enabled(monkeypatch, tmp_path, enabled):
    log = []
    _patch_callback_environment(monkeypatch, log)
    optimizer = FakeOptimizer(log)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(optimizer=optimizer))
    callback = zero2_comm.Zero2CommCallback(trainer, True, True, output_dir=tmp_path, pack_timing=enabled)
    callback.on_train_begin(None, None, SimpleNamespace())
    callback.installation.comm_stats["pack_events"] = [(_TimedEvent(), _TimedEvent(error=OSError("x")))]
    callback.close()
    comm = json.loads((tmp_path / "zero2_comm" / "runtime-rank-00.json").read_text())["comm"]
    assert comm["pack_timing"] is enabled and "pack_events" not in comm
    if enabled:
        assert comm["pack_ms"] is None and comm["pack_ms_error"] == "OSError: x"
    else:
        assert "pack_ms" not in comm and "pack_ms_error" not in comm
