from gr00t.Replacement.zero2_allgather import (
    GroupAllGatherPipeline,
    bind_instance_step,
    launch_group_all_gather,
    regroup_optimizer_parameter_groups,
    regroup_torch_optimizer,
    describe_partition_layout,
    validate_zero2_optimizer_attributes,
    validate_vendor_build,
)






def test_allgather_callback_supports_transformers_lifecycle_events():
    from types import SimpleNamespace

    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.trainer_callback import CallbackHandler

    from gr00t.Replacement.zero2_allgather import Zero2AllGatherCallback

    callback = Zero2AllGatherCallback(SimpleNamespace(), "pipeline", "unused")
    handler = CallbackHandler([callback], None, None, None, None)
    for event in vars(TrainerCallback):
        if event.startswith("on_") and event not in {"on_train_begin", "on_train_end"}:
            handler.call_event(
                event, SimpleNamespace(), TrainerState(), TrainerControl(), metrics={}
            )


def test_allgather_rank_agreement_includes_parameter_plan_and_optimizer_options(monkeypatch):
    from copy import deepcopy

    import pytest
    import torch
    from accelerate.utils import DeepSpeedPlugin

    from gr00t.Replacement import zero2_allgather

    # Use Accelerate's real config preprocessing, including its infinite
    # steps_per_print sentinel, rather than a hand-written finite-only fixture.
    plugin = DeepSpeedPlugin(hf_ds_config={"zero_optimization": {"stage": 2}})
    assert plugin.deepspeed_config["steps_per_print"] == float("inf")
    record = {
        "rank": 0, "mode": "pipeline", "stage": 2, "dtype": "torch.bfloat16",
        "gradient_accumulation_steps": 1, "group_elements": [4], "vendor": None,
        "partition_layout": [{"parameter_elements": 4, "flat_elements": 8,
                              "padding_elements": 4, "partition_elements": [1] * 8}],
        "group_plan": [{"source_group_index": 0, "parameter_names": ["a", "b"]}],
        "optimizer_options": [{"lr": 1e-4, "weight_decay": 0.01}],
        "effective_config": plugin.deepspeed_config,
    }
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(zero2_allgather, "_validate_engine", lambda *args: deepcopy(record))
    remote = deepcopy(record)

    def gather(records, local):
        records[:] = [local, {"ok": True, "record": deepcopy(remote)}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert len(zero2_allgather.validate_all_ranks(None, "pipeline", [])) == 2
    for field, value in (
        ("group_plan", [{"source_group_index": 0, "parameter_names": ["b", "a"]}]),
        ("group_plan", [{"source_group_index": 1, "parameter_names": ["a", "b"]}]),
        ("optimizer_options", [{"lr": 2e-4, "weight_decay": 0.01}]),
        ("effective_config", {"zero_optimization": {"stage": 2, "overlap_comm": False}}),
    ):
        remote = {**deepcopy(record), field: value}
        with pytest.raises(RuntimeError, match="disagree"):
            zero2_allgather.validate_all_ranks(None, "pipeline", [])


def test_pipeline_overflow_clears_gradients_without_updating_or_gathering():
    from types import SimpleNamespace

    from gr00t.Replacement.zero2_allgather import _pipeline_zero2_step

    events = []
    timer = SimpleNamespace(start=lambda: events.append("timer_start"),
                            stop=lambda: events.append("timer_stop"))
    optimizer = SimpleNamespace(
        micro_step_id=9, check_grad_overflow=True, partition_gradients=True,
        loss_scale=2.0, overflow=True,
        check_overflow=lambda **kwargs: events.append(("check_overflow", kwargs)),
        _update_scale=lambda overflow: events.append(("update_scale", overflow)),
        zero_grad=lambda **kwargs: events.append(("zero_grad", kwargs)),
        averaged_gradients={0: object(), 1: object()},
        all_grad_tensors={0: object(), 1: object()},
        timers=lambda _: timer,
    )

    _pipeline_zero2_step(optimizer)

    assert optimizer.micro_step_id != 9
    assert optimizer.averaged_gradients == {0: None, 1: None}
    assert optimizer.all_grad_tensors == {0: None, 1: None}
    assert events[:3] == [
        ("check_overflow", {"partition_gradients": True}),
        ("update_scale", True),
        ("zero_grad", {"set_to_none": True}),
    ]
    assert events.count("timer_start") == events.count("timer_stop")








def test_regroup_keeps_parameters_whole_ordered_and_hyperparameters():
    import torch

    first = [torch.nn.Parameter(torch.zeros(size)) for size in (6, 5, 2, 12)]
    second = [torch.nn.Parameter(torch.zeros(size)) for size in (4, 4)]
    groups = [
        {"params": first, "weight_decay": 0.1, "lr": 0.01, "group_name": "decay"},
        {"params": second, "weight_decay": 0.0, "lr": 0.01, "group_name": "no_decay"},
    ]

    regrouped = regroup_optimizer_parameter_groups(groups, target_elements=10)

    assert [len(group["params"]) for group in regrouped] == [1, 2, 1, 2]
    assert [param for group in regrouped for param in group["params"]] == first + second
    assert [group["group_name"] for group in regrouped] == ["decay"] * 3 + ["no_decay"]
    assert [group["weight_decay"] for group in regrouped] == [0.1] * 3 + [0.0]
    assert sum(param.numel() for group in groups for param in group["params"]) == sum(
        param.numel() for group in regrouped for param in group["params"]
    )
    assert len(groups) == 2


def test_regroup_rejects_empty_parameter_groups():
    import torch

    groups = [{"params": [torch.nn.Parameter(torch.ones(1))]}, {"params": []}]

    try:
        regroup_optimizer_parameter_groups(groups, target_elements=10)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty source parameter groups must be rejected")


def test_pipeline_waits_for_all_gathers_after_all_groups_are_submitted():
    events = []

    class Work:
        def __init__(self, group_index):
            self.group_index = group_index

        def wait(self):
            events.append(("wait", self.group_index))

    pipeline = GroupAllGatherPipeline(lambda group_index: (events.append(("submit", group_index)) or Work(group_index)))
    for group_index in range(3):
        events.append(("update", group_index))
        events.append(("copy", group_index))
        pipeline.submit(group_index)

    pipeline.wait_all()
    events.append(("map_weights", None))

    assert events == [
        ("update", 0),
        ("copy", 0),
        ("submit", 0),
        ("update", 1),
        ("copy", 1),
        ("submit", 1),
        ("update", 2),
        ("copy", 2),
        ("submit", 2),
        ("wait", 0),
        ("wait", 1),
        ("wait", 2),
        ("map_weights", None),
    ]


def test_pipeline_retains_tensors_until_work_completes():
    released = []

    class Work:
        def wait(self):
            released.append("wait")

    pipeline = GroupAllGatherPipeline(lambda _: (released.append("submit") or Work()))
    tensor_refs = (object(), object())
    pipeline.submit(0, tensor_refs=tensor_refs)

    assert pipeline.in_flight_references == [tensor_refs]
    pipeline.wait_all()
    assert released == ["submit", "wait"]
    assert pipeline.in_flight_references == []


def test_pipeline_keeps_failed_work_references_for_error_cleanup():
    class Work:
        def wait(self):
            raise RuntimeError("device communication failed")

    pipeline = GroupAllGatherPipeline(lambda _: Work())
    refs = (object(), object())
    pipeline.submit(0, tensor_refs=refs)

    try:
        pipeline.wait_all()
    except RuntimeError as exc:
        assert "communication failed" in str(exc)
    else:
        raise AssertionError("failed work must propagate its error")

    assert pipeline.in_flight_references == [refs]


def test_instance_step_binding_is_reversible_and_idempotent():
    class Optimizer:
        def step(self):
            return "vendor"

    optimizer = Optimizer()

    def experiment_step(self):
        return "experiment"

    restore = bind_instance_step(optimizer, experiment_step)
    assert optimizer.step() == "experiment"

    restore()
    restore()
    assert optimizer.step() == "vendor"
    assert "step" not in optimizer.__dict__


def test_group_gather_uses_rank_partition_alias_and_async_backend():
    import torch

    flat = torch.arange(8, dtype=torch.bfloat16)
    partitions = [flat.narrow(0, 0, 4), flat.narrow(0, 4, 4)]
    calls = []
    expected_work = object()

    class Comm:
        @staticmethod
        def get_world_size(group):
            assert group == "dp"
            return 2

        @staticmethod
        def get_rank(group):
            assert group == "dp"
            return 1

        @staticmethod
        def all_gather_into_tensor(**kwargs):
            calls.append(kwargs)
            return expected_work

    zero = type(
        "Zero",
        (),
        {
            "real_dp_process_group": ["dp"],
            "bit16_groups_flat": [flat],
            "parallel_partitioned_bit16_groups": [partitions],
        },
    )()

    request = launch_group_all_gather(zero, 0, Comm)

    assert request.work is expected_work
    assert request.output is flat
    assert request.input is partitions[1]
    assert request.input.data_ptr() == flat.narrow(0, 4, 4).data_ptr()
    assert calls == [{"output_tensor": flat, "tensor": partitions[1], "group": "dp", "async_op": True}]


def test_group_gather_rejects_nonaliased_input_partition():
    import torch

    class Comm:
        @staticmethod
        def get_world_size(group):
            return 2

        @staticmethod
        def get_rank(group):
            return 0

    zero = type(
        "Zero",
        (),
        {
            "real_dp_process_group": [None],
            "bit16_groups_flat": [torch.zeros(8, dtype=torch.bfloat16)],
            "parallel_partitioned_bit16_groups": [
                [torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.bfloat16)]
            ],
        },
    )()

    try:
        launch_group_all_gather(zero, 0, Comm)
    except RuntimeError as exc:
        assert "alias" in str(exc)
    else:
        raise AssertionError("nonaliased gather storage must fail closed")


def test_partition_layout_records_offsets_padding_and_checks_equal_parts():
    import torch

    flat = torch.zeros(12, dtype=torch.bfloat16)
    parts = [flat.narrow(0, 0, 4), flat.narrow(0, 4, 4), flat.narrow(0, 8, 4)]
    params = [torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(6))]

    layout = describe_partition_layout(2, params, flat, parts, rank=1)

    assert layout == {
        "group_index": 2,
        "parameter_elements": 11,
        "flat_elements": 12,
        "padding_elements": 1,
        "partition_elements": [4, 4, 4],
        "local_partition_offset": 4,
        "local_partition_elements": 4,
    }

    unequal = [flat.narrow(0, 0, 3), flat.narrow(0, 3, 5), flat.narrow(0, 8, 4)]
    try:
        describe_partition_layout(2, params, flat, unequal, rank=1)
    except RuntimeError as exc:
        assert "equal" in str(exc)
    else:
        raise AssertionError("AllGather must reject unequal partition sizes")


def test_torch_optimizer_regroup_preserves_decay_groups_names_and_parameter_order():
    import torch

    decay = [torch.nn.Parameter(torch.ones(6)), torch.nn.Parameter(torch.ones(3))]
    no_decay = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(2))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1, "lr": 1e-4},
            {"params": no_decay, "weight_decay": 0.0, "lr": 1e-4},
        ],
        eps=1e-8,
    )
    names = {id(parameter): f"p{index}" for index, parameter in enumerate(decay + no_decay)}

    manifest = regroup_torch_optimizer(optimizer, target_elements=8, parameter_names=names)

    assert [parameter for group in optimizer.param_groups for parameter in group["params"]] == decay + no_decay
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.1, 0.0]
    assert [entry["parameter_names"] for entry in manifest] == [["p0"], ["p1"], ["p2", "p3"]]
    assert all(group["eps"] == 1e-8 and group["lr"] == 1e-4 for group in optimizer.param_groups)


def test_torch_optimizer_regroup_rejects_existing_state():
    import torch

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter]["step"] = torch.tensor(1.0)

    try:
        regroup_torch_optimizer(optimizer, target_elements=1, parameter_names={id(parameter): "p"})
    except ValueError as exc:
        assert "before the first optimizer step" in str(exc)
    else:
        raise AssertionError("optimizer groups cannot change after state creation")


def test_pipeline_adapter_rejects_unpinned_vendor_build_and_source():
    expected = {
        "step": "80b29d8459a480b5adf9214f857ee435fb22f80f699912c0fbf2991119c4a49f",
        "_optimizer_step": "d7e21affbbf9b696584576042b21bc6147bc15e7ff0baa35b78140da4a484347",
        "_update_model_bit16_weights": "c12510aa0cb2a82ac5b9ae5c44b15ba260c4ef791dce9529249cd90df6a66607",
    }

    assert validate_vendor_build(
        "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df",
        "2.10.0+das.opt1.dtk2604.2608131652.g28498c",
        expected,
    )

    for version, torch_version, hashes in [
        ("0.18.4", "2.10.0+das.opt1.dtk2604.2608131652.g28498c", expected),
        ("0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df", "2.10.0", expected),
        (
            "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df",
            "2.10.0+das.opt1.dtk2604.2608131652.g28498c",
            {**expected, "step": "different"},
        ),
    ]:
        try:
            validate_vendor_build(version, torch_version, hashes)
        except RuntimeError as exc:
            assert "pinned" in str(exc)
        else:
            raise AssertionError("pipeline adapter must reject changed vendor implementations")


def test_pipeline_runtime_guard_accepts_only_the_tested_zero2_shape():
    import torch

    good = {
        "stage": 2,
        "dtype": torch.bfloat16,
        "cpu_offload": False,
        "partition_gradients": True,
        "reduce_scatter": True,
        "gradient_accumulation_steps": 1,
        "optimizer_type": torch.optim.AdamW,
        "optimizer_group_count": 3,
        "zero_group_count": 3,
        "has_all_gather_into_tensor": True,
    }
    assert validate_zero2_optimizer_attributes(**good)

    for field, value in [
        ("stage", 3),
        ("dtype", torch.float16),
        ("cpu_offload", True),
        ("partition_gradients", False),
        ("reduce_scatter", False),
        ("gradient_accumulation_steps", 2),
        ("optimizer_type", torch.optim.SGD),
        ("zero_group_count", 2),
        ("has_all_gather_into_tensor", False),
    ]:
        candidate = {**good, field: value}
        try:
            validate_zero2_optimizer_attributes(**candidate)
        except RuntimeError as exc:
            assert "unsupported" in str(exc).lower()
        else:
            raise AssertionError(f"unsupported runtime field was accepted: {field}")
