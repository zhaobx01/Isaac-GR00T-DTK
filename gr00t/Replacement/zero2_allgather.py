"""Opt-in ZeRO-2 optimizer group preparation and AllGather scheduling."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from types import MethodType
from typing import Any, Callable

from transformers import TrainerCallback

from gr00t.experiment.trainer import Gr00tTrainer


_SUPPORTED_DEEPSPEED_VERSION = "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df"
_SUPPORTED_TORCH_VERSION = "2.10.0+das.opt1.dtk2604.2608131652.g28498c"
_SUPPORTED_DEEPSPEED_METHOD_HASHES = {
    "step": "80b29d8459a480b5adf9214f857ee435fb22f80f699912c0fbf2991119c4a49f",
    "_optimizer_step": "d7e21affbbf9b696584576042b21bc6147bc15e7ff0baa35b78140da4a484347",
    "_update_model_bit16_weights": "c12510aa0cb2a82ac5b9ae5c44b15ba260c4ef791dce9529249cd90df6a66607",
}


def validate_vendor_build(deepspeed_version: str, torch_version: str, method_hashes: dict[str, str]) -> bool:
    """Require the exact tested vendor builds before binding a copied optimizer step."""
    if (
        deepspeed_version != _SUPPORTED_DEEPSPEED_VERSION
        or torch_version != _SUPPORTED_TORCH_VERSION
        or method_hashes != _SUPPORTED_DEEPSPEED_METHOD_HASHES
    ):
        raise RuntimeError(
            "ZeRO-2 AllGather pipeline requires the pinned DeepSpeed/Torch build and method hashes"
        )
    return True


def validate_zero2_optimizer_attributes(
    *,
    stage: int,
    dtype: Any,
    cpu_offload: bool,
    partition_gradients: bool,
    reduce_scatter: bool,
    gradient_accumulation_steps: int,
    optimizer_type: type,
    optimizer_group_count: int,
    zero_group_count: int,
    has_all_gather_into_tensor: bool,
) -> bool:
    import torch

    unsupported = []
    if stage != 2:
        unsupported.append("ZeRO stage must be 2")
    if dtype is not torch.bfloat16:
        unsupported.append("parameter dtype must be BF16")
    if cpu_offload:
        unsupported.append("CPU offload is unsupported")
    if not partition_gradients or not reduce_scatter:
        unsupported.append("ZeRO-2 gradient partitioning and reduce-scatter must be enabled")
    if gradient_accumulation_steps != 1:
        unsupported.append("gradient accumulation steps must be 1")
    if optimizer_type is not torch.optim.AdamW:
        unsupported.append("optimizer must be torch.optim.AdamW")
    if optimizer_group_count != zero_group_count:
        unsupported.append("optimizer and ZeRO parameter group counts differ")
    if not has_all_gather_into_tensor:
        unsupported.append("backend must support all_gather_into_tensor")
    if unsupported:
        raise RuntimeError("Unsupported ZeRO-2 AllGather pipeline runtime: " + "; ".join(unsupported))
    return True


def _pipeline_zero2_step(self, closure=None):
    """Pinned ZeRO-2 step copy that submits each group AllGather after its update."""
    import torch
    from deepspeed import comm
    from deepspeed.runtime.utils import see_memory_usage
    from deepspeed.runtime.zero.stage_1_and_2 import (
        INITIAL_MICRO_STEP_ID,
        OPTIMIZER_ALLGATHER_TIMER,
        OPTIMIZER_GRADIENTS_TIMER,
        OPTIMIZER_STEP_TIMER,
        OPTIMIZER_TIMERS,
    )

    self.micro_step_id = INITIAL_MICRO_STEP_ID
    see_memory_usage("In step before checking overflow")
    if self.check_grad_overflow:
        self.check_overflow(partition_gradients=self.partition_gradients)

    prev_scale = self.loss_scale
    self._update_scale(self.overflow)
    if self.overflow:
        see_memory_usage("After overflow before clearing gradients")
        self.zero_grad(set_to_none=True)
        for key in self.averaged_gradients:
            self.averaged_gradients[key] = None
            self.all_grad_tensors[key] = None
        see_memory_usage("After overflow after clearing gradients")
        for timer in OPTIMIZER_TIMERS:
            self.timers(timer).start()
            self.timers(timer).stop()
        return

    see_memory_usage("Before norm calculation")
    scaled_global_grad_norm = self.scaled_global_norm()
    self._global_grad_norm = scaled_global_grad_norm / prev_scale
    see_memory_usage("After norm before optimizer")

    pipeline = GroupAllGatherPipeline(lambda index: launch_group_all_gather(self, index, comm))
    self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
    try:
        for index, group in enumerate(self.bit16_groups):
            with torch.profiler.record_function(f"zero2_allgather::optimizer_group_{index}"):
                self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
                partition_id = comm.get_rank(group=self.real_dp_process_group[index])
                self.free_grad_in_param_list(self.params_not_in_partition[index])
                if partition_id == comm.get_world_size(group=self.real_dp_process_group[index]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[index], int(self.partition_size[index])
                    ).to(self.single_partition_of_fp32_groups[index].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[index]).to(
                        self.single_partition_of_fp32_groups[index].dtype
                    )
                assert single_grad_partition.numel() == self.partition_size[index], (
                    "averaged gradients have different numbers of elements than partition size "
                    f"{single_grad_partition.numel()} {self.partition_size[index]} {index} {partition_id}"
                )

                self.single_partition_of_fp32_groups[index].grad = single_grad_partition
                self.free_grad_in_param_list(self.params_in_partition[index])
                self.averaged_gradients[index] = None
                self.all_grad_tensors[index] = None
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                self.timers(OPTIMIZER_STEP_TIMER).start()
                with torch.profiler.record_function(f"zero2_allgather::adamw_update_{index}"):
                    self._optimizer_step(index)
                self.single_partition_of_fp32_groups[index].grad = None
                del single_grad_partition

                bit16_partitions = self.parallel_partitioned_bit16_groups[index]
                fp32_partition = self.single_partition_of_fp32_groups[index]
                with torch.profiler.record_function(f"zero2_allgather::bf16_copy_{index}"):
                    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

                with torch.profiler.record_function(f"zero2_allgather::submit_{index}"):
                    pipeline.submit(index)

        see_memory_usage("After optimizer before all-gather completion")
        if self.cpu_offload:
            raise RuntimeError("CPU offload became active during the AllGather pipeline step")
        with torch.profiler.record_function("zero2_allgather::wait_all"):
            pipeline.wait_all()
    finally:
        try:
            if pipeline.in_flight_references:
                with torch.profiler.record_function("zero2_allgather::wait_after_error"):
                    pipeline.wait_all()
        finally:
            self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

    see_memory_usage("After optimizer before weight mapping")
    for index in range(len(self.bit16_groups)):
        with torch.profiler.record_function(f"zero2_allgather::weight_mapping_{index}"):
            self._update_model_bit16_weights(index)
    self.timers.log(OPTIMIZER_TIMERS)
    see_memory_usage("After zero_optimizer step")


def _current_vendor_versions_and_hashes() -> tuple[str, str, dict[str, str]]:
    from importlib.metadata import version

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    hashes = {
        name: hashlib.sha256(inspect.getsource(getattr(DeepSpeedZeroOptimizer, name)).encode()).hexdigest()
        for name in _SUPPORTED_DEEPSPEED_METHOD_HASHES
    }
    return version("deepspeed"), version("torch"), hashes


def describe_partition_layout(
    group_index: int, parameters: list[Any], flat: Any, partitions: list[Any], rank: int
) -> dict:
    """Validate and describe a ZeRO flat-buffer partition, including padding and aliasing."""
    if not partitions or not 0 <= rank < len(partitions):
        raise RuntimeError(f"Invalid AllGather partition rank for group {group_index}")
    partition_sizes = [part.numel() for part in partitions]
    if len(set(partition_sizes)) != 1:
        raise RuntimeError(f"AllGather partitions must have equal sizes for group {group_index}")
    partition_elements = partition_sizes[rank]
    if flat.numel() != len(partitions) * partition_elements:
        raise RuntimeError(f"AllGather output size mismatch in group {group_index}")
    if flat.dtype != partitions[rank].dtype or flat.device != partitions[rank].device:
        raise RuntimeError(f"AllGather dtype/device mismatch for group {group_index}")
    offset = sum(partition_sizes[:rank])
    expected = flat.narrow(0, offset, partition_elements)
    if partitions[rank].data_ptr() != expected.data_ptr():
        raise RuntimeError(f"AllGather input partition does not alias output in group {group_index}")
    parameter_elements = sum(parameter.numel() for parameter in parameters)
    if parameter_elements > flat.numel():
        raise RuntimeError(f"AllGather flat buffer is smaller than parameters in group {group_index}")
    return {
        "group_index": group_index,
        "parameter_elements": parameter_elements,
        "flat_elements": flat.numel(),
        "padding_elements": flat.numel() - parameter_elements,
        "partition_elements": partition_sizes,
        "local_partition_offset": offset,
        "local_partition_elements": partition_elements,
    }


def _validate_engine(engine: Any, mode: str, group_plan: list[dict]) -> dict:
    from deepspeed import comm
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    zero_optimizer = engine.optimizer
    config = engine._config._param_dict
    stage = config.get("zero_optimization", {}).get("stage")
    inner_optimizer = zero_optimizer.optimizer
    zero_groups = zero_optimizer.bit16_groups
    optimizer_groups = inner_optimizer.param_groups
    optimizer_options = json.loads(json.dumps(
        [{key: value for key, value in group.items() if key != "params"}
         for group in optimizer_groups],
        allow_nan=False,
    ))
    attributes = {
        "stage": stage,
        "dtype": zero_optimizer.dtype,
        "cpu_offload": zero_optimizer.cpu_offload,
        "partition_gradients": zero_optimizer.partition_gradients,
        "reduce_scatter": zero_optimizer.reduce_scatter,
        "gradient_accumulation_steps": zero_optimizer.gradient_accumulation_steps,
        "optimizer_type": type(inner_optimizer),
        "optimizer_group_count": len(optimizer_groups),
        "zero_group_count": len(zero_groups),
        "has_all_gather_into_tensor": comm.has_all_gather_into_tensor(),
    }
    validate_zero2_optimizer_attributes(**attributes)
    if not isinstance(zero_optimizer, DeepSpeedZeroOptimizer):
        raise RuntimeError(f"Unsupported ZeRO optimizer type: {type(zero_optimizer)}")
    if zero_optimizer.torch_autocast_gradscaler or zero_optimizer.zenflow:
        raise RuntimeError("Unsupported ZeRO-2 AllGather pipeline scaler or ZenFlow mode")
    if len(group_plan) != len(zero_groups):
        raise RuntimeError("Trainer optimizer group plan does not match DeepSpeed parameter groups")
    actual_elements = [sum(parameter.numel() for parameter in group) for group in zero_groups]
    planned_elements = [entry["elements"] for entry in group_plan]
    if actual_elements != planned_elements:
        raise RuntimeError("Trainer optimizer group elements do not match DeepSpeed parameter groups")

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"This prototype requires 8 data-parallel ranks, got {world_size}")
    partition_layout = []
    for index in range(len(zero_groups)):
        process_group = zero_optimizer.real_dp_process_group[index]
        group_rank = comm.get_rank(group=process_group)
        group_world_size = comm.get_world_size(group=process_group)
        if group_world_size != world_size:
            raise RuntimeError(f"AllGather group {index} does not span all 8 ranks")
        flat = zero_optimizer.bit16_groups_flat[index]
        parts = zero_optimizer.parallel_partitioned_bit16_groups[index]
        if len(parts) != group_world_size:
            raise RuntimeError(f"Invalid AllGather partition count in group {index}")
        partition_layout.append(
            describe_partition_layout(index, zero_groups[index], flat, parts, group_rank)
        )

    version_record = None
    if mode == "pipeline":
        versions = _current_vendor_versions_and_hashes()
        validate_vendor_build(*versions)
        version_record = {"deepspeed": versions[0], "torch": versions[1], "method_hashes": versions[2]}
    return {
        "rank": rank,
        "mode": mode,
        "stage": stage,
        "dtype": str(zero_optimizer.dtype),
        "gradient_accumulation_steps": zero_optimizer.gradient_accumulation_steps,
        "group_elements": actual_elements,
        "group_plan": group_plan,
        "optimizer_options": optimizer_options,
        "partition_layout": partition_layout,
        "vendor": version_record,
        "effective_config": config,
    }


def validate_all_ranks(engine: Any, mode: str, group_plan: list[dict]) -> list[dict]:
    """Have every rank validate locally, then agree before installing any patch."""
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Distributed process group must be initialized before AllGather validation")
    local = None
    try:
        local = {"ok": True, "record": _validate_engine(engine, mode, group_plan)}
    except BaseException as exc:
        local = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    records = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(records, local)
    errors = [f"rank {index}: {record['error']}" for index, record in enumerate(records) if not record["ok"]]
    if errors:
        raise RuntimeError("AllGather runtime validation failed on all ranks: " + " | ".join(errors))
    signatures = [
        (
            record["record"]["mode"],
            record["record"]["stage"],
            record["record"]["dtype"],
            record["record"]["gradient_accumulation_steps"],
            record["record"]["group_elements"],
            tuple(
                (
                    group["parameter_elements"],
                    group["flat_elements"],
                    group["padding_elements"],
                    tuple(group["partition_elements"]),
                )
                for group in record["record"]["partition_layout"]
            ),
            record["record"]["vendor"],
            json.dumps(record["record"]["group_plan"], sort_keys=True, allow_nan=False),
            json.dumps(record["record"]["optimizer_options"], sort_keys=True, allow_nan=False),
            # Accelerate intentionally sets steps_per_print=inf to silence logging.
            # Compare config structure directly; tensor/optimizer finite checks are separate.
            record["record"]["effective_config"],
        )
        for record in records
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("All ranks disagree on the ZeRO-2 AllGather pipeline configuration")
    return [record["record"] for record in records]


class AllGatherStepInstallation:
    """Install a step on one ZeRO optimizer instance and restore it on close."""

    def __init__(self, optimizer: Any):
        self.restore = bind_instance_step(optimizer, _pipeline_zero2_step)
        self.optimizer = optimizer

    def close(self) -> None:
        if self.restore is not None:
            restore, self.restore = self.restore, None
            restore()


class Zero2AllGatherCallback(TrainerCallback):
    """Validate rank agreement, install the optional adapter, and restore it at train end."""

    def __init__(self, trainer: Any, mode: str, output_dir: Any):
        if mode != "pipeline":
            raise ValueError(f"Invalid AllGather callback mode: {mode}")
        self.trainer = trainer
        self.mode = mode
        self.output_dir = output_dir
        self.installation = None
        self.started = False

    def on_train_begin(self, args, state, control, **kwargs):
        import json
        from pathlib import Path

        import torch

        if self.started:
            raise RuntimeError("ZeRO-2 AllGather callback cannot be started twice")
        self.started = True
        engine = self.trainer.model_wrapped
        records = validate_all_ranks(engine, self.mode, self.trainer.zero2_allgather_group_plan)
        write_status = {"ok": True}
        if torch.distributed.get_rank() == 0:
            path = Path(self.output_dir) / "zero2_allgather"
            try:
                path.mkdir(parents=True, exist_ok=True)
                (path / "runtime.json").write_text(
                    json.dumps(
                        {
                            "mode": self.mode,
                            "group_elements": self.trainer.zero2_allgather_group_elements,
                            "group_plan": self.trainer.zero2_allgather_group_plan,
                            "ranks": records,
                        },
                        indent=2,
                        default=str,
                    )
                    + "\n"
                )
            except BaseException as exc:
                write_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        write_statuses = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(write_statuses, write_status)
        write_errors = [
            f"rank {index}: {status['error']}"
            for index, status in enumerate(write_statuses)
            if not status["ok"]
        ]
        if write_errors:
            raise RuntimeError("AllGather runtime manifest could not be saved: " + " | ".join(write_errors))
        if self.mode == "pipeline":
            install_status = {"ok": True}
            try:
                self.installation = AllGatherStepInstallation(engine.optimizer)
            except BaseException as exc:
                install_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            install_statuses = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(install_statuses, install_status)
            install_errors = [
                f"rank {index}: {status['error']}"
                for index, status in enumerate(install_statuses)
                if not status["ok"]
            ]
            if install_errors:
                self.close()
                raise RuntimeError("AllGather step installation failed: " + " | ".join(install_errors))

    def on_train_end(self, args, state, control, **kwargs):
        self.close()

    def close(self) -> None:
        if self.installation is not None:
            installation, self.installation = self.installation, None
            installation.close()


def regroup_optimizer_parameter_groups(parameter_groups: list[dict], target_elements: int) -> list[dict]:
    """Split optimizer groups by element count without splitting Parameters.

    Source group boundaries are preserved, so parameters with different
    optimizer hyperparameters can never be combined.
    """
    if target_elements <= 0:
        raise ValueError("target_elements must be positive")

    result = []
    for source_index, source in enumerate(parameter_groups):
        parameters = list(source["params"])
        if not parameters:
            raise ValueError(f"Cannot regroup empty optimizer parameter group {source_index}")

        current = []
        current_elements = 0

        def flush():
            nonlocal current, current_elements
            if current:
                group = copy(source)
                group["params"] = current
                result.append(group)
                current = []
                current_elements = 0

        for parameter in parameters:
            parameter_elements = parameter.numel()
            if current and current_elements + parameter_elements > target_elements:
                flush()
            current.append(parameter)
            current_elements += parameter_elements
            if parameter_elements >= target_elements:
                flush()
        flush()
    return result


def regroup_torch_optimizer(optimizer: Any, target_elements: int, parameter_names: dict[int, str]) -> list[dict]:
    """Regroup an unstepped Torch AdamW while preserving source-group options."""
    import torch

    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError(f"Expected torch.optim.AdamW before DeepSpeed initialization, got {type(optimizer)}")
    if optimizer.state:
        raise ValueError("Optimizer groups can only be changed before the first optimizer step")

    source_groups = list(optimizer.param_groups)
    regrouped = regroup_optimizer_parameter_groups(source_groups, target_elements)
    manifest = []
    source_index = 0
    sub_index = 0
    for group_index, group in enumerate(regrouped):
        while source_index < len(source_groups) and not any(
            parameter is candidate
            for parameter in group["params"]
            for candidate in source_groups[source_index]["params"]
        ):
            source_index += 1
            sub_index = 0
        if source_index >= len(source_groups):
            raise RuntimeError("Regrouped parameter no longer maps to an original optimizer group")
        names = []
        for parameter in group["params"]:
            name = parameter_names.get(id(parameter))
            if name is None:
                raise ValueError("Every trainable parameter must have a stable model name")
            names.append(name)
        manifest.append(
            {
                "group_index": group_index,
                "source_group_index": source_index,
                "subgroup_index": sub_index,
                "elements": sum(parameter.numel() for parameter in group["params"]),
                "parameter_names": names,
            }
        )
        sub_index += 1

    optimizer.param_groups = regrouped
    return manifest


@dataclass(frozen=True)
class PendingAllGather:
    work: Any
    output: Any
    input: Any
    process_group: Any


def launch_group_all_gather(zero_optimizer: Any, group_index: int, comm: Any) -> PendingAllGather | None:
    """Submit one ZeRO parameter group's in-place-alias all-gather asynchronously."""
    process_group = zero_optimizer.real_dp_process_group[group_index]
    world_size = comm.get_world_size(group=process_group)
    if world_size == 1:
        return None

    rank = comm.get_rank(group=process_group)
    output = zero_optimizer.bit16_groups_flat[group_index]
    partitions = zero_optimizer.parallel_partitioned_bit16_groups[group_index]
    if not 0 <= rank < len(partitions) or len(partitions) != world_size:
        raise RuntimeError(f"Invalid ZeRO partition layout for group {group_index}")
    input_tensor = partitions[rank]
    offset = sum(part.numel() for part in partitions[:rank])
    expected_input = output.narrow(0, offset, input_tensor.numel())
    if output.dtype != input_tensor.dtype or output.device != input_tensor.device:
        raise RuntimeError(f"AllGather dtype/device mismatch for group {group_index}")
    if output.numel() != world_size * input_tensor.numel():
        raise RuntimeError(f"AllGather tensor sizes mismatch for group {group_index}")
    if input_tensor.data_ptr() != expected_input.data_ptr():
        raise RuntimeError(f"AllGather input partition must alias its output flat buffer (group {group_index})")

    work = comm.all_gather_into_tensor(
        output_tensor=output,
        tensor=input_tensor,
        group=process_group,
        async_op=True,
    )
    if work is None:
        raise RuntimeError(f"Asynchronous AllGather returned no Work for group {group_index}")
    return PendingAllGather(work, output, input_tensor, process_group)


class GroupAllGatherPipeline:
    """Retain asynchronous gather requests and their tensors until completion."""

    def __init__(self, submit: Callable[[int], Any]):
        self._submit = submit
        self._works = []
        self._references = []

    @property
    def in_flight_references(self):
        return list(self._references)

    def submit(self, group_index: int, tensor_refs: tuple[Any, ...] = ()) -> None:
        result = self._submit(group_index)
        if isinstance(result, PendingAllGather):
            work = result.work
            tensor_refs = (result.output, result.input, result.process_group)
        else:
            work = result
        if work is not None:
            self._works.append(work)
            self._references.append(tensor_refs)

    def wait_all(self) -> None:
        first_error = None
        pending_works = []
        pending_references = []
        for work, references in zip(self._works, self._references):
            try:
                work.wait()
            except BaseException as exc:  # retain storage until failed work is observed again
                pending_works.append(work)
                pending_references.append(references)
                if first_error is None:
                    first_error = exc
        self._works = pending_works
        self._references = pending_references
        if first_error is not None:
            raise first_error


def bind_instance_step(optimizer: Any, step_implementation: Callable) -> Callable[[], None]:
    """Bind an experimental step to one optimizer instance and return cleanup."""
    missing = object()
    previous = optimizer.__dict__.get("step", missing)
    optimizer.step = MethodType(step_implementation, optimizer)
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        if previous is missing:
            del optimizer.__dict__["step"]
        else:
            optimizer.step = previous

    return restore


def validate_training_config(config) -> None:
    """Reject unsupported training paths before allocating model/device resources."""
    t = config.training
    if t.deepspeed_stage != 2 or t.use_ddp or t.num_gpus != 8:
        raise ValueError("AllGather pipeline requires 8-GPU DeepSpeed ZeRO-2")
    if not t.bf16 or t.fp16 or t.optim != "adamw_torch":
        raise ValueError("AllGather pipeline requires BF16 and adamw_torch")
    if t.gradient_accumulation_steps != 1:
        raise ValueError("AllGather pipeline requires gradient accumulation steps=1")
    if t.resume_from_checkpoint:
        raise ValueError("AllGather pipeline does not support optimizer checkpoint resume")
    if t.save_strategy != "no":
        raise ValueError("AllGather pipeline requires save_strategy='no'")


class PipelineTrainer(Gr00tTrainer):
    """Keep optimizer preparation and callback installation outside the base Trainer."""

    def __init__(self, *args, **kwargs):
        self.zero2_allgather_group_elements = 100_000_000
        self.zero2_allgather_group_plan = []
        self._pipeline_groups_prepared = False
        super().__init__(*args, **kwargs)
        self._pipeline_callback = Zero2AllGatherCallback(self, "pipeline", self.args.output_dir)
        self.add_callback(self._pipeline_callback)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if not self._pipeline_groups_prepared:
            if optimizer is None:
                raise RuntimeError("Trainer did not create an optimizer for AllGather pipeline")
            names = {id(parameter): name for name, parameter in self.model.named_parameters()}
            self.zero2_allgather_group_plan = regroup_torch_optimizer(
                optimizer, self.zero2_allgather_group_elements, names
            )
            self._pipeline_groups_prepared = True
        return optimizer

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self.close_pipeline()

    def close_pipeline(self):
        self._pipeline_callback.close()
