"""Opt-in ZeRO-2 gradient reduction scheduling for DeepSpeed 0.18.4 (DTK build).

Two independent switches, installed on a live ``DeepSpeedZeroOptimizer`` instance:

* overlap: DeepSpeed makes the compute stream wait for every bucket reduction. Here the compute
  stream waits only on a per-buffer event when a half of the double buffer is about to be reused.
* reduce_scatter: the per-owner slices of a bucket are summed with ReduceScatter or a directed
  Reduce instead of an AllReduce of the whole bucket.

The replaced methods mirror the pinned vendor source, so installation is refused on any other build.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from gr00t.Replacement.zero2_overlap import BucketEventScheduler
from gr00t.Replacement.zero2_reduce_scatter import (
    clear_staging_buffers,
    preallocate_staging,
    reduce_by_owner,
)


_SUPPORTED_DEEPSPEED_VERSION = "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df"
_SUPPORTED_TORCH_VERSION = "2.10.0+das.opt1.dtk2604.2608131652.g28498c"
_SUPPORTED_METHOD_HASHES = {
    "reduce_independent_p_g_buckets_and_remove_grads": "5f8288dd62441a9a806f7c490254bb4e59da08e784308887a6964873a9443bea",
    "average_tensor": "ede6b0fed00157f5b2ce4b8da088cf4ab7e171664ba81481f36be0cfa1b2c6cd",
    "reduce_ipg_grads": "c33644762e4656d0eb6ea8f4d3b85d9ec6cbb378f5b6b295e740e5be17f6e971",
    "setup_buckets": "f844a862e97d613d7dc22df0ce8f0cc17545b88e1b3e83f0ff69cc67fc66b971",
    "allreduce_and_scatter": "b2a54dd6ad1d31370b6af8fad0d6a4c11a9c7af15f1d0f6fcbefb3efed9af8c4",
}
_EXPECTED_SIGNATURES = {
    "reduce_independent_p_g_buckets_and_remove_grads": ["self", "param", "i"],
    "average_tensor": ["self", "tensor", "communication_data_type"],
    "reduce_ipg_grads": ["self"],
    "setup_buckets": ["self"],
    "allreduce_and_scatter": [
        "self", "bucket", "communication_data_type", "numel_per_bucket", "log", "divide", "process_group",
    ],
}


def validate_vendor_build(deepspeed_version: str, torch_version: str, method_hashes: dict[str, str]) -> bool:
    if (
        deepspeed_version != _SUPPORTED_DEEPSPEED_VERSION
        or torch_version != _SUPPORTED_TORCH_VERSION
        or method_hashes != _SUPPORTED_METHOD_HASHES
    ):
        raise RuntimeError("ZeRO-2 communication switches require the pinned DeepSpeed/Torch build and method hashes")
    return True


def current_vendor_versions_and_hashes() -> tuple[str, str, dict[str, str]]:
    from importlib.metadata import version

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    hashes = {
        name: hashlib.sha256(inspect.getsource(getattr(DeepSpeedZeroOptimizer, name)).encode()).hexdigest()
        for name in _SUPPORTED_METHOD_HASHES
    }
    return version("deepspeed"), version("torch"), hashes


def validate_training_config(config, overlap: bool, reduce_scatter: bool) -> None:
    """Reject unsupported training setups before any model or device resources are allocated."""
    if not (overlap or reduce_scatter):
        return
    t = config.training
    if t.deepspeed_stage != 2 or t.use_ddp or t.num_gpus < 2:
        raise ValueError("ZeRO-2 communication switches require multi-GPU DeepSpeed ZeRO-2")
    if not t.bf16 or t.fp16:
        raise ValueError("ZeRO-2 communication switches require BF16")
    if t.gradient_accumulation_steps != 1:
        raise ValueError("ZeRO-2 communication switches require gradient accumulation steps=1")


def get_accelerator():
    from deepspeed.accelerator import get_accelerator as _get_accelerator

    return _get_accelerator()


def validate_optimizer(optimizer: Any, accelerator: Any = None, require_deepspeed_type: bool = True) -> None:
    unsupported = []
    cls = type(optimizer)
    if require_deepspeed_type and (
        cls.__module__ != "deepspeed.runtime.zero.stage_1_and_2" or cls.__name__ != "DeepSpeedZeroOptimizer"
    ):
        unsupported.append("optimizer must be DeepSpeedZeroOptimizer (ZeRO stage 1/2)")
    required_true = (
        "partition_gradients", "overlap_comm", "contiguous_gradients", "reduce_scatter",
        "use_multi_rank_bucket_allreduce",
    )
    for name in required_true:
        if not getattr(optimizer, name, False):
            unsupported.append(f"{name} must be enabled")
    for name in ("cpu_offload", "zenflow"):
        if getattr(optimizer, name, False):
            unsupported.append(f"{name} is unsupported")
    if getattr(optimizer, "sequence_parallel_size", 1) != 1:
        unsupported.append("sequence parallelism is unsupported")
    if getattr(optimizer, "gradient_predivide_factor", 1.0) != 1.0:
        unsupported.append("gradient_predivide_factor must be 1.0")
    if len(getattr(optimizer, "ipg_buckets", {})) != 1:
        unsupported.append("exactly one gradient communication dtype is supported")
    accelerator = accelerator or get_accelerator()
    if accelerator.resolves_data_dependency() or getattr(optimizer, "reduction_stream", None) is None:
        unsupported.append("a device with an asynchronous reduction stream is required")
    for name, expected in _EXPECTED_SIGNATURES.items():
        method = getattr(cls, name, None)
        if method is None or list(inspect.signature(method).parameters) != expected:
            unsupported.append(f"{name} signature differs from the supported DeepSpeed build")
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size(group=optimizer.dp_process_group) != dist.get_world_size():
            unsupported.append("data-parallel group must span all ranks")
    if unsupported:
        raise RuntimeError("Unsupported ZeRO-2 communication runtime: " + "; ".join(unsupported))


def build_rank_and_offsets(optimizer: Any, bucket: Any) -> tuple[list[tuple[int, int, int]], int]:
    """Map a bucket to (owner, offset, numel) slices, merging adjacent runs with the same owner."""
    result: list[tuple[int, int, int]] = []
    current = 0
    previous_owner = None
    for group_index, param_index, param_id in bucket.params:
        param = optimizer.bit16_groups[group_index][param_index]
        partitions = sorted(
            ((pid, optimizer.grad_start_offset[group_index][pid][param_id])
             for pid in optimizer.param_to_partition_ids[group_index][param_id]),
            key=lambda item: item[1],
        )
        for position, (owner, offset) in enumerate(partitions):
            if position == len(partitions) - 1:
                numel = param.numel() - offset
            else:
                numel = partitions[position + 1][1] - offset
            if owner == previous_owner:
                last_owner, last_offset, last_numel = result[-1]
                result[-1] = (last_owner, last_offset, last_numel + numel)
            else:
                result.append((owner, current, numel))
            current += numel
            previous_owner = owner
    return result, current


def layout_digest(optimizer: Any) -> str:
    """Hash everything that decides which rank owns which gradient slice and where buckets split."""
    digest = hashlib.sha256()
    digest.update(repr(getattr(optimizer, "reduce_bucket_size", None)).encode())
    group = getattr(optimizer, "dp_process_group", None)
    if dist.is_available() and dist.is_initialized():
        digest.update(repr(dist.get_world_size(group=group)).encode())
    for group_index, params in enumerate(getattr(optimizer, "bit16_groups", [])):
        partitions = optimizer.param_to_partition_ids[group_index]
        offsets = optimizer.grad_start_offset[group_index]
        digest.update(repr(("group", group_index, [p.numel() for p in params])).encode())
        for param_id in sorted(partitions):
            owners = sorted(partitions[param_id])
            digest.update(repr((param_id, owners, [offsets[o][param_id] for o in owners])).encode())
    return digest.hexdigest()


def verify_switch_consistency(overlap: bool, reduce_scatter: bool) -> None:
    """Collective check, run on every rank whether or not a switch is on, so no rank is left waiting."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (overlap, reduce_scatter))
    if len(set(gathered)) != 1:
        detail = ", ".join(f"rank {i}: overlap={o} reduce_scatter={r}" for i, (o, r) in enumerate(gathered))
        raise RuntimeError("ZeRO-2 communication switches differ across ranks: " + detail)


_MISSING = object()


class Zero2CommInstallation:
    """Bind the selected switches to one optimizer instance; ``close`` restores the originals."""

    def __init__(
        self,
        optimizer: Any,
        *,
        overlap: bool,
        reduce_scatter: bool,
        accelerator: Any = None,
        event_factory: Any = None,
        pack_timing: bool = False,
    ):
        if not (overlap or reduce_scatter):
            raise ValueError("At least one ZeRO-2 communication switch must be enabled")
        self.optimizer = optimizer
        self.overlap = overlap
        self.reduce_scatter = reduce_scatter
        self.pack_timing = pack_timing
        self.accelerator = accelerator or get_accelerator()
        self.scheduler = BucketEventScheduler(
            lambda: self.accelerator.current_stream(), event_factory or torch.cuda.Event
        )
        self.stats = {"average_tensor": 0, "reduce_scatter": 0, "reduce": 0, "empty": 0}
        self.comm_stats: dict = {}
        self._previous: dict[str, Any] = {}
        self._bind("average_tensor", self.average_tensor)
        if overlap:
            self._bind("reduce_independent_p_g_buckets_and_remove_grads",
                       self.reduce_independent_p_g_buckets_and_remove_grads)
            self._bind("setup_buckets", self.setup_buckets)

    def _bind(self, name: str, implementation: Any) -> None:
        self._previous[name] = self.optimizer.__dict__.get(name, _MISSING)
        setattr(self.optimizer, name, implementation)

    def close(self) -> None:
        for name, previous in self._previous.items():
            if previous is _MISSING:
                self.optimizer.__dict__.pop(name, None)
            else:
                setattr(self.optimizer, name, previous)
        self._previous.clear()

    def setup_buckets(self) -> None:
        fresh = not self.optimizer.ready_for_gradients
        type(self.optimizer).setup_buckets(self.optimizer)
        if fresh:
            self.scheduler.reset()

    def reduce_independent_p_g_buckets_and_remove_grads(self, param: Any, i: int) -> None:
        optimizer = self.optimizer
        grad_reduc = optimizer.get_gradient_for_reduction(param)
        comm_dtype = optimizer.get_param_comm_dtype(param)
        bucket = optimizer.ipg_buckets[comm_dtype]
        if bucket.elements + param.numel() > optimizer.reduce_bucket_size:
            optimizer.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
            reduced_index = bucket.index
            optimizer.reduce_ipg_grads()
            if optimizer.contiguous_gradients and optimizer.overlap_comm:
                # bucket.clear() reset the index; alternate halves explicitly so compute never
                # writes the half that is still being reduced.
                self.scheduler.record_consumed((comm_dtype, reduced_index), optimizer.reduction_stream)
                bucket.index = 1 - reduced_index
                self.scheduler.wait_before_reuse((comm_dtype, bucket.index))
            optimizer.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        param_id = optimizer.get_param_id(param)
        assert optimizer.params_already_reduced[param_id] == False, (
            f"Gradient computed twice for this partition (param {param_id}); "
            "multiple gradient reductions are currently not supported"
        )

        if optimizer.contiguous_gradients:
            if param.numel() > optimizer.reduce_bucket_size:
                optimizer.extra_large_param_to_reduce[comm_dtype] = param
            else:
                new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())
                new_grad_tensor.copy_(grad_reduc.view(-1))
                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        bucket.elements += param.numel()

        assert grad_reduc is not None, f"Invalid to reduce param {param_id} with None gradient"

        if getattr(param, "ds_grad_is_ready", True):
            bucket.grads.append(grad_reduc)
            bucket.params.append((i, param.param_idx_in_group, param_id))

        from deepspeed.moe.utils import is_moe_param

        if is_moe_param(param):
            bucket.has_moe_params = True

        optimizer.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def average_tensor(self, tensor: torch.Tensor, communication_data_type: torch.dtype) -> None:
        optimizer = self.optimizer
        accelerator = self.accelerator
        current = accelerator.current_stream()
        if optimizer.overlap_comm:
            stream = optimizer.reduction_stream
            stream.wait_stream(current)
            if self.overlap:
                # The tensor may be a gradient that compute frees; keep it alive for the reduction.
                tensor.record_stream(stream)
            else:
                current.wait_stream(stream)
        else:
            stream = current

        with accelerator.stream(stream):
            bucket = optimizer.ipg_buckets[communication_data_type]
            if bucket.has_moe_params:
                raise RuntimeError("MoE parameters are unsupported by the ZeRO-2 communication switches")
            spec, total = build_rank_and_offsets(optimizer, bucket)
            if total != tensor.numel():
                raise RuntimeError(f"Bucket layout covers {total} elements but the tensor has {tensor.numel()}")
            group = optimizer.dp_process_group
            world_size = dist.get_world_size(group=group)
            tensor.div_(world_size / float(optimizer.sequence_parallel_size))
            slices = [tensor.narrow(0, int(offset), int(numel)) for _, offset, numel in spec]
            owners = [owner for owner, _, _ in spec]
            self.stats["average_tensor"] += 1
            if self.reduce_scatter:
                path = reduce_by_owner(
                    slices,
                    owners,
                    group=group,
                    rank=dist.get_rank(group=group),
                    world_size=world_size,
                    dtype=communication_data_type,
                    stats=self.comm_stats,
                    time_packing=self.pack_timing,
                )
                self.stats[path] += 1
            else:
                optimizer.allreduce_and_scatter(
                    list(zip(owners, slices)),
                    communication_data_type,
                    numel_per_bucket=optimizer.reduce_bucket_size,
                    divide=False,
                    process_group=group,
                )


def _gather_statuses(status: dict) -> list[dict]:
    if dist.is_available() and dist.is_initialized():
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, status)
        return statuses
    return [status]


def _read_pack_timing(pairs) -> tuple[float | None, str | None]:
    """Wait for every end event, then sum the intervals; on failure return the reason."""
    total = 0.0
    try:
        for begin, end in pairs:
            end.synchronize()
            total += begin.elapsed_time(end)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return total, None


class Zero2CommCallback(TrainerCallback):
    """Install the switches once the DeepSpeed engine exists; ``close`` restores them and is safe to call repeatedly."""

    def __init__(
        self, trainer: Any, overlap: bool, reduce_scatter: bool, output_dir: Any = None, pack_timing: bool = False
    ):
        self.trainer = trainer
        self.overlap = overlap
        self.reduce_scatter = reduce_scatter
        self.output_dir = output_dir
        self.pack_timing = pack_timing
        self.installation: Zero2CommInstallation | None = None
        self.layout: str | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.installation is not None:
            raise RuntimeError("ZeRO-2 communication callback cannot be started twice")
        status = {"ok": True}
        try:
            validate_vendor_build(*current_vendor_versions_and_hashes())
            optimizer = self.trainer.model_wrapped.optimizer
            validate_optimizer(optimizer)
            self.installation = Zero2CommInstallation(
                optimizer, overlap=self.overlap, reduce_scatter=self.reduce_scatter,
                pack_timing=self.pack_timing,
            )
            self.layout = layout_digest(optimizer)
            if self.reduce_scatter:
                self._preallocate(optimizer)
            status = {"ok": True, "config": (self.overlap, self.reduce_scatter, self.layout)}
        except BaseException as exc:
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        statuses = _gather_statuses(status)
        errors = [
            f"rank {index}: {item['error']}" for index, item in enumerate(statuses) if not item["ok"]
        ]
        if not errors and len({item["config"] for item in statuses}) != 1:
            errors = [
                f"rank {index}: switches/layout {item['config']}" for index, item in enumerate(statuses)
            ]
            errors.insert(0, "ranks disagree on switches or bucket/owner layout")
        if errors:
            self.close()
            raise RuntimeError("ZeRO-2 communication switches could not be installed: " + " | ".join(errors))
        return control

    @staticmethod
    def _preallocate(optimizer: Any) -> None:
        groups = getattr(optimizer, "bit16_groups", None)
        if not groups or not groups[0]:
            return
        (dtype,) = optimizer.ipg_buckets
        preallocate_staging(
            groups[0][0].device,
            dtype,
            int(optimizer.reduce_bucket_size),
            dist.get_world_size(group=optimizer.dp_process_group),
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.close()
        return control

    def close(self) -> None:
        installation, self.installation = self.installation, None
        if installation is None:
            return
        installation.close()
        clear_staging_buffers()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if self.output_dir is None:
            return
        comm = dict(installation.comm_stats)
        pairs = comm.pop("pack_events", [])
        comm["pack_timing"] = self.pack_timing
        if self.pack_timing:
            comm["pack_ms"], comm["pack_ms_error"] = _read_pack_timing(pairs)
        path = Path(self.output_dir) / "zero2_comm"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"runtime-rank-{rank:02d}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "overlap": self.overlap,
                    "reduce_scatter": self.reduce_scatter,
                    "layout_digest": self.layout,
                    "stats": installation.stats,
                    "comm": comm,
                    "event_records": installation.scheduler.records,
                    "event_waits": installation.scheduler.waits,
                },
                indent=2,
            )
            + "\n"
        )
