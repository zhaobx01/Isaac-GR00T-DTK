"""Opt-in sequence shape adaptation; no tensor padding or model mutation."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
import types
from functools import wraps
from pathlib import Path

import torch


def resolve_policy(environ=None):
    environ = os.environ if environ is None else environ
    policy = environ.get("GR00T_COMPILE_SEQUENCE_POLICY", "auto")
    if policy not in {"auto", "dynamic-buckets", "pad-152-156", "pad-156"}:
        raise ValueError("Unsupported GR00T_COMPILE_SEQUENCE_POLICY")
    diagnostic = environ.get("GR00T_COMPILE_LENGTH_DIAGNOSTICS", "0")
    if diagnostic not in {"0", "1"}:
        raise ValueError("GR00T_COMPILE_LENGTH_DIAGNOSTICS must be 0 or 1")
    path = environ.get("GR00T_COMPILE_BUCKET_CONFIG")
    if policy == "dynamic-buckets" and (not path or not Path(path).is_absolute()):
        raise ValueError("GR00T_COMPILE_BUCKET_CONFIG must be an absolute configuration path")
    return policy, path, diagnostic == "1"


def validate_activation(policy, diagnostic, variant, legacy):
    if (policy != "auto" or diagnostic) and legacy is not None:
        raise ValueError("Sequence diagnostics and dynamic buckets do not support legacy CLI compilation")
    if (policy != "auto" or diagnostic) and variant != "dit-model-vl":
        raise ValueError("Sequence diagnostics and dynamic buckets require dit-model-vl")


class DiagnosticRecorder:
    """Write diagnostic metadata from the training process, not data-loader workers."""

    def __init__(self, output_dir, *, rank, physical_device):
        self.path = Path(output_dir) / f"rank-{rank:02d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.physical_device = physical_device
        self.step = None
        self.call_id = 0

    def emit(self, event, **fields):
        payload = {"event": event, "rank": self.rank, "physical_device": self.physical_device,
                   "step": self.step, "wall_time_ns": time.time_ns(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        print("[SEQUENCE_DIAGNOSTIC] " + json.dumps(payload), file=sys.stderr, flush=True)

    def record_batch(self, batch):
        values = batch.get("inputs", batch)
        ids = values.get("input_ids")
        mask = values.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("Length diagnostics require input_ids and attention_mask")
        ids = ids.detach().cpu().contiguous()
        mask = mask.detach().cpu().contiguous()
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("Input IDs and attention mask must have matching rank-2 shapes")
        self.emit("batch", padded_length=ids.shape[1], valid_lengths=mask.sum(dim=1).tolist(),
                  input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())

    def record_call(self, target, args, kwargs, stage, target_path=None):
        tensors = {}
        for name, value in [(f"arg{i}", x) for i, x in enumerate(args)] + list(kwargs.items()):
            if isinstance(value, torch.Tensor):
                tensors[name] = {"shape": list(value.shape), "stride": list(value.stride()),
                                 "dtype": str(value.dtype), "device": str(value.device)}
        self.emit("compile_boundary", target=target, target_path=target_path, stage=stage,
                  call_id=self.call_id, tensors=tensors)


def load_buckets(path):
    document = json.loads(Path(path).read_text())
    buckets = document.get("buckets")
    if document.get("schema_version") != 1 or not isinstance(buckets, list) or not buckets:
        raise ValueError("Invalid sequence bucket configuration")
    required = {"target", "batch", "feature_dim", "dtype", "min_length", "max_length"}
    for bucket in buckets:
        if not isinstance(bucket, dict) or set(bucket) != required:
            raise ValueError("Invalid sequence bucket fields")
        if (bucket["target"] not in {"vl", "dit"} or not isinstance(bucket["dtype"], str)
                or any(type(bucket[x]) is not int for x in ("batch", "feature_dim", "min_length", "max_length"))
                or bucket["batch"] < 1 or bucket["feature_dim"] < 1
                or bucket["min_length"] < 2 or bucket["min_length"] >= bucket["max_length"]):
            raise ValueError("Invalid sequence bucket bounds or structure")
    for i, left in enumerate(buckets):
        for right in buckets[i + 1:]:
            if (all(left[k] == right[k] for k in ("target", "batch", "feature_dim", "dtype"))
                    and max(left["min_length"], right["min_length"]) <= min(left["max_length"], right["max_length"])):
                raise ValueError("Overlapping sequence buckets")
    return buckets


def select_bucket(buckets, target, features):
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("Sequence features must be a rank-3 tensor")
    if not features.is_contiguous():
        raise ValueError("Sequence features must have contiguous layout")
    batch, length, dim = features.shape
    candidates = [b for b in buckets if b["target"] == target and b["batch"] == batch
                  and b["feature_dim"] == dim and b["dtype"] == str(features.dtype)
                  and b["min_length"] <= length <= b["max_length"]]
    if len(candidates) != 1:
        raise ValueError(f"No sequence bucket for {target}: shape={tuple(features.shape)} dtype={features.dtype}")
    return candidates[0]


def mark_arguments(buckets, target, args, kwargs):
    if target == "vl":
        features = kwargs.get("hidden_states", args[0] if args else None)
        masks = []
    elif target == "dit":
        features = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        masks = [kwargs.get(name, args[pos] if len(args) > pos else None) for name, pos in (
            ("encoder_attention_mask", 3), ("image_mask", 5), ("backbone_attention_mask", 6))]
    else:
        raise ValueError(f"Unsupported sequence target {target}")
    bucket = select_bucket(buckets, target, features)
    for mask in masks:
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 2
                                 or tuple(mask.shape) != tuple(features.shape[:2])):
            raise ValueError("Mask sequence dimension does not match encoder features")
        if mask is not None and (mask.dtype != torch.bool or not mask.is_contiguous()):
            raise ValueError("Sequence mask must be contiguous and boolean")
    if target == "dit" and any(mask is None for mask in masks):
        raise ValueError("Sequence mask inputs are required for AlternateVLDiT")
    seen = set()
    for tensor in (features, *masks):
        if tensor is not None and id(tensor) not in seen:
            seen.add(id(tensor))
            torch._dynamo.mark_dynamic(tensor, 1, min=bucket["min_length"], max=bucket["max_length"])
    return bucket


class SequenceAdapter:
    """Wrap existing instance compile callables and restore them on exit."""

    def __init__(self, targets, buckets, recorder=None):
        self.targets = list(targets)
        self.buckets = buckets
        self.recorder = recorder
        self.originals = []

    def __enter__(self):
        try:
            for item in self.targets:
                target_type, target = item[:2]
                target_path = item[2] if len(item) == 3 else target_type
                original = getattr(target, "_compiled_call_impl", None)
                if not callable(original):
                    raise ValueError(f"{target_type} target is not compiled")
                self.originals.append((target, original))

                @wraps(original)
                def adapted(*args, _original=original, _target=target_type, _path=target_path, **kwargs):
                    if torch.compiler.is_compiling():
                        raise RuntimeError("Sequence annotations must run outside a compiled region")
                    if self.recorder is not None:
                        self.recorder.call_id += 1
                        self.recorder.record_call(_target, args, kwargs, "begin", _path)
                    try:
                        if self.buckets is not None:
                            mark_arguments(self.buckets, _target, args, kwargs)
                        result = _original(*args, **kwargs)
                    except BaseException:
                        if self.recorder is not None:
                            self.recorder.record_call(_target, args, kwargs, "error", _path)
                        raise
                    if self.recorder is not None:
                        self.recorder.record_call(_target, args, kwargs, "end", _path)
                    return result

                target._compiled_call_impl = adapted
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for target, original in reversed(self.originals):
            target._compiled_call_impl = original
        self.originals.clear()

    def __exit__(self, *_exc):
        self.close()


class TrainerDiagnostics:
    """Capture prepared batches and steps without changing model inputs."""

    def __init__(self, trainer, recorder):
        self.trainer = trainer
        self.recorder = recorder
        self.original = None
        self.had_instance_method = False
        self.callback = None

    def install(self):
        from transformers import TrainerCallback

        recorder = self.recorder

        class StepCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                recorder.step = state.global_step + 1

        self.callback = StepCallback()
        self.original = self.trainer._prepare_inputs
        self.had_instance_method = "_prepare_inputs" in vars(self.trainer)
        original = self.original

        def prepare(_trainer, inputs):
            if recorder.step is None:
                raise RuntimeError("Cannot record a batch without a training step")
            recorder.record_batch(inputs)
            return original(inputs)

        self.trainer._prepare_inputs = types.MethodType(prepare, self.trainer)
        try:
            self.trainer.add_callback(self.callback)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.original is not None:
            if self.had_instance_method:
                self.trainer._prepare_inputs = self.original
            else:
                del self.trainer._prepare_inputs
            self.original = None
        if self.callback is not None:
            self.trainer.remove_callback(self.callback)
            self.callback = None
