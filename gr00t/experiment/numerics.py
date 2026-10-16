"""Opt-in numerical diagnostics at rank-symmetric training boundaries."""

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import TrainerCallback


def tensor_items(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            if isinstance(key, torch.Tensor):
                key = f"parameter-{index}"
            yield from tensor_items(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}.{index}")


def nonfinite_counts(value):
    """Bound temporary allocation even for large flattened optimizer partitions."""
    counts = torch.zeros(2, dtype=torch.int64, device=value.device)
    for chunk in value.detach().reshape(-1).split(4 * 1024 * 1024):
        counts += torch.stack((torch.isnan(chunk).sum(), torch.isinf(chunk).sum()))
    return counts


def scan_checkpoint(directory):
    """Read all safetensors on CPU without constructing a model or using HIP."""
    from safetensors import safe_open

    root = Path(directory)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        names = sorted(set(mapping.values()))
    else:
        mapping = None
        names = [p.name for p in sorted(root.glob("*.safetensors"))]
    if not names:
        raise FileNotFoundError(f"No safetensors found in {root}")
    counts = Counter(tensors=0, stored_elements=0, bad_tensors=0, nan=0, inf=0)
    failures = []
    dtypes = Counter()
    found = set()
    for name in names:
        with safe_open(root / name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if mapping is not None and mapping.get(key) != name:
                    raise ValueError(f"Index/shard mismatch for {key}")
                found.add(key)
                value = handle.get_tensor(key)
                counts["tensors"] += 1
                counts["stored_elements"] += value.numel()
                dtypes[str(value.dtype)] += value.numel()
                if not value.is_floating_point():
                    continue
                nan, inf = nonfinite_counts(value).tolist()
                if nan or inf:
                    counts["bad_tensors"] += 1
                    counts["nan"] += nan
                    counts["inf"] += inf
                    failures.append({"name": key, "dtype": str(value.dtype),
                                     "shape": list(value.shape), "nan": nan, "inf": inf})
    if mapping is not None and found != set(mapping):
        raise ValueError(f"Missing indexed tensors: {sorted(set(mapping) - found)}")
    return {"directory": str(root), "passed": not failures, "counts": dict(counts),
            "dtypes": dict(dtypes), "failures": failures}


class NumericalError(RuntimeError):
    pass


def witness_indices(numel, device):
    """Integer arithmetic avoids rounded-up endpoints on large parameter tensors."""
    count = min(16, numel)
    indices = torch.arange(count, device=device, dtype=torch.long)
    return indices * (numel - 1) // max(1, count - 1)


class NumericalMonitor(TrainerCallback):
    """Hooks record locally; only explicit, rank-symmetric boundaries communicate."""

    def __init__(self, output_dir):
        self.root = Path(output_dir) / "numerics"
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.step = 0
        self.pending = []
        self.handles = []
        self.restorations = []
        self.norms = []
        self.witnesses = {}
        self.saw_update = False
        self.saw_gradient = False
        self.has_gradient = []
        self.trainer = None

    def event(self, **values):
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"rank": self.rank, "step": self.step, **values}
        with (self.root / f"rank-{self.rank:02d}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def observe(self, stage, value):
        for name, tensor in tensor_items(value, stage):
            if tensor.is_floating_point() or tensor.is_complex():
                self.pending.append((name, str(tensor.dtype), list(tensor.shape),
                                     nonfinite_counts(tensor), "nonfinite"))

    def condition(self, name, valid):
        counts = torch.stack(((~valid).long().sum(), valid.new_zeros((), dtype=torch.long)))
        self.pending.append((name, str(valid.dtype), list(valid.shape),
                             counts, "condition"))

    def boundary(self, stage):
        failures = []
        grouped = {}
        for name, dtype, shape, counts, kind in self.pending:
            grouped.setdefault(counts.device, []).append((name, dtype, shape, counts, kind))
        self.pending.clear()
        for records in grouped.values():
            counts = torch.stack([r[3] for r in records]).cpu().tolist()
            for (name, dtype, shape, _, kind), (nan, inf) in zip(records, counts):
                if nan or inf:
                    failures.append({"name": name, "dtype": dtype, "shape": shape,
                                     **({"nan": nan, "inf": inf} if kind == "nonfinite"
                                        else {"invalid": nan})})
        device = self.device
        if dist.is_initialized() and dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        flag = torch.tensor(bool(failures), device=device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            report = {"stage": stage, "failures": failures[:20], "failed_tensors": len(failures)}
            self.event(**report)
            reports = [report]
            if dist.is_initialized():
                reports = [None] * dist.get_world_size()
                dist.all_gather_object(reports, {"rank": self.rank, **report})
            raise NumericalError("Non-finite training state: " + json.dumps(reports))

    def attach(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.scan_model("loaded", all_parameters=True)
        self.boundary("loaded")
        dtypes = Counter()
        for name, parameter in model.named_parameters():
            dtypes[f"{parameter.dtype}:trainable={parameter.requires_grad}"] += parameter.numel()
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(self.gradient_hook(name)))
        self.event(stage="loaded", dtypes=dict(dtypes))
        boundaries = {"backbone", "action_head.state_encoder", "action_head.action_encoder",
                      "action_head.model", "action_head.action_decoder"}
        for name, module in model.named_modules():
            if name in boundaries:
                self.handles.append(module.register_forward_hook(self.forward_hook(name)))

    def forward_hook(self, name):
        def hook(module, inputs, output):
            self.observe(f"forward.{name}", output)
        return hook

    def gradient_hook(self, name):
        def hook(gradient):
            self.observe(f"gradient.{name}", gradient)
            self.has_gradient.append(torch.any(gradient.detach() != 0))
        return hook

    def scan_model(self, stage, all_parameters=False):
        for name, parameter in self.model.named_parameters():
            if all_parameters or parameter.requires_grad:
                self.observe(f"{stage}.parameter.{name}", parameter)
        for name, buffer in self.model.named_buffers():
            self.observe(f"{stage}.buffer.{name}", buffer)

    def prepare_batch(self, inputs):
        self.observe("input", inputs)
        for name, value in tensor_items(inputs):
            if name.split(".")[-1] == "action_mask":
                self.condition("input.action_mask_empty", value.bool().any())
        self.boundary("input")
        return inputs

    def scan_optimizer(self, optimizer, stage):
        self.observe(f"{stage}.optimizer_state", optimizer.state)
        for index, group in enumerate(optimizer.param_groups):
            for number, parameter in enumerate(group["params"]):
                self.observe(f"{stage}.master.{index}.{number}", parameter)
                if parameter.grad is not None:
                    self.observe(f"{stage}.optimizer_gradient.{index}.{number}", parameter.grad)

    def check_updates(self):
        changes = []
        for name, parameter in self.model.named_parameters():
            if name in self.witnesses:
                indices, initial = self.witnesses[name]
                changes.append(torch.any(parameter.detach().reshape(-1)[indices] != initial))
        if changes:
            self.saw_update |= bool(torch.stack(changes).any().item())

    def on_train_begin(self, args, state, control, **kwargs):
        self.device = next(self.model.parameters()).device
        self.scan_model("prepared", all_parameters=True)
        self.boundary("prepared")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                indices = witness_indices(parameter.numel(), parameter.device)
                self.witnesses[name] = (indices, parameter.detach().reshape(-1)[indices].clone())
        engine = getattr(self.trainer, "deepspeed", None)
        optimizer = engine.optimizer if engine is not None else self.trainer.optimizer
        seen = set()
        while hasattr(optimizer, "optimizer") and id(optimizer) not in seen:
            seen.add(id(optimizer))
            optimizer = optimizer.optimizer
        self.event(stage="optimizer", optimizer=type(optimizer).__name__,
                   parameter_dtypes=dict(Counter(str(p.dtype) for g in optimizer.param_groups
                                                 for p in g["params"])),
                   state_dtypes=dict(Counter(str(t.dtype)
                                             for _, t in tensor_items(optimizer.state))))
        self.scan_optimizer(optimizer, "optimizer_ready")
        self.boundary("optimizer_ready")

        def pre_step(opt, unused_args, unused_kwargs):
            if self.has_gradient:
                self.saw_gradient |= bool(torch.stack(self.has_gradient).any().item())
                self.has_gradient.clear()
            self.scan_optimizer(opt, "pre_update")
            self.boundary("pre_update")

        def post_step(opt, unused_args, unused_kwargs):
            self.scan_optimizer(opt, "post_update")
            self.boundary("post_update")
            if engine is None:
                self.scan_model("post_update")
                self.boundary("post_update_model")
                self.check_updates()

        self.handles.extend((optimizer.register_step_pre_hook(pre_step),
                             optimizer.register_step_post_hook(post_step)))
        if engine is not None:
            original = engine.step

            def checked_step(*step_args, **step_kwargs):
                self.observe("backward.partition_gradient", engine.optimizer.averaged_gradients)
                self.observe("backward.master", engine.optimizer.single_partition_of_fp32_groups)
                self.boundary("backward")
                result = original(*step_args, **step_kwargs)
                self.scan_model("post_update")
                self.boundary("post_update_model")
                self.check_updates()
                return result

            engine.step = checked_step
            self.restorations.append((engine, "step", original))
            if hasattr(engine.optimizer, "get_grad_norm_direct"):
                original_norm = engine.optimizer.get_grad_norm_direct

                def checked_norm(*norm_args, **norm_kwargs):
                    norm = original_norm(*norm_args, **norm_kwargs)
                    self.observe("gradient_group_norm", norm)
                    self.condition("negative_gradient_group_norm", norm >= 0)
                    self.norms.append(float(norm.detach().cpu()))
                    return norm

                engine.optimizer.get_grad_norm_direct = checked_norm
                self.restorations.append((engine.optimizer, "get_grad_norm_direct", original_norm))

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = state.global_step + 1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 5 or state.global_step % args.logging_steps == 0:
            self.event(stage="step_complete", gradient_groups=self.norms,
                       saw_nonzero_gradient=self.saw_gradient, saw_parameter_update=self.saw_update)
        self.norms.clear()

    def on_train_end(self, args, state, control, **kwargs):
        self.condition("no_nonzero_gradient", torch.tensor(self.saw_gradient, device=self.device))
        self.condition("no_parameter_update", torch.tensor(self.saw_update, device=self.device))
        self.scan_model("final", all_parameters=True)
        self.boundary("final")
        self.event(stage="training_numerics_passed", steps=state.global_step,
                   saw_nonzero_gradient=self.saw_gradient, saw_parameter_update=self.saw_update)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for target, name, original in reversed(self.restorations):
            setattr(target, name, original)
