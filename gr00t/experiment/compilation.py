# SPDX-License-Identifier: Apache-2.0
"""Opt-in instance compilation after checkpoint loading."""

import logging
import types

import torch
from torch import nn


_STATE_ATTR = "_gr00t_dit_compile"
_MODEL_STATE_ATTR = "_gr00t_model_compile"


def _check_installed(model: nn.Module, scope: str | None, mode: str) -> list[str] | None:
    """Return installed paths for an identical request; raise if the request changed."""
    combined = getattr(model, _MODEL_STATE_ATTR, None)
    if combined is not None:
        request = combined["request"]
        combined_scope = "model" if request["whole"] else "blocks" if request["blocks"] else None
        if combined["failed"] or combined_scope != scope or request["mode"] != mode:
            raise RuntimeError(
                "Compile configuration changed or installation failed; restart the process"
            )
    installed = getattr(model, _STATE_ATTR, None)
    if installed is None:
        return None
    if installed.get("failed", False) or installed["scope"] != scope or installed["mode"] != mode:
        raise RuntimeError(
            "DiT compile configuration changed or installation failed; restart the process"
        )
    return list(installed["paths"])


def _dit_blocks(model: nn.Module):
    from gr00t.model.modules.dit import BasicTransformerBlock

    blocks = model.action_head.model.transformer_blocks
    if len(blocks) != 32 or not all(isinstance(b, BasicTransformerBlock) for b in blocks):
        raise ValueError("DiT compilation requires exactly 32 BasicTransformerBlock instances")
    if len({id(block) for block in blocks}) != 32:
        raise ValueError("DiT compilation requires 32 distinct block instances")
    return blocks


def _reject_existing_wrappers(modules) -> None:
    for module in modules:
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError("DiT module already compiled by another entry point; restart")
        if not callable(getattr(module, "compile", None)):
            raise RuntimeError("This PyTorch build lacks Module.compile()")


def compile_dit_blocks(model: nn.Module, *, enabled: bool = False, mode: str = "default") -> list[str]:
    """Install lazy wrappers, preserving module identity and checkpoint keys.

    Turning compilation off or changing modes after installation requires a new
    process. Existing visual compilation is independent of this switch.
    """
    installed = _check_installed(model, "blocks" if enabled else None, mode)
    if installed is not None:
        return installed
    if not enabled:
        logging.info("DiT compile disabled; keeping existing DiT eager path")
        return []

    blocks = _dit_blocks(model)
    _reject_existing_wrappers([model.action_head.model, *blocks])

    paths = []
    state = {"scope": "blocks", "mode": mode, "paths": (), "failed": True}
    setattr(model, _STATE_ATTR, state)
    for index, block in enumerate(blocks):
        path = f"action_head.model.transformer_blocks.{index}"
        block.compile(mode=mode, fullgraph=False)
        paths.append(path)
        logging.info(
            "Installed lazy compile wrapper: %s mode=%s dynamic=auto fullgraph=False", path, mode
        )
    state.update(paths=tuple(paths), failed=False)
    logging.info(
        "DiT wrappers installed: %d; capture/execution must be verified at runtime", len(paths)
    )
    return paths


def compile_dit_model(model: nn.Module, *, enabled: bool = False, mode: str = "default") -> list[str]:
    """Compile the whole action DiT instance (``action_head.model``) as one region."""
    installed = _check_installed(model, "model" if enabled else None, mode)
    if installed is not None:
        return installed
    if not enabled:
        logging.info("DiT compile disabled; keeping existing DiT eager path")
        return []

    from gr00t.model.modules.dit import DiT

    dit = model.action_head.model
    if not isinstance(dit, DiT):
        raise ValueError("Whole-DiT compilation requires action_head.model to be a DiT instance")
    _reject_existing_wrappers([dit, *_dit_blocks(model)])

    path = "action_head.model"
    state = {"scope": "model", "mode": mode, "paths": (path,), "failed": True}
    setattr(model, _STATE_ATTR, state)
    dit.compile(mode=mode, fullgraph=False)
    state["failed"] = False
    logging.info(
        "Installed lazy compile wrapper: %s (%s) mode=%s dynamic=auto fullgraph=False; "
        "capture/execution must be verified at runtime",
        path,
        type(dit).__name__,
        mode,
    )
    return [path]


def install_dit_compile(
    model: nn.Module, *, blocks: bool = False, whole: bool = False, mode: str = "default"
) -> list[str]:
    """Single entry point; per-block and whole-DiT compilation are mutually exclusive."""
    if blocks and whole:
        raise ValueError("--compile-dit-blocks and --compile-dit-model are mutually exclusive")
    if whole:
        return compile_dit_model(model, enabled=True, mode=mode)
    return compile_dit_blocks(model, enabled=blocks, mode=mode)


def _typed_layers(layers, layer_type, count: int, path: str, boundary: str = "module"):
    if len(layers) != count or not all(isinstance(layer, layer_type) for layer in layers):
        raise ValueError(
            f"{path} compilation requires exactly {count} {layer_type.__name__} instances"
        )
    if len({id(layer) for layer in layers}) != count:
        raise ValueError(f"{path} compilation requires {count} distinct instances")
    return [(f"{path}.{index}", layer, boundary) for index, layer in enumerate(layers)]


def _reject_class_compile(modules) -> None:
    for module in modules:
        if hasattr(type(module).__call__, "_torchdynamo_orig_callable"):
            raise RuntimeError(f"{type(module).__name__} class already compiled; restart")


def _reject_foreign_forward(module: nn.Module) -> None:
    """Allow no instance ``forward``, or the class forward bound to itself.

    The latter is what transformers' output capture restores after an eager call.
    """
    forward = vars(module).get("forward")
    if forward is None:
        return
    if (isinstance(forward, types.MethodType) and forward.__self__ is module
            and forward.__func__ is type(module).forward):
        return
    raise RuntimeError(f"{type(module).__name__} has a replaced instance forward; restart")


def _preflight_targets(model: nn.Module, targets) -> None:
    if len({id(module) for _, module, _ in targets}) != len(targets):
        raise ValueError("Compile targets must be distinct instances across scopes")
    for path, module, boundary in targets:
        # Reject overlapping wrappers on both ancestors and descendants.
        ancestors = [model]
        parts = path.split(".")
        ancestors.extend(model.get_submodule(".".join(parts[:i])) for i in range(1, len(parts)))
        _reject_existing_wrappers([*ancestors, *module.modules()])
        _reject_class_compile(module.modules())
        if boundary == "forward":
            _reject_foreign_forward(module)


def install_model_compile(
    model: nn.Module, *, blocks: bool = False, whole: bool = False,
    text_layers: bool = False, text_forwards: bool = False, vl_blocks: bool = False,
    mode: str = "default",
) -> list[str]:
    """Install independently selected scopes, with a single preflight before mutation.

    ``text_forwards`` compiles the text decoder layer ``forward`` once and binds it per
    instance. Transformers' hidden-state capture swaps each layer's instance ``forward`` for
    a closure whose collected-output count differs per layer; compiling inside that closure
    keeps it eager and lets all 16 layers share one graph. This relies on the capture
    reading the instance ``forward`` at call time and restoring it on normal return; after
    an exception the closures stay installed, so the model must not be reused. Per-instance
    binding keeps ``deepcopy`` correct; the model object is not picklable, so save via
    ``state_dict`` and re-install on the reloaded instance.

    Disabling/changing an installed configuration requires a fresh process. No
    model computation, precision, cache policy or module identity is changed.
    """
    if blocks and whole:
        raise ValueError("--compile-dit-blocks and --compile-dit-model are mutually exclusive")
    if text_layers and text_forwards:
        raise ValueError(
            "--compile-text-layers and --compile-text-layer-forwards are mutually exclusive"
        )
    request = dict(
        blocks=blocks, whole=whole, text_layers=text_layers, text_forwards=text_forwards,
        vl_blocks=vl_blocks, mode=mode,
    )
    installed = getattr(model, _MODEL_STATE_ATTR, None)
    if installed is not None:
        if installed["failed"] or installed["request"] != request:
            raise RuntimeError(
                "Compile configuration changed or installation failed; restart the process"
            )
        return list(installed["paths"])

    dit_scope = "model" if whole else "blocks" if blocks else None
    dit_installed = _check_installed(model, dit_scope, mode)
    if not any((blocks, whole, text_layers, text_forwards, vl_blocks)):
        logging.info(
            "Optional model compile disabled; keeping existing eager paths and visual compilation"
        )
        return []

    targets = []
    if text_layers or text_forwards:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

        path = "backbone.model.language_model.layers"
        targets.extend(_typed_layers(
            model.get_submodule(path), Qwen3VLTextDecoderLayer, 16, path,
            "forward" if text_forwards else "module",
        ))
    if vl_blocks:
        from gr00t.model.modules.dit import BasicTransformerBlock

        path = "action_head.vl_self_attention.transformer_blocks"
        targets.extend(_typed_layers(model.get_submodule(path), BasicTransformerBlock, 4, path))
    dit_targets = []
    if blocks:
        dit_targets = [
            (f"action_head.model.transformer_blocks.{i}", b, "module")
            for i, b in enumerate(_dit_blocks(model))
        ]
    elif whole:
        from gr00t.model.modules.dit import DiT

        dit = model.action_head.model
        if not isinstance(dit, DiT):
            raise ValueError(
                "Whole-DiT compilation requires action_head.model to be a DiT instance"
            )
        _dit_blocks(model)
        dit_targets = [("action_head.model", dit, "module")]
    if dit_installed is None:
        targets.extend(dit_targets)
    _preflight_targets(model, targets)

    paths = [path for path, _, _ in targets] + (dit_installed or [])
    # Mark incomplete installation before the first wrapper: failures must not be retried in-place.
    state = {"request": request, "paths": tuple(paths), "failed": True}
    setattr(model, _MODEL_STATE_ATTR, state)
    logging.info("Optional compile request: %s; matched %d paths: %s", request, len(paths), paths)
    try:
        shared_forwards = {}
        for path, module, boundary in targets:
            if boundary == "forward":
                cls = type(module)
                if cls not in shared_forwards:
                    shared_forwards[cls] = torch.compile(cls.forward, mode=mode, fullgraph=False)
                module.forward = types.MethodType(shared_forwards[cls], module)
            else:
                module.compile(mode=mode, fullgraph=False)
            logging.info(
                "Installed lazy compile wrapper: %s (%s) boundary=%s mode=%s dynamic=auto "
                "fullgraph=False; capture/execution must be verified at runtime",
                path, type(module).__name__, boundary, mode,
            )
    except Exception:
        logging.exception("Compile installation incomplete; restart the process before retrying")
        raise
    if dit_targets:
        setattr(model, _STATE_ATTR, {
            "scope": dit_scope, "mode": mode, "paths": tuple(path for path, _, _ in dit_targets),
        })
    state["failed"] = False
    return paths
