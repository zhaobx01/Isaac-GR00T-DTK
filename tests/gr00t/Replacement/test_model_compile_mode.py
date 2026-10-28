# SPDX-License-Identifier: Apache-2.0
"""Compile-mode dispatch tests use CPU modules and stub compilation only."""
import json
import logging
import sys
from types import SimpleNamespace

import pytest
from torch import nn

from gr00t.Replacement import model_compile


@pytest.fixture
def setup_model(monkeypatch):
    class Block(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(32)])

    monkeypatch.setitem(sys.modules, "gr00t.model.modules.dit",
                        SimpleNamespace(BasicTransformerBlock=Block, DiT=DiT))
    calls = []

    def compile_stub(module, **kwargs):
        calls.append((module, kwargs))
        module._compiled_call_impl = lambda *args, **kw: None

    monkeypatch.setattr(nn.Module, "compile", compile_stub)
    monkeypatch.delenv("GR00T_COMPILE_MODE", raising=False)
    monkeypatch.delenv("GR00T_COMPILE_TRACE_MARKERS", raising=False)
    model = nn.Module()
    model.action_head = nn.Module()
    model.action_head.model = DiT()
    model.action_head.vl_self_attention = nn.Module()
    model.action_head.vl_self_attention.transformer_blocks = nn.ModuleList(
        [Block() for _ in range(4)]
    )
    return model, calls


def test_mode_default_unchanged(setup_model):
    model, calls = setup_model
    info = model_compile.install(model, "dit-model-vl")
    assert info["mode"] == "default"
    assert len(calls) == 5
    assert all(kwargs == {"mode": "default", "fullgraph": False} for _, kwargs in calls)


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
def test_environment_mode_reaches_every_target(setup_model, monkeypatch, tmp_path, caplog, mode):
    model, calls = setup_model
    monkeypatch.setenv("GR00T_COMPILE_MODE", mode)
    monkeypatch.setenv("RANK", "0")
    with caplog.at_level(logging.INFO):
        info = model_compile.install(model, "dit-model-vl")
    assert len(calls) == 5
    assert all(kwargs == {"mode": mode, "fullgraph": False} for _, kwargs in calls)
    assert info["mode"] == mode
    assert f"mode={mode}" in caplog.text
    model_compile.write_manifest(tmp_path, info)
    assert json.loads((tmp_path / "compilation/rank-00.json").read_text())["mode"] == mode


def test_mode_change_requires_new_process(setup_model, monkeypatch):
    model, calls = setup_model
    model_compile.install(model, "dit-model")
    monkeypatch.setenv("GR00T_COMPILE_MODE", "max-autotune")
    with pytest.raises(RuntimeError, match="restart"):
        model_compile.install(model, "dit-model")
    assert len(calls) == 1


def test_same_nondefault_request_is_idempotent(setup_model, monkeypatch):
    model, calls = setup_model
    monkeypatch.setenv("GR00T_COMPILE_MODE", "max-autotune-no-cudagraphs")
    first = model_compile.install(model, "vl")
    assert model_compile.install(model, "vl") == first
    assert len(calls) == 4


@pytest.mark.parametrize("mode", ["", "DEFAULT", "unknown-mode"])
def test_invalid_mode_rejected_before_wrapping(setup_model, monkeypatch, mode):
    model, calls = setup_model
    monkeypatch.setenv("GR00T_COMPILE_MODE", mode)
    with pytest.raises(ValueError, match="GR00T_COMPILE_MODE"):
        model_compile.install(model, "vl")
    assert not calls
    assert not hasattr(model, "_gr00t_optional_compile")


def test_explicit_mode_overrides_environment(setup_model, monkeypatch):
    model, calls = setup_model
    monkeypatch.setenv("GR00T_COMPILE_MODE", "reduce-overhead")
    info = model_compile.install(model, "dit-model", mode="max-autotune")
    assert info["mode"] == "max-autotune"
    assert calls[0][1]["mode"] == "max-autotune"


def test_backend_specific_mode_is_supported(setup_model, monkeypatch):
    from torch._inductor import list_mode_options

    extras = set(list_mode_options()) - {
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    }
    if not extras:
        pytest.skip("Installed backend has no additional modes")
    model, calls = setup_model
    mode = sorted(extras)[0]
    monkeypatch.setenv("GR00T_COMPILE_MODE", mode)
    assert model_compile.install(model, "vl")["mode"] == mode
    assert all(kwargs["mode"] == mode for _, kwargs in calls)
