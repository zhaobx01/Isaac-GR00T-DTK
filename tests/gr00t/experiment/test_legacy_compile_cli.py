# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of restored CLI parsing and compile dispatch; no GPU initialization."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tyro

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.experiment import training_runtime

REPO = Path(__file__).resolve().parents[3]
FIELDS = (
    "compile_dit_blocks", "compile_dit_model", "compile_text_layers",
    "compile_text_layer_forwards", "compile_vl_self_attention_blocks", "compile_mode",
)


@pytest.mark.parametrize("args,enabled", [
    (["--compile-dit-blocks"], {"compile_dit_blocks"}),
    (["--compile-dit-model"], {"compile_dit_model"}),
    (["--compile-dit-model", "--compile-vl-self-attention-blocks"],
     {"compile_dit_model", "compile_vl_self_attention_blocks"}),
    (["--compile-text-layer-forwards"], {"compile_text_layer_forwards"}),
    (["--compile-text-layers"], {"compile_text_layers"}),
    (["--compile-vl-self-attention-blocks"], {"compile_vl_self_attention_blocks"}),
    (["--no-compile-dit-blocks", "--no-compile-dit-model", "--no-compile-text-layers",
      "--no-compile-text-layer-forwards", "--no-compile-vl-self-attention-blocks"], set()),
])
def test_historical_cli_flags(args, enabled):
    config = tyro.cli(FinetuneConfig, args=[
        "--base-model-path", "/unused/model", "--dataset-path", "/unused/data",
        "--embodiment-tag", "LIBERO_PANDA", *args, "--compile-mode", "default",
    ])
    assert {field for field in FIELDS[:-1] if getattr(config, field)} == enabled
    assert config.compile_mode == "default"


def test_config_defaults_and_launch_passthrough():
    ft = FinetuneConfig("/unused/model", "/unused/data", "LIBERO_PANDA")
    training = TrainingConfig()
    for field in FIELDS:
        assert getattr(ft, field) == getattr(training, field) == (
            "default" if field == "compile_mode" else False
        )
    ft.compile_dit_model = True
    ft.compile_vl_self_attention_blocks = True
    ft.compile_mode = "reduce-overhead"
    tree = ast.parse((REPO / "gr00t/experiment/launch_finetune.py").read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1
                   and ast.unparse(node.targets[0]) in {f"config.training.{f}" for f in FIELDS}]
    assert len(assignments) == len(FIELDS)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "<launch-fields>", "exec"),
         {"config": SimpleNamespace(training=training), "ft_config": ft})
    assert all(getattr(ft, f) == getattr(training, f) for f in FIELDS)


def test_default_off_and_environment_only():
    config = TrainingConfig()
    assert training_runtime.resolve_legacy_compile_request(config, "off") is None
    assert training_runtime.resolve_legacy_compile_request(config, "dit-model-vl") is None


def test_legacy_request_preserves_scope_and_mode():
    config = TrainingConfig(compile_dit_model=True, compile_vl_self_attention_blocks=True,
                            compile_mode="reduce-overhead")
    assert training_runtime.resolve_legacy_compile_request(config, "off") == {
        "blocks": False, "whole": True, "text_layers": False, "text_forwards": False,
        "vl_blocks": True, "mode": "reduce-overhead",
    }


def test_conflicting_environment_and_cli_rejected():
    with pytest.raises(ValueError, match="GR00T_COMPILE_VARIANT"):
        training_runtime.resolve_legacy_compile_request(
            TrainingConfig(compile_dit_model=True), "dit-model")


@pytest.mark.parametrize("fields", [
    {"compile_dit_blocks": True, "compile_dit_model": True},
    {"compile_text_layers": True, "compile_text_layer_forwards": True},
])
def test_mutually_exclusive_legacy_scopes(fields):
    with pytest.raises(ValueError, match="mutually exclusive"):
        training_runtime.resolve_legacy_compile_request(TrainingConfig(**fields), "off")


@pytest.mark.parametrize("variant,legacy", [("off", None), ("off", {"whole": True}),
                                           ("dit-model-vl", None)])
def test_experiment_calls_exactly_one_installer(monkeypatch, variant, legacy):
    """Execute the actual dispatch block without importing heavy experiment dependencies."""
    tree = ast.parse((REPO / "gr00t/experiment/experiment.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    init = next(n for n in run.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "compile_info" for t in n.targets))
    dispatch = run.body[run.body.index(init) + 1]
    legacy_install, env_install, write = Mock(return_value=["legacy-path"]), Mock(), Mock()
    env_install.return_value = {"variant": variant, "paths": ["env-path"]}
    monkeypatch.setitem(sys.modules, "gr00t.experiment.compilation",
                        SimpleNamespace(install_model_compile=legacy_install))
    monkeypatch.setitem(sys.modules, "gr00t.Replacement.model_compile",
                        SimpleNamespace(install=env_install, write_manifest=write))
    scope = {"compile_variant": variant, "legacy_compile_request": legacy,
             "model": object(), "output_dir": "/unused"}
    exec(compile(ast.Module(body=[init, dispatch], type_ignores=[]), "<dispatch>", "exec"), scope)
    if legacy:
        legacy_install.assert_called_once_with(scope["model"], **legacy)
        env_install.assert_not_called()
        assert scope["compile_info"]["paths"] == ["legacy-path"]
        assert scope["compile_info"]["variant"] == "legacy-cli"
    elif variant != "off":
        env_install.assert_called_once_with(scope["model"], variant)
        legacy_install.assert_not_called()
    else:
        legacy_install.assert_not_called()
        env_install.assert_not_called()
