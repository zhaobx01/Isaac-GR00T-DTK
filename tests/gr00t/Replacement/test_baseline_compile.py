# SPDX-License-Identifier: Apache-2.0
"""Baseline compiler configuration and process-global lifecycle, on CPU only."""
from functools import wraps
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gr00t.Replacement import model_compile as extension


@pytest.fixture
def setup_baseline(monkeypatch):
    import transformers.integrations.flash_attention as flash
    import transformers.modeling_utils as modeling

    class Block(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

    class Registry(dict):
        def register(self, key, value):
            self[key] = value

    def original(value):
        return value + 2

    registry = Registry(flash_attention_2=original, flash_attention_3=original)
    monkeypatch.setattr(flash, 'flash_attention_forward', original)
    monkeypatch.setattr(modeling, 'ALL_ATTENTION_FUNCTIONS', registry)
    monkeypatch.setattr(extension, '_BASELINE_PROCESS', None, raising=False)
    calls = []

    def compiler(cls, **kwargs):
        calls.append((cls, kwargs))
        old = cls.__call__
        @wraps(old)
        def compiled(self, *args, **kw):
            return old(self, *args, **kw)
        compiled._torchdynamo_orig_callable = old
        cls.__call__ = compiled
        return cls

    monkeypatch.setattr(torch, 'compile', compiler)
    monkeypatch.delenv('GR00T_COMPILE_VISION_BLOCKS', raising=False)
    monkeypatch.delenv('GR00T_FLASH_ATTN_DYNAMO_DISABLE', raising=False)

    def backbone(block_type=Block):
        model = nn.Module()
        model.visual = nn.Module()
        model.visual.blocks = nn.ModuleList([block_type(), block_type()])
        return SimpleNamespace(model=model)

    return SimpleNamespace(backbone=backbone, Block=Block, calls=calls,
                           flash=flash, registry=registry, original=original)


def test_defaults_install_and_repeat_is_idempotent(setup_baseline):
    s = setup_baseline
    backbone = s.backbone()
    info = extension.install_baseline_overrides(backbone)
    assert info['requested'] == {'vision_blocks': True, 'flash_attention_dynamo_disable': True}
    assert info['installed'] == info['requested']
    assert info['installation_only']
    assert s.calls == [(s.Block, {'mode': 'max-autotune-no-cudagraphs'})]
    assert s.flash.flash_attention_forward._torchdynamo_disable
    assert s.registry['flash_attention_2'] is s.flash.flash_attention_forward
    assert s.registry['flash_attention_3'] is s.flash.flash_attention_forward
    assert extension.install_baseline_overrides(backbone) == info
    assert extension.install_baseline_overrides(s.backbone()) == info
    assert len(s.calls) == 1


@pytest.mark.parametrize('vision,flash', [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')])
def test_independent_switches(setup_baseline, vision, flash):
    s = setup_baseline
    info = extension.install_baseline_overrides(s.backbone(), {
        'GR00T_COMPILE_VISION_BLOCKS': vision, 'GR00T_FLASH_ATTN_DYNAMO_DISABLE': flash,
        'GR00T_COMPILE_MODE': 'reduce-overhead'})
    assert bool(s.calls) == (vision == '1')
    assert (s.flash.flash_attention_forward is not s.original) == (flash == '1')
    assert info['installed']['vision_blocks'] == (vision == '1')
    assert info['installed']['flash_attention_dynamo_disable'] == (flash == '1')


@pytest.mark.parametrize('name', ['GR00T_COMPILE_VISION_BLOCKS',
                                 'GR00T_FLASH_ATTN_DYNAMO_DISABLE'])
@pytest.mark.parametrize('bad', ['', 'true', '2'])
def test_invalid_flags_do_not_mutate(setup_baseline, name, bad):
    s = setup_baseline
    with pytest.raises(ValueError, match=name):
        extension.install_baseline_overrides(s.backbone(), {name: bad})
    assert not s.calls and s.flash.flash_attention_forward is s.original
    assert extension._BASELINE_PROCESS is None


def test_changed_config_requires_restart(setup_baseline):
    s = setup_baseline
    extension.install_baseline_overrides(s.backbone())
    with pytest.raises(RuntimeError, match='restart'):
        extension.install_baseline_overrides(s.backbone(), {'GR00T_COMPILE_VISION_BLOCKS': '0'})


def test_disabled_configuration_is_also_fixed_for_process(setup_baseline):
    s = setup_baseline
    extension.install_baseline_overrides(s.backbone(), {
        'GR00T_COMPILE_VISION_BLOCKS': '0', 'GR00T_FLASH_ATTN_DYNAMO_DISABLE': '0'})
    with pytest.raises(RuntimeError, match='restart'):
        extension.install_baseline_overrides(s.backbone())


def test_baseline_manifest_records_both_flags_without_optional_paths(setup_baseline, tmp_path,
                                                                  monkeypatch):
    s = setup_baseline
    info = extension.install_baseline_overrides(s.backbone())
    monkeypatch.setenv('RANK', '2')
    extension.write_baseline_manifest(tmp_path, info)
    payload = json.loads((tmp_path / 'compilation/baseline-rank-02.json').read_text())
    assert payload['rank'] == 2 and payload['requested'] == info['installed']
    assert 'paths' not in payload


def test_training_records_baseline_for_off_and_cli_paths(setup_baseline, tmp_path, monkeypatch):
    s = setup_baseline
    model = nn.Module()
    model.backbone = s.backbone()
    info = extension.install_baseline_overrides(model.backbone)
    monkeypatch.setenv('RANK', '0')
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / 'gr00t/experiment/experiment.py').read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    assignment = next(node for node in run.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'baseline_compile_info'
                              for t in node.targets))
    index = run.body.index(assignment)
    # Execute the unconditional recording block, independently of optional compile dispatch.
    nodes = run.body[index - 1:index + 2]
    scope = {'model': model, 'output_dir': tmp_path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<baseline-record>', 'exec'), scope)
    assert scope['baseline_compile_info'] == info
    payload = json.loads((tmp_path / 'compilation/baseline-rank-00.json').read_text())
    assert payload['installed'] == info['installed']
    metrics = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'StepMetricsCallback')
    run_info = next(kw.value for kw in metrics.keywords if kw.arg == 'run_info')
    fields = {key.value: value for key, value in zip(run_info.keys, run_info.values)}
    assert ast.unparse(fields['baseline_compile']) == 'baseline_compile_info'


def test_new_visual_class_is_installed_with_same_process_configuration(setup_baseline):
    s = setup_baseline
    extension.install_baseline_overrides(s.backbone())
    class NewBlock(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
    extension.install_baseline_overrides(s.backbone(NewBlock))
    assert [call[0] for call in s.calls] == [s.Block, NewBlock]


def test_custom_registry_entries_and_missing_fa3_preserved(setup_baseline):
    s = setup_baseline
    custom = lambda value: value
    s.registry['flash_attention_2'] = custom
    del s.registry['flash_attention_3']
    extension.install_baseline_overrides(s.backbone())
    assert s.registry['flash_attention_2'] is custom
    assert 'flash_attention_3' not in s.registry


@pytest.mark.parametrize('local_original,global_original', [(True, True), (True, False),
                                                         (False, True)])
def test_real_attention_interface_preserves_each_mapping_layer(setup_baseline, monkeypatch,
                                                            local_original, global_original):
    from transformers.modeling_utils import AttentionInterface
    import transformers.modeling_utils as modeling
    s = setup_baseline
    custom = lambda x: x
    class Registry(AttentionInterface):
        _global_mapping = {'flash_attention_2': s.original if global_original else custom}
    registry = Registry()
    registry['flash_attention_2'] = s.original if local_original else custom
    monkeypatch.setattr(modeling, 'ALL_ATTENTION_FUNCTIONS', registry)
    extension.install_baseline_overrides(s.backbone())
    wrapped = s.flash.flash_attention_forward
    assert registry._local_mapping['flash_attention_2'] is (
        wrapped if local_original else custom)
    assert Registry._global_mapping['flash_attention_2'] is (
        wrapped if global_original else custom)
    assert registry['flash_attention_2'] is (wrapped if local_original else custom)
    extension.install_baseline_overrides(s.backbone())


def test_foreign_class_wrapper_rejected_before_flash_mutation(setup_baseline):
    s = setup_baseline
    s.Block.__call__ = torch._dynamo.optimize('eager')(s.Block.__call__)
    with pytest.raises(RuntimeError, match='conflict|wrapper'):
        extension.install_baseline_overrides(s.backbone())
    assert not s.calls and s.flash.flash_attention_forward is s.original


def test_foreign_descendant_class_wrapper_rejected_atomically(setup_baseline):
    s = setup_baseline
    class Child(nn.Identity):
        pass
    torch._dynamo.optimize('eager')(Child)
    backbone = s.backbone()
    backbone.model.visual.blocks[0].child = Child()
    with pytest.raises(RuntimeError, match='wrapper'):
        extension.install_baseline_overrides(backbone)
    assert not s.calls and s.flash.flash_attention_forward is s.original


def test_foreign_flash_wrapper_rejected_before_visual_mutation(setup_baseline):
    s = setup_baseline
    s.flash.flash_attention_forward = torch._dynamo.disable(s.original)
    with pytest.raises(RuntimeError, match='conflict|wrapper'):
        extension.install_baseline_overrides(s.backbone())
    assert not s.calls


def test_partial_failure_is_poisoned(setup_baseline):
    s = setup_baseline
    def fail(*args):
        raise RuntimeError('registry failure')
    s.registry.register = fail
    with pytest.raises(RuntimeError, match='registry failure'):
        extension.install_baseline_overrides(s.backbone())
    assert s.calls
    with pytest.raises(RuntimeError, match='restart'):
        extension.install_baseline_overrides(s.backbone())


def test_tampered_owned_wrapper_rejected(setup_baseline):
    s = setup_baseline
    extension.install_baseline_overrides(s.backbone())
    s.registry['flash_attention_2'] = lambda x: x
    with pytest.raises(RuntimeError, match='conflict|wrapper'):
        extension.install_baseline_overrides(s.backbone())


def test_real_cpu_compile_preserves_values_and_parameters(setup_baseline, monkeypatch):
    s = setup_baseline
    # Use Dynamo's actual eager backend, without executing device kernels or Inductor.
    monkeypatch.setattr(torch, 'compile', lambda cls, **kw: torch._dynamo.optimize('eager')(cls))
    backbone = s.backbone()
    block = backbone.model.visual.blocks[0]
    x = torch.randn(2, 2)
    expected = block(x)
    keys = list(backbone.model.state_dict())
    parameters = [(id(p), p.dtype, p.requires_grad) for p in backbone.model.parameters()]
    extension.install_baseline_overrides(backbone)
    torch.testing.assert_close(block(x), expected)
    torch.testing.assert_close(s.registry['flash_attention_2'](x), x + 2)
    assert list(backbone.model.state_dict()) == keys
    assert [(id(p), p.dtype, p.requires_grad) for p in backbone.model.parameters()] == parameters
    torch._dynamo.reset()


def test_direct_backbone_construction_delegates_to_replacement(setup_baseline, monkeypatch):
    from gr00t.model.modules import qwen3_backbone as source
    s = setup_baseline
    model = s.backbone().model
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
    monkeypatch.setattr(source, 'Qwen3VLForConditionalGeneration',
                        SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    monkeypatch.setattr(source.Qwen3Backbone, '_reset_rotary_inv_freq', lambda self: None)
    monkeypatch.setattr(source.Qwen3Backbone, '_apply_cpu_position_preprocessing', lambda self: None)
    backbone = source.Qwen3Backbone(select_layer=1)
    assert backbone._gr00t_baseline_compile['installed']['vision_blocks']
    assert backbone._gr00t_baseline_compile['installed']['flash_attention_dynamo_disable']
    assert len(s.calls) == 1
