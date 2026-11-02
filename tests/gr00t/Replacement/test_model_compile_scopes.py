# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of Replacement scopes, preflight, and callable boundaries."""
import copy
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gr00t.Replacement import model_compile
from gr00t.experiment import training_runtime


@pytest.fixture
def model_setup(monkeypatch):
    class Block(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(32)])

    class TextLayer(Block):
        pass

    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen
    monkeypatch.setattr(qwen, 'Qwen3VLTextDecoderLayer', TextLayer)
    monkeypatch.setitem(sys.modules, 'gr00t.model.modules.dit',
                        SimpleNamespace(BasicTransformerBlock=Block, DiT=DiT))
    model = nn.Module()
    model.action_head = nn.Module()
    model.action_head.model = DiT()
    model.action_head.vl_self_attention = nn.Module()
    model.action_head.vl_self_attention.transformer_blocks = nn.ModuleList(
        [Block() for _ in range(4)])
    model.backbone = nn.Module()
    model.backbone.model = nn.Module()
    model.backbone.model.language_model = nn.Module()
    model.backbone.model.language_model.layers = nn.ModuleList([TextLayer() for _ in range(16)])
    calls = []

    def module_compile(target, **kwargs):
        calls.append(('module', target, kwargs))
        target._compiled_call_impl = target.forward

    def forward_compile(fn, **kwargs):
        calls.append(('forward', fn, kwargs))
        def compiled(self, *args, **kw):
            return fn(self, *args, **kw)
        return compiled

    monkeypatch.setattr(nn.Module, 'compile', module_compile)
    monkeypatch.setattr(torch, 'compile', forward_compile)
    monkeypatch.delenv('GR00T_COMPILE_MODE', raising=False)
    monkeypatch.delenv('GR00T_COMPILE_TRACE_MARKERS', raising=False)
    return model, calls


@pytest.mark.parametrize('variant,count,boundary', [
    ('dit-blocks', 32, 'module'), ('text-layers', 16, 'module'),
    ('text-forwards', 16, 'forward'), ('vl', 4, 'module'),
    ('dit-model', 1, 'module'), ('dit-model-vl', 5, 'module'),
])
def test_scope_dispatch(model_setup, variant, count, boundary):
    model, calls = model_setup
    assert model_compile.resolve_variant({'GR00T_COMPILE_VARIANT': variant}) == variant
    assert training_runtime.resolve_compile_variant({'GR00T_COMPILE_VARIANT': variant}) == variant
    info = model_compile.install(model, variant)
    assert len(info['paths']) == count
    assert info['boundaries'] == dict.fromkeys(info['paths'], boundary)
    assert len(calls) == (1 if boundary == 'forward' else count)
    assert all(c[0] == boundary and c[2] == {'mode': 'default', 'fullgraph': False}
               for c in calls)
    assert not info['failed'] and info['installation_only']
    assert model_compile.install(model, variant) == info
    assert len(calls) == (1 if boundary == 'forward' else count)
    if variant == 'dit-blocks':
        assert info['paths'] == [f'action_head.model.transformer_blocks.{i}' for i in range(32)]
    elif variant.startswith('text-'):
        assert info['paths'] == [f'backbone.model.language_model.layers.{i}' for i in range(16)]
    if boundary == 'forward':
        layers = model.backbone.model.language_model.layers
        assert len({layer.forward.__func__ for layer in layers}) == 1
        assert all(layer.forward.__self__ is layer for layer in layers)


@pytest.mark.parametrize('variant', ['dit-blocks', 'text-layers', 'text-forwards'])
def test_new_scopes_conflict_with_legacy_cli(variant):
    with pytest.raises(ValueError, match='GR00T_COMPILE_VARIANT'):
        training_runtime.resolve_legacy_compile_request(
            SimpleNamespace(compile_dit_blocks=True, compile_dit_model=False,
                            compile_text_layers=False, compile_text_layer_forwards=False,
                            compile_vl_self_attention_blocks=False, compile_mode='default'),
            variant)


@pytest.mark.parametrize('bad', ['', 'text', 'dit-blocks-vl'])
def test_invalid_variant(bad):
    with pytest.raises(ValueError):
        model_compile.expected_paths(bad)
    with pytest.raises(ValueError):
        training_runtime.resolve_compile_variant({'GR00T_COMPILE_VARIANT': bad})


@pytest.mark.parametrize('variant', ['dit-blocks', 'text-layers', 'text-forwards'])
@pytest.mark.parametrize('problem', ['count', 'duplicate', 'type', 'wrapper'])
def test_preflight_does_not_mutate_on_failure(model_setup, variant, problem):
    model, calls = model_setup
    layers = (model.action_head.model.transformer_blocks if variant == 'dit-blocks'
              else model.backbone.model.language_model.layers)
    if problem == 'count':
        del layers[-1]
    elif problem == 'duplicate':
        layers[-1] = layers[0]
    elif problem == 'type':
        layers[-1] = nn.Identity()
    else:
        layers[-1]._compiled_call_impl = lambda x: x
    with pytest.raises((ValueError, RuntimeError)):
        model_compile.install(model, variant)
    assert calls == []
    assert not hasattr(model, '_gr00t_optional_compile')


def test_foreign_forward_rejected_but_original_bound_method_allowed(model_setup):
    model, calls = model_setup
    layer = model.backbone.model.language_model.layers[0]
    layer.forward = lambda x: x
    with pytest.raises(RuntimeError, match='forward'):
        model_compile.install(model, 'text-forwards')
    assert not calls
    layer.forward = types.MethodType(type(layer).forward, layer)
    model_compile.install(model, 'text-forwards')


def test_forward_wrapper_blocks_other_scope_installation(model_setup):
    model, _ = model_setup
    model_compile.install(model, 'text-forwards')
    delattr(model, '_gr00t_optional_compile')
    with pytest.raises(RuntimeError, match='forward|wrapper'):
        model_compile.install(model, 'text-layers')


@pytest.mark.parametrize('position', ['ancestor', 'descendant'])
def test_standard_compiled_forward_overlap_rejected_atomically(model_setup, position):
    model, calls = model_setup
    text = model.backbone.model.language_model
    if position == 'ancestor':
        target = text
    else:
        text.layers[-1].child = nn.Identity()
        target = text.layers[-1].child
    target.forward = torch._dynamo.optimize('eager')(target.forward)
    with pytest.raises(RuntimeError, match='Overlapping'):
        model_compile.install(model, 'text-forwards')
    assert not calls
    assert not hasattr(model, '_gr00t_optional_compile')


def test_partial_failure_requires_restart(model_setup, monkeypatch):
    model, _ = model_setup
    installed = []
    def fail(target, **kwargs):
        installed.append(target)
        if len(installed) == 2:
            raise RuntimeError('compiler failure')
        target._compiled_call_impl = target.forward
    monkeypatch.setattr(nn.Module, 'compile', fail)
    with pytest.raises(RuntimeError, match='compiler failure'):
        model_compile.install(model, 'dit-blocks')
    assert model._gr00t_optional_compile['failed']
    assert model.action_head.model.transformer_blocks[0]._compiled_call_impl is not None
    with pytest.raises(RuntimeError, match='restart'):
        model_compile.install(model, 'dit-blocks')
    assert len(installed) == 2


@pytest.mark.parametrize('variant', ['dit-blocks', 'text-layers', 'text-forwards'])
def test_configuration_change_requires_restart(model_setup, variant):
    model, _ = model_setup
    model_compile.install(model, variant)
    with pytest.raises(RuntimeError, match='restart'):
        model_compile.install(model, 'off')
    with pytest.raises(RuntimeError, match='restart'):
        model_compile.install(model, variant, mode='reduce-overhead')


@pytest.mark.parametrize('variant', ['text-layers', 'text-forwards'])
def test_markers_callable_deepcopy_and_evidence(model_setup, monkeypatch, tmp_path, variant):
    model, _ = model_setup
    layers = model.backbone.model.language_model.layers
    inputs = torch.ones(1, 2)
    expected = layers[0](inputs)
    keys = list(model.state_dict())
    monkeypatch.setenv('GR00T_COMPILE_TRACE_MARKERS', '1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.delenv('GR00T_CAPTURE_DEVICE_MAPPING', raising=False)
    info = model_compile.install(model, variant)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        torch.testing.assert_close(layers[0](inputs), expected)
    assert any(event.name == f"gr00t_compile::{info['paths'][0]}" for event in prof.events())
    assert list(model.state_dict()) == keys
    cloned = copy.deepcopy(model)
    cloned_layer = cloned.backbone.model.language_model.layers[0]
    if variant == 'text-forwards':
        assert cloned_layer.forward.__self__ is cloned_layer
    torch.testing.assert_close(cloned_layer(inputs), expected)
    if variant == 'text-forwards':
        with torch.no_grad():
            cloned_layer.weight.zero_()
            cloned_layer.bias.zero_()
        torch.testing.assert_close(cloned_layer(inputs), torch.zeros_like(expected))
        torch.testing.assert_close(layers[0](inputs), expected)
    callback = model_compile.evidence_callback(model, info)
    callback.on_train_end(SimpleNamespace(output_dir=tmp_path),
                          SimpleNamespace(global_step=40), None)
    report = json.loads((tmp_path / 'compilation/rank-00.json').read_text())
    assert report['runtime']['compiled_callables_present']
    assert report['installation_only']
    assert report['boundaries'] == info['boundaries']
    if variant == 'text-forwards':
        layers[0].forward = types.MethodType(type(layers[0]).forward, layers[0])
    else:
        layers[0]._compiled_call_impl = None
    callback.on_train_end(SimpleNamespace(output_dir=tmp_path),
                          SimpleNamespace(global_step=40), None)
    report = json.loads((tmp_path / 'compilation/rank-00.json').read_text())
    assert not report['runtime']['compiled_callables_present']


@pytest.mark.parametrize('markers', [False, True])
@pytest.mark.parametrize('variant', ['text-layers', 'text-forwards'])
def test_real_text_model_output_capture_cpu(monkeypatch, variant, markers):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    config = Qwen3VLTextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=16,
                              num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                              vocab_size=32,
                              rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]})
    config._attn_implementation = 'eager'
    text = Qwen3VLTextModel(config).eval()
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.model = nn.Module()
    model.backbone.model.language_model = text
    original = torch.compile
    monkeypatch.setattr(torch, 'compile',
                        lambda fn, **kwargs: original(fn, backend='eager', **kwargs))
    monkeypatch.setenv('GR00T_COMPILE_TRACE_MARKERS', '1' if markers else '0')
    monkeypatch.delenv('GR00T_COMPILE_MODE', raising=False)
    inputs = torch.tensor([[1, 2, 3]])
    keys = list(model.state_dict())
    use_cache = text.config.use_cache
    parameters = [(id(p), p.dtype, p.requires_grad) for p in model.parameters()]
    graphs_before = counters['stats']['unique_graphs']
    with torch.no_grad():
        expected = text(inputs, output_hidden_states=True)
        info = model_compile.install(model, variant)
        actual = text(inputs, output_hidden_states=True)
    assert text.config.use_cache == use_cache
    assert [(id(p), p.dtype, p.requires_grad) for p in model.parameters()] == parameters
    assert counters['stats']['unique_graphs'] > graphs_before
    assert all(model_compile._callable_present(model.get_submodule(path), boundary)
               for path, boundary in info['boundaries'].items())
    assert len(actual.hidden_states) == len(expected.hidden_states) == 17
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    for left, right in zip(actual.hidden_states, expected.hidden_states):
        torch.testing.assert_close(left, right)
    assert list(model.state_dict()) == keys
    torch._dynamo.reset()
