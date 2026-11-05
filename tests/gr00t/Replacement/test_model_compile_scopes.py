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


SINGLE_SCOPES = {
    'dit-blocks': {'GR00T_COMPILE_DIT': 'blocks'}, 'text-layers': {'GR00T_COMPILE_TEXT': 'layers'},
    'text-forwards': {'GR00T_COMPILE_TEXT': 'forwards'}, 'vl': {'GR00T_COMPILE_VL': '1'},
    'dit-model': {'GR00T_COMPILE_DIT': 'model'},
    'dit-model+vl': {'GR00T_COMPILE_DIT': 'model', 'GR00T_COMPILE_VL': '1'},
}


@pytest.mark.parametrize('variant,count,boundary', [
    ('dit-blocks', 32, 'module'), ('text-layers', 16, 'module'),
    ('text-forwards', 16, 'forward'), ('vl', 4, 'module'),
    ('dit-model', 1, 'module'), ('dit-model+vl', 5, 'module'),
])
def test_scope_dispatch(model_setup, variant, count, boundary):
    model, calls = model_setup
    name = variant
    assert model_compile.resolve_variant(SINGLE_SCOPES[variant]) == name
    assert training_runtime.resolve_compile_variant(SINGLE_SCOPES[variant]) == name
    info = model_compile.install(model, variant)
    assert info['variant'] == name
    assert len(info['paths']) == count
    assert info['boundaries'] == dict.fromkeys(info['paths'], boundary)
    assert len(calls) == (1 if boundary == 'forward' else count)
    assert all(c[0] == boundary and c[2] == {'mode': 'max-autotune-no-cudagraphs', 'fullgraph': False}
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


def test_legacy_cli_removed():
    from gr00t.configs.finetune_config import FinetuneConfig
    from gr00t.configs.training.training_config import TrainingConfig
    assert not hasattr(training_runtime, 'resolve_legacy_compile_request')
    for cls in (FinetuneConfig, TrainingConfig):
        assert not [f for f in cls.__dataclass_fields__ if f.startswith('compile_')]
    with pytest.raises(ModuleNotFoundError):
        __import__('gr00t.experiment.compilation')


@pytest.mark.parametrize('bad', ['', 'text', 'dit-blocks-vl', 'off+vl', 'vl+vl',
                                 'dit-model+dit-blocks', 'text-layers+text-forwards'])
def test_invalid_variant(bad):
    with pytest.raises(ValueError):
        model_compile.expected_paths(bad)


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


def test_foreign_forward_rejected_for_text_layers_too(model_setup):
    model, calls = model_setup
    layer = model.backbone.model.language_model.layers[3]
    layer.forward = lambda x: x
    with pytest.raises(RuntimeError, match='forward'):
        model_compile.install(model, 'text-layers')
    with pytest.raises(RuntimeError, match='forward'):
        model_compile.install(model, 'dit-model+vl+text-layers')
    assert not calls and not hasattr(model, '_gr00t_optional_compile')
    layer.forward = types.MethodType(type(layer).forward, layer)
    model_compile.install(model, 'text-layers')


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


COMBOS = {
    'dit-model+vl': ({'GR00T_COMPILE_DIT': 'model', 'GR00T_COMPILE_VL': '1'}, 5),
    'dit-model+vl+text-forwards': ({'GR00T_COMPILE_DIT': 'model', 'GR00T_COMPILE_VL': '1',
                                    'GR00T_COMPILE_TEXT': 'forwards'}, 21),
    'dit-blocks+text-layers': ({'GR00T_COMPILE_DIT': 'blocks', 'GR00T_COMPILE_TEXT': 'layers'}, 48),
    'dit-blocks+vl+text-forwards': ({'GR00T_COMPILE_DIT': 'blocks', 'GR00T_COMPILE_VL': '1',
                                     'GR00T_COMPILE_TEXT': 'forwards'}, 52),
}


@pytest.mark.parametrize('name', COMBOS)
def test_stacked_scopes_install_exact_targets(model_setup, name):
    model, calls = model_setup
    env, count = COMBOS[name]
    assert model_compile.resolve_variant(env) == name
    assert training_runtime.resolve_compile_variant(env) == name
    scopes = model_compile.resolve_scopes(env)
    expected = {}
    if scopes.vl:
        expected.update({f'action_head.vl_self_attention.transformer_blocks.{i}': 'module'
                         for i in range(4)})
    if scopes.dit == 'model':
        expected['action_head.model'] = 'module'
    if scopes.dit == 'blocks':
        expected.update({f'action_head.model.transformer_blocks.{i}': 'module' for i in range(32)})
    if scopes.text != 'off':
        expected.update({f'backbone.model.language_model.layers.{i}':
                         'forward' if scopes.text == 'forwards' else 'module' for i in range(16)})
    info = model_compile.install(model, scopes)
    assert len(info['paths']) == count
    assert info['boundaries'] == expected and info['paths'] == list(expected)
    assert info['variant'] == name and info['scopes'] == scopes.as_dict()
    json.dumps(info)
    module_targets = {id(c[1]) for c in calls if c[0] == 'module'}
    assert module_targets == {id(model.get_submodule(p)) for p, b in expected.items() if b == 'module'}
    assert sum(c[0] == 'forward' for c in calls) == (1 if scopes.text == 'forwards' else 0)
    for path, boundary in expected.items():
        assert model_compile._callable_present(model.get_submodule(path), boundary)
    assert model_compile.install(model, name) == info


def test_full_stack_is_21_paths_matching_legacy_combination():
    paths = model_compile.expected_paths(model_compile.CompileScopes('model', True, 'forwards'))
    assert len(paths) == 21 and len(set(paths)) == 21
    assert set(paths) == (
        {'action_head.model'}
        | {f'action_head.vl_self_attention.transformer_blocks.{i}' for i in range(4)}
        | {f'backbone.model.language_model.layers.{i}' for i in range(16)})


def test_stacked_preflight_failure_does_not_mutate(model_setup):
    model, calls = model_setup
    del model.backbone.model.language_model.layers[-1]
    with pytest.raises(ValueError):
        model_compile.install(model, 'dit-model+vl+text-forwards')
    assert calls == [] and not hasattr(model, '_gr00t_optional_compile')


@pytest.mark.parametrize('env', [
    {'GR00T_COMPILE_DIT': 'both'}, {'GR00T_COMPILE_VL': '2'}, {'GR00T_COMPILE_TEXT': 'all'},
])
def test_invalid_scope_environment(env):
    with pytest.raises(ValueError):
        model_compile.resolve_scopes(env)


def test_manifest_round_trip_is_json_serializable(model_setup, tmp_path, monkeypatch):
    model, _ = model_setup
    monkeypatch.setenv('RANK', '0')
    info = model_compile.install(model, 'dit-model+vl+text-forwards')
    model_compile.write_manifest(tmp_path, info)
    report = json.loads((tmp_path / 'compilation/rank-00.json').read_text())
    assert report['variant'] == 'dit-model+vl+text-forwards'
    assert report['scopes'] == {'dit': 'model', 'vl': '1', 'text': 'forwards'}


# Fixed snapshot of the former legacy-CLI combinations (dit, text, vl) -> (path count, forward count).
LEGACY_SNAPSHOT = {
    ('model', 'off', False): (1, 0), ('blocks', 'off', False): (32, 0),
    ('off', 'layers', False): (16, 0), ('off', 'forwards', False): (16, 1),
    ('off', 'off', True): (4, 0), ('model', 'off', True): (5, 0), ('blocks', 'off', True): (36, 0),
    ('model', 'layers', False): (17, 0), ('model', 'forwards', False): (17, 1),
    ('blocks', 'layers', False): (48, 0), ('blocks', 'forwards', False): (48, 1),
    ('off', 'layers', True): (20, 0), ('off', 'forwards', True): (20, 1),
    ('model', 'layers', True): (21, 0), ('model', 'forwards', True): (21, 1),
    ('blocks', 'layers', True): (52, 0), ('blocks', 'forwards', True): (52, 1),
}


@pytest.mark.parametrize('combo', LEGACY_SNAPSHOT)
def test_scopes_match_former_legacy_combinations(model_setup, combo):
    dit, text, vl = combo
    count, forwards = LEGACY_SNAPSHOT[combo]
    model, calls = model_setup
    scopes = model_compile.CompileScopes(dit, vl, text)
    expected = model_compile.expected_boundaries(scopes)
    assert len(expected) == count and len(set(expected)) == count
    assert sum(b == 'forward' for b in expected.values()) == (16 if forwards else 0)
    info = model_compile.install(model, scopes)
    assert info['paths'] == list(expected)
    assert sum(c[0] == 'forward' for c in calls) == forwards
    assert sum(c[0] == 'module' for c in calls) == count - (16 if forwards else 0)


@pytest.mark.parametrize('variant', ['off', 'dit-model+vl+text-forwards'])
def test_experiment_dispatch_calls_only_replacement_installer(monkeypatch, variant):
    """Run the actual dispatch block of experiment.run without importing heavy dependencies."""
    import ast
    from pathlib import Path
    from unittest.mock import Mock

    repo = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo / 'gr00t/experiment/experiment.py').read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    init = next(n for n in run.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'compile_info' for t in n.targets))
    dispatch = run.body[run.body.index(init) + 1]
    install, write = Mock(return_value={'variant': variant, 'paths': ['p']}), Mock()
    monkeypatch.setitem(sys.modules, 'gr00t.Replacement.model_compile',
                        SimpleNamespace(install=install, write_manifest=write))
    scope = {'compile_variant': variant, 'model': object(), 'output_dir': '/unused'}
    exec(compile(ast.Module(body=[init, dispatch], type_ignores=[]), '<dispatch>', 'exec'), scope)
    if variant == 'off':
        install.assert_not_called()
        assert scope['compile_info'] == {'variant': 'off', 'paths': []}
    else:
        install.assert_called_once_with(scope['model'], variant)
        write.assert_called_once()
