# SPDX-License-Identifier: Apache-2.0
"""Opt-in DiT/VL/text compilation; preserve module identity and model behavior."""
from __future__ import annotations

import json
import logging
import os
import types
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import RLock

_STATE = '_gr00t_optional_compile'
_FORWARD_TAG = '_gr00t_compiled_forward'
_BASELINE_PROCESS = None
_BASELINE_LOCK = RLock()
_BASELINE_MODE = 'max-autotune-no-cudagraphs'
_BASELINE_FLAGS = {
    'vision_blocks': 'GR00T_COMPILE_VISION_BLOCKS',
    'flash_attention_dynamo_disable': 'GR00T_FLASH_ATTN_DYNAMO_DISABLE',
}


def resolve_baseline_overrides(environ=None):
    """Resolve the two baseline switches independently of optional compilation."""
    environ = os.environ if environ is None else environ
    config = {}
    for key, name in _BASELINE_FLAGS.items():
        value = environ.get(name, '1')
        if value not in {'0', '1'}:
            raise ValueError(f'{name} must be 0 or 1')
        config[key] = value == '1'
    return config


def _flash_registry_layers(registry):
    """Transformers reads local overrides before global entries; register writes global."""
    local = getattr(registry, '_local_mapping', None)
    global_mapping = getattr(type(registry), '_global_mapping', None)
    if local is not None or global_mapping is not None:
        if not isinstance(local, MutableMapping) or not isinstance(global_mapping, MutableMapping):
            raise ValueError('Unsupported FlashAttention registry mapping layout')
        return [('local', local), ('global', global_mapping)]
    return [('registry', registry)]


def install_baseline_overrides(backbone, environ=None):
    """Install baseline process-global wrappers before the first backbone call.

    This preserves the previous class-level vision boundary and FlashAttention registry
    scope. Configuration changes and partial failures require a new process; no global
    wrapper is uninstalled. All preflight checks happen before the first mutation.
    """
    import torch
    from transformers.integrations import flash_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    global _BASELINE_PROCESS
    config = resolve_baseline_overrides(environ)
    with _BASELINE_LOCK:
        state = _BASELINE_PROCESS
        if state is not None and (state['status'] != 'ready' or state['config'] != config):
            raise RuntimeError('Baseline compile configuration changed or failed; restart the process')

        visual = getattr(backbone.model, 'visual', None)
        blocks = list(getattr(visual, 'blocks', ()))
        if config['vision_blocks'] and (
            not blocks or not all(isinstance(block, torch.nn.Module) for block in blocks)
        ):
            raise ValueError('Baseline vision compilation requires visual Block instances')
        classes = list(dict.fromkeys(type(block) for block in blocks))
        owned_classes = state['classes'] if state is not None else {}
        for cls in classes:
            call = cls.__call__
            if cls in owned_classes:
                if call is not owned_classes[cls]:
                    raise RuntimeError('Baseline visual class wrapper conflict; restart the process')
            elif hasattr(call, '_torchdynamo_orig_callable'):
                raise RuntimeError('Foreign visual class compile wrapper conflict; restart the process')
        for block in blocks:
            for module in block.modules():
                if (getattr(module, '_compiled_call_impl', None) is not None
                        or (type(module) not in owned_classes
                            and hasattr(type(module).__call__, '_torchdynamo_orig_callable'))
                        or hasattr(module.forward, '_torchdynamo_orig_callable')
                        or getattr(module.forward, _FORWARD_TAG, False)):
                    raise RuntimeError('Foreign visual instance wrapper conflict; restart the process')

        original = flash_attention.flash_attention_forward
        layers = _flash_registry_layers(ALL_ATTENTION_FUNCTIONS)
        owned_flash = state['flash'] if state is not None else None
        if owned_flash is not None:
            if (flash_attention is not state['flash_module'] or original is not owned_flash
                    or ALL_ATTENTION_FUNCTIONS is not state['registry']):
                raise RuntimeError('Baseline FlashAttention wrapper conflict; restart the process')
            if any(current_kind != previous_kind or current_map is not previous_map
                   for (current_kind, current_map), (previous_kind, previous_map)
                   in zip(layers, state['registry_layers'])) or len(layers) != len(
                       state['registry_layers']):
                raise RuntimeError('Baseline FlashAttention registry layers changed; restart')
            for _, mapping, name in state['registry_owned']:
                if mapping.get(name) is not owned_flash:
                    raise RuntimeError('Baseline FlashAttention registry conflict; restart the process')
            original = state['original_flash']
        elif (getattr(original, '_torchdynamo_disable', False)
              or hasattr(original, '_torchdynamo_orig_callable')
              or hasattr(original, '__wrapped__')):
            raise RuntimeError('Foreign FlashAttention wrapper conflict; restart the process')
        updates = []
        if config['flash_attention_dynamo_disable']:
            if not callable(original):
                raise ValueError('FlashAttention integration function is not callable')
            for kind, mapping in layers:
                for name in ('flash_attention_2', 'flash_attention_3'):
                    if mapping.get(name) is original:
                        if kind != 'local' and not callable(
                            getattr(ALL_ATTENTION_FUNCTIONS, 'register', None)
                        ):
                            raise ValueError('FlashAttention registry has no callable register')
                        updates.append((kind, mapping, name))

        if state is None:
            state = {'config': dict(config), 'status': 'installing', 'classes': {},
                     'flash': None, 'original_flash': original, 'registry_owned': [],
                     'flash_module': flash_attention, 'registry': ALL_ATTENTION_FUNCTIONS,
                     'registry_layers': layers}
            _BASELINE_PROCESS = state
        state['status'] = 'installing'
        try:
            if config['vision_blocks']:
                for cls in classes:
                    if cls not in state['classes']:
                        torch.compile(cls, mode=_BASELINE_MODE)
                        if not hasattr(cls.__call__, '_torchdynamo_orig_callable'):
                            raise RuntimeError('Vision compile wrapper was not installed; restart')
                        state['classes'][cls] = cls.__call__
            if config['flash_attention_dynamo_disable']:
                if state['flash'] is None:
                    state['flash'] = torch._dynamo.disable(original)
                    flash_attention.flash_attention_forward = state['flash']
                for kind, mapping, name in updates:
                    if kind == 'local':
                        ALL_ATTENTION_FUNCTIONS[name] = state['flash']
                    else:
                        ALL_ATTENTION_FUNCTIONS.register(name, state['flash'])
                    if mapping.get(name) is not state['flash']:
                        raise RuntimeError('FlashAttention registry update did not take effect; restart')
                    state['registry_owned'].append((kind, mapping, name))
            info = {
                'requested': dict(config), 'installed': dict(config), 'status': 'ready',
                'vision_mode': _BASELINE_MODE,
                'vision_classes': [f'{cls.__module__}.{cls.__qualname__}' for cls in classes]
                if config['vision_blocks'] else [],
                'installation_only': True,
            }
            backbone._gr00t_baseline_compile = info
            state['status'] = 'ready'
        except BaseException:
            state['status'] = 'failed'
            raise
        logging.info('Baseline compiler wrappers configured: %s; runtime compilation unverified', info)
        return dict(info)


def write_baseline_manifest(output_dir, info):
    """Record baseline installation separately, including variant=off and legacy CLI."""
    rank = int(os.environ.get('RANK', '0'))
    path = Path(output_dir) / 'compilation' / f'baseline-rank-{rank:02d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(info, rank=rank), indent=2) + '\n')
    temporary.replace(path)


DIT_MODES = ('off', 'model', 'blocks')
TEXT_MODES = ('off', 'layers', 'forwards')
_SCOPE_ENV = {'dit': 'GR00T_COMPILE_DIT', 'vl': 'GR00T_COMPILE_VL', 'text': 'GR00T_COMPILE_TEXT'}


@dataclass(frozen=True)
class CompileScopes:
    """Independent compile scopes; mutually exclusive choices share one field.

    Process-internal only: serialize ``name`` or ``as_dict()``, never the instance.
    """
    dit: str = 'off'
    vl: bool = False
    text: str = 'off'

    def __post_init__(self):
        if self.dit not in DIT_MODES:
            raise ValueError(f'{_SCOPE_ENV["dit"]} must be one of {DIT_MODES}, got {self.dit!r}')
        if self.text not in TEXT_MODES:
            raise ValueError(f'{_SCOPE_ENV["text"]} must be one of {TEXT_MODES}, got {self.text!r}')
        if not isinstance(self.vl, bool):
            raise ValueError(f'{_SCOPE_ENV["vl"]} must be 0 or 1')

    @property
    def enabled(self):
        return self != CompileScopes()

    @property
    def name(self):
        parts = []
        if self.dit != 'off':
            parts.append(f'dit-{self.dit}')
        if self.vl:
            parts.append('vl')
        if self.text != 'off':
            parts.append(f'text-{self.text}')
        return '+'.join(parts) or 'off'

    def as_dict(self):
        return {'dit': self.dit, 'vl': '1' if self.vl else '0', 'text': self.text}


def scopes_from_name(name):
    """Parse ``off`` or a ``+``-joined canonical name; ``dit-model-vl`` is a legacy alias."""
    if name == 'off':
        return CompileScopes()
    fields = {}

    def put(key, value):
        if key in fields:
            raise ValueError(f'Compile scope {key!r} given twice in {name!r}')
        fields[key] = value

    for part in str(name).split('+'):
        if part == 'vl':
            put('vl', True)
        elif part == 'dit-model-vl':
            put('dit', 'model')
            put('vl', True)
        elif part in ('dit-model', 'dit-blocks'):
            put('dit', part[4:])
        elif part in ('text-layers', 'text-forwards'):
            put('text', part[5:])
        else:
            raise ValueError(f'Unknown compile variant: {name!r}')
    return CompileScopes(**fields)


def as_scopes(value):
    return value if isinstance(value, CompileScopes) else scopes_from_name(value)


def resolve_scopes(environ=None):
    """Read the GR00T_COMPILE_DIT/VL/TEXT environment variables."""
    environ = os.environ if environ is None else environ
    vl = environ.get(_SCOPE_ENV['vl'], '0')
    if vl not in {'0', '1'}:
        raise ValueError(f'{_SCOPE_ENV["vl"]} must be 0 or 1, got {vl!r}')
    return CompileScopes(dit=environ.get(_SCOPE_ENV['dit'], 'off'), vl=vl == '1',
                         text=environ.get(_SCOPE_ENV['text'], 'off'))


def resolve_variant(environ=None):
    """Canonical scope name used in manifests and run records."""
    return resolve_scopes(environ).name


def expected_boundaries(scopes):
    """Ordered ``{path: 'module'|'forward'}`` for every compile target."""
    scopes = as_scopes(scopes)
    boundaries = {}
    if scopes.vl:
        boundaries.update(
            dict.fromkeys((f'action_head.vl_self_attention.transformer_blocks.{i}' for i in range(4)),
                          'module'))
    if scopes.dit == 'model':
        boundaries['action_head.model'] = 'module'
    elif scopes.dit == 'blocks':
        boundaries.update(
            dict.fromkeys((f'action_head.model.transformer_blocks.{i}' for i in range(32)), 'module'))
    if scopes.text != 'off':
        boundary = 'forward' if scopes.text == 'forwards' else 'module'
        boundaries.update(
            dict.fromkeys((f'backbone.model.language_model.layers.{i}' for i in range(16)), boundary))
    return boundaries


def expected_paths(scopes):
    return list(expected_boundaries(scopes))


def _check_layers(layers, count, layer_type, description):
    if (len(layers) != count or len({id(layer) for layer in layers}) != count
            or not all(isinstance(layer, layer_type) for layer in layers)):
        raise ValueError(f'{description} requires {count} distinct {layer_type.__name__} instances')


def _check_forward(target):
    forward = vars(target).get('forward')
    if forward is not None and not (
        isinstance(forward, types.MethodType) and forward.__self__ is target
        and forward.__func__ is type(target).forward
    ):
        raise RuntimeError('Target has a replaced instance forward; restart')


def _marked_forward(optimized, path):
    # Bind this function per instance; never close over the original target instance.
    @wraps(optimized)
    def marked(self, *args, **kwargs):
        from torch.profiler import record_function
        with record_function(f'gr00t_compile::{path}'):
            return optimized(self, *args, **kwargs)
    return marked


def _callable_present(target, boundary):
    if boundary == 'forward':
        forward = target.forward
        return (isinstance(forward, types.MethodType) and forward.__self__ is target
                and bool(getattr(forward.__func__, _FORWARD_TAG, False)))
    return callable(getattr(target, '_compiled_call_impl', None))


def resolve_mode(environ=None):
    """Validate against the modes supported by the installed Inductor backend."""
    from torch._inductor import list_mode_options

    value = (os.environ if environ is None else environ).get('GR00T_COMPILE_MODE', 'max-autotune-no-cudagraphs')
    supported = list_mode_options()
    if value not in supported:
        raise ValueError(
            f'GR00T_COMPILE_MODE must be one of {tuple(supported)}, got {value!r}'
        )
    return value


def install(model, variant, *, mode=None):
    """Preflight all targets, then install lazy instance wrappers exactly once.

    ``variant`` is a ``CompileScopes`` or its canonical name (``off``, ``dit-model+vl``, ...).
    """
    scopes = as_scopes(variant)
    boundaries = expected_boundaries(scopes)
    paths = list(boundaries)
    mode = resolve_mode() if mode is None else resolve_mode({'GR00T_COMPILE_MODE': mode})
    previous = getattr(model, _STATE, None)
    if previous is not None:
        if previous['variant'] != scopes.name or previous['mode'] != mode or previous['failed']:
            raise RuntimeError('Compile request changed or failed; restart the process')
        return dict(previous)
    info = {'variant': scopes.name, 'scopes': scopes.as_dict(), 'paths': paths,
            'boundaries': boundaries, 'mode': mode, 'fullgraph': False,
            'dynamic': 'auto', 'failed': False, 'installation_only': True}
    if not scopes.enabled:
        return info
    if scopes.vl or scopes.dit != 'off':
        from gr00t.model.modules.dit import BasicTransformerBlock, DiT

    if scopes.vl:
        blocks = model.action_head.vl_self_attention.transformer_blocks
        _check_layers(blocks, 4, BasicTransformerBlock, 'VL compilation')
    if scopes.dit != 'off':
        dit = model.action_head.model
        if not isinstance(dit, DiT):
            raise ValueError('DiT compilation requires a DiT instance')
        _check_layers(dit.transformer_blocks, 32, BasicTransformerBlock, 'DiT compilation')
    if scopes.text != 'off':
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        layers = model.get_submodule('backbone.model.language_model.layers')
        _check_layers(layers, 16, Qwen3VLTextDecoderLayer, 'Text compilation')
    targets = [model.get_submodule(path) for path in paths]
    if len({id(t) for t in targets}) != len(targets):
        raise ValueError('Compile targets must be distinct')
    for path, target in zip(paths, targets):
        parents = [model] + [model.get_submodule('.'.join(path.split('.')[:i]))
                             for i in range(1, len(path.split('.')))]
        for module in [*parents, *target.modules()]:
            if (getattr(module, '_compiled_call_impl', None) is not None
                    or hasattr(type(module).__call__, '_torchdynamo_orig_callable')
                    or hasattr(getattr(module, 'forward', None), '_torchdynamo_orig_callable')
                    or getattr(getattr(module, 'forward', None), _FORWARD_TAG, False)):
                raise RuntimeError(f'Overlapping compile wrapper at {path}; restart')
        if scopes.text != 'off' and path.startswith('backbone.model.language_model.layers.'):
            _check_forward(target)
        if boundaries[path] == 'module' and not callable(getattr(target, 'compile', None)):
            raise RuntimeError('This PyTorch build does not expose Module.compile')
    info['failed'] = True
    setattr(model, _STATE, info)
    info['trace_markers'] = os.environ.get('GR00T_COMPILE_TRACE_MARKERS', '0') == '1'
    shared_forwards = {}
    for path, target in zip(paths, targets):
        boundary = boundaries[path]
        if boundary == 'forward':
            import torch
            cls = type(target)
            if cls not in shared_forwards:
                compiled = torch.compile(cls.forward, mode=mode, fullgraph=False)
                setattr(compiled, _FORWARD_TAG, True)
                shared_forwards[cls] = compiled
            optimized = shared_forwards[cls]
            if info['trace_markers']:
                optimized = _marked_forward(optimized, path)
            target.forward = types.MethodType(optimized, target)
        else:
            target.compile(mode=mode, fullgraph=False)
            if info['trace_markers']:
                from torch.profiler import record_function
                optimized = target._compiled_call_impl

                @wraps(optimized)
                def marked(*args, _optimized=optimized, _path=path, **kwargs):
                    with record_function(f'gr00t_compile::{_path}'):
                        return _optimized(*args, **kwargs)

                target._compiled_call_impl = marked
        logging.info(
            'Installed optional lazy compile wrapper: %s, boundary=%s '
            'mode=%s fullgraph=False; runtime compilation unverified', path, boundary, mode)
    info['failed'] = False
    return dict(info)


def write_manifest(output_dir, info, *, runtime=None):
    rank = int(os.environ.get('RANK', '0'))
    path = Path(output_dir) / 'compilation' / f'rank-{rank:02d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(info, rank=rank)
    if runtime is not None:
        payload['runtime'] = runtime
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def evidence_callback(model, info):
    """Record compiler observations; process-wide counters are not per-scope timing."""
    from transformers import TrainerCallback

    class CompileEvidenceCallback(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            from torch._dynamo.utils import counters
            from torch._inductor import metrics

            runtime = {
                'steps': state.global_step,
                'compiled_callables_present': all(
                    _callable_present(model.get_submodule(p), info['boundaries'][p])
                    for p in info['paths']),
                'unique_graphs': int(counters['stats']['unique_graphs']),
                'generated_kernel_count': int(metrics.generated_kernel_count),
                'counters': {str(group): {str(k): int(v) for k, v in values.items()}
                             for group, values in counters.items()},
                'counter_scope': 'Whole process, including existing visual compilation; not target-only.',
            }
            if os.environ.get('GR00T_CAPTURE_DEVICE_MAPPING') == '1':
                import ctypes
                import torch
                hip = ctypes.CDLL('libamdhip64.so')
                hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                bus = ctypes.create_string_buffer(32)
                device = torch.cuda.current_device()
                rc = hip.hipDeviceGetPCIBusId(bus, len(bus), device)
                if rc:
                    raise RuntimeError(f'hipDeviceGetPCIBusId failed: {rc}')
                runtime['device'] = {'local_rank': int(os.environ['LOCAL_RANK']),
                                     'logical_device': device, 'pci_bus': bus.value.decode().lower(),
                                     'visible': os.environ.get('HIP_VISIBLE_DEVICES')}
            write_manifest(args.output_dir, info, runtime=runtime)
            return control

    return CompileEvidenceCallback()
