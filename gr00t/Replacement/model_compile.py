# SPDX-License-Identifier: Apache-2.0
"""Opt-in DiT/VL compilation; preserve module identity, weights and cache policy."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

VARIANTS = ('off', 'vl', 'dit-model', 'dit-model-vl')
_STATE = '_gr00t_optional_compile'


def resolve_variant(environ=None):
    value = (os.environ if environ is None else environ).get('GR00T_COMPILE_VARIANT', 'off')
    if value not in VARIANTS:
        raise ValueError(f'GR00T_COMPILE_VARIANT must be one of {VARIANTS}, got {value!r}')
    return value


def expected_paths(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown compile variant: {variant}')
    paths = []
    if variant in ('vl', 'dit-model-vl'):
        paths += [f'action_head.vl_self_attention.transformer_blocks.{i}' for i in range(4)]
    if variant in ('dit-model', 'dit-model-vl'):
        paths += ['action_head.model']
    return paths


def resolve_mode(environ=None):
    """Validate against the modes supported by the installed Inductor backend."""
    from torch._inductor import list_mode_options

    value = (os.environ if environ is None else environ).get('GR00T_COMPILE_MODE', 'default')
    supported = list_mode_options()
    if value not in supported:
        raise ValueError(
            f'GR00T_COMPILE_MODE must be one of {tuple(supported)}, got {value!r}'
        )
    return value


def install(model, variant, *, mode=None):
    """Preflight all targets, then install lazy instance wrappers exactly once."""
    paths = expected_paths(variant)
    mode = resolve_mode() if mode is None else resolve_mode({'GR00T_COMPILE_MODE': mode})
    previous = getattr(model, _STATE, None)
    if previous is not None:
        if previous['variant'] != variant or previous['mode'] != mode or previous['failed']:
            raise RuntimeError('Compile request changed or failed; restart the process')
        return dict(previous)
    info = {'variant': variant, 'paths': paths, 'mode': mode, 'fullgraph': False,
            'dynamic': 'auto', 'failed': False, 'installation_only': True}
    if variant == 'off':
        return info
    from gr00t.model.modules.dit import BasicTransformerBlock, DiT

    if variant in ('vl', 'dit-model-vl'):
        blocks = model.action_head.vl_self_attention.transformer_blocks
        if (len(blocks) != 4 or len({id(b) for b in blocks}) != 4
                or not all(isinstance(b, BasicTransformerBlock) for b in blocks)):
            raise ValueError('VL compilation requires four distinct BasicTransformerBlock instances')
    if variant in ('dit-model', 'dit-model-vl'):
        dit = model.action_head.model
        blocks = dit.transformer_blocks
        if (not isinstance(dit, DiT) or len(blocks) != 32
                or len({id(b) for b in blocks}) != 32
                or not all(isinstance(b, BasicTransformerBlock) for b in blocks)):
            raise ValueError('DiT compilation requires a DiT with 32 distinct blocks')
    targets = [model.get_submodule(path) for path in paths]
    if len({id(t) for t in targets}) != len(targets):
        raise ValueError('Compile targets must be distinct')
    for path, target in zip(paths, targets):
        parents = [model] + [model.get_submodule('.'.join(path.split('.')[:i]))
                             for i in range(1, len(path.split('.')))]
        for module in [*parents, *target.modules()]:
            if (getattr(module, '_compiled_call_impl', None) is not None
                    or hasattr(type(module).__call__, '_torchdynamo_orig_callable')):
                raise RuntimeError(f'Overlapping compile wrapper at {path}; restart')
        if not callable(getattr(target, 'compile', None)):
            raise RuntimeError('This PyTorch build does not expose Module.compile')
    info['failed'] = True
    setattr(model, _STATE, info)
    for path, target in zip(paths, targets):
        target.compile(mode=mode, fullgraph=False)
        logging.info(
            'Installed optional lazy compile wrapper: %s, mode=%s fullgraph=False', path, mode
        )
    info['failed'] = False
    info['trace_markers'] = os.environ.get('GR00T_COMPILE_TRACE_MARKERS', '0') == '1'
    if info['trace_markers']:
        from functools import wraps
        from torch.profiler import record_function

        for path, target in zip(paths, targets):
            optimized = target._compiled_call_impl

            @wraps(optimized)
            def marked(*args, _optimized=optimized, _path=path, **kwargs):
                with record_function(f'gr00t_compile::{_path}'):
                    return _optimized(*args, **kwargs)

            target._compiled_call_impl = marked
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
                    getattr(model.get_submodule(p), '_compiled_call_impl', None) is not None
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
