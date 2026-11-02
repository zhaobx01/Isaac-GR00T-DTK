#!/usr/bin/env python3
"""Run GR00T batch-768 single-feature and combined benchmarks."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import threading

REPO = Path('/workspace/Isaac-GR00T')
DATA = Path('/workspace/data')
# 已确认有效的优化：输入 BF16 + FusedAdam + 梯度归约重叠；所有 combo_* 以此为公共基础
RUNTIME = {'GR00T_PIXEL_VALUES_BF16': '1', 'GR00T_APEX_FUSED_ADAM': '1', 'GR00T_ZERO2_OVERLAP': '1'}
CASES = {
    'baseline': {},
    # 'pixel_bf16': {'GR00T_PIXEL_VALUES_BF16': '1'},         # 输入图像提前转 BF16
    # 'fused_adam': {'GR00T_APEX_FUSED_ADAM': '1'},           # 使用 Apex 融合 AdamW 优化器
    # 'gro': {'GR00T_ZERO2_OVERLAP': '1'},                    # 梯度归约与反向计算重叠
    # 'obr': {'GR00T_ZERO2_REDUCE_SCATTER': '1'},             # 按梯度所属参数分片的 owner 归约
    # 'allgather': {'GR00T_ZERO2_ALLGATHER_PIPELINE': '1'},   # 参数更新与 AllGather 流水重叠
    # 'vl': {'GR00T_COMPILE_VARIANT': 'vl'},                  # 启用 VL 自注意力模块编译
    # 'dit_model': {'GR00T_COMPILE_VARIANT': 'dit-model'},          # 启用整体动作 DiT 编译
    # 'dit_vl': {'GR00T_COMPILE_VARIANT': 'dit-model-vl'},    # 同时启用整体 DiT 和 VL 编译
    # 'dit_blocks': {'GR00T_COMPILE_VARIANT': 'dit-blocks'},  # 动作 DiT 的 32 个 Block 逐块编译
    # 'text_layers': {'GR00T_COMPILE_VARIANT': 'text-layers'},      # 文本 Decoder 的 16 层按完整模块编译
    # 'text_forwards': {'GR00T_COMPILE_VARIANT': 'text-forwards'},  # 文本层仅编译 forward，按类型共享编译函数
    # 'combo_runtime': dict(RUNTIME),                                      # 已确认有效的运行时组合
    # 'combo_dit': {**RUNTIME, 'GR00T_COMPILE_VARIANT': 'dit-model'},      # 运行时组合 + 整体 DiT 编译
    # 'combo_full': {**RUNTIME, 'GR00T_COMPILE_VARIANT': 'dit-model-vl'},  # 运行时组合 + 整体 DiT/VL 编译
    # 'combo_text_layers': {**RUNTIME, 'GR00T_COMPILE_VARIANT': 'text-layers'},      # 运行时组合 + 文本层模块编译
    # 'combo_text_forwards': {**RUNTIME, 'GR00T_COMPILE_VARIANT': 'text-forwards'},  # 运行时组合 + 文本层 forward 编译
    # 'combo_allgather': {**RUNTIME, 'GR00T_COMPILE_VARIANT': 'dit-model-vl',
    #                     'GR00T_ZERO2_ALLGATHER_PIPELINE': '1'},          # combo_full + AllGather 流水
    'combo_obr': {**RUNTIME, 'GR00T_ZERO2_REDUCE_SCATTER': '1'},
    'combo_obr_dit_vl': {**RUNTIME, 'GR00T_ZERO2_REDUCE_SCATTER': '1',
                         'GR00T_COMPILE_VARIANT': 'dit-model-vl'},
    'combo_obr_dit_vl_text': {**RUNTIME, 'GR00T_ZERO2_REDUCE_SCATTER': '1'},
    'combo_obr_dit_vl_text_allgather': {**RUNTIME, 'GR00T_ZERO2_REDUCE_SCATTER': '1',
                                        'GR00T_ZERO2_ALLGATHER_PIPELINE': '1'},
}


LEGACY_COMPILE_CASES = {'combo_obr_dit_vl_text', 'combo_obr_dit_vl_text_allgather'}
LEGACY_COMPILE_FLAGS = ['--compile-dit-model', '--compile-vl-self-attention-blocks',
                        '--compile-text-layer-forwards']
DEFAULTS = {'GR00T_COMPILE_VISION_BLOCKS': '1', 'GR00T_FLASH_ATTN_DYNAMO_DISABLE': '1', 'GR00T_VISION_PATCH_EMBED_CHANNELS_LAST': '1', 'GR00T_PIXEL_VALUES_BF16': '0', 'GR00T_APEX_FUSED_ADAM': '0', 'GR00T_ZERO2_OVERLAP': '0', 'GR00T_ZERO2_REDUCE_SCATTER': '0', 'GR00T_ZERO2_ALLGATHER_PIPELINE': '0', 'GR00T_ZERO2_PACK_TIMING': '0', 'GR00T_COMPILE_VARIANT': 'off', 'GR00T_COMPILE_SEQUENCE_POLICY': 'pad-156'}

def measure(records):
    if len(records) != 8 or {r.get('rank') for r in records} != set(range(8)) or any(r.get('world_size') != 8 for r in records):
        raise ValueError('expected eight unique ranks with world_size=8')
    for r in records:
        a = r.get('step_seconds', [])
        if len(a) != 40 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in a):
            raise ValueError('expected exactly 40 positive finite step durations per rank')
    step = [max(r['step_seconds'][i] for r in records) for i in range(40)]
    steady = step[10:40]
    cv = 100 * statistics.pstdev(steady) / statistics.mean(steady)
    drift = 100 * (statistics.mean(steady[15:]) / statistics.mean(steady[:15]) - 1)
    mean = statistics.mean(step[30:40])
    return {'valid': cv <= 5 and abs(drift) <= 5,
            'step35_ms': 1000 * step[34],
            'step35_samples_per_second': 768 / step[34],
            'mean31_40_ms': 1000 * mean,
            'mean31_40_samples_per_second': 768 / mean,
            'cv_pct': cv, 'drift_pct': drift}

def device_check():
    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--json'], text=True, capture_output=True, check=True, timeout=20)
    return json.loads(result.stdout)


MEMORY_START_THRESHOLD_PCT = 5.0


def device_statistics(cards):
    """Normalize eight-card telemetry; GPU utilization is observational only."""
    used_mib, memory_pct, gpu_util, per_card = [], [], [], {}
    for i in range(8):
        try:
            card = cards[f'card{i}']
            unit = 'B' if 'vram Total Used Memory (B)' in card else 'MiB'
            used = float(card[f'vram Total Used Memory ({unit})'])
            total = float(card[f'vram Total Memory ({unit})'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'card {i}: incomplete or invalid memory query') from exc
        if not all(map(math.isfinite, (used, total))) or total <= 0 or not 0 <= used <= total:
            raise ValueError(f'card {i}: invalid memory readings: {card}')
        used_mib.append(used / (1024 ** 2) if unit == 'B' else used)
        memory_pct.append(100 * used / total)
        try:
            util = float(card.get('HCU use (%)', card.get('GPU use (%)')))
            gpu_util.append(util if math.isfinite(util) and 0 <= util <= 100 else None)
        except (TypeError, ValueError):
            gpu_util.append(None)
        per_card[f'card{i}'] = {'used_mib': used_mib[-1], 'memory_pct': memory_pct[-1]}
    return {'memory_by_card': per_card, 'average_used_memory_mib': statistics.mean(used_mib),
            'average_memory_pct': statistics.mean(memory_pct),
            'max_memory_pct': max(memory_pct),
            'average_gpu_util_pct': statistics.mean(gpu_util)
            if all(util is not None for util in gpu_util) else None}


def devices_idle(cards):
    """各卡显存低于配置阈值时放行，GPU 利用率不参与判断。"""
    return device_statistics(cards)['max_memory_pct'] < MEMORY_START_THRESHOLD_PCT


def wait_for_idle_gpus(log_path, *, query=device_check, sleep=time.sleep):
    """每 10 秒检查，各卡显存低于配置阈值时启动。"""
    while True:
        row = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'idle': False}
        try:
            row['cards'] = query()
            row.update(device_statistics(row['cards']))
            row['idle'] = row['max_memory_pct'] < MEMORY_START_THRESHOLD_PCT
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        with Path(log_path).open('a') as log:
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
        if 'error' not in row:
            util = row['average_gpu_util_pct']
            gpu_text = f'{util:.2f}%' if util is not None else '未知'
            print(f"[{row['time_utc']}] 八卡平均显存 {row['average_memory_pct']:.2f}% | GPU 利用率 {gpu_text} | 单卡最高显存 {row['max_memory_pct']:.2f}%", flush=True)
        else:
            print(f"[{row['time_utc']}] 查询失败：{row['error']}", flush=True)
        if row['idle']:
            print('资源空闲，启动训练', flush=True)
            return row['cards']
        print('等待空闲，10 秒后重试', flush=True)
        sleep(10)

def validate_run(run, case, records):
    env = {**DEFAULTS, **CASES[case]}
    expected = {'zero2_allgather_pipeline': env['GR00T_ZERO2_ALLGATHER_PIPELINE'] == '1', 'zero2_overlap': env['GR00T_ZERO2_OVERLAP'] == '1', 'zero2_reduce_scatter': env['GR00T_ZERO2_REDUCE_SCATTER'] == '1', 'compile_variant': 'legacy-cli' if case in LEGACY_COMPILE_CASES else env['GR00T_COMPILE_VARIANT'], 'global_batch_size': 768, 'num_gpus': 8}
    for record in records:
        info = record.get('run_info', {})
        for key, value in expected.items():
            if info.get(key) != value:
                raise ValueError(f'rank {record.get("rank")} effective {key}: {info.get(key)!r} != {value!r}')
        if env['GR00T_COMPILE_VARIANT'] != 'off' and not info.get('compiled_paths'):
            raise ValueError('compile enabled but no compiled paths reported')
        if case in LEGACY_COMPILE_CASES:
            required = {'action_head.model'} | {f'action_head.vl_self_attention.transformer_blocks.{i}' for i in range(4)} | {f'backbone.model.language_model.layers.{i}' for i in range(16)}
            if not required.issubset(set(info.get('compiled_paths') or [])):
                raise ValueError('legacy compile missing required compiled paths')
    log = (run / 'train.log').read_text(errors='replace')
    if env['GR00T_PIXEL_VALUES_BF16'] == '1' and 'pixel_values is cast to BF16 in DataLoader workers' not in log:
        raise ValueError('BF16 pixel collator activation not observed')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return str(s.getsockname()[1])

def build_env(case):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GR00T_') and k not in {'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_LOGS', 'TORCH_TRACE', 'CUDA_LAUNCH_BLOCKING', 'HIP_LAUNCH_BLOCKING'}}
    env.update(DEFAULTS)
    env.update(CASES[case])
    env.update({'NUM_GPUS': '8', 'MAX_STEPS': '40', 'GLOBAL_BATCH_SIZE': '768', 'DATALOADER_NUM_WORKERS': '4', 'USE_WANDB': '0', 'MASTER_PORT': free_port(), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLM_MODEL_PATH': str(DATA / 'checkpoints/nvidia/Cosmos-Reason2-2B')})
    return env

def build_command(run, *, trace=False, case=None):
    cmd = ['bash', 'examples/finetune.sh', '--base-model-path',
           str(DATA / 'checkpoints/GR00T-N1.7-3B-local-cosmos'), '--dataset-path',
           str(DATA / 'libero_10_no_noops_1.0.0_lerobot_h264'), '--embodiment-tag',
           'LIBERO_PANDA', '--output-dir', str(run), '--state-dropout-prob', '0.2',
           '--save-only-model', '--', '--gradient-accumulation-steps', '1',
           '--log-step-metrics', '--save-strategy', 'no', '--no-save-final-model',
           '--no-check-numerics']
    if case in LEGACY_COMPILE_CASES:
        cmd += LEGACY_COMPILE_FLAGS
    if trace:
        # First active training step = skip_first + wait + warmup + 1 = 35.
        cmd += ['--enable-profiling', '--profiler-skip-first', '28', '--profiler-wait', '1',
                '--profiler-warmup', '5', '--profiler-active', '1', '--profiler-repeat', '1',
                '--profiler-with-stack']
    else:
        cmd += ['--no-enable-profiling']
    return cmd


def monitor_memory(path, stop, *, query, interval):
    """每 10 秒记录设备显存，查询失败单独记错。"""
    with path.open('a') as log:
        while not stop.is_set():
            row = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                row['cards'] = query()
                row.update(device_statistics(row['cards']))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if 'error' in row:
                print(f"显存查询失败：{row['error']}", flush=True)
            else:
                util = row['average_gpu_util_pct']
                util_text = f'{util:.2f}%' if util is not None else '未知'
                print(f"八卡平均显存 {row['average_memory_pct']:.2f}% | GPU 利用率 {util_text}",
                      flush=True)
            if stop.wait(interval):
                break


def run_training(cmd, env, run, *, monitor=True, query=None, interval=10, timeout=3600):
    stop = threading.Event()
    worker = None
    with (run / 'train.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if monitor:
                worker = threading.Thread(target=monitor_memory,
                    args=(run / 'memory.jsonl', stop),
                    kwargs={'query': query or device_check, 'interval': interval})
                worker.start()
            proc.wait(timeout=timeout)
            return proc
        finally:
            try:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    except ProcessLookupError:
                        proc.wait()
            finally:
                # Also clean workers when the launcher exited first, including on failure.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    stop.set()
                    if worker is not None:
                        worker.join()  # rocm-smi has a bounded query timeout.


def summarize_device_memory(path):
    peaks, percentages, samples, errors = {}, {}, 0, 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if 'error' in row:
                    errors += 1
                    continue
                stats = device_statistics(row['cards'])
            except (ValueError, KeyError, TypeError):
                errors += 1
                continue
            samples += 1
            for card, value in stats['memory_by_card'].items():
                peaks[card] = max(peaks.get(card, 0), value['used_mib'])
                percentages[card] = max(percentages.get(card, 0), value['memory_pct'])
    return {'memory_samples': samples, 'memory_query_errors': errors,
            'device_peak_used_mib_by_card': peaks,
            'device_peak_memory_pct_by_card': percentages,
            'device_max_used_mib': max(peaks.values()) if peaks else None}


def summarize_pytorch_memory(records):
    summary = {}
    for source, name in [('max_memory_allocated_bytes', 'allocated'),
                         ('max_memory_reserved_bytes', 'reserved')]:
        peaks = {str(r['rank']): r[source] / 1024 ** 2 for r in records
                 if isinstance(r.get(source), (int, float)) and math.isfinite(r[source])
                 and r[source] >= 0}
        summary[f'pytorch_peak_{name}_mib_by_rank'] = peaks
        summary[f'pytorch_max_{name}_mib'] = max(peaks.values()) if len(peaks) == 8 else None
    return summary


def validate_trace(run):
    profile = run / 'profiling'
    for rank in range(8):
        trace = profile / f'trace_rank_{rank}_iter_35.json'
        if not trace.is_file() or trace.stat().st_size == 0:
            raise ValueError(f'missing step-35 Trace for rank {rank}')
        stopped = json.loads((profile / f'profiler-stopped-rank-{rank}.json').read_text())
        if stopped != {'rank': rank, 'step': 35, 'stopped_after_trace': True}:
            raise ValueError(f'profiler stop evidence mismatch for rank {rank}')
    table = profile / 'rank0-key-averages.txt'
    if not table.is_file() or not table.read_text().strip():
        raise ValueError('missing rank 0 key_averages table')
    return {'trace_step': 35, 'trace_ranks': list(range(8)),
            'key_averages': 'profiling/rank0-key-averages.txt'}


def write_report(root, rows, *, update_results=True):
    if update_results:
        (root / 'results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    speed_rows = [r for r in rows if r.get('kind', 'speed') == 'speed']
    traces = [r for r in rows if r.get('kind') == 'trace']
    lines = ['# GR00T batch 768 单项与组合测速', '',
             '8 rank、每轮 40 步；视觉布局、视觉 Block 编译及 FlashAttention Dynamo 边界隔离纳入所有方案基线。每步耗时取该步 8 rank 最大值。吞吐量为 768 ÷ 对应步耗时；十步吞吐为 768 ÷ 十步平均耗时。', '',
             '各组至少两轮有效才计算跨轮中位数；相对值与同批基线比较。有效性检查采用第 11–40 步逐步最慢 rank：CV ≤5%，前后各 15 步均值漂移绝对值 ≤5%。测速不证明数值等价。', '',
             f"combo_runtime = BF16 + FusedAdam + GRO；combo_obr 系列依次叠加 OBR、DiT/VL 编译、Text Forwards 编译和 AllGather。含 Text Forwards 的新组合使用旧 CLI 多范围编译入口；其余组合使用单范围环境变量入口。默认序列策略为 {DEFAULTS['GR00T_COMPILE_SEQUENCE_POLICY']}。", '',
             '每个 case 的 N 轮测速后独立采集一次第 35 步 Trace（with_stack），Trace 不计入测速。编译磁盘缓存沿用框架默认策略，在进程间共享。']
    for title, time_key, speed_key in [
        ('第 35 步', 'step35_ms', 'step35_samples_per_second'),
        ('第 31–40 步均值', 'mean31_40_ms', 'mean31_40_samples_per_second'),
    ]:
        baseline = [r[speed_key] for r in speed_rows
                    if r['case'] == 'baseline' and r.get('valid')]
        base = statistics.median(baseline) if len(baseline) >= 2 else None
        lines += ['', '## ' + title, '',
                  '| 方案 | 有效/总轮数 | 耗时中位数 ms | 吞吐中位数 samples/s | 相对基线 |',
                  '|---|---:|---:|---:|---:|']
        for case in CASES:
            group = [r for r in speed_rows if r['case'] == case]
            valid = [r for r in group if r.get('valid')]
            if len(valid) < 2:
                lines.append(f'| {case} | {len(valid)}/{len(group)} | — | — | — |')
                continue
            t = statistics.median(r[time_key] for r in valid)
            speed = statistics.median(r[speed_key] for r in valid)
            relative = f'{(speed / base - 1) * 100:+.2f}%' if base else '—'
            lines.append(f'| {case} | {len(valid)}/{len(group)} | {t:.3f} | {speed:.3f} | {relative} |')
    lines += ['', '## 每轮明细', '',
              '| 轮次 | 有效 | 第 35 步 ms | 第 31–40 步均值 ms | 第 35 步吞吐量 samples/s | CV % | 漂移 % | 错误 |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    def fmt(row, key):
        return f"{row[key]:.3f}" if row.get(key) is not None else '—'
    for r in speed_rows:
        lines.append(f"| {r['case']}-{r['round']} | {r.get('valid', False)} | {fmt(r, 'step35_ms')} | {fmt(r, 'mean31_40_ms')} | {fmt(r, 'step35_samples_per_second')} | {fmt(r, 'cv_pct')} | {fmt(r, 'drift_pct')} | {r.get('error', '')} |")
    lines += ['', '## 测速显存', '',
              '设备值为每 10 秒采样的单卡最大占用，包含 PyTorch 外占用，可能漏过瞬时峰值；PyTorch 值为训练期间各 rank 分配器峰值的最大值。各卡和各 rank 数据保存在 results.json；设备原始采样保存在对应测速目录 memory.jsonl。查询失败不改变测速有效性；缺失显示为 —。', '',
              '| 轮次 | 设备采样最大 MiB | PyTorch allocated 峰值 MiB | PyTorch reserved 峰值 MiB | 成功/失败采样数 |',
              '|---|---:|---:|---:|---:|']
    for r in speed_rows:
        lines.append(f"| {r['case']}-{r['round']} | {fmt(r, 'device_max_used_mib')} | {fmt(r, 'pytorch_max_allocated_mib')} | {fmt(r, 'pytorch_max_reserved_mib')} | {r.get('memory_samples', 0)}/{r.get('memory_query_errors', 0)} |")
    lines += ['', '## Trace', '', '| 采集轮次 | 成功 | 数据目录 | 错误 |', '|---|---|---|---|']
    for r in traces:
        name = f"{r['case']}-trace"
        path = f"{r['case']}/{name}/profiling"
        lines.append(f"| {name} | {r.get('valid', False)} | [{path}]({path}/) | {r.get('error', '')} |")
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def load_resume_rows(root, *, rounds=None):
    """仅续跑新版目录；已记录失败项不自动重试。"""
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata.get('layout_version') != 2:
        raise ValueError('unsupported output layout; old flat directories cannot resume')
    if rounds is not None and metadata.get('rounds') != rounds:
        raise ValueError('--rounds must match the original run')
    rows = json.loads((root / 'results.json').read_text())
    if not isinstance(rows, list):
        raise ValueError('results.json must contain a list of measured rounds')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed completed round in results.json')
        kind, n = row.get('kind'), row.get('round')
        valid_round = ((kind == 'speed' and type(n) is int and n >= 1)
                       or (kind == 'trace' and n == 'trace'))
        if (row.get('case') not in CASES or not valid_round
                or type(row.get('valid')) is not bool):
            raise ValueError('Malformed completed round in results.json')
        key = (row['case'], n)
        if key in seen:
            raise ValueError(f'Duplicate completed round: {key}')
        seen.add(key)
        case = row['case']
        launch = root / case / f'{case}-{n}' / 'launch.json'
        if launch.exists():
            launch_data = json.loads(launch.read_text())
            recorded = launch_data.get('environment', {})
            expected = {**DEFAULTS, **CASES[case]}
            changed = [key for key, value in expected.items() if recorded.get(key) != value]
            if changed:
                detail = 'sequence policy ' if 'GR00T_COMPILE_SEQUENCE_POLICY' in changed else ''
                raise ValueError(f'{detail}configuration changed for {case}-{n}: {changed}; use a new output directory')
            if case in LEGACY_COMPILE_CASES and not all(flag in launch_data.get('command', []) for flag in LEGACY_COMPILE_FLAGS):
                raise ValueError(f'compile configuration changed for {case}-{n}; use a new output directory')
        else:
            raise ValueError(f'cannot verify sequence policy for {case}-{n}; use a new output directory')
    return rows


def pending_runs(cases, rounds, rows):
    completed = {(row['case'], row['round']) for row in rows}
    return [(case, n) for case in cases for n in [*range(1, rounds + 1), 'trace']
            if (case, n) not in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--rounds', type=int, default=2, help='speed rounds per case, plus one Trace run')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--sequence-policy', choices=('auto', 'pad-156', 'pad-152-156'),
                        help='override the default sequence policy for this run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='continue missing runs in a new-layout directory')
    args = parser.parse_args()
    if args.sequence_policy is not None:
        DEFAULTS['GR00T_COMPILE_SEQUENCE_POLICY'] = args.sequence_policy
    if args.rounds < 2 or 'baseline' not in args.cases or len(set(args.cases)) != len(args.cases):
        parser.error('need at least two rounds, unique cases, and baseline')
    if args.dry_run:
        rows = load_resume_rows(args.output, rounds=args.rounds) if args.resume else []
        print(json.dumps({'completed': len(rows),
                          'pending': pending_runs(args.cases, args.rounds, rows),
                          'environments': {c: {**DEFAULTS, **CASES[c]} for c in args.cases},
                          'commands': {c: build_command(Path('/tmp/benchmark-preview') / c, case=c) for c in args.cases},
                          'trace': {'step': 35, 'with_stack': True, 'ranks': list(range(8))}},
                         indent=2))
        return
    for path in [REPO / 'examples/finetune.sh', DATA / 'checkpoints/GR00T-N1.7-3B-local-cosmos', DATA / 'checkpoints/nvidia/Cosmos-Reason2-2B', DATA / 'libero_10_no_noops_1.0.0_lerobot_h264']:
        if not path.exists():
            raise SystemExit(f'missing input: {path}')
    if args.resume:
        if not args.output.is_dir():
            raise SystemExit('--resume requires an existing output directory')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    runner_lock = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another benchmark is already writing to this output directory')
        _run_benchmarks(args, root)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        os.close(runner_lock)


def _run_benchmarks(args, root):
    rows = load_resume_rows(root, rounds=args.rounds) if args.resume else []
    if not args.resume:
        metadata = {'layout_version': 2, 'rounds': args.rounds,
                    'cases': args.cases,
                    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                              text=True, capture_output=True).stdout.strip(),
                    'git_status': subprocess.run(['git', 'status', '--short'], cwd=REPO,
                                                text=True, capture_output=True).stdout}
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        write_report(root, rows)
    # Check all pending directories before launching any training.
    pending = pending_runs(args.cases, args.rounds, rows)
    for case, n in pending:
        run = root / case / f'{case}-{n}'
        if run.exists():
            raise SystemExit(f'Unrecorded existing round {run}; refusing to overwrite')
    for case, n in pending:
        trace = n == 'trace'
        run = root / case / f'{case}-{n}'
        run.mkdir(parents=True)
        env = build_env(case)
        if trace:
            env['GR00T_PROFILER_STOP_AFTER_TRACE'] = '1'
        cmd = build_command(run, trace=trace, case=case)
        keys = list(DEFAULTS) + ['NUM_GPUS', 'GLOBAL_BATCH_SIZE', 'MAX_STEPS',
                                 'DATALOADER_NUM_WORKERS', 'MASTER_PORT']
        if trace:
            keys.append('GR00T_PROFILER_STOP_AFTER_TRACE')
        (run / 'launch.json').write_text(json.dumps(
            {'command': cmd, 'environment': {k: env[k] for k in keys}}, indent=2) + '\n')
        row = {'case': case, 'round': n, 'kind': 'trace' if trace else 'speed', 'valid': False}
        print(f'{case}-{n}: preflight and run', flush=True)
        records = []
        interrupted = False
        try:
            cards = wait_for_idle_gpus(run / 'resource-wait.jsonl')
            (run / 'preflight.json').write_text(json.dumps(cards, indent=2) + '\n')
            proc = run_training(cmd, env, run, monitor=not trace)
            if proc.returncode:
                raise RuntimeError(f'training exit {proc.returncode}; inspect train.log')
            records = [json.loads((run / 'step_metrics' / f'rank-{i:02d}.json').read_text())
                       for i in range(8)]
            validate_run(run, case, records)
            if trace:
                # Check all 40 steps completed, but do not apply speed stability to Trace.
                measure(records)
                row.update(validate_trace(run))
                row['valid'] = True
            else:
                row.update(measure(records))
                if not row['valid']:
                    row['error'] = 'CV or drift > 5%'
        except KeyboardInterrupt:
            row['error'] = 'Interrupted: training stopped by user or SIGTERM'
            interrupted = True
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if not trace:
                row.update(summarize_device_memory(run / 'memory.jsonl'))
                if not records:
                    for i in range(8):
                        path = run / 'step_metrics' / f'rank-{i:02d}.json'
                        try:
                            records.append(json.loads(path.read_text()))
                        except (OSError, ValueError):
                            pass
                row.update(summarize_pytorch_memory(records))
        rows.append(row)
        write_report(root, rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if interrupted:
            raise KeyboardInterrupt
    write_report(root, rows)
    print(f'Report: {root / "report.md"}')
    if any(not r.get('valid') for r in rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
