#!/usr/bin/env bash
# 单文件编排：组合配置 Layer3(40步) -> GEMM选优 -> 全部CASES各2轮测速，无Trace。
# 不修改 benchmark.py / run_background.sh，不自动重试，不永久覆盖原库。
# 预览：bash scripts/benchmark_all/run_gemm_workflow.sh plan
# 后台：bash scripts/benchmark_all/run_gemm_workflow.sh start --output /workspace/data/outputs/gemm-full-001
# 前台：bash scripts/benchmark_all/run_gemm_workflow.sh run --output /workspace/data/outputs/gemm-full-001
# 续跑：bash scripts/benchmark_all/run_gemm_workflow.sh start --output /workspace/data/outputs/gemm-full-001 --resume
# 状态：bash scripts/benchmark_all/run_gemm_workflow.sh status --output /workspace/data/outputs/gemm-full-001
# 停止：bash scripts/benchmark_all/run_gemm_workflow.sh stop --output /workspace/data/outputs/gemm-full-001
# 手动续跑：跳过成功阶段；失败的调优阶段备份中间结果后重新选优，测速只补失败/未完成轮次。
# 空闲门槛仅为启动检查，运行时仍需人为预留8卡，不能阻止其他程序抢占。
# 运行时仅在独立输出目录生成状态/归档/python3适配入口；Python仅使用标准库。
set -euo pipefail
workflow_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
exec "${GEMM_WORKFLOW_PYTHON:-$(command -v python3)}" -u - "$workflow_script" "$@" <<'WORKFLOW_PY'
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

SCRIPT = Path(sys.argv[1]).resolve()
REPO = SCRIPT.parents[2]
BENCH = REPO / 'scripts/benchmark_all/benchmark.py'
BACKGROUND = BENCH.with_name('run_background.sh')
AUTOTUNE = REPO / 'getting_started/optimization/dcu_gemm_autotune.sh'
DTK = Path(os.environ.get('DTKROOT', '/opt/dtk')).resolve()
TOOL = DTK / 'lib/rocblas/auto_select_test/auto_select_tools'
COMBO = 'combo_obr_dit_vl_text_allgather'
TUNING = {'strategy': 'common', 'chunk_size': '10', 'tune_warmup_times': '2',
          'tune_loop_times': '4', 'bench_warmup_times': '2', 'bench_loop_times': '10',
          'test_times': '2', 'enable_bestkernel_csv': '0', 'enable_mk_n_threshold': '0',
          'update_target_config': '0'}


def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')


def log(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def backup_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists():
        archive(temporary, destination.parent / 'incomplete_backups')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    temporary.rename(destination)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_digest(path):
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f'缺少目录: {path}')
    files = {p.relative_to(path).as_posix(): sha(p)
             for p in sorted(path.rglob('*')) if p.is_file()}
    if not files:
        raise RuntimeError(f'目录为空: {path}')
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def load_benchmark():
    spec = importlib.util.spec_from_file_location('gemm_benchmark', BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(module):
    # Hash dirty file CONTENTS as well as tracked source; documentation/test edits do not alter training.
    paths = set()
    for folder in ('gr00t', 'examples', 'configs', 'scripts/benchmark_all'):
        for p in (REPO / folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.yaml', '.yml', '.toml'):
                if 'test' not in p.name and '__pycache__' not in p.parts:
                    paths.add(p)
    paths.update((SCRIPT, AUTOTUNE))
    for name in ('pyproject.toml', 'uv.lock', 'start.sh'):
        if (REPO / name).is_file():
            paths.add(REPO / name)
    for folder in ('src', 'source-all', 'source-test'):
        for p in (TOOL / folder).rglob('*'):
            if p.is_file() and (p.suffix in ('.py', '.cpp', '.hpp', '.h') or p.name == 'Makefile'):
                paths.add(p)
    paths.update((DTK / 'env.sh', TOOL / 'Auto_tuning.py'))
    return {'version': 1, 'repo': str(REPO), 'dtk': str(DTK),
            'sources': {str(p): sha(p) for p in sorted(paths)},
            'installed_library_digest': tree_digest(DTK / 'lib/rocblas/library_gpu5'),
            'cases': list(module.CASES), 'defaults': module.DEFAULTS,
            'combo': module.CASES[COMBO], 'capture_steps': 40, 'rounds': 2, 'trace': False,
            'idle': {'memory_pct_below': 2, 'gpu_pct_at_most': 5, 'continuous_seconds': 60},
            'tuning': TUNING, 'hardware': 'BW', 'cu_count': 0, 'gpus': list(range(8))}


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('GR00T_', 'ROCBLAS_LOG_')) and k not in {
               'ROCBLAS_LAYER', 'ROCBLAS_TENSILE_LIBPATH', 'TENSILE_DB', 'KERNEL_TEST',
               'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
               'GPU_DEVICE_ORDINAL', 'TORCH_LOGS', 'TORCH_TRACE', 'TORCHINDUCTOR_CACHE_DIR',
               'TRITON_CACHE_DIR', 'CUDA_LAUNCH_BLOCKING', 'HIP_LAUNCH_BLOCKING'}}
    env['DTKROOT'] = str(DTK)
    return env


def wait_idle(path, query=None, sleep=time.sleep, clock=time.monotonic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    since = None
    log('等待8卡各自显存<2%、利用率<=5%，连续60秒；每10秒检查')
    while True:
        row = {'time': datetime.now(timezone.utc).isoformat()}
        try:
            if query is None:
                result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--json'],
                                        capture_output=True, text=True, check=True, timeout=20)
                cards = json.loads(result.stdout)
            else:
                cards = query()
            row['cards'] = cards
            mem, util = [], []
            for i in range(8):
                c = cards[f'card{i}']
                unit = 'B' if 'vram Total Memory (B)' in c else 'MiB'
                total = float(c[f'vram Total Memory ({unit})'])
                used = float(c[f'vram Total Used Memory ({unit})'])
                gpu = float(c.get('HCU use (%)', c.get('GPU use (%)')))
                if not all(map(math.isfinite, (total, used, gpu))) or not (total > 0 and 0 <= used <= total and 0 <= gpu <= 100):
                    raise ValueError(f'card{i} telemetry invalid')
                mem.append(100 * used / total)
                util.append(gpu)
            idle = max(mem) < 2 and max(util) <= 5
            now = clock()
            since = (now if since is None else since) if idle else None
            row.update(max_memory_pct=max(mem), max_gpu_pct=max(util),
                       stable_seconds=0 if since is None else now - since)
            log(f'最高显存 {max(mem):.2f}% / 最高GPU {max(util):.2f}% / 连续空闲 {row["stable_seconds"]:.0f}s')
        except Exception as exc:
            since = None
            row['error'] = str(exc)
            log(f'资源查询失败，继续等待: {exc}')
        with path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if since is not None and row.get('stable_seconds', 0) >= 60:
            return row['cards']
        sleep(10)


def clean_layer3(source, dest):
    commands = set()
    for line in Path(source).read_text(errors='replace').splitlines():
        matches = list(re.finditer(r'\./rocblas-bench(?=\s)', line))
        for i, match in enumerate(matches):
            command = line[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(line)].strip()
            args = shlex.split(command)
            if not all(flag in args for flag in ('-f', '-m', '-n', '-k')):
                raise RuntimeError('Layer3命令缺少GEMM参数: ' + command[:200])
            commands.add(command)
    if not commands:
        raise RuntimeError('Layer3日志没有有效rocblas-bench调用')
    Path(dest).write_text('\n'.join(sorted(commands)) + '\n')
    return len(commands)


def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # A launcher can exit before its workers; always clean the remaining process group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def command_run(cmd, env, stdout, stderr=None):
    log('执行: ' + shlex.join(list(map(str, cmd))))
    with Path(stdout).open('ab') as out, contextlib.ExitStack() as stack:
        err = stack.enter_context(Path(stderr).open('ab')) if stderr else subprocess.STDOUT
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = proc.wait()
            if code:
                raise RuntimeError(f'命令退出码 {code}，查看 {stdout}' + (f' / {stderr}' if stderr else ''))
        finally:
            stop_group(proc)


def archive(path, archive_root):
    path = Path(path)
    if path.exists():
        target = Path(archive_root) / (path.name + '-' + stamp())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)


def prepare_benchmark_resume(root, cases):
    root = Path(root)
    rows = read(root / 'results.json') if (root / 'results.json').exists() else []
    valid = [r for r in rows if r.get('valid')]
    if len({(r['case'], r['round']) for r in rows}) != len(rows):
        raise RuntimeError('重复的benchmark轮次记录，拒绝续跑')
    done = {(r['case'], r['round']) for r in valid}
    failed = root / 'failed_attempts' / stamp()
    # Journal and atomically remove failed records BEFORE moving their directories.
    # An interrupted archive then leaves only harmless orphan directories for the next resume.
    if rows != valid:
        save(failed / 'results-before-resume.json', rows)
    save(root / 'results.json', valid)
    # Never delete failures or orphan directories, including interruption before results.json was written.
    for case in cases:
        for n in (1, 2):
            run = root / case / f'{case}-{n}'
            if run.exists() and (case, n) not in done:
                archive(run, failed / case)


def patch_benchmark(module, root, provenance, config):
    original_report = module.write_report
    original_env = getattr(module, 'build_env', None)
    original_training = getattr(module, 'run_training', None)

    def report(path, rows, **kwargs):
        original_report(path, rows, **kwargs)
        # The shipped report assumes a trace even with --no-trace; clarify actual workload.
        report_path = Path(path) / 'report.md'
        if report_path.exists():
            text = report_path.read_text()
            text = text.replace('每个 case 的 N 轮测速后独立采集一次第 35 步 Trace（with_stack），Trace 不计入测速。',
                                '本次跳过 Trace，每个 case 仅采集两轮40步测速。')
            report_path.write_text(text)
        if rows and not rows[-1].get('valid'):
            raise SystemExit('本轮失败或稳定性不合格：已保存记录并停止，没有自动重试。')

    def environment(case):
        # Preserve benchmark's defaults/case generation while removing inherited device/log overrides.
        env = original_env(case)
        for key in list(env):
            if key.startswith('ROCBLAS_LOG_') or key in {'ROCBLAS_LAYER', 'TENSILE_DB', 'KERNEL_TEST',
                    'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'}:
                env.pop(key, None)
        env['ROCBLAS_TENSILE_LIBPATH'] = str(config)
        return env

    def training(cmd, env, run, **kwargs):
        if tree_digest(config) != provenance['config_digest']:
            raise RuntimeError('配置归档内容已变化，拒绝运行')
        launch = Path(run) / 'launch.json'
        data = read(launch)
        data['environment']['ROCBLAS_TENSILE_LIBPATH'] = str(config)
        data['gemm_provenance'] = provenance
        save(launch, data)
        return original_training(cmd, env, run, **kwargs)

    module.write_report = report
    module.wait_for_idle_gpus = wait_idle
    if original_env:
        module.build_env = environment
    if original_training:
        module.run_training = training


def benchmark_adapter(root):
    state = read(root / 'state.json')
    config = root / 'selected_config/library_gpu5'
    if tree_digest(config) != state['config_digest']:
        raise RuntimeError('归档配置变化，拒绝benchmark')
    module = load_benchmark()
    target = root / 'benchmark'
    resume = target.exists()
    if resume:
        # Validate existing completed records before archiving failed/orphan attempts.
        if (target / 'metadata.json').exists() and (target / 'results.json').exists():
            module.load_resume_rows(target, rounds=2)
        else:
            archive(target, root / 'failed_attempts')
            resume = False
    if resume:
        prepare_benchmark_resume(target, list(module.CASES))
    provenance = {'config_digest': state['config_digest'], 'workflow': str(root),
                  'manifest_sha256': sha(root / 'manifest.json')}
    patch_benchmark(module, target, provenance, config)
    sys.argv = [str(BENCH), '--output', str(target), '--rounds', '2', '--no-trace']
    if resume:
        sys.argv.append('--resume')
    module.main()


def validate_capture(root, module):
    capture = root / 'capture'
    records = [read(capture / 'training/step_metrics' / f'rank-{rank:02d}.json') for rank in range(8)]
    module.measure(records)  # validates 8 ranks, 40 finite positive durations; speed stability not required for capture
    expected_name, expected_paths = module.compile_expectation({**module.DEFAULTS, **module.CASES[COMBO]})
    for rank, record in enumerate(records):
        info = record['run_info']
        if record['rank'] != rank or info.get('compile_variant') != expected_name or info.get('global_batch_size') != 768 or info.get('num_gpus') != 8:
            raise RuntimeError(f'采集rank{rank}训练配置不符合组合配置')
        for name in ('zero2_overlap', 'zero2_reduce_scatter', 'zero2_allgather_pipeline'):
            if info.get(name) is not True:
                raise RuntimeError(f'采集rank{rank}缺少{name}')
        if set(info.get('compiled_paths', [])) != set(expected_paths):
            raise RuntimeError(f'采集rank{rank}编译目标不匹配')
    text = (capture / 'train.log').read_text(errors='replace') + (capture / 'layer3.raw.log').read_text(errors='replace')
    if 'Training completed!' not in text:
        raise RuntimeError('缺少Training completed!，拒绝使用部分日志')
    return clean_layer3(capture / 'layer3.raw.log', capture / 'layer3.clean.log')


def tune_work_digest():
    paths = [TOOL / 'configs/tuning_args.txt']
    for name in ('optimization_configs', 'optimization_artifacts', 'rocblas_layer3_size'):
        paths.extend(p for p in (TOOL / name).rglob('*') if p.is_file())
    return hashlib.sha256(json.dumps({str(p): sha(p) for p in sorted(paths) if p.exists()}, sort_keys=True).encode()).hexdigest()


def run_workflow(root, resume):
    root.parent.mkdir(parents=True, exist_ok=True)
    # Global advisory lock between instances of this orchestrator. External jobs still need reserved GPUs.
    lockpath = Path('/workspace/data/outputs/.gemm-workflow.lock')
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有本编排脚本正在运行')
        if resume:
            if not (root / 'manifest.json').is_file():
                raise RuntimeError('续跑目录缺少manifest.json')
        else:
            root.mkdir(exist_ok=False)
        save(root / 'process.json', {'pid': os.getpid(), 'start': process_start(os.getpid())})
        state = read(root / 'state.json') if (root / 'state.json').exists() else {'completed': []}
        state.update(status='checking')
        state.pop('error', None)
        (root / 'exit.code').unlink(missing_ok=True)
        save(root / 'state.json', state)
        try:
            module = load_benchmark()
            current = identity(module)
            if resume:
                if read(root / 'manifest.json') != current:
                    raise RuntimeError('源码/原库/训练参数已变化，必须使用新的运行目录')
            else:
                save(root / 'manifest.json', current)
            state['status'] = 'running'
            save(root / 'state.json', state)
            for stage in ('capture', 'tune', 'benchmark'):
                if stage in state['completed']:
                    if stage == 'capture' and sha(root / 'capture/layer3.clean.log') != state['input_digest']:
                        raise RuntimeError('采集输入已变化')
                    if stage == 'tune' and tree_digest(root / 'selected_config/library_gpu5') != state['config_digest']:
                        raise RuntimeError('选优配置归档已变化')
                    log(f'跳过已成功阶段: {stage}')
                    continue
                state['stage'] = stage
                save(root / 'state.json', state)
                if stage == 'capture':
                    archive(root / 'capture', root / 'failed_attempts')
                    capture = root / 'capture'
                    capture.mkdir()
                    wait_idle(capture / 'resource-wait.jsonl')
                    # Reuse the current benchmark combination directly, not duplicated env defaults.
                    env = clean_env()
                    old = dict(os.environ)
                    try:
                        os.environ.clear(); os.environ.update(env)
                        env = module.build_env(COMBO)
                    finally:
                        os.environ.clear(); os.environ.update(old)
                    env['ROCBLAS_LAYER'] = '3'
                    cmd = module.build_command(capture / 'training', trace=False, case=COMBO)
                    save(capture / 'launch.json', {'command': cmd, 'environment': {k: v for k, v in env.items() if k.startswith('GR00T_') or k in {'ROCBLAS_LAYER', 'NUM_GPUS', 'GLOBAL_BATCH_SIZE', 'MAX_STEPS', 'MASTER_PORT', 'VLM_MODEL_PATH', 'DATALOADER_NUM_WORKERS', 'USE_WANDB', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'}}})
                    command_run(cmd, env, capture / 'train.log', capture / 'layer3.raw.log')
                    count = validate_capture(root, module)
                    state['input_digest'] = sha(capture / 'layer3.clean.log')
                    log(f'采集成功：清理去重后{count}条命令')
                elif stage == 'tune':
                    wait_idle(root / 'tune-resource-wait.jsonl')
                    continuing = 'tune_work_digest' in state
                    if continuing and tune_work_digest() != state['tune_work_digest']:
                        raise RuntimeError('DTK调优中间产物已被改变，拒绝混用，需新建运行目录')
                    if not (root / 'backup-complete.json').exists():
                        for name in ('optimization_configs', 'optimization_artifacts', 'optimization_results',
                                     'rocblas_layer3_size', 'configs'):
                            source = TOOL / name
                            dest = root / 'before_tuning' / name
                            if source.exists() and not dest.exists():
                                backup_tree(source, dest)
                        save(root / 'backup-complete.json', {'time': stamp()})
                    cmd = ['bash', str(AUTOTUNE), 'run', '--source', 'layer3', '--hardware', 'BW',
                           '--gpus', '0 1 2 3 4 5 6 7', '--cu-count', '0']
                    # Explicit resume repeats this failed stage from its validated input.
                    # Never infer that a prelaunch failure initialized DTK's own resume state.
                    if continuing:
                        archive(root / 'tuning.log', root / 'failed_attempts')
                        evidence = root / 'failed_attempts' / ('tune-' + stamp())
                        for name in ('optimization_configs', 'optimization_artifacts', 'optimization_results'):
                            if (TOOL / name).exists():
                                backup_tree(TOOL / name, evidence / name)
                    cmd += ['--input', str(root / 'capture/layer3.clean.log')]
                    for key, value in TUNING.items():
                        cmd += ['--set', f'{key}={value}']
                    # Load DTK before wrapper checks/compilation as well as its internal launch.
                    shell = ['bash', '-c', 'set -e; source "$1"; shift; exec "$@"', 'gemm', str(DTK / 'env.sh'), *cmd]
                    # Backups may take long enough for another job to enter after the first check.
                    # Check again immediately before launching the tuning process, including on resume.
                    log('备份与输入准备完成，启动GEMM调优前再次检查资源')
                    wait_idle(root / 'tune-launch-resource-wait.jsonl')
                    try:
                        command_run(shell, clean_env(), root / 'tuning.log')
                    finally:
                        state['tune_work_digest'] = tune_work_digest()
                        save(root / 'state.json', state)
                    if '所有流程执行完毕' not in (root / 'tuning.log').read_text(errors='replace'):
                        raise RuntimeError('调优未确认六步完成')
                    snapshot = root / 'selected_config/library_gpu5'
                    archive(snapshot, root / 'failed_attempts')
                    shutil.copytree(TOOL / 'optimization_configs/new/config/library_gpu5', snapshot)
                    if not list(snapshot.glob('G*.dat')):
                        raise RuntimeError('没有生成G*.dat配置文件')
                    for name in ('optimization_results', 'configs'):
                        shutil.copytree(TOOL / name, root / 'tuning_evidence' / name, dirs_exist_ok=True)
                    state['config_digest'] = tree_digest(snapshot)
                else:
                    # Original run_background.sh is invoked unchanged. A private python3 shim selects
                    # this single-file adapter ONLY for its exact benchmark.py command.
                    runtime = root / 'runtime'
                    runtime.mkdir(exist_ok=True)
                    shim = runtime / 'python3'
                    actual = shlex.quote(sys.executable)
                    shim.write_text('#!/usr/bin/env bash\nset -e\n' +
                        f'if [[ "${{1:-}}" == -u && "${{2:-}}" == {shlex.quote(str(BENCH))} ]]; then\n' +
                        f'  exec bash {shlex.quote(str(SCRIPT))} _benchmark --output {shlex.quote(str(root))}\nfi\n' +
                        f'exec {actual} "$@"\n')
                    shim.chmod(0o755)
                    env = clean_env()
                    env.update(PATH=str(runtime) + os.pathsep + env['PATH'],
                               GEMM_WORKFLOW_PYTHON=sys.executable,
                               GR00T_BENCH_OUT_ROOT=str(runtime / ('background-' + stamp())),
                               ROCBLAS_TENSILE_LIBPATH=str(root / 'selected_config/library_gpu5'))
                    background_run(root, env, state)
                    rows = read(root / 'benchmark/results.json')
                    wanted = {(case, n) for case in module.CASES for n in (1, 2)}
                    if len(rows) != len(wanted) or {(r['case'], r['round']) for r in rows if r['valid']} != wanted:
                        raise RuntimeError('benchmark未完成全部CASES的两轮有效数据')
                state['completed'].append(stage)
                save(root / 'state.json', state)
            state.update(status='complete', stage='complete')
            save(root / 'state.json', state)
            (root / 'exit.code').write_text('0\n')
            log(f'全部完成，报告: {root / "benchmark/report.md"}')
        except BaseException as exc:
            state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            save(root / 'state.json', state)
            (root / 'exit.code').write_text('143\n' if isinstance(exc, KeyboardInterrupt) else '1\n')
            raise


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def alive(record):
    return record.get('start') is not None and process_start(record['pid']) == record['start']


def background_run(root, env, state):
    command_run(['bash', str(BACKGROUND), '--no-trace', '--rounds', '2'], env, root / 'background-launch.log')
    side = Path(env['GR00T_BENCH_OUT_ROOT'])
    # Use only the newly launched worker; never confuse a previous .exit with this invocation.
    pidfiles = sorted(side.glob('all-optimizations-*.pid'), key=lambda p: p.stat().st_mtime_ns)
    if not pidfiles:
        raise RuntimeError('后台入口未写出PID')
    pidfile = pidfiles[-1]
    pid = int(pidfile.read_text())
    record = {'pid': pid, 'start': process_start(pid)}
    state['background_worker'] = record
    save(root / 'state.json', state)
    exitfile = pidfile.with_suffix('.exit')
    try:
        while not exitfile.exists():
            if not alive(record):
                raise RuntimeError('后台worker退出且未写.exit；查看runtime/background下的日志')
            time.sleep(2)
        code = int(exitfile.read_text())
        if code:
            raise RuntimeError(f'benchmark退出码{code}，没有自动重试；查看{root / "benchmark"}')
    finally:
        if exitfile.exists():
            deadline = time.monotonic() + 5
            while alive(record) and time.monotonic() < deadline:
                time.sleep(.1)
        if alive(record):
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 40
            while alive(record) and time.monotonic() < deadline:
                time.sleep(.2)
            if alive(record):
                raise RuntimeError(f'后台worker {pid} 未及时退出，请检查其子进程；暂勿重启')


def main():
    parser = argparse.ArgumentParser(description='组合GEMM采集/选优，再按顺序后台采集全部CASES。无自动重试。')
    parser.add_argument('action', choices=('plan', 'start', 'run', 'status', 'stop', '_benchmark'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    if args.action == 'plan':
        module = load_benchmark()
        print(json.dumps({'capture': COMBO, 'capture_steps': 40, 'gpus': list(range(8)),
              'global_batch_size': 768, 'sequence_policy': 'pad-156', 'tuning': TUNING,
              'schedule': [[case, n] for case in module.CASES for n in (1, 2)],
              'trace': False, 'auto_retry': False, 'idle': 'each VRAM<2%, GPU<=5%, continuous60s',
              'configuration': 'archive + ROCBLAS_TENSILE_LIBPATH; no installed-library replacement'},
              ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    root = args.output.expanduser().resolve()
    if args.action == '_benchmark':
        benchmark_adapter(root)
        return
    if args.action in ('status', 'stop'):
        record = read(root / 'process.json') if (root / 'process.json').exists() else {}
        if args.action == 'status':
            print(json.dumps({'running': alive(record), 'state': read(root / 'state.json') if (root / 'state.json').exists() else None,
                              'exit_code': (root / 'exit.code').read_text().strip() if (root / 'exit.code').exists() else None},
                              ensure_ascii=False, indent=2))
        elif alive(record):
            os.kill(record['pid'], signal.SIGTERM)
            log('已请求停止；通过status确认进程退出')
        else:
            log('该运行没有存活的编排进程')
        return
    if args.action == 'start':
        if (root / 'process.json').exists() and alive(read(root / 'process.json')):
            raise RuntimeError('该输出目录已有运行进程')
        if root.exists() and not args.resume:
            raise RuntimeError('输出目录已存在；续跑请使用--resume')
        if args.resume and not (root / 'manifest.json').exists():
            raise RuntimeError('续跑目录缺少manifest.json')
        root.parent.mkdir(parents=True, exist_ok=True)
        logfile = root.with_name(root.name + '.log')
        cmd = ['bash', str(SCRIPT), 'run', '--output', str(root)] + (['--resume'] if args.resume else [])
        with logfile.open('ab') as stream:
            proc = subprocess.Popen(cmd, cwd=REPO, env={**os.environ, 'GEMM_WORKFLOW_PYTHON': sys.executable},
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError(f'后台启动失败，查看{logfile}')
            if (root / 'process.json').exists() and read(root / 'process.json')['pid'] == proc.pid and (root / 'state.json').exists():
                log(f'后台PID={proc.pid}；日志={logfile}；状态目录={root}')
                return
            time.sleep(.1)
        raise RuntimeError(f'尚未确认启动，请检查{logfile}，不要重复启动')
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'收到信号{signum}')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    run_workflow(root, args.resume)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('流程已停止')
        sys.exit(143)
    except Exception as exc:
        log(f'失败: {exc}')
        sys.exit(1)
WORKFLOW_PY
