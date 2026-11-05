import json
import importlib.util
import pathlib
import tempfile
import unittest

path = pathlib.Path(__file__).with_name('benchmark.py')
spec = importlib.util.spec_from_file_location('benchmark', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        port = patch.object(module, 'free_port', return_value='29599')
        port.start()
        self.addCleanup(port.stop)

    def test_case_major_schedule_includes_one_trace_and_skips_recorded_failures(self):
        rows = [{'case': 'baseline', 'round': 1, 'kind': 'speed', 'valid': False},
                {'case': 'baseline', 'round': 'trace', 'kind': 'trace', 'valid': True}]
        self.assertEqual(module.pending_runs(['baseline', 'dit_model'], 2, rows),
                         [('baseline', 2), ('dit_model', 1), ('dit_model', 2),
                          ('dit_model', 'trace')])

    def test_resume_rejects_old_layout_without_changing_it(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text('{}')
            (root / 'results.json').write_text('[]')
            (root / 'baseline-01').mkdir()
            with self.assertRaisesRegex(ValueError, 'layout'):
                module.load_resume_rows(root)
            self.assertEqual((root / 'results.json').read_text(), '[]')
            self.assertTrue((root / 'baseline-01').is_dir())

    def test_resume_keeps_trace_status_and_original_metadata(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            metadata = '{"layout_version": 2, "rounds": 2}'
            (root / 'metadata.json').write_text(metadata)
            rows = [{'case': 'baseline', 'round': 1, 'kind': 'speed', 'valid': False,
                     'error': 'unstable'}]
            (root / 'results.json').write_text(json.dumps(rows))
            (root / 'baseline/baseline-1').mkdir(parents=True)
            (root / 'baseline/baseline-1/keep').write_text('keep')
            (root / 'baseline/baseline-1/launch.json').write_text(json.dumps(
                {'environment': dict(module.DEFAULTS)}))
            launched = []
            def training(cmd, env, run, **kwargs):
                launched.append(run.relative_to(root).as_posix())
                (run / 'train.log').write_text('training')
                (run / 'step_metrics').mkdir()
                for i in range(8):
                    record = {'rank': i, 'world_size': 8, 'step_seconds': [1.] * 40,
                              'max_memory_allocated_bytes': (i + 1) * 1024 ** 2,
                              'max_memory_reserved_bytes': (i + 2) * 1024 ** 2,
                              'run_info': {'zero2_allgather_pipeline': False,
                                'zero2_overlap': False, 'zero2_reduce_scatter': False,
                                'compile_variant': 'off', 'global_batch_size': 768,
                                'num_gpus': 8}}
                    (run / 'step_metrics' / f'rank-{i:02d}.json').write_text(json.dumps(record))
                if run.name.endswith('trace'):
                    profile = run / 'profiling'
                    profile.mkdir()
                    for i in range(8):
                        (profile / f'trace_rank_{i}_iter_35.json').write_text('{"traceEvents": []}')
                        (profile / f'profiler-stopped-rank-{i}.json').write_text(
                            json.dumps({'rank': i, 'step': 35, 'stopped_after_trace': True}))
                    (profile / 'rank0-key-averages.txt').write_text('Self CUDA time total')
                return SimpleNamespace(returncode=0)
            args = SimpleNamespace(resume=True, rounds=2, cases=['baseline'])
            with patch.object(module, 'wait_for_idle_gpus', return_value=self.idle_cards()), \
                 patch.object(module, 'run_training', side_effect=training):
                with self.assertRaises(SystemExit):
                    module._run_benchmarks(args, root)
            self.assertEqual(launched, ['baseline/baseline-2', 'baseline/baseline-trace'])
            result = json.loads((root / 'results.json').read_text())
            self.assertEqual(result[0], rows[0])
            self.assertEqual(result[1]['pytorch_max_allocated_mib'], 8)
            self.assertTrue(result[2]['valid'])
            self.assertNotIn('step35_ms', result[2])
            self.assertEqual((root / 'metadata.json').read_text(), metadata)
            report = (root / 'report.md').read_text()
            self.assertIn('| baseline | 1/2 |', report)
            self.assertIn('baseline-trace', report)
            self.assertEqual((root / 'baseline/baseline-1/keep').read_text(), 'keep')

    def test_resume_rejects_duplicate_rows(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            row = {'case': 'baseline', 'round': 1, 'kind': 'speed', 'valid': True}
            (root / 'metadata.json').write_text('{"layout_version": 2, "rounds": 2}')
            (root / 'results.json').write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                module.load_resume_rows(root)

    def test_all_cases_include_enabled_baseline_compiler_flags(self):
        from unittest.mock import patch
        with patch.object(module, 'free_port', return_value='29599'):
            for case in module.CASES:
                with self.subTest(case=case):
                    env = module.build_env(case)
                    self.assertEqual(env['GR00T_COMPILE_VISION_BLOCKS'], '1')
                    self.assertEqual(env['GR00T_FLASH_ATTN_DYNAMO_DISABLE'], '1')

    def test_new_compile_cases_reach_training_environment(self):
        from unittest.mock import patch
        expected = {'combo_obr_dit_vl': 'dit-model+vl'}
        with patch.object(module, 'free_port', return_value='29599'):
            for case, variant in expected.items():
                with self.subTest(case=case):
                    self.assertIn(case, module.CASES)
                    env = module.build_env(case)
                    self.assertEqual(module.compile_expectation(env)[0], variant)
                    self.assertNotIn('GR00T_COMPILE_VARIANT', env)
                    self.assertEqual(env['MAX_STEPS'], '40')
                    self.assertEqual(env['GR00T_VISION_PATCH_EMBED_CHANNELS_LAST'], '1')

    def test_obr_combinations_build_expected_commands(self):
        expected = {
            'combo_obr': ('off', False),
            'combo_obr_dit_vl': ('dit-model+vl', False),
            'combo_obr_dit_vl_text': ('dit-model+vl+text-forwards', False),
            'combo_obr_dit_vl_text_allgather': ('dit-model+vl+text-forwards', True),
        }
        for case, (variant, gather) in expected.items():
            with self.subTest(case=case):
                self.assertIn(case, module.CASES)
                env = module.build_env(case)
                for key in ('GR00T_PIXEL_VALUES_BF16', 'GR00T_APEX_FUSED_ADAM',
                            'GR00T_ZERO2_OVERLAP', 'GR00T_ZERO2_REDUCE_SCATTER'):
                    self.assertEqual(env[key], '1')
                self.assertEqual(module.compile_expectation(env)[0], variant)
                self.assertEqual(env['GR00T_ZERO2_ALLGATHER_PIPELINE'], str(int(gather)))
                cmd = module.build_command(pathlib.Path('/tmp/run'), case=case)
                self.assertFalse([arg for arg in cmd if arg.startswith('--compile-')])

    def test_compile_expectation_matches_replacement_scopes(self):
        from gr00t.Replacement import model_compile
        for dit in ('off', 'model', 'blocks'):
            for vl in ('0', '1'):
                for text in ('off', 'layers', 'forwards'):
                    env = {'GR00T_COMPILE_DIT': dit, 'GR00T_COMPILE_VL': vl, 'GR00T_COMPILE_TEXT': text}
                    scopes = model_compile.resolve_scopes(env)
                    name, paths = module.compile_expectation(env)
                    self.assertEqual(name, scopes.name)
                    self.assertEqual(paths, model_compile.expected_boundaries(scopes))
                    self.assertEqual(list(paths), model_compile.expected_paths(scopes))

    def test_compile_validation_requires_exact_paths_and_boundaries(self):
        case = 'combo_obr_dit_vl_text'
        name, paths = module.compile_expectation({**module.DEFAULTS, **module.CASES[case]})
        self.assertEqual(len(paths), 21)
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            (run / 'train.log').write_text('pixel_values is cast to BF16 in DataLoader workers')
            (run / 'compilation').mkdir()
            info = {'zero2_allgather_pipeline': False, 'zero2_overlap': True,
                    'zero2_reduce_scatter': True, 'compile_variant': name,
                    'global_batch_size': 768, 'num_gpus': 8,
                    'compiled_paths': [p for p in paths if not p.startswith('backbone')]}
            records = [{'rank': i, 'run_info': info} for i in range(8)]
            with self.assertRaisesRegex(ValueError, 'compiled paths'):
                module.validate_run(run, case, records)
            info['compiled_paths'] = list(paths)
            with self.assertRaisesRegex(ValueError, 'boundaries'):
                module.validate_run(run, case, records)
            for i in range(8):
                (run / 'compilation' / f'rank-{i:02d}.json').write_text(
                    json.dumps({'boundaries': {**paths, 'action_head.model': 'forward'}}))
            with self.assertRaisesRegex(ValueError, 'boundaries'):
                module.validate_run(run, case, records)
            for i in range(8):
                (run / 'compilation' / f'rank-{i:02d}.json').write_text(
                    json.dumps({'boundaries': paths}))
            module.validate_run(run, case, records)

    @staticmethod
    def idle_cards():
        return {f'card{i}': {'vram Total Used Memory (MiB)': '0',
                            'vram Total Memory (MiB)': '65536',
                            'HCU use (%)': '0'} for i in range(8)}

    def test_device_idle_uses_each_cards_memory_below_configured_threshold(self):
        cards = self.idle_cards()
        self.assertTrue(module.devices_idle(cards))
        cards['card7']['vram Total Used Memory (MiB)'] = str(65536 * (module.MEMORY_START_THRESHOLD_PCT - .01) / 100)
        cards['card7']['HCU use (%)'] = '100'
        self.assertTrue(module.devices_idle(cards))
        cards['card7']['vram Total Used Memory (MiB)'] = str(65536 * module.MEMORY_START_THRESHOLD_PCT / 100)
        self.assertFalse(module.devices_idle(cards))
        cards['card7']['vram Total Used Memory (MiB)'] = str(65536 * (module.MEMORY_START_THRESHOLD_PCT + .01) / 100)
        self.assertFalse(module.devices_idle(cards))
        del cards['card7']
        with self.assertRaises(ValueError):
            module.devices_idle(cards)

    def test_device_statistics_normalizes_bytes_and_ignores_missing_gpu_util(self):
        cards = {f'card{i}': {'vram Total Used Memory (B)': str(1024 * 1024),
                            'vram Total Memory (B)': str(100 * 1024 * 1024),
                            'GPU use (%)': str(i * 10)} for i in range(8)}
        stats = module.device_statistics(cards)
        self.assertEqual(stats['average_used_memory_mib'], 1)
        self.assertEqual(stats['average_memory_pct'], 1)
        self.assertEqual(stats['average_gpu_util_pct'], 35)
        del cards['card0']['GPU use (%)']
        self.assertIsNone(module.device_statistics(cards)['average_gpu_util_pct'])
        self.assertTrue(module.devices_idle(cards))

    def test_invalid_memory_is_never_ready(self):
        for field, value in [('vram Total Memory (MiB)', '0'),
                             ('vram Total Used Memory (MiB)', 'nan'),
                             ('vram Total Used Memory (MiB)', '-1')]:
            with self.subTest(field=field, value=value):
                cards = self.idle_cards()
                cards['card0'][field] = value
                with self.assertRaises(ValueError):
                    module.devices_idle(cards)

    def test_wait_retries_busy_and_query_error_every_ten_seconds(self):
        idle = self.idle_cards()
        busy = self.idle_cards()
        busy['card0']['vram Total Used Memory (MiB)'] = str(65536 * (module.MEMORY_START_THRESHOLD_PCT + 1) / 100)
        responses = iter([busy, RuntimeError('query failed'), idle])
        sleeps = []
        def query():
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / 'wait.jsonl'
            from contextlib import redirect_stdout
            from io import StringIO
            output = StringIO()
            with redirect_stdout(output):
                result = module.wait_for_idle_gpus(log, query=query, sleep=sleeps.append)
            self.assertEqual(result, idle)
            self.assertEqual(sleeps, [10, 10])
            import json
            polls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([p['idle'] for p in polls], [False, False, True])
            self.assertAlmostEqual(polls[0]['average_memory_pct'], (module.MEMORY_START_THRESHOLD_PCT + 1) / 8)
            self.assertIn('八卡平均', output.getvalue())
            self.assertIn('GPU 利用率', output.getvalue())

    def test_combination_cases_are_distinct_and_support_controlled_comparisons(self):
        combinations = {case for case in module.CASES if case.startswith('combo_')}
        self.assertEqual(combinations, {'combo_runtime', 'combo_dit', 'combo_text_layers',
                                        'combo_text_forwards', 'combo_obr', 'combo_obr_dit_vl',
                                        'combo_obr_dit_vl_text', 'combo_obr_dit_vl_text_allgather'})
        envs = {case: module.build_env(case) for case in module.CASES}
        keys = set(module.DEFAULTS)
        self.assertEqual(len({tuple(env[k] for k in sorted(keys)) for env in envs.values()}), len(envs))
        for env in envs.values():
            self.assertEqual(env['GR00T_COMPILE_SEQUENCE_POLICY'], 'pad-156')
        self.assertEqual({k for k in keys if envs['combo_obr'][k] != envs['combo_obr_dit_vl'][k]},
                         {'GR00T_COMPILE_DIT', 'GR00T_COMPILE_VL'})
        self.assertEqual({k for k in keys if envs['combo_obr_dit_vl'][k] !=
                          envs['combo_obr_dit_vl_text'][k]}, {'GR00T_COMPILE_TEXT'})
        self.assertEqual({k for k in keys if envs['combo_obr_dit_vl_text'][k] !=
                          envs['combo_obr_dit_vl_text_allgather'][k]}, {'GR00T_ZERO2_ALLGATHER_PIPELINE'})

    def test_sequence_policy_default_applies_to_all_cases_and_report(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with patch.dict(module.DEFAULTS, {'GR00T_COMPILE_SEQUENCE_POLICY': 'auto'}):
                for case in module.CASES:
                    self.assertEqual(module.build_env(case)['GR00T_COMPILE_SEQUENCE_POLICY'], 'auto')
                module.write_report(root, [])
            self.assertIn('auto', (root / 'report.md').read_text())

    def test_cli_auto_policy_overrides_default_for_every_case(self):
        import json
        import subprocess
        import sys
        result = subprocess.run([sys.executable, str(path), '--output', '/tmp/benchmark-preview',
                                 '--cases', 'baseline', 'combo_obr_dit_vl', '--sequence-policy', 'auto',
                                 '--dry-run'], text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual({env['GR00T_COMPILE_SEQUENCE_POLICY']
                          for env in plan['environments'].values()}, {'auto'})

    def test_background_runs_once_and_records_failure(self):
        import os
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            fake_python = fake_bin / 'python3'
            fake_python.write_text('''#!/bin/bash
case " $* " in
  *" --sequence-policy auto "*) mode=auto ;;
  *) mode=default ;;
esac
printf 'begin %s %s\\n' "$mode" "$*" >> "$BENCHMARK_TEST_EVENTS"
sleep .1
printf 'end %s\\n' "$mode" >> "$BENCHMARK_TEST_EVENTS"
if [ "$mode" = default ]; then exit 1; fi
''')
            fake_python.chmod(0o755)
            shell = root / 'run_background.sh'
            source = path.with_name('run_background.sh').read_text()
            shell.write_text(source.replace('/workspace/data/outputs/', f'{root}/'))
            (root / 'benchmark.py').touch()
            events = root / 'events'
            env = dict(os.environ, PATH=f'{fake_bin}:{os.environ["PATH"]}',
                       BENCHMARK_TEST_EVENTS=str(events), GR00T_BENCH_OUT_ROOT=str(root))
            result = subprocess.run(['bash', str(shell)], env=env, text=True,
                                    capture_output=True, check=True, timeout=5)
            for _ in range(100):
                if list(root.glob('all-optimizations-*.exit')):
                    break
                time.sleep(.05)
            else:
                self.fail('background collection did not finish')
            lines = events.read_text().splitlines()
            self.assertEqual([line.split()[0:2] for line in lines],
                             [['begin', 'default'], ['end', 'default']])
            self.assertNotIn('--sequence-policy', lines[0])
            first_output = lines[0].split('--output ', 1)[1].split()[0]
            self.assertIn(first_output, result.stdout)
            self.assertEqual(pathlib.Path(first_output + '.exit').read_text().strip(), '1')

    def test_stopping_background_runner_stops_child_and_skips_auto(self):
        import os
        import signal
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            fake_python = fake_bin / 'python3'
            fake_python.write_text('''#!/bin/bash
trap 'echo stopped >> "$BENCHMARK_TEST_EVENTS"; exit 143' TERM
echo started >> "$BENCHMARK_TEST_EVENTS"
while true; do sleep .1; done
''')
            fake_python.chmod(0o755)
            shell = root / 'run_background.sh'
            shell.write_text(path.with_name('run_background.sh').read_text().replace(
                '/workspace/data/outputs/', f'{root}/'))
            (root / 'benchmark.py').touch()
            events = root / 'events'
            env = dict(os.environ, PATH=f'{fake_bin}:{os.environ["PATH"]}',
                       BENCHMARK_TEST_EVENTS=str(events), GR00T_BENCH_OUT_ROOT=str(root))
            result = subprocess.run(['bash', str(shell)], env=env, text=True,
                                    capture_output=True, check=True, timeout=5)
            pid = int(result.stdout.split('进程 PID：', 1)[1].splitlines()[0])
            try:
                for _ in range(100):
                    if events.exists() and 'started' in events.read_text():
                        break
                    time.sleep(.05)
                else:
                    self.fail('first run did not start')
                os.kill(pid, signal.SIGTERM)
                for _ in range(100):
                    if 'stopped' in events.read_text():
                        break
                    time.sleep(.05)
                else:
                    self.fail('child did not receive termination')
                self.assertEqual(events.read_text().splitlines(), ['started', 'stopped'])
            finally:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def test_resume_rejects_missing_launch_config(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text('{"layout_version": 2, "rounds": 2}')
            (root / 'results.json').write_text(json.dumps([
                {'case': 'baseline', 'round': 1, 'kind': 'speed', 'valid': True}]))
            with self.assertRaisesRegex(ValueError, 'sequence policy'):
                module.load_resume_rows(root, rounds=2)

    def test_resume_rejects_old_allgather_optimizer_configuration(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text('{"layout_version": 2, "rounds": 2}')
            (root / 'results.json').write_text(json.dumps([
                {'case': 'combo_obr_dit_vl_text_allgather', 'round': 1, 'kind': 'speed', 'valid': True}]))
            run = root / 'combo_obr_dit_vl_text_allgather/combo_obr_dit_vl_text_allgather-1'
            run.mkdir(parents=True)
            env = {**module.DEFAULTS, **module.CASES['combo_obr_dit_vl_text_allgather'], 'GR00T_APEX_FUSED_ADAM': '0'}
            (run / 'launch.json').write_text(json.dumps({'environment': env}))
            with self.assertRaisesRegex(ValueError, 'GR00T_APEX_FUSED_ADAM'):
                module.load_resume_rows(root, rounds=2)

    def test_resume_rejects_recorded_case_with_different_sequence_policy(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text('{"layout_version": 2, "rounds": 2}')
            (root / 'results.json').write_text(json.dumps([
                {'case': 'combo_obr_dit_vl', 'round': 1, 'kind': 'speed', 'valid': True}]))
            run = root / 'combo_obr_dit_vl/combo_obr_dit_vl-1'
            run.mkdir(parents=True)
            env = {**module.DEFAULTS, **module.CASES['combo_obr_dit_vl'], 'GR00T_COMPILE_SEQUENCE_POLICY': 'auto'}
            (run / 'launch.json').write_text(json.dumps({'environment': env}))
            with self.assertRaisesRegex(ValueError, 'sequence policy'):
                module.load_resume_rows(root, rounds=2)

    def test_trace_command_collects_only_step35_with_stack(self):
        cmd = module.build_command(pathlib.Path('/tmp/case-trace'), trace=True)
        for flag, value in {'--profiler-skip-first': '28', '--profiler-wait': '1',
                            '--profiler-warmup': '5', '--profiler-active': '1',
                            '--profiler-repeat': '1'}.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--profiler-with-stack', cmd)
        self.assertIn('--enable-profiling', cmd)
        speed = module.build_command(pathlib.Path('/tmp/case-1'))
        self.assertIn('--no-enable-profiling', speed)
        self.assertNotIn('--profiler-with-stack', speed)

    def test_training_monitor_samples_real_child_and_records_errors(self):
        import json
        import sys
        import os
        from unittest.mock import patch
        calls = []
        def query():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError('temporary telemetry failure')
            cards = self.idle_cards()
            cards['card7']['vram Total Used Memory (MiB)'] = '1234'
            return cards
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            with patch.object(module, 'REPO', run):
                result = module.run_training([sys.executable, '-c',
                    'import time; print("child log"); time.sleep(.25)'],
                    dict(os.environ), run, query=query, interval=.02)
            self.assertEqual(result.returncode, 0)
            rows = [json.loads(line) for line in (run / 'memory.jsonl').read_text().splitlines()]
            self.assertIn('temporary telemetry failure', rows[0]['error'])
            self.assertEqual(rows[1]['cards']['card7']['vram Total Used Memory (MiB)'], '1234')
            self.assertIn('child log', (run / 'train.log').read_text())
            summary = module.summarize_device_memory(run / 'memory.jsonl')
            self.assertEqual(summary['device_peak_used_mib_by_card']['card7'], 1234)
            self.assertEqual(summary['device_max_used_mib'], 1234)
            self.assertEqual(summary['memory_query_errors'], 1)

    def test_cpu_profiler_records_step35_only_and_training_continues(self):
        import torch
        cmd = module.build_command(pathlib.Path('/tmp/case-trace'), trace=True)
        values = {name: int(cmd[cmd.index('--profiler-' + name.replace('_', '-')) + 1])
                  for name in ('skip_first', 'wait', 'warmup', 'active', 'repeat')}
        captured, steps = [], []
        def ready(prof):
            captured.append({event.key for event in prof.key_averages()
                             if event.key.startswith('train_step_')})
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(**values), with_stack=True,
                on_trace_ready=ready) as prof:
            for step in range(1, 41):
                with torch.profiler.record_function(f'train_step_{step}'):
                    torch.ones(2).sum()
                prof.step()
                steps.append(step)
        self.assertEqual(captured, [{'train_step_35'}])
        self.assertEqual(steps[-1], 40)

    def test_failed_training_preserves_device_memory_and_all_run_statuses(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            def training(cmd, env, run, **kwargs):
                self.assertEqual(env['MAX_STEPS'], '40')
                if run.name.endswith('trace'):
                    self.assertEqual(env['GR00T_PROFILER_STOP_AFTER_TRACE'], '1')
                    self.assertFalse(kwargs['monitor'])
                else:
                    self.assertTrue(kwargs['monitor'])
                    (run / 'memory.jsonl').write_text(json.dumps({'cards': self.idle_cards()}) + '\n')
                return SimpleNamespace(returncode=1)
            args = SimpleNamespace(resume=False, rounds=2, cases=['baseline'])
            with patch.object(module, 'wait_for_idle_gpus', return_value=self.idle_cards()), \
                 patch.object(module, 'run_training', side_effect=training):
                with self.assertRaises(SystemExit):
                    module._run_benchmarks(args, root)
            result = json.loads((root / 'results.json').read_text())
            self.assertEqual([r['round'] for r in result], [1, 2, 'trace'])
            self.assertEqual(result[0]['memory_samples'], 1)
            self.assertEqual(result[0]['device_max_used_mib'], 0)
            self.assertIn('training exit 1', result[2]['error'])
            metadata = json.loads((root / 'metadata.json').read_text())
            self.assertEqual(metadata['layout_version'], 2)
            self.assertEqual(metadata['rounds'], 2)
            self.assertTrue((root / 'baseline/baseline-trace/launch.json').is_file())

    def test_training_timeout_stops_child_and_monitor(self):
        import os
        import sys
        import subprocess
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            with patch.object(module, 'REPO', run):
                with self.assertRaises(subprocess.TimeoutExpired):
                    module.run_training([sys.executable, '-c',
                        'import os,time; open("pid", "w").write(str(os.getpid())); time.sleep(30)'],
                        dict(os.environ), run, query=self.idle_cards, interval=.02, timeout=.3)
            pid = int((run / 'pid').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertTrue((run / 'memory.jsonl').stat().st_size > 0)

    def test_timeout_kills_workers_even_if_launcher_exits_on_term(self):
        import os
        import sys
        import subprocess
        from unittest.mock import patch
        worker_code = ('import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                       'open("worker-pid", "w").write(str(os.getpid())); time.sleep(30)')
        parent_code = (f'import subprocess,sys,time; subprocess.Popen([sys.executable, "-c", '
                       f'{worker_code!r}]); time.sleep(30)')
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            pid = None
            try:
                with patch.object(module, 'REPO', run):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        module.run_training([sys.executable, '-c', parent_code], dict(os.environ),
                                            run, monitor=False, timeout=.4)
                pid = int((run / 'worker-pid').read_text())
                import time
                for _ in range(50):
                    stat = pathlib.Path(f'/proc/{pid}/stat')
                    if not stat.exists() or stat.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('training worker survived launcher timeout')
            finally:
                if pid is not None:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass

    def test_exited_launcher_cannot_leave_workers_running(self):
        import os
        import sys
        import time
        from unittest.mock import patch
        worker = ('import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  'open("worker-pid", "w").write(str(os.getpid())); time.sleep(30)')
        parent = ('import subprocess,sys,time; from pathlib import Path; '
                  f'subprocess.Popen([sys.executable,"-c",{worker!r}]); '
                  '\nwhile not Path("worker-pid").exists(): time.sleep(.01)'
                  '\nsys.exit(1)')
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            with patch.object(module, 'REPO', run):
                result = module.run_training([sys.executable, '-c', parent], dict(os.environ),
                                             run, monitor=False)
            self.assertEqual(result.returncode, 1)
            pid = int((run / 'worker-pid').read_text())
            try:
                for _ in range(50):
                    stat = pathlib.Path(f'/proc/{pid}/stat')
                    if not stat.exists() or stat.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('worker survived exited launcher')
            finally:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass

    def test_interruption_records_failed_run_so_resume_can_continue(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            args = SimpleNamespace(resume=False, rounds=2, cases=['baseline'])
            with patch.object(module, 'wait_for_idle_gpus', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    module._run_benchmarks(args, root)
            rows = module.load_resume_rows(root, rounds=2)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]['valid'])
            self.assertIn('Interrupted', rows[0]['error'])
            self.assertEqual(module.pending_runs(['baseline'], 2, rows),
                             [('baseline', 2), ('baseline', 'trace')])

    def test_main_sigterm_cleans_training_process_and_records_interruption(self):
        import subprocess
        import sys
        import time
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for relative in ['repo/examples', 'data/checkpoints/GR00T-N1.7-3B-local-cosmos',
                             'data/checkpoints/nvidia/Cosmos-Reason2-2B',
                             'data/libero_10_no_noops_1.0.0_lerobot_h264']:
                (root / relative).mkdir(parents=True)
            (root / 'repo/examples/finetune.sh').touch()
            child_code = ('import os,time; open("training-pid", "w").write(str(os.getpid())); '
                          'time.sleep(30)')
            code = f"""
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('benchmark',{str(path)!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.REPO=Path({str(root / 'repo')!r}); m.DATA=Path({str(root / 'data')!r})
m.free_port=lambda:'29599'
m.wait_for_idle_gpus=lambda *a:{{}}
m.build_command=lambda *a,**kw:[sys.executable,'-c',{child_code!r}]
sys.argv=['benchmark.py','--output',{str(root / 'out')!r},'--cases','baseline']
m.main()
"""
            proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            pid = None
            try:
                pid_path = root / 'repo/training-pid'
                for _ in range(100):
                    if pid_path.exists() and pid_path.read_text():
                        break
                    time.sleep(.02)
                pid = int(pid_path.read_text())
                proc.terminate()
                proc.wait(timeout=5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                self.assertEqual(module.load_resume_rows(root / 'out')[0]['valid'], False)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if pid is not None:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass

    def test_malformed_telemetry_does_not_lose_valid_samples(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'memory.jsonl'
            path.write_text(json.dumps({'cards': self.idle_cards()}) + '\n' +
                            '{"cards": {}}\n' + '{partial')
            result = module.summarize_device_memory(path)
            self.assertEqual(result['memory_samples'], 1)
            self.assertEqual(result['memory_query_errors'], 2)

    def test_all_telemetry_errors_are_missing_instead_of_zero(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'memory.jsonl'
            path.write_text(json.dumps({'error': 'unavailable'}) + '\n')
            result = module.summarize_device_memory(path)
            self.assertIsNone(result['device_max_used_mib'])
            self.assertEqual(result['memory_samples'], 0)
            self.assertEqual(result['memory_query_errors'], 1)

    def test_trace_wrong_step_and_missing_rank_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            profile = run / 'profiling'
            profile.mkdir()
            for rank in range(8):
                (profile / f'trace_rank_{rank}_iter_35.json').write_text('{}')
                (profile / f'profiler-stopped-rank-{rank}.json').write_text(json.dumps(
                    {'rank': rank, 'step': 35, 'stopped_after_trace': True}))
            (profile / 'rank0-key-averages.txt').write_text('table')
            self.assertEqual(module.validate_trace(run)['trace_step'], 35)
            stop = profile / 'profiler-stopped-rank-7.json'
            stop.write_text(json.dumps({'rank': 7, 'step': 36, 'stopped_after_trace': True}))
            with self.assertRaisesRegex(ValueError, 'rank 7'):
                module.validate_trace(run)
            (profile / 'trace_rank_7_iter_35.json').unlink()
            with self.assertRaisesRegex(ValueError, 'rank 7'):
                module.validate_trace(run)

    def test_resume_refuses_changed_round_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text('{"layout_version": 2, "rounds": 2}')
            (root / 'results.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, '--rounds'):
                module.load_resume_rows(root, rounds=3)

    def test_missing_trace_artifact_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((ValueError, FileNotFoundError)):
                module.validate_trace(pathlib.Path(directory))

    def test_trace_does_not_enter_speed_tables(self):
        rows = [dict(module.measure([{'rank': r, 'world_size': 8,
                    'step_seconds': [1.] * 40} for r in range(8)]),
                    case='baseline', round=i, kind='speed') for i in (1, 2)]
        rows.append({'case': 'baseline', 'round': 'trace', 'kind': 'trace', 'valid': True})
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            module.write_report(root, rows)
            report = (root / 'report.md').read_text()
            self.assertIn('| baseline | 2/2 | 1000.000 | 768.000 |', report)
            self.assertIn('baseline-trace', report)

    def test_35_and_mean_31_40_use_step_slowest_rank(self):
        rows = [{'rank': r, 'world_size': 8, 'step_seconds': [1.0] * 40} for r in range(8)]
        rows[2]['step_seconds'][34] = 2.0
        rows[3]['step_seconds'][30] = 3.0
        result = module.measure(rows)
        self.assertAlmostEqual(result['step35_ms'], 2000)
        self.assertAlmostEqual(result['step35_samples_per_second'], 384)
        self.assertAlmostEqual(result['mean31_40_ms'], 1300)
        self.assertAlmostEqual(result['mean31_40_samples_per_second'], 768 / 1.3)

    def test_missing_rank_rejected(self):
        with self.assertRaises(ValueError):
            module.measure([{'rank': 0, 'world_size': 8, 'step_seconds': [1.0] * 40}])

    def test_report_has_two_tables_and_no_layout_off(self):
        self.assertNotIn('layout_off', module.CASES)
        self.assertEqual(module.DEFAULTS['GR00T_VISION_PATCH_EMBED_CHANNELS_LAST'], '1')
        rows = [{'rank': r, 'world_size': 8, 'step_seconds': [1.0] * 40} for r in range(8)]
        result = module.measure(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            module.write_report(root, [dict(result, case='baseline', round=i) for i in (1, 2)])
            report = (root / 'report.md').read_text()
            self.assertIn('第 35 步', report)
            self.assertIn('第 31–40 步', report)
            self.assertIn('768.000', report)
            self.assertIn('第 35 步吞吐量 samples/s', report)


if __name__ == '__main__': unittest.main()
