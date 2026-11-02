# SPDX-License-Identifier: Apache-2.0
"""Profiler shutdown must occur once, after the exported trace and before later steps."""
import pytest
import torch

from gr00t.experiment import profiling


def test_cpu_profiler_exports_once_then_stops_before_later_steps(tmp_path):
    assert hasattr(profiling, "StopAfterTrace"), "Missing stop-after-trace controller"
    traces, stops = [], []
    holder = {}

    def ready(prof):
        trace = tmp_path / f"trace-{len(traces)}.json"
        prof.export_chrome_trace(str(trace))
        traces.append(trace)
        holder["controller"].mark_trace_exported()

    raw = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=ready,
        with_stack=True,
    )
    controller = profiling.StopAfterTrace(raw, on_stopped=lambda: stops.append(len(traces)))
    holder["controller"] = controller
    with controller as prof:
        for _ in range(10):
            with torch.profiler.record_function("cpu_work"):
                torch.ones(2).sum()
            prof.step()
    assert len(traces) == 1
    assert traces[0].stat().st_size > 0
    assert stops == [1]
    assert raw.step_num == 4


def test_profiler_is_closed_once_if_training_raises():
    assert hasattr(profiling, "StopAfterTrace"), "Missing stop-after-trace controller"

    class Profiler:
        def __init__(self): self.started = self.stopped = 0
        def start(self): self.started += 1
        def stop(self): self.stopped += 1
        def step(self): pass

    raw = Profiler()
    with pytest.raises(ValueError, match="training failed"):
        with profiling.StopAfterTrace(raw):
            raise ValueError("training failed")
    assert (raw.started, raw.stopped) == (1, 1)
