"""Event bookkeeping that lets backward compute run while a gradient bucket is being reduced."""

from __future__ import annotations

from typing import Any, Callable, Hashable


class BucketEventScheduler:
    """Track when each half of a double-buffered gradient bucket is free for reuse.

    The reduction stream records an event once a buffer has been fully consumed. The compute
    stream waits on that event only when it is about to write the same buffer again, instead of
    waiting for every reduction as DeepSpeed's second ``wait_stream`` does.
    """

    def __init__(self, current_stream: Callable[[], Any], event_factory: Callable[[], Any]):
        self._current_stream = current_stream
        self._event_factory = event_factory
        self._events: dict[Hashable, Any] = {}
        self.records = 0
        self.waits = 0

    def reset(self) -> None:
        self._events.clear()

    def record_consumed(self, key: Hashable, reduction_stream: Any) -> None:
        event = self._event_factory()
        event.record(reduction_stream)
        self._events[key] = event
        self.records += 1

    def wait_before_reuse(self, key: Hashable) -> bool:
        event = self._events.pop(key, None)
        if event is None:
            return False
        self._current_stream().wait_event(event)
        self.waits += 1
        return True
