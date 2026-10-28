from gr00t.Replacement.zero2_overlap import BucketEventScheduler


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))


class _Event:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream):
        self.log.append(("record", self.name, stream))


def _scheduler(log):
    counter = iter(range(100))
    return BucketEventScheduler(
        lambda: _Stream("compute", log), lambda: _Event(f"e{next(counter)}", log)
    )


def test_wait_only_after_the_same_half_was_consumed():
    log = []
    scheduler = _scheduler(log)
    assert scheduler.wait_before_reuse(("bf16", 0)) is False
    scheduler.record_consumed(("bf16", 0), "reduction")
    assert scheduler.wait_before_reuse(("bf16", 1)) is False
    assert scheduler.wait_before_reuse(("bf16", 0)) is True
    assert log == [("record", "e0", "reduction"), ("wait_event", "compute", "e0")]
    assert (scheduler.records, scheduler.waits) == (1, 1)


def test_event_is_consumed_by_the_wait():
    log = []
    scheduler = _scheduler(log)
    scheduler.record_consumed("k", "reduction")
    assert scheduler.wait_before_reuse("k") is True
    assert scheduler.wait_before_reuse("k") is False


def test_reset_drops_pending_events():
    log = []
    scheduler = _scheduler(log)
    scheduler.record_consumed("k", "reduction")
    scheduler.reset()
    assert scheduler.wait_before_reuse("k") is False
    assert scheduler.waits == 0
