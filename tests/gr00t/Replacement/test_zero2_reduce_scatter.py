import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gr00t.Replacement import zero2_reduce_scatter as rs


def _flat(rank: int, total: int, dtype=torch.float32) -> torch.Tensor:
    base = torch.arange(1, total + 1, dtype=torch.float64)
    return (base * (rank + 1)).to(dtype)


def _expected(total: int, world: int) -> torch.Tensor:
    return torch.arange(1, total + 1, dtype=torch.float64) * (world * (world + 1) // 2)


def _slices(flat: torch.Tensor, spec):
    return [flat.narrow(0, offset, numel) for _, offset, numel in spec], [owner for owner, _, _ in spec]


def _worker(rank, world, init_file, spec, total, max_elements, expected_path, slice_dtype, comm_dtype):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        flat = _flat(rank, total, slice_dtype)
        slices, owners = _slices(flat, spec)
        path = rs.reduce_by_owner(
            slices, owners, group=None, rank=rank, world_size=world, dtype=comm_dtype, max_elements=max_elements
        )
        assert path == expected_path
        expected = _expected(total, world).to(slice_dtype)
        for (owner, offset, numel) in spec:
            if owner == rank:
                assert torch.equal(flat[offset : offset + numel], expected[offset : offset + numel]), (rank, owner)
    finally:
        dist.destroy_process_group()


def _run(world, spec, total, *, max_elements=rs.DEFAULT_MAX_ELEMENTS, expected_path,
         slice_dtype=torch.float32, comm_dtype=torch.float32):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(world, os.path.join(directory, "store"), spec, total, max_elements, expected_path,
                  slice_dtype, comm_dtype),
            nprocs=world,
        )


def test_balanced_buckets_use_reduce_scatter():
    _run(2, [(0, 0, 6), (1, 6, 6)], 12, expected_path="reduce_scatter")


def test_interleaved_owners_cross_chunk_boundaries():
    spec = [(0, 0, 5), (1, 5, 3), (0, 8, 4), (1, 12, 6), (0, 18, 2)]
    _run(2, spec, 20, max_elements=6, expected_path="reduce_scatter")


def test_unbalanced_buckets_use_directed_reduce():
    spec = [(0, 0, 40), (1, 40, 1), (2, 41, 1), (3, 42, 1)]
    _run(4, spec, 43, expected_path="reduce")


def test_directed_reduce_converts_dtype_and_chunks():
    spec = [(0, 0, 40), (1, 40, 1), (2, 41, 1), (3, 42, 1)]
    _run(4, spec, 43, max_elements=7, expected_path="reduce",
         slice_dtype=torch.float32, comm_dtype=torch.float64)


def test_owner_without_slices_still_participates():
    spec = [(0, 0, 8), (2, 8, 8)]
    _run(3, spec, 16, expected_path="reduce_scatter")


def test_empty_bucket_is_a_no_op():
    assert rs.reduce_by_owner([], [], group=None, rank=0, world_size=2, dtype=torch.float32) == "empty"


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        rs.reduce_by_owner([torch.zeros(2)], [], group=None, rank=0, world_size=2, dtype=torch.float32)


def test_balance_rule():
    assert rs.is_balanced([5, 5, 5, 5])
    assert rs.is_balanced([10, 1])
    assert not rs.is_balanced([40, 1, 1, 1])


def test_window_helpers_round_trip_with_zero_padding():
    tensors = [torch.arange(1.0, 4.0), torch.arange(10.0, 12.0)]
    window = torch.full((4,), -1.0)
    rs._fill_window(tensors, 3, 4, window)
    assert window.tolist() == [10.0, 11.0, 0.0, 0.0]
    window = torch.full((6,), -1.0)
    rs._fill_window(tensors, 0, 6, window)
    assert window.tolist() == [1.0, 2.0, 3.0, 10.0, 11.0, 0.0]
    rs._store_window(tensors, 2, 3, torch.tensor([7.0, 8.0, 9.0]))
    assert tensors[0].tolist() == [1.0, 2.0, 7.0]
    assert tensors[1].tolist() == [8.0, 9.0]


def test_staging_buffer_is_reused_and_grows():
    rs.clear_staging_buffers()
    first = rs._staging_buffer("cpu", torch.float32, 8)
    again = rs._staging_buffer("cpu", torch.float32, 4)
    assert first.data_ptr() == again.data_ptr()
    larger = rs._staging_buffer("cpu", torch.float32, 16)
    assert larger.numel() == 16
    other_slot = rs._staging_buffer("cpu", torch.float32, 8, slot=1)
    assert other_slot.data_ptr() != larger.data_ptr()
    rs.clear_staging_buffers()


def test_preallocation_covers_a_balanced_bucket_without_growth():
    rs.clear_staging_buffers()
    rs.preallocate_staging("cpu", torch.float32, bucket_elements=40, world_size=4)
    sizes = {key: buffer.numel() for key, buffer in rs._staging.items()}
    assert sizes == {(torch.device("cpu"), torch.float32, 0): 80, (torch.device("cpu"), torch.float32, 1): 20}
    rs._staging_buffer("cpu", torch.float32, 80, slot=0)
    assert rs._staging[(torch.device("cpu"), torch.float32, 0)].numel() == 80
    rs.clear_staging_buffers()


def test_stats_count_elements_and_padding():
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=0, world_size=1)
        try:
            stats = {}
            slices = [torch.ones(6), torch.ones(2)]
            assert rs.reduce_by_owner(slices, [0, 0], group=None, rank=0, world_size=1,
                                      dtype=torch.float32, stats=stats) == "reduce_scatter"
            assert stats == {"elements": 8, "padding": 0, "collective_elements": 8}
        finally:
            dist.destroy_process_group()


def test_directed_reduce_counts_collective_elements_without_padding():
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=0, world_size=1)
        try:
            stats = {}
            rs._directed_reduce([torch.ones(5), torch.ones(0), torch.ones(3)], [0, 0, 0], group=None, rank=0,
                                dtype=torch.float32, max_elements=4, stats=stats)
            assert stats == {"collective_elements": 8}
        finally:
            dist.destroy_process_group()


def test_pack_events_are_never_collected_without_time_packing():
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=0, world_size=1)
        try:
            stats = {}
            rs.reduce_by_owner([torch.ones(4)], [0], group=None, rank=0, world_size=1,
                               dtype=torch.float32, stats=stats, time_packing=False)
            assert "pack_events" not in stats
        finally:
            dist.destroy_process_group()
