"""Owner-based gradient reduction for ZeRO-2 buckets: ReduceScatter or directed Reduce."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist


DEFAULT_MAX_ELEMENTS = 64 * 1024 * 1024

_staging: dict[tuple, torch.Tensor] = {}


def _staging_buffer(device, dtype: torch.dtype, numel: int, slot: int = 0) -> torch.Tensor:
    """Return a persistent scratch view. ``preallocate_staging`` sizes it up front; it only grows
    afterwards for a bucket larger than the preallocation (for example one oversized parameter)."""
    key = (torch.device(device), dtype, slot)
    buffer = _staging.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(numel, dtype=dtype, device=device)
        _staging[key] = buffer
    return buffer[:numel]


def clear_staging_buffers() -> None:
    _staging.clear()


def preallocate_staging(device, dtype: torch.dtype, bucket_elements: int, world_size: int,
                        max_elements: int = DEFAULT_MAX_ELEMENTS) -> None:
    """Allocate the scratch buffers before backward starts.

    A balanced bucket pads every owner to the longest one (under 2x the bucket), so the packed
    buffer needs at most 2 * bucket_elements, capped by the window limit.
    """
    packed = min(max_elements, 2 * bucket_elements)
    _staging_buffer(device, dtype, packed, slot=0)
    _staging_buffer(device, dtype, max(1, packed // world_size), slot=1)


def owner_lengths(slices: Sequence[torch.Tensor], owners: Sequence[int], world_size: int) -> list[int]:
    lengths = [0] * world_size
    for tensor, owner in zip(slices, owners):
        lengths[owner] += tensor.numel()
    return lengths


def is_balanced(lengths: Sequence[int]) -> bool:
    """ReduceScatter pads every owner to the longest one; use it only when padding stays under 2x."""
    return max(lengths) * len(lengths) <= 2 * sum(lengths)


def _window_overlaps(tensors: Sequence[torch.Tensor], start: int, width: int):
    """Yield (tensor, tensor_lo, tensor_hi, window_lo) for the part of the concatenation in [start, start+width)."""
    offset = 0
    end = start + width
    for tensor in tensors:
        length = tensor.numel()
        lo = max(start, offset)
        hi = min(end, offset + length)
        if lo < hi:
            yield tensor, lo - offset, hi - offset, lo - start
        offset += length
        if offset >= end:
            break


def _fill_window(tensors: Sequence[torch.Tensor], start: int, width: int, destination: torch.Tensor) -> None:
    filled = 0
    for tensor, lo, hi, window_lo in _window_overlaps(tensors, start, width):
        destination[window_lo : window_lo + hi - lo].copy_(tensor[lo:hi])
        filled = window_lo + hi - lo
    if filled < width:
        destination[filled:].zero_()


def _store_window(tensors: Sequence[torch.Tensor], start: int, width: int, source: torch.Tensor) -> None:
    for tensor, lo, hi, window_lo in _window_overlaps(tensors, start, width):
        tensor[lo:hi].copy_(source[window_lo : window_lo + hi - lo])


def _reduce_scatter(per_owner, lengths, *, group, rank, dtype, max_elements, stats=None,
                    time_packing=False) -> None:
    world_size = len(per_owner)
    width = max(1, max_elements // world_size)
    longest = max(lengths)
    device = next(t.device for tensors in per_owner for t in tensors)
    chunk = min(width, longest)
    packed = _staging_buffer(device, dtype, world_size * chunk, slot=0)
    output = _staging_buffer(device, dtype, chunk, slot=1)
    for start in range(0, longest, width):
        window = min(width, longest - start)
        rows = packed[: world_size * window].view(world_size, window)
        timed = time_packing and stats is not None and device.type == "cuda"
        if timed:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        for owner in range(world_size):
            _fill_window(per_owner[owner], start, window, rows[owner])
        if timed:
            end.record()
            stats.setdefault("pack_events", []).append((begin, end))
        dist.reduce_scatter_tensor(output[:window], rows.view(-1), group=group)
        _store_window(per_owner[rank], start, window, output[:window])


def _directed_reduce(slices, owners, *, group, rank, dtype, max_elements, stats=None) -> None:
    for tensor, owner in zip(slices, owners):
        if tensor.numel() == 0:
            continue
        if stats is not None:
            stats["collective_elements"] = stats.get("collective_elements", 0) + tensor.numel()
        destination = dist.get_global_rank(group, owner) if group is not None else owner
        if tensor.dtype == dtype and tensor.is_contiguous():
            dist.reduce(tensor, destination, group=group)
            continue
        for start in range(0, tensor.numel(), max_elements):
            part = tensor[start : start + max_elements]
            scratch = _staging_buffer(tensor.device, dtype, part.numel(), slot=0)
            scratch.copy_(part)
            dist.reduce(scratch, destination, group=group)
            if owner == rank:
                part.copy_(scratch)


def reduce_by_owner(
    slices: Sequence[torch.Tensor],
    owners: Sequence[int],
    *,
    group,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    stats: dict | None = None,
    time_packing: bool = False,
) -> str:
    """Sum 1-D gradient slices across ranks so that each slice's owner holds the result.

    Only the owner's slices are valid afterwards; other ranks' copies are scratch. Returns the
    path taken: "empty", "reduce_scatter" or "reduce". When ``stats`` is given, it accumulates
    ``elements`` (logical gradient elements, not link traffic), ``padding`` (zero elements added by
    ReduceScatter) and ``collective_elements`` (elements fed to the collectives, padding included).
    Only with ``time_packing`` on a CUDA device does it also collect ``pack_events`` (begin/end
    event pairs around the packing copies); leave it off when measuring throughput.
    """
    if len(slices) != len(owners):
        raise ValueError("slices and owners must have the same length")
    lengths = owner_lengths(slices, owners, world_size)
    if sum(lengths) == 0:
        return "empty"
    if stats is not None:
        stats["elements"] = stats.get("elements", 0) + sum(lengths)
    if is_balanced(lengths):
        if stats is not None:
            stats["padding"] = stats.get("padding", 0) + max(lengths) * world_size - sum(lengths)
            stats["collective_elements"] = stats.get("collective_elements", 0) + max(lengths) * world_size
        per_owner = [[] for _ in range(world_size)]
        for tensor, owner in zip(slices, owners):
            per_owner[owner].append(tensor)
        _reduce_scatter(per_owner, lengths, group=group, rank=rank, dtype=dtype, max_elements=max_elements,
                        stats=stats, time_packing=time_packing)
        return "reduce_scatter"
    _directed_reduce(slices, owners, group=group, rank=rank, dtype=dtype, max_elements=max_elements,
                     stats=stats)
    return "reduce"
