from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _compact_topk_ids_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    ids = tl.load(topk_ids_ptr + pair_slots, mask=pair_slots < total_pairs, other=0).to(tl.int32)

    first_flags = tl.zeros((BLOCK,), dtype=tl.int32)
    for pair_idx in range(BLOCK):
        is_valid = pair_idx < total_pairs
        expert_id = tl.load(topk_ids_ptr + pair_idx, mask=is_valid, other=0).to(tl.int32)
        prior_same = is_valid & (pair_slots < pair_idx) & (ids == expert_id)
        has_prior = tl.sum(prior_same.to(tl.int32), axis=0) > 0
        is_first = is_valid & (~has_prior)
        first_flags = tl.where(pair_slots == pair_idx, is_first.to(tl.int32), first_flags)

    for pair_idx in range(BLOCK):
        is_valid = pair_idx < total_pairs
        expert_id = tl.load(topk_ids_ptr + pair_idx, mask=is_valid, other=0).to(tl.int32)
        prior_same = is_valid & (pair_slots < pair_idx) & (ids == expert_id)
        prior_slots = tl.where(prior_same, pair_slots, BLOCK)
        first_match = tl.min(prior_slots, axis=0)
        first_slot = tl.where(first_match < BLOCK, first_match, pair_idx)
        compact_id = tl.sum(tl.where(pair_slots <= first_slot, first_flags, 0), axis=0) - 1
        tl.store(compact_topk_ids_ptr + pair_idx, compact_id, mask=is_valid)
        is_first = tl.sum(tl.where(pair_slots == pair_idx, first_flags, 0), axis=0) != 0
        tl.store(weight_expert_ids_ptr + compact_id, expert_id, mask=is_valid & is_first)

    active_expert_count = tl.sum(first_flags, axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


def compact_topk_ids(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
) -> None:
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        active_expert_count.zero_()
        return
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    if weight_expert_ids.numel() < total_pairs:
        raise ValueError("weight_expert_ids must have at least total_pairs elements")
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")

    block = triton.next_power_of_2(total_pairs)
    _compact_topk_ids_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        total_pairs,
        BLOCK=block,
        num_warps=1,
    )
