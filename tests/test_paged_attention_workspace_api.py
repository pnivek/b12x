from __future__ import annotations

import pytest
import torch

from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import (
    allocate_paged_attention_workspace,
    allocate_paged_attention_workspace_pool,
    b12x_paged_attention_forward,
    clear_attention_caches,
)

from .helpers import require_sm120


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _make_paged_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    page_table_width: int | None = None,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(q_seqlens) != len(cache_seqlens):
        raise ValueError("q_seqlens and cache_seqlens must have the same length")
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4

    pages_per_request = [(cache_len + page_size - 1) // page_size for cache_len in cache_seqlens]
    max_pages = max(pages_per_request, default=0)
    if page_table_width is not None:
        if page_table_width < max_pages:
            raise ValueError(
                f"page_table_width={page_table_width} is smaller than the required max_pages={max_pages}"
            )
        max_pages = page_table_width
    total_pages_needed = sum(pages_per_request)
    if num_pages is None:
        num_pages = max(1, total_pages_needed * 2)
    if num_pages < total_pages_needed:
        raise ValueError(f"num_pages={num_pages} is smaller than the required total {total_pages_needed}")

    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def test_paged_workspace_matches_reference_for_qwen_like_extend_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[33, 29, 41, 20],
        page_size=16,
        seed=23,
    )
    workspace = allocate_paged_attention_workspace(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    out, lse = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (lse - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_paged_workspace_pool_reuses_output_buffers() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 96, 48, 80],
        page_size=16,
        seed=31,
    )
    pool = allocate_paged_attention_workspace_pool()

    out0, lse0 = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=pool,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    out1, lse1 = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=pool,
    )
    torch.cuda.synchronize()

    assert len(pool.workspaces) == 1
    assert out0.data_ptr() == out1.data_ptr()
    assert lse0.data_ptr() == lse1.data_ptr()
    assert (out1 - ref_out).abs().max().item() <= 0.02
    assert (lse1 - ref_lse).abs().max().item() <= 0.03


def test_exact_paged_workspace_rejects_larger_cache_requirement() -> None:
    require_sm120()
    clear_attention_caches()

    q0, k0, v0, page_table0, cache_seqlens0, cu_seqlens_q0 = _make_paged_inputs(
        q_seqlens=[2, 2],
        cache_seqlens=[17, 17],
        page_size=16,
        seed=37,
        page_table_width=3,
        num_pages=10,
    )
    q1, k1, v1, page_table1, cache_seqlens1, cu_seqlens_q1 = _make_paged_inputs(
        q_seqlens=[2, 2],
        cache_seqlens=[33, 17],
        page_size=16,
        seed=41,
        page_table_width=3,
        num_pages=10,
    )
    workspace = allocate_paged_attention_workspace(
        q0,
        k0,
        v0,
        page_table0,
        cache_seqlens0,
        cu_seqlens_q0,
        causal=True,
    )

    with pytest.raises(ValueError, match="paged workspace capacity mismatch"):
        b12x_paged_attention_forward(
            q1,
            k1,
            v1,
            page_table1,
            cache_seqlens1,
            cu_seqlens_q1,
            workspace=workspace,
        )


def test_single_token_single_key_paged_corner_is_rejected() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[1],
        page_size=16,
        seed=53,
    )

    with pytest.raises(ValueError, match="single-token single-key corner"):
        allocate_paged_attention_workspace(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
        )
