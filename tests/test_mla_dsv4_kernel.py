"""Level-2 DSV4-Flash kernel tests: exercise actual compiled CuTe kernels.

All tests use synthetic random data (no model weights required).
Each test compares the CUDA kernel output against dense_mla_reference.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.mla.kernel import run_sparse_mla_kernel
from b12x.attention.mla.kernel_onepass import run_sparse_mla_kernel as run_sparse_mla_kernel_onepass
from b12x.attention.mla.split import run_sparse_mla_split_decode
from b12x.attention.mla.reference import (
    dense_mla_reference,
    pack_mla_kv_cache_reference,
)

from .helpers import require_sm120


_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_DSV4_HEAD_DIM = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM   # 512
_DSV4_V_HEAD_DIM = 448
_DSV4_SM_SCALE = _DSV4_HEAD_DIM ** -0.5
_DSV4_NUM_HEADS = 8


def _make_dsv4_tensors(
    *,
    num_tokens: int,
    num_q: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q_all = (
        torch.randn(
            (num_q, _DSV4_NUM_HEADS, _DSV4_HEAD_DIM),
            generator=g,
            dtype=torch.bfloat16,
        ).to(device)
        * 0.02
    )
    k_nope = (
        torch.randn(
            (num_tokens, 1, _DSV4_NOPE_DIM),
            generator=g,
            dtype=torch.bfloat16,
        ).to(device)
        * 0.02
    )
    k_rope = (
        torch.randn(
            (num_tokens, 1, _DSV4_ROPE_DIM),
            generator=g,
            dtype=torch.bfloat16,
        ).to(device)
        * 0.02
    )
    return q_all, k_nope, k_rope


def _compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a - b).float()
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return diff.abs().max().item(), diff.square().mean().sqrt().item(), cos


def _make_sm_scale(device: torch.device) -> torch.Tensor:
    return torch.tensor([_DSV4_SM_SCALE], dtype=torch.float32, device=device)


@pytest.mark.parametrize("num_tokens", [63, 64, 65, 128, 129])
def test_dsv4_kernel_decode_matches_dense_oracle(num_tokens: int) -> None:
    """run_sparse_mla_kernel (kernel.py) produces correct output for DSV4 decode."""
    device = require_sm120()
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=10_000 + num_tokens, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(num_tokens, dtype=torch.int32, device=device).unsqueeze(0)
    active_token_counts = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    output = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    run_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=_make_sm_scale(device),
        output=output,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
    )
    torch.cuda.synchronize(device)

    assert output.shape == (1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(output, expected)
    assert cos >= 0.99, f"num_tokens={num_tokens}: cos={cos:.6f} (max_abs={max_abs:.4f} rmse={rmse:.4f})"
    assert max_abs <= 0.15, f"num_tokens={num_tokens}: max_abs={max_abs:.6f}"


@pytest.mark.parametrize("num_tokens", [63, 64, 65, 128, 129])
def test_dsv4_kernel_onepass_decode_matches_dense_oracle(num_tokens: int) -> None:
    """run_sparse_mla_kernel (kernel_onepass.py) produces correct output for DSV4 decode."""
    device = require_sm120()
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=20_000 + num_tokens, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(num_tokens, dtype=torch.int32, device=device).unsqueeze(0)
    active_token_counts = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    output = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    run_sparse_mla_kernel_onepass(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=_make_sm_scale(device),
        output=output,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
    )
    torch.cuda.synchronize(device)

    assert output.shape == (1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(output, expected)
    assert cos >= 0.99, f"num_tokens={num_tokens}: cos={cos:.6f} (max_abs={max_abs:.4f} rmse={rmse:.4f})"
    assert max_abs <= 0.15, f"num_tokens={num_tokens}: max_abs={max_abs:.6f}"


@pytest.mark.parametrize("num_q", [2, 4, 8])
def test_dsv4_kernel_mtp_matches_dense_oracle(num_q: int) -> None:
    """Multi-token prediction: multiple query tokens attending the same cache."""
    device = require_sm120()
    num_tokens = 128
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=num_q, seed=30_000 + num_q, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(num_tokens, dtype=torch.int32, device=device).repeat(num_q, 1)
    active_token_counts = torch.full((num_q,), num_tokens, dtype=torch.int32, device=device)
    output = torch.zeros(num_q, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    run_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=_make_sm_scale(device),
        output=output,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
    )
    torch.cuda.synchronize(device)

    assert output.shape == (num_q, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(output, expected)
    assert cos >= 0.99, f"num_q={num_q}: cos={cos:.6f} (max_abs={max_abs:.4f} rmse={rmse:.4f})"
    assert max_abs <= 0.15, f"num_q={num_q}: max_abs={max_abs:.6f}"


def test_dsv4_kernel_sparse_page_table_matches_dense_oracle() -> None:
    """Kernel handles sparse page table with -1 padding correctly for DSV4."""
    device = require_sm120()
    num_tokens = 128
    width = 64
    valid_count = 37
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=40_000, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)

    g = torch.Generator(device="cpu")
    g.manual_seed(40_001)
    valid = torch.randperm(num_tokens, generator=g, dtype=torch.int32)[:valid_count]
    page_table_1 = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table_1[0, :valid_count] = valid.to(device)
    active_token_counts = torch.tensor([valid_count], dtype=torch.int32, device=device)
    output = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    run_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=_make_sm_scale(device),
        output=output,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(output, expected)
    assert cos >= 0.99, f"cos={cos:.6f} (max_abs={max_abs:.4f} rmse={rmse:.4f})"
    assert max_abs <= 0.15, f"max_abs={max_abs:.6f}"


@pytest.mark.parametrize("width", [129, 257, 512])
def test_dsv4_split_kernel_matches_dense_oracle(width: int) -> None:
    """run_sparse_mla_split_decode produces correct output for wide DSV4 contexts."""
    device = require_sm120()
    num_tokens = max(width + 17, 300)
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=50_000 + width, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    active_token_counts = torch.tensor([width], dtype=torch.int32, device=device)

    chunk_size = 64
    num_chunks = (width + chunk_size - 1) // chunk_size

    tmp_output = torch.zeros(
        1, _DSV4_NUM_HEADS, num_chunks, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    tmp_lse = torch.full(
        (1, _DSV4_NUM_HEADS, num_chunks), float("-inf"), dtype=torch.float32, device=device
    )
    output = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    kv_chunk_size_ptr = torch.tensor([chunk_size], dtype=torch.int32, device=device)
    num_chunks_ptr = torch.tensor([num_chunks], dtype=torch.int32, device=device)
    sm_scale_t = _make_sm_scale(device)

    run_sparse_mla_split_decode(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale_t,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output=output,
        launch_num_chunks=num_chunks,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
    )
    torch.cuda.synchronize(device)

    assert output.shape == (1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(output, expected)
    assert cos >= 0.99, f"width={width}: cos={cos:.6f} (max_abs={max_abs:.4f} rmse={rmse:.4f})"
    assert max_abs <= 0.15, f"width={width}: max_abs={max_abs:.6f}"
