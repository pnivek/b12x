"""Smoke tests for DSV4-Flash MLA support (head_dim=512, v_head_dim=448).

Uses synthetic random data; no model weights required.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.mla.reference import (
    dense_mla_reference,
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
    unpack_mla_kv_cache_reference,
)
from b12x.attention.mla.traits import select_sparse_mla_traits

from .helpers import require_sm120


_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_DSV4_HEAD_DIM = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM   # 512
_DSV4_V_HEAD_DIM = 448
_DSV4_SM_SCALE = _DSV4_HEAD_DIM ** -0.5
_DSV4_NUM_HEADS = 8
_PACKED_WIDTH = 656


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


def test_dsv4_traits_selected_correctly() -> None:
    device = require_sm120()
    _, k_nope, k_rope = _make_dsv4_tensors(num_tokens=64, num_q=1, seed=4000, device=device)
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    q_all = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device)
    page_table_1 = torch.zeros(1, 64, dtype=torch.int32, device=device)

    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        output_dtype=torch.bfloat16,
        v_head_dim=_DSV4_V_HEAD_DIM,
    )
    assert traits is not None, "DSV4-Flash traits should be selected"
    assert traits.head_dim == _DSV4_HEAD_DIM
    assert traits.v_head_dim == _DSV4_V_HEAD_DIM
    assert traits.nope_logical_dim == _DSV4_NOPE_DIM
    assert traits.rope_dim == _DSV4_ROPE_DIM


def test_dsv4_traits_rejected_for_wrong_v_head_dim() -> None:
    device = require_sm120()
    _, k_nope, k_rope = _make_dsv4_tensors(num_tokens=64, num_q=1, seed=5000, device=device)
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    q_all = torch.zeros(1, _DSV4_NUM_HEADS, _DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device)
    page_table_1 = torch.zeros(1, 64, dtype=torch.int32, device=device)

    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        output_dtype=torch.bfloat16,
        v_head_dim=512,  # unsupported for head_dim=512
    )
    assert traits is None


@pytest.mark.parametrize("num_tokens", [63, 64, 65, 128, 256, 320])
def test_dsv4_pack_roundtrip(num_tokens: int) -> None:
    device = require_sm120()
    _, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=1000 + num_tokens, device=device
    )

    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    assert packed.shape == (num_tokens, 1, _PACKED_WIDTH)

    unpacked = unpack_mla_kv_cache_reference(packed).squeeze(1)  # [T, 576]
    expected_nope = k_nope.squeeze(1).float()
    expected_rope = k_rope.squeeze(1).float()

    recovered_nope = unpacked[:, :_DSV4_NOPE_DIM].float()
    recovered_pad = unpacked[:, _DSV4_NOPE_DIM:512].float()
    recovered_rope = unpacked[:, 512:].float()

    max_abs, rmse, cos = _compare(recovered_nope, expected_nope)
    assert max_abs <= 0.08, f"nope max_abs={max_abs:.6f}"
    assert rmse <= 0.004, f"nope rmse={rmse:.6f}"
    assert cos >= 0.9995, f"nope cos={cos:.6f}"

    assert recovered_pad.abs().max().item() < 1e-6, "zero-padded dims 448-511 must be zero after unpack"

    max_abs, rmse, cos = _compare(recovered_rope, expected_rope)
    assert max_abs <= 0.08, f"rope max_abs={max_abs:.6f}"
    assert rmse <= 0.004, f"rope rmse={rmse:.6f}"
    assert cos >= 0.9995, f"rope cos={cos:.6f}"


@pytest.mark.parametrize("num_tokens", [63, 64, 65, 128, 256])
def test_dsv4_sparse_reference_matches_dense_oracle_decode(num_tokens: int) -> None:
    device = require_sm120()
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=2000 + num_tokens, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(num_tokens, dtype=torch.int32, device=device).unsqueeze(0)

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
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

    assert actual.shape == (1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"num_tokens={num_tokens}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"num_tokens={num_tokens}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"num_tokens={num_tokens}: cos={cos:.6f}"


@pytest.mark.parametrize("num_q", [2, 4, 8])
def test_dsv4_sparse_reference_matches_dense_oracle_mtp(num_q: int) -> None:
    """Multi-token prediction: multiple query tokens attending the same KV cache."""
    device = require_sm120()
    num_tokens = 128
    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=num_q, seed=3000 + num_q, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table_1 = torch.arange(num_tokens, dtype=torch.int32, device=device).repeat(num_q, 1)

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
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

    assert actual.shape == (num_q, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"num_q={num_q}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"num_q={num_q}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"num_q={num_q}: cos={cos:.6f}"


def test_dsv4_sparse_reference_handles_sparse_page_table() -> None:
    """Sparse page table with padding (-1 entries) should work correctly."""
    device = require_sm120()
    num_tokens = 128
    width = 64
    valid_per_row = 37

    q_all, k_nope, k_rope = _make_dsv4_tensors(
        num_tokens=num_tokens, num_q=1, seed=6000, device=device
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)

    g = torch.Generator(device="cpu")
    g.manual_seed(6001)
    valid = torch.randperm(num_tokens, generator=g, dtype=torch.int32)[:valid_per_row]
    page_table_1 = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table_1[0, :valid_per_row] = valid.to(device)

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=_DSV4_SM_SCALE,
        v_head_dim=_DSV4_V_HEAD_DIM,
        nope_logical_dim=_DSV4_NOPE_DIM,
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

    assert actual.shape == (1, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM)
    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"
