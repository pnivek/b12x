from __future__ import annotations

import torch

from b12x.integration.mla import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    MLAWorkspace,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)


def _make_workspace(*, mode: str, topk: int = 4) -> MLAWorkspace:
    return MLAWorkspace.for_fixed_capacity(
        mode=mode,
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=topk,
        max_total_q=8,
        max_batch=4,
    )


def test_sparse_mla_decode_keeps_query_head_shape(monkeypatch) -> None:
    workspace = _make_workspace(mode="decode")
    captured: dict[str, torch.Tensor | float | int] = {}

    def fake_sparse_mla_reference(*, q_all, kv_cache, page_table_1, sm_scale, v_head_dim):
        captured["q"] = q_all
        captured["page_table_1"] = page_table_1
        captured["kv_cache"] = kv_cache
        captured["sm_scale"] = sm_scale
        captured["d_v"] = v_head_dim
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=0.5,
        v_head_dim=256,
    )

    assert output.shape == (2, 8, 256)
    assert captured["q"].shape == (2, 8, 256)
    assert captured["page_table_1"].shape == (2, 4)
    assert captured["sm_scale"] == 0.5
    assert captured["d_v"] == 256
    assert workspace.page_table_1 is not page_table_1
    assert torch.equal(workspace.page_table_1, page_table_1)


def test_sparse_mla_extend_uses_bound_metadata(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=6)

    def fake_sparse_mla_reference(*, q_all, kv_cache, page_table_1, sm_scale, v_head_dim):
        del kv_cache, page_table_1, sm_scale, v_head_dim
        return q_all[:, :8, :].clone()

    monkeypatch.setattr(
        "b12x.attention.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((3, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([12, 12, 12], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=1,
        max_seq_len_k=12,
        mode="draft_extend",
    )

    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (3, 8, 256)
    assert workspace.cache_seqlens_int32 is not cache_seqlens
    assert workspace.nsa_cache_seqlens_int32 is not cache_seqlens
    assert torch.equal(workspace.cache_seqlens_int32, cache_seqlens)
    assert torch.equal(workspace.nsa_cache_seqlens_int32, cache_seqlens)


def test_mla_workspace_graph_mode_copies_runtime_metadata() -> None:
    workspace = MLAWorkspace.for_contract(
        mode="decode",
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=4,
        max_total_q=8,
        max_batch=4,
        use_cuda_graph=True,
    )

    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([4, 4], dtype=torch.int32)
    workspace.prepare_decode(page_table_1, cache_seqlens, nsa_cache_seqlens)

    assert workspace.page_table_1 is not page_table_1
    assert workspace.cache_seqlens_int32 is not cache_seqlens
    assert workspace.nsa_cache_seqlens_int32 is not nsa_cache_seqlens
    assert torch.equal(workspace.page_table_1, page_table_1)
    assert torch.equal(workspace.cache_seqlens_int32, cache_seqlens)
    assert torch.equal(workspace.nsa_cache_seqlens_int32, nsa_cache_seqlens)


def test_mla_decode_workspace_allocates_split_buffers_and_chunk_scalars() -> None:
    workspace = MLAWorkspace.for_fixed_capacity(
        mode="decode",
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=2048,
        max_total_q=8,
        max_batch=4,
    )

    assert workspace.tmp_output is not None
    assert workspace.tmp_output.shape == (8 * workspace.max_chunks_per_row, 8, 256)
    assert workspace.tmp_lse is not None
    assert workspace.tmp_lse.shape == (8 * workspace.max_chunks_per_row, 8)
    workspace.set_decode_chunk_config(kv_chunk_size=256, num_chunks=8)
    assert workspace.kv_chunk_size_ptr is not None
    assert workspace.num_chunks_ptr is not None
    assert int(workspace.kv_chunk_size_ptr[0].item()) == 256
    assert int(workspace.num_chunks_ptr[0].item()) == 8


def test_mla_workspace_enforces_capacity_limits() -> None:
    workspace = _make_workspace(mode="decode", topk=4)
    with torch.no_grad():
        too_wide = torch.zeros((2, 5), dtype=torch.int32)
        cache_seqlens = torch.zeros((2,), dtype=torch.int32)
        try:
            workspace.prepare_decode(too_wide, cache_seqlens, cache_seqlens)
        except ValueError as exc:
            assert "topk capacity" in str(exc)
        else:
            raise AssertionError("expected capacity validation to fail")
