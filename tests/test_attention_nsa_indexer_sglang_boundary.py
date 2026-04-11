from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch

from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerExtendMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
)


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_nsa_indexer():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa.nsa_indexer")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA indexer: {exc}")
    return module


class _FakePool:
    page_size = 64

    def __init__(self, index_k_cache: torch.Tensor):
        self._index_k_cache = index_k_cache

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._index_k_cache


class _FakeDecodeMode:
    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        del include_v2
        return False


class _FakeExtendMode:
    def is_decode_or_idle(self) -> bool:
        return False

    def is_extend_without_speculative(self) -> bool:
        return True


class _FakeAttnBackend:
    nsa_decode_impl = "b12x_mla"
    nsa_prefill_impl = "b12x_mla"


class _FakePagedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        real_page_table: torch.Tensor,
        seqlens_int32: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        self._page_table_1 = page_table_1
        self._real_page_table = real_page_table
        self._seqlens_int32 = seqlens_int32
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens
        self.paged_mqa_schedule_metadata = None

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_page_table_64(self) -> torch.Tensor:
        return self._real_page_table

    def get_seqlens_int32(self) -> torch.Tensor:
        return self._seqlens_int32

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens

    def topk_transform(self, logits: torch.Tensor, topk: int) -> torch.Tensor:
        rows = logits.shape[0]
        output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
        lengths = self._seqlens_expanded[:rows]
        page_table_1 = self._page_table_1[:rows]
        for row_idx in range(rows):
            seq_len = min(int(lengths[row_idx].item()), page_table_1.shape[1])
            if seq_len <= 0:
                continue
            row_logits = logits[row_idx, :seq_len]
            topk_pos = torch.argsort(row_logits, descending=True, stable=True)[: min(topk, seq_len)]
            output[row_idx, : topk_pos.numel()] = page_table_1[row_idx, topk_pos.to(torch.long)]
        return output


class _FakeRaggedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        self._page_table_1 = page_table_1
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens


def _make_fake_indexer(module, *, topk: int):
    class _FakeIndexer:
        index_topk = topk
        _use_b12x_mla_indexer = staticmethod(module.Indexer._use_b12x_mla_indexer)
        _get_b12x_paged_topk = module.Indexer._get_b12x_paged_topk
        _get_b12x_ragged_topk = module.Indexer._get_b12x_ragged_topk

    return _FakeIndexer()


def _make_paged_candidate_tables(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = len(seqlens)
    page_table_1 = torch.full((rows, width_blocks * 64), -1, dtype=torch.int32, device=device)
    real_page_table = torch.full((rows, width_blocks), -1, dtype=torch.int32, device=device)
    for row_idx, (page_start, seq_len) in enumerate(zip(page_starts, seqlens, strict=True)):
        block_count = (int(seq_len) + 63) // 64
        if block_count == 0:
            continue
        real_page_table[row_idx, :block_count] = torch.arange(
            page_start,
            page_start + block_count,
            dtype=torch.int32,
            device=device,
        )
        token_ids = torch.arange(
            page_start * 64,
            (page_start + block_count) * 64,
            dtype=torch.int32,
            device=device,
        )
        page_table_1[row_idx, :seq_len] = token_ids[:seq_len]
    return page_table_1, real_page_table


def test_sglang_b12x_nsa_indexer_paged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_nsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_100)

    page_starts = [4, 8, 12]
    seqlens = torch.tensor([65, 96, 127], dtype=torch.int32)
    width_blocks = 2
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 3
    topk = 4
    q_rows = len(page_starts)

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=torch.device("cpu"),
    )

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1],
    )

    actual = module.Indexer._get_topk_paged(
        fake_indexer,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )

    assert torch.equal(actual, expected)


def test_sglang_b12x_nsa_indexer_paged_boundary_respects_active_decode_rows() -> None:
    module = _import_sglang_nsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_102)

    page_starts = [2, 6, 10, 14]
    seqlens = torch.tensor([70, 129, 66, 95], dtype=torch.int32)
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 4
    topk = 5
    q_rows = len(page_starts)
    active_rows = 3

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=torch.device("cpu"),
    )

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1, 0],
    )

    actual = module.Indexer._get_topk_paged(
        fake_indexer,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1[:active_rows],
            cache_seqlens_int32=seqlens[:active_rows],
        ),
        topk=topk,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual[active_rows], torch.full((topk,), -1, dtype=torch.int32))


def test_sglang_b12x_nsa_indexer_ragged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_nsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_101)

    num_tokens = 128
    num_heads = 2
    topk = 5

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1 = torch.full((2, 12), -1, dtype=torch.int32)
    page_table_1[0, :7] = torch.tensor([1, 3, 5, 7, 9, 11, 13], dtype=torch.int32)
    page_table_1[1, :9] = torch.tensor([64, 66, 68, 70, 72, 74, 76, 78, 80], dtype=torch.int32)
    seqlens_expanded = torch.tensor([6, 7, 7, 8, 9], dtype=torch.int32)
    extend_lens = [2, 3]

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeExtendMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakeRaggedMetadata(
        page_table_1=page_table_1,
        seqlens_expanded=seqlens_expanded,
        extend_lens=extend_lens,
    )

    actual = module.Indexer._get_topk_ragged(
        fake_indexer,
        False,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected = sparse_nsa_index_extend_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerExtendMetadata(
            page_table_1=page_table_1,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_lens,
        ),
        topk=topk,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture coverage")
def test_sglang_b12x_nsa_indexer_paged_boundary_cuda_graph_capture() -> None:
    module = _import_sglang_nsa_indexer()
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_103)

    page_starts = [4, 8, 12, 16]
    seqlens = torch.tensor([96, 130, 71, 80], dtype=torch.int32, device=device)
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 8
    topk = 6
    q_rows = len(page_starts)
    active_rows = 3

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32).to(
        device=device
    )
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1, 0],
    )

    clear_nsa_indexer_caches()
    captured_out = None

    def run() -> None:
        nonlocal captured_out
        captured_out = module.Indexer._get_topk_paged(
            fake_indexer,
            fake_forward_batch,
            0,
            q_fp8,
            weights,
            metadata,
        )

    run()
    torch.cuda.synchronize(device)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    expected = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1[:active_rows],
            cache_seqlens_int32=seqlens[:active_rows],
        ),
        topk=topk,
    )

    assert captured_out is not None
    assert torch.equal(captured_out, expected)
