"""NSA indexer API aligned with the DeepGEMM-style logits contracts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from .kernel import (
    clear_sparse_nsa_indexer_kernel_cache,
    run_sparse_nsa_paged_logits_kernel,
    supports_sparse_nsa_paged_logits_kernel,
)
from .extend_kernel import (
    run_sparse_nsa_extend_logits_kernel,
    supports_sparse_nsa_extend_logits_kernel,
)
from .reference import sparse_nsa_extend_logits_reference, sparse_nsa_paged_logits_reference


_INDEX_HEAD_DIM = 128


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _infer_active_width_hint(cache_seqlens_int32: torch.Tensor) -> int | None:
    if cache_seqlens_int32.numel() == 0:
        return 0
    if _is_cuda_graph_capture_active(cache_seqlens_int32.device):
        return None
    return max(int(cache_seqlens_int32.amax().item()), 0)


@dataclass(frozen=True)
class NSAIndexerPagedDecodeMetadata:
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    paged_mqa_schedule_metadata: torch.Tensor | None = None
    active_width_hint: int | None = None

    def __post_init__(self) -> None:
        if self.active_width_hint is None:
            object.__setattr__(
                self,
                "active_width_hint",
                _infer_active_width_hint(self.cache_seqlens_int32),
            )


@dataclass(frozen=True)
class NSAIndexerExtendLogitsMetadata:
    k_start: torch.Tensor
    k_end: torch.Tensor


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int | None = None,
) -> torch.Tensor:
    """Return a placeholder paged-MQA schedule tensor matching DeepGEMM's API.

    The current b12x paged decode kernel does not consume schedule metadata yet,
    but serving code should still flow through the same two-step contract:
    build metadata first, then hand it back into the decode logits call.
    """

    if context_lens.ndim not in (1, 2):
        raise ValueError(
            f"context_lens must be rank-1 or rank-2, got {tuple(context_lens.shape)}"
        )
    if context_lens.dtype != torch.int32:
        raise ValueError(
            f"context_lens must have dtype torch.int32, got {context_lens.dtype}"
        )
    if not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous")
    if block_kv <= 0:
        raise ValueError(f"block_kv must be positive, got {block_kv}")
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(context_lens.device).multi_processor_count
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    return torch.zeros(
        (num_sms + 1, 2),
        dtype=torch.int32,
        device=context_lens.device,
    )


def clear_nsa_indexer_caches() -> None:
    """Clear any cached NSA indexer runtime state."""
    clear_sparse_nsa_indexer_kernel_cache()
    _cached_width_cap_tensor.cache_clear()


def _normalize_weights(weights: torch.Tensor, *, q_rows: int, num_heads: int) -> torch.Tensor:
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weights = weights.squeeze(2)
    if weights.ndim != 2:
        raise ValueError(f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}")
    if weights.shape != (q_rows, num_heads):
        raise ValueError(f"weights shape must be {(q_rows, num_heads)}, got {tuple(weights.shape)}")
    return weights.to(torch.float32)


@lru_cache(maxsize=64)
def _cached_width_cap_tensor(
    width: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor([width], dtype=torch.int32, device=torch.device(device_type, device_index))


def _make_active_width_tensor(
    *,
    seqlens_per_query: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            "seqlens_per_query must be rank-1 when computing active width, got "
            f"{tuple(seqlens_per_query.shape)}"
        )
    width_cap = _cached_width_cap_tensor(
        int(width),
        seqlens_per_query.device.type,
        seqlens_per_query.device.index,
    )
    return torch.minimum(seqlens_per_query.amax().reshape(1), width_cap)


def _validate_paged_decode_inputs(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    paged_mqa_schedule_metadata: torch.Tensor | None,
) -> torch.Tensor:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    if real_page_table.ndim != 2:
        raise ValueError(f"real_page_table must be rank-2, got {tuple(real_page_table.shape)}")
    if real_page_table.dtype != torch.int32:
        raise ValueError(
            f"real_page_table must have dtype torch.int32, got {real_page_table.dtype}"
        )
    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, got "
            f"{tuple(cache_seqlens_int32.shape)}"
        )
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} do not match "
            f"cache_seqlens rows {cache_seqlens_int32.shape[0]}"
        )
    if real_page_table.shape[0] > q_fp8.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} exceed q rows {q_fp8.shape[0]}"
        )
    if real_page_table.device != q_fp8.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match q_fp8 device {q_fp8.device}"
        )
    if cache_seqlens_int32.device != q_fp8.device:
        raise ValueError(
            f"cache_seqlens_int32 device {cache_seqlens_int32.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if paged_mqa_schedule_metadata is not None:
        if paged_mqa_schedule_metadata.ndim != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must be rank-2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.shape[1] != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata trailing dimension must be 2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.dtype != torch.int32:
            raise ValueError(
                "paged_mqa_schedule_metadata must have dtype torch.int32, got "
                f"{paged_mqa_schedule_metadata.dtype}"
            )
        if not paged_mqa_schedule_metadata.is_contiguous():
            raise ValueError("paged_mqa_schedule_metadata must be contiguous")
        if paged_mqa_schedule_metadata.device != q_fp8.device:
            raise ValueError(
                "paged_mqa_schedule_metadata device "
                f"{paged_mqa_schedule_metadata.device} does not match q_fp8 device {q_fp8.device}"
            )
    return _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])


def sparse_nsa_index_decode_logits_paged(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: NSAIndexerPagedDecodeMetadata,
    page_size: int = 64,
) -> torch.Tensor:
    weights_f = _validate_paged_decode_inputs(
        q_fp8=q_fp8,
        weights=weights,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
    )

    valid_q_rows = metadata.real_page_table.shape[0]
    full_q_rows = q_fp8.shape[0]
    width_tokens = metadata.real_page_table.shape[1] * page_size
    if valid_q_rows == 0 or width_tokens == 0:
        return torch.full(
            (full_q_rows, width_tokens),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )

    seqlens_valid = metadata.cache_seqlens_int32.contiguous()
    active_width = _make_active_width_tensor(seqlens_per_query=seqlens_valid, width=width_tokens)
    max_page_capacity = index_k_cache.shape[0]

    if not _is_cuda_graph_capture_active(q_fp8.device):
        active_width_host = min(width_tokens, int(active_width.item()))
        if active_width_host > 0:
            positions = torch.arange(
                active_width_host,
                dtype=torch.int32,
                device=q_fp8.device,
            ).unsqueeze(0)
            page_cols = torch.div(positions, page_size, rounding_mode="floor").to(torch.long)
            page_cols = page_cols.expand(valid_q_rows, -1)
            candidate_pages = metadata.real_page_table.gather(1, page_cols)
            candidate_valid_mask = (positions < seqlens_valid.unsqueeze(1)) & (candidate_pages >= 0)
            overflow_mask = candidate_valid_mask & (candidate_pages >= max_page_capacity)
            if torch.any(overflow_mask):
                bad = int(candidate_pages[overflow_mask].max().item())
                raise ValueError(
                    f"real_page_table page id {bad} exceeds index_k_cache page capacity {max_page_capacity}"
                )

    if not supports_sparse_nsa_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        page_size=page_size,
    ):
        query_row_to_batch = torch.arange(valid_q_rows, dtype=torch.int32, device=q_fp8.device)
        return sparse_nsa_paged_logits_reference(
            q_fp8=q_fp8,
            weights=weights_f,
            index_k_cache=index_k_cache,
            real_page_table=metadata.real_page_table,
            query_row_to_batch=query_row_to_batch,
            seqlens_per_query=seqlens_valid,
            page_size=page_size,
        )

    logits_valid = run_sparse_nsa_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        active_width=active_width,
        active_width_hint=metadata.active_width_hint,
        page_size=page_size,
    )
    if valid_q_rows == full_q_rows:
        return logits_valid

    logits = torch.full(
        (full_q_rows, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    logits[:valid_q_rows].copy_(logits_valid)
    return logits


def sparse_nsa_index_extend_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: NSAIndexerExtendLogitsMetadata,
) -> torch.Tensor:
    k_start = metadata.k_start
    k_end = metadata.k_end
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])
    if k_start.ndim != 1 or k_end.ndim != 1:
        raise ValueError(
            f"k_start and k_end must be rank-1, got {tuple(k_start.shape)} and {tuple(k_end.shape)}"
        )
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got {tuple(k_start.shape)} vs {tuple(k_end.shape)}"
        )
    if k_start.device != q_fp8.device or k_end.device != q_fp8.device:
        raise ValueError("k_start and k_end must be on the same device as q_fp8")

    weights_f = _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])
    k_quant, k_scale = kv_fp8
    if supports_sparse_nsa_extend_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        return run_sparse_nsa_extend_logits_kernel(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
        )

    return sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights_f,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )
