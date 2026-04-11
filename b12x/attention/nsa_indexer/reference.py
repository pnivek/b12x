"""PyTorch NSA indexer references aligned with DeepGEMM FP8 MQA semantics."""

from __future__ import annotations

import math

import torch


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_INDEX_HEAD_DIM = 128
_SCALE_BYTES = 4


def _stable_topk_positions(logits: torch.Tensor, *, topk: int) -> torch.Tensor:
    return torch.argsort(logits, descending=True, stable=True)[:topk]


def _as_2d_index_k_cache(index_k_cache: torch.Tensor, *, page_size: int) -> torch.Tensor:
    if index_k_cache.ndim != 2:
        raise ValueError(f"index_k_cache must be rank-2, got {tuple(index_k_cache.shape)}")
    expected_width = page_size * (_INDEX_HEAD_DIM + _SCALE_BYTES)
    if index_k_cache.shape[1] != expected_width:
        raise ValueError(
            f"index_k_cache width must be {expected_width} for page_size={page_size}, "
            f"got {index_k_cache.shape[1]}"
        )
    if index_k_cache.dtype != torch.uint8:
        raise ValueError(f"index_k_cache must have dtype torch.uint8, got {index_k_cache.dtype}")
    return index_k_cache.contiguous()


def _as_k_matrix(k: torch.Tensor) -> torch.Tensor:
    if k.ndim == 3:
        if k.shape[1] != 1:
            raise ValueError(f"k must have middle dimension 1, got {tuple(k.shape)}")
        k = k[:, 0, :]
    if k.ndim != 2:
        raise ValueError(f"k must be rank-2 or rank-3, got {tuple(k.shape)}")
    if k.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(f"k last dimension must be {_INDEX_HEAD_DIM}, got {k.shape[1]}")
    return k.contiguous()


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
        raise ValueError(
            f"weights shape must be {(q_rows, num_heads)}, got {tuple(weights.shape)}"
        )
    return weights.to(torch.float32)


def _split_index_k_cache_reference(
    index_k_cache: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = _as_2d_index_k_cache(index_k_cache, page_size=page_size)
    num_pages = cache.shape[0]
    data_bytes = page_size * _INDEX_HEAD_DIM
    k_quant = (
        cache[:, :data_bytes]
        .contiguous()
        .view(num_pages, page_size, _INDEX_HEAD_DIM)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    k_scale = (
        cache[:, data_bytes : data_bytes + page_size * _SCALE_BYTES]
        .contiguous()
        .view(torch.float32)
        .view(num_pages, page_size)
    )
    return k_quant, k_scale


def pack_nsa_index_k_cache_reference(
    k: torch.Tensor,
    *,
    page_size: int = 64,
) -> torch.Tensor:
    """Pack dequantized index K rows into the paged FP8+scale cache layout."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    k_matrix = _as_k_matrix(k).to(torch.float32)
    num_tokens = k_matrix.shape[0]
    num_pages = max(1, math.ceil(num_tokens / page_size))
    cache = torch.zeros(
        (num_pages, page_size * (_INDEX_HEAD_DIM + _SCALE_BYTES)),
        dtype=torch.uint8,
        device=k_matrix.device,
    )

    data_bytes = page_size * _INDEX_HEAD_DIM
    for token_idx in range(num_tokens):
        page_idx = token_idx // page_size
        slot_idx = token_idx % page_size
        row = k_matrix[token_idx]
        scale = row.abs().amax() / _FP8_E4M3_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quant = (row / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        cache[page_idx, slot_idx * _INDEX_HEAD_DIM : (slot_idx + 1) * _INDEX_HEAD_DIM] = (
            quant.view(torch.uint8)
        )
        scale_offset = data_bytes + slot_idx * _SCALE_BYTES
        cache[page_idx, scale_offset : scale_offset + _SCALE_BYTES] = (
            scale.reshape(1).view(torch.uint8)
        )
    return cache.contiguous()


def unpack_nsa_index_k_cache_reference(
    index_k_cache: torch.Tensor,
    *,
    num_tokens: int,
    page_size: int = 64,
) -> torch.Tensor:
    """Unpack the paged FP8+scale cache layout back into dequantized K rows."""

    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    k_quant, k_scale = _split_index_k_cache_reference(index_k_cache, page_size=page_size)
    max_tokens = k_quant.shape[0] * page_size
    if num_tokens > max_tokens:
        raise ValueError(f"num_tokens {num_tokens} exceeds cache capacity {max_tokens}")
    if num_tokens == 0:
        return torch.empty((0, _INDEX_HEAD_DIM), dtype=torch.float32, device=index_k_cache.device)
    token_ids = torch.arange(num_tokens, device=index_k_cache.device, dtype=torch.long)
    page_idx = torch.div(token_ids, page_size, rounding_mode="floor")
    slot_idx = token_ids % page_size
    return k_quant[page_idx, slot_idx] * k_scale[page_idx, slot_idx].unsqueeze(1)


def sparse_nsa_index_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    topk: int,
    page_size: int = 64,
) -> torch.Tensor:
    """Return top-k token ids using the DeepGEMM MQA logits reference formula."""

    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    if page_table_1.ndim != 2:
        raise ValueError(f"page_table_1 must be rank-2, got {tuple(page_table_1.shape)}")
    if page_table_1.dtype != torch.int32:
        raise ValueError(f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}")
    if page_table_1.device != q_fp8.device:
        raise ValueError(
            f"page_table_1 device {page_table_1.device} does not match q_fp8 device {q_fp8.device}"
        )
    if query_row_to_batch.ndim != 1:
        raise ValueError(
            f"query_row_to_batch must be rank-1, got {tuple(query_row_to_batch.shape)}"
        )
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            f"seqlens_per_query must be rank-1, got {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.shape != seqlens_per_query.shape:
        raise ValueError(
            "query_row_to_batch and seqlens_per_query must have the same shape, got "
            f"{tuple(query_row_to_batch.shape)} vs {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.device != q_fp8.device:
        raise ValueError(
            f"query_row_to_batch device {query_row_to_batch.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if seqlens_per_query.device != q_fp8.device:
        raise ValueError(
            f"seqlens_per_query device {seqlens_per_query.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    num_queries, num_heads, _ = q_fp8.shape
    weights_f = _normalize_weights(weights, q_rows=num_queries, num_heads=num_heads)
    k_quant, k_scale = _split_index_k_cache_reference(index_k_cache, page_size=page_size)

    valid_q_rows = int(query_row_to_batch.numel())
    if valid_q_rows > num_queries:
        raise ValueError(f"metadata describes {valid_q_rows} query rows, but q_fp8 has {num_queries}")

    output = torch.full(
        (num_queries, topk),
        -1,
        dtype=torch.int32,
        device=q_fp8.device,
    )
    q_fp32 = q_fp8.to(torch.float32)
    max_token_capacity = k_quant.shape[0] * page_size

    for query_row in range(valid_q_rows):
        batch_row = int(query_row_to_batch[query_row].item())
        if batch_row < 0 or batch_row >= page_table_1.shape[0]:
            raise ValueError(
                f"query_row_to_batch[{query_row}]={batch_row} is out of bounds for "
                f"{page_table_1.shape[0]} batch rows"
            )
        seq_len = int(seqlens_per_query[query_row].item())
        if seq_len <= 0:
            continue

        candidate_tokens = page_table_1[batch_row, : min(seq_len, page_table_1.shape[1])]
        candidate_tokens = candidate_tokens[candidate_tokens >= 0]
        if candidate_tokens.numel() == 0:
            continue
        if torch.any(candidate_tokens >= max_token_capacity):
            bad = int(candidate_tokens.max().item())
            raise ValueError(
                f"page_table_1 token id {bad} exceeds index_k_cache capacity {max_token_capacity}"
            )

        token_ids = candidate_tokens.to(torch.long)
        page_idx = torch.div(token_ids, page_size, rounding_mode="floor")
        slot_idx = token_ids % page_size
        k_selected = k_quant[page_idx, slot_idx]
        scale_selected = k_scale[page_idx, slot_idx]

        score = torch.matmul(q_fp32[query_row], k_selected.transpose(0, 1))
        logits = (torch.relu(score) * weights_f[query_row].unsqueeze(1)).sum(dim=0)
        logits = logits * scale_selected

        topk_count = min(topk, logits.numel())
        if topk_count == 0:
            continue
        topk_pos = _stable_topk_positions(logits, topk=topk_count)
        output[query_row, :topk_count] = token_ids[topk_pos].to(torch.int32)

    return output


def sparse_nsa_paged_logits_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    """Return dense token-position logits from the paged/block-table contract."""

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
    if real_page_table.device != q_fp8.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match q_fp8 device {q_fp8.device}"
        )
    if query_row_to_batch.ndim != 1:
        raise ValueError(
            f"query_row_to_batch must be rank-1, got {tuple(query_row_to_batch.shape)}"
        )
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            f"seqlens_per_query must be rank-1, got {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.shape != seqlens_per_query.shape:
        raise ValueError(
            "query_row_to_batch and seqlens_per_query must have the same shape, got "
            f"{tuple(query_row_to_batch.shape)} vs {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.device != q_fp8.device:
        raise ValueError(
            f"query_row_to_batch device {query_row_to_batch.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if seqlens_per_query.device != q_fp8.device:
        raise ValueError(
            f"seqlens_per_query device {seqlens_per_query.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )

    num_queries, num_heads, _ = q_fp8.shape
    weights_f = _normalize_weights(weights, q_rows=num_queries, num_heads=num_heads)
    k_quant, k_scale = _split_index_k_cache_reference(index_k_cache, page_size=page_size)

    valid_q_rows = int(query_row_to_batch.numel())
    if valid_q_rows > num_queries:
        raise ValueError(f"metadata describes {valid_q_rows} query rows, but q_fp8 has {num_queries}")

    width_tokens = real_page_table.shape[1] * page_size
    logits_out = torch.full(
        (num_queries, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    q_fp32 = q_fp8.to(torch.float32)
    max_page_capacity = k_quant.shape[0]

    for query_row in range(valid_q_rows):
        batch_row = int(query_row_to_batch[query_row].item())
        if batch_row < 0 or batch_row >= real_page_table.shape[0]:
            raise ValueError(
                f"query_row_to_batch[{query_row}]={batch_row} is out of bounds for "
                f"{real_page_table.shape[0]} batch rows"
            )
        seq_len = int(seqlens_per_query[query_row].item())
        if seq_len <= 0:
            continue
        seq_len = min(seq_len, width_tokens)
        token_pos = torch.arange(seq_len, device=q_fp8.device, dtype=torch.long)
        page_col = torch.div(token_pos, page_size, rounding_mode="floor")
        slot_idx = token_pos % page_size
        page_ids = real_page_table[batch_row, page_col].to(torch.long)
        valid_mask = page_ids >= 0
        if not torch.any(valid_mask):
            continue
        if torch.any(page_ids[valid_mask] >= max_page_capacity):
            bad = int(page_ids[valid_mask].max().item())
            raise ValueError(
                f"real_page_table page id {bad} exceeds index_k_cache page capacity {max_page_capacity}"
            )

        valid_token_pos = token_pos[valid_mask]
        valid_page_ids = page_ids[valid_mask]
        valid_slot_idx = slot_idx[valid_mask]
        k_selected = k_quant[valid_page_ids, valid_slot_idx]
        scale_selected = k_scale[valid_page_ids, valid_slot_idx]

        score = torch.matmul(q_fp32[query_row], k_selected.transpose(0, 1))
        logits = (torch.relu(score) * weights_f[query_row].unsqueeze(1)).sum(dim=0)
        logits = logits * scale_selected
        logits_out[query_row, valid_token_pos] = logits

    return logits_out
