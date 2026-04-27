"""Convert vLLM's fp8_ds_mla KV cache layout into b12x's packed format.

vLLM fp8_ds_mla cache layout (per-block, block_size tokens):
    Token data area: block_size * 576 bytes
        Token i: bytes [i*576 .. i*576+575]
            [0..447]    FP8 E4M3 NoPE (448 elements, quantized per-64 with UE8M0)
            [448..575]  BF16 RoPE (64 elements * 2 bytes)
    Scale area:      block_size * 8 bytes
        Token i scales: bytes [block_size*576 + i*8 .. block_size*576 + i*8 + 7]
            [0..6]      7 UE8M0 exponent bytes (one per 64-element nope group)
            [7]         pad/unused
    Total per block: block_size * 584 bytes.

b12x packed layout (per-token, contiguous):
    [0..511]    FP8 E4M3 NoPE (512 elements; bytes [448..511] zero-padded for DSV4)
    [512..527]  4 FP32 scales (one per 128-element nope group)
    [528..655]  BF16 RoPE (64 elements * 2 bytes)
    Total: 656 bytes per token.

Two format mismatches handled here:
    1. Layout: vLLM block-segregated -> b12x contiguous-per-token (handled by gather).
    2. Scale granularity: vLLM per-64 -> b12x per-128 (handled by dequant->requant
       round-trip via the existing ``pack_mla_kv_cache_reference``, which tolerates
       partial-group nope_logical_dim=448).
"""

from __future__ import annotations

import torch

from b12x.attention.mla.reference import pack_mla_kv_cache_reference, ue8m0_to_fp32


_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_VLLM_TOKEN_DATA_BYTES = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM * 2  # 576
_VLLM_TOKEN_SCALE_BYTES = 8                                    # 7 real + 1 pad
_VLLM_NOPE_QUANT_BLOCK = 64                                    # vLLM scale granularity
_VLLM_REAL_SCALES = _DSV4_NOPE_DIM // _VLLM_NOPE_QUANT_BLOCK   # 7


def _validate_inputs(
    vllm_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    block_size: int,
) -> tuple[int, torch.Tensor]:
    """Normalise inputs and return (num_blocks, vllm_cache_2d_uint8)."""
    if vllm_cache.dtype != torch.uint8:
        raise TypeError(f"vllm_cache must be uint8, got {vllm_cache.dtype}")
    expected_block_bytes = block_size * (_VLLM_TOKEN_DATA_BYTES + _VLLM_TOKEN_SCALE_BYTES)
    flat = vllm_cache.contiguous().view(vllm_cache.shape[0], -1)
    if flat.shape[1] != expected_block_bytes:
        raise ValueError(
            f"vllm_cache per-block bytes {flat.shape[1]} does not match expected "
            f"block_size({block_size}) * 584 = {expected_block_bytes}"
        )
    if page_table_1.dtype != torch.int32:
        raise TypeError(f"page_table_1 must be int32, got {page_table_1.dtype}")
    if page_table_1.ndim != 2:
        raise ValueError(f"page_table_1 must be rank-2, got shape {tuple(page_table_1.shape)}")
    return int(flat.shape[0]), flat


def _gather_token_rows(
    vllm_cache_flat: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (token_data, token_scale) bytes for each slot id.

    Returns:
        token_data: (num_rows, 576) uint8
        token_scales: (num_rows, 8) uint8
    """
    block_idx = slot_ids // block_size
    pos_in_block = slot_ids % block_size

    token_data_offset = (pos_in_block.to(torch.int64) * _VLLM_TOKEN_DATA_BYTES)
    scale_base_offset = block_size * _VLLM_TOKEN_DATA_BYTES
    token_scale_offset = scale_base_offset + pos_in_block.to(torch.int64) * _VLLM_TOKEN_SCALE_BYTES

    block_rows = vllm_cache_flat.index_select(0, block_idx.to(torch.long))  # (num_rows, block_bytes)

    arange_data = torch.arange(_VLLM_TOKEN_DATA_BYTES, device=block_rows.device)
    arange_scale = torch.arange(_VLLM_TOKEN_SCALE_BYTES, device=block_rows.device)

    data_idx = token_data_offset.unsqueeze(-1) + arange_data.unsqueeze(0)   # (num_rows, 576)
    scale_idx = token_scale_offset.unsqueeze(-1) + arange_scale.unsqueeze(0)  # (num_rows, 8)

    token_data = torch.gather(block_rows, 1, data_idx)
    token_scales = torch.gather(block_rows, 1, scale_idx)
    return token_data, token_scales


def _dequant_nope(
    token_data: torch.Tensor,
    scale_bytes: torch.Tensor,
) -> torch.Tensor:
    """vLLM token-data (FP8 NoPE + BF16 RoPE) + UE8M0 scales -> BF16 K_nope.

    Returns shape ``(num_rows, 1, 448)``.
    """
    num_rows = token_data.shape[0]
    fp8_bytes = token_data[:, :_DSV4_NOPE_DIM].contiguous()
    fp8 = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)  # (num_rows, 448)

    real_scale_bytes = scale_bytes[:, :_VLLM_REAL_SCALES]
    scales_fp32 = ue8m0_to_fp32(real_scale_bytes)                 # (num_rows, 7)

    fp8_grouped = fp8.view(num_rows, _VLLM_REAL_SCALES, _VLLM_NOPE_QUANT_BLOCK)
    nope_fp32 = fp8_grouped * scales_fp32.unsqueeze(-1)
    nope_bf16 = nope_fp32.reshape(num_rows, _DSV4_NOPE_DIM).to(torch.bfloat16)
    return nope_bf16.unsqueeze(1)  # (num_rows, 1, 448)


def _extract_rope(token_data: torch.Tensor) -> torch.Tensor:
    """Extract BF16 K_rope of shape (num_rows, 1, 64) from token data bytes."""
    rope_bytes = token_data[:, _DSV4_NOPE_DIM:].contiguous()
    return rope_bytes.view(torch.bfloat16).reshape(rope_bytes.shape[0], 1, _DSV4_ROPE_DIM)


def gather_and_dequant_fp8ds(
    vllm_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast path: gather rows + dequant to BF16 only — skip the pack/unpack round-trip.

    Returns:
        k_nope_bf16: ``(num_q*topk, 448)`` bf16
        k_rope_bf16: ``(num_q*topk, 64)`` bf16
        new_page_table: ``(num_q, topk)`` int32 — arange-based, unmasked
            (caller is responsible for masking via active_token_counts).
    """
    num_blocks, vllm_cache_flat = _validate_inputs(vllm_cache, page_table_1, block_size)
    num_q, topk = page_table_1.shape
    num_rows = num_q * topk
    flat_slots = page_table_1.reshape(-1)

    valid = (flat_slots >= 0) & (flat_slots < num_blocks * block_size)
    safe_slots = torch.where(valid, flat_slots, torch.zeros_like(flat_slots))

    token_data, token_scales = _gather_token_rows(vllm_cache_flat, safe_slots, block_size)
    k_nope = _dequant_nope(token_data, token_scales).squeeze(1)  # (num_rows, 448)
    k_rope = _extract_rope(token_data).squeeze(1)                # (num_rows, 64)

    if (~valid).any():
        invalid_rows = (~valid).nonzero(as_tuple=False).squeeze(-1)
        k_nope[invalid_rows] = 0
        k_rope[invalid_rows] = 0

    new_page_table = torch.arange(
        num_rows, dtype=torch.int32, device=page_table_1.device
    ).reshape(num_q, topk)
    return k_nope, k_rope, new_page_table


def convert_fp8ds_to_b12x_gathered(
    vllm_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather + repack vLLM fp8_ds_mla KV rows into b12x packed format.

    Args:
        vllm_cache: vLLM KV cache tensor with last dim totaling
            ``block_size * 584`` bytes per block (any leading dims are flattened
            to ``num_blocks``).
        page_table_1: ``(num_q, topk)`` int32 — global slot ids into vllm_cache.
        block_size: tokens per block in the vLLM layout.

    Returns:
        b12x_cache: ``(num_q*topk, 1, 656)`` uint8 packed b12x cache.
        new_page_table: ``(num_q, topk)`` int32 indices into ``b12x_cache``
            (each query row sees its own contiguous block of length ``topk``).
    """
    num_blocks, vllm_cache_flat = _validate_inputs(vllm_cache, page_table_1, block_size)

    num_q, topk = page_table_1.shape
    num_rows = num_q * topk
    flat_slots = page_table_1.reshape(-1)

    valid = (flat_slots >= 0) & (flat_slots < num_blocks * block_size)
    safe_slots = torch.where(valid, flat_slots, torch.zeros_like(flat_slots))

    token_data, token_scales = _gather_token_rows(vllm_cache_flat, safe_slots, block_size)
    k_nope = _dequant_nope(token_data, token_scales)
    k_rope = _extract_rope(token_data)

    if (~valid).any():
        invalid_rows = (~valid).nonzero(as_tuple=False).squeeze(-1)
        k_nope[invalid_rows] = 0
        k_rope[invalid_rows] = 0

    packed = pack_mla_kv_cache_reference(
        k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM
    )  # (num_rows, 1, 656)

    new_page_table = torch.arange(
        num_rows, dtype=torch.int32, device=page_table_1.device
    ).reshape(num_q, topk)
    return packed, new_page_table
