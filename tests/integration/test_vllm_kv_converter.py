"""Unit tests for the vLLM fp8_ds_mla -> b12x KV cache converter.

These tests build synthetic vLLM caches in the documented block-segregated layout,
run the converter, and validate the resulting b12x packed cache by unpacking it
and comparing against a BF16 reference.  No real vLLM dependency required.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.mla.reference import unpack_mla_kv_cache_reference
from b12x.integration.vllm_kv_converter import (
    _DSV4_NOPE_DIM,
    _DSV4_ROPE_DIM,
    _VLLM_NOPE_QUANT_BLOCK,
    _VLLM_REAL_SCALES,
    _VLLM_TOKEN_DATA_BYTES,
    _VLLM_TOKEN_SCALE_BYTES,
    convert_fp8ds_to_b12x_gathered,
)
from tests.helpers import require_sm120


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _quantize_fp8_ue8m0(values_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-style UE8M0 quantization of a (..., quant_block) tensor.

    Returns:
        fp8_bytes: uint8 tensor with same shape as input
        scale_bytes: uint8 tensor with shape (...) (one byte per quant_block)
    """
    block_max = values_bf16.float().abs().amax(dim=-1).clamp_min(1e-4)
    raw_scale = block_max / _FP8_E4M3_MAX
    log_scale = torch.log2(raw_scale)
    exponent = torch.ceil(log_scale)
    scale = torch.exp2(exponent)
    scale_bytes = (exponent + 127.0).clamp(0, 255).to(torch.uint8)

    quantized = (values_bf16.float() / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    fp8 = quantized.to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8)
    return fp8_bytes, scale_bytes


def _build_synthetic_vllm_cache(
    *,
    num_blocks: int,
    block_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a vLLM fp8_ds_mla cache with known BF16 ground truth.

    Returns:
        vllm_cache: (num_blocks, block_size * 584) uint8
        ground_truth_nope: (num_blocks * block_size, 448) bf16
        ground_truth_rope: (num_blocks * block_size, 64) bf16
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_tokens = num_blocks * block_size

    nope_bf16 = (
        torch.randn((num_tokens, _DSV4_NOPE_DIM), generator=g, dtype=torch.bfloat16) * 0.05
    ).to(device)
    rope_bf16 = (
        torch.randn((num_tokens, _DSV4_ROPE_DIM), generator=g, dtype=torch.bfloat16) * 0.05
    ).to(device)

    nope_grouped = nope_bf16.view(num_tokens, _VLLM_REAL_SCALES, _VLLM_NOPE_QUANT_BLOCK)
    fp8_bytes, scale_bytes = _quantize_fp8_ue8m0(nope_grouped)
    fp8_bytes = fp8_bytes.view(num_tokens, _DSV4_NOPE_DIM)
    pad_byte = torch.zeros((num_tokens, 1), dtype=torch.uint8, device=device)
    full_scales = torch.cat([scale_bytes, pad_byte], dim=-1)  # (num_tokens, 8)

    rope_bytes = rope_bf16.view(torch.uint8).view(num_tokens, _DSV4_ROPE_DIM * 2)
    token_data = torch.cat([fp8_bytes, rope_bytes], dim=-1)  # (num_tokens, 576)

    token_data_per_block = token_data.view(num_blocks, block_size, _VLLM_TOKEN_DATA_BYTES)
    token_scales_per_block = full_scales.view(num_blocks, block_size, _VLLM_TOKEN_SCALE_BYTES)
    block_data_flat = token_data_per_block.reshape(num_blocks, block_size * _VLLM_TOKEN_DATA_BYTES)
    block_scales_flat = token_scales_per_block.reshape(num_blocks, block_size * _VLLM_TOKEN_SCALE_BYTES)
    vllm_cache = torch.cat([block_data_flat, block_scales_flat], dim=-1).contiguous()

    expected_block_bytes = block_size * (_VLLM_TOKEN_DATA_BYTES + _VLLM_TOKEN_SCALE_BYTES)
    assert vllm_cache.shape == (num_blocks, expected_block_bytes), vllm_cache.shape

    return vllm_cache, nope_bf16, rope_bf16


def _unpack_b12x(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack b12x cache to (nope_512, rope_64) float tensors."""
    unpacked = unpack_mla_kv_cache_reference(packed).squeeze(1)  # (N, 576)
    nope = unpacked[:, :512]
    rope = unpacked[:, 512:]
    return nope, rope


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_blocks", [2, 4])
def test_converter_full_block_roundtrip(num_blocks: int, block_size: int) -> None:
    """Convert all rows of a small cache; verify against ground-truth BF16."""
    device = require_sm120()
    vllm_cache, gt_nope, gt_rope = _build_synthetic_vllm_cache(
        num_blocks=num_blocks, block_size=block_size, seed=1000 + num_blocks * 10 + block_size, device=device
    )

    num_tokens = num_blocks * block_size
    page_table = torch.arange(num_tokens, dtype=torch.int32, device=device).view(1, num_tokens)

    b12x_cache, new_page_table = convert_fp8ds_to_b12x_gathered(
        vllm_cache, page_table, block_size=block_size
    )
    assert b12x_cache.shape == (num_tokens, 1, 656), b12x_cache.shape
    assert new_page_table.dtype == torch.int32
    assert new_page_table.shape == (1, num_tokens)
    assert torch.equal(new_page_table.view(-1), torch.arange(num_tokens, dtype=torch.int32, device=device))

    recovered_nope_512, recovered_rope = _unpack_b12x(b12x_cache)

    # Real nope dims (0..447): expect close to ground truth (FP8 quant + per-128 re-scale combine)
    nope_diff = (recovered_nope_512[:, :_DSV4_NOPE_DIM].float() - gt_nope.float()).abs()
    assert nope_diff.max().item() < 0.10, f"nope max_abs={nope_diff.max().item()}"
    # Loose RMSE check; precision drop from per-64 -> per-128 grouping is bounded
    nope_rmse = (nope_diff ** 2).mean().sqrt().item()
    assert nope_rmse < 0.01, f"nope rmse={nope_rmse}"

    # Padded nope dims (448..511) must be exactly zero
    pad = recovered_nope_512[:, _DSV4_NOPE_DIM:]
    assert pad.abs().max().item() < 1e-6

    # RoPE: bit-exact (BF16 copy, no requantization)
    rope_diff = (recovered_rope.to(torch.bfloat16) - gt_rope).abs()
    assert rope_diff.max().item() < 1e-3, f"rope max_abs={rope_diff.max().item()}"


def test_converter_gather_with_repeats_and_invalid() -> None:
    """Page table referencing the same slot twice + an invalid -1 slot."""
    device = require_sm120()
    block_size = 32
    vllm_cache, gt_nope, gt_rope = _build_synthetic_vllm_cache(
        num_blocks=2, block_size=block_size, seed=99, device=device
    )

    page_table = torch.tensor(
        [[5, 5, 33, -1], [12, 33, 0, 5]], dtype=torch.int32, device=device
    )
    b12x_cache, new_page_table = convert_fp8ds_to_b12x_gathered(
        vllm_cache, page_table, block_size=block_size
    )
    assert b12x_cache.shape == (2 * 4, 1, 656)
    assert torch.equal(new_page_table, torch.arange(8, dtype=torch.int32, device=device).view(2, 4))

    nope_512, rope = _unpack_b12x(b12x_cache)

    # Row 0 and 1 both gather slot 5; their nope/rope must match (within FP8 rounding identity)
    assert torch.allclose(nope_512[0], nope_512[1])
    assert torch.allclose(rope[0], rope[1])

    # Slot 5 should match ground truth for slot 5
    diff = (nope_512[0, :_DSV4_NOPE_DIM].float() - gt_nope[5].float()).abs()
    assert diff.max().item() < 0.10

    # Row 3 (invalid -1) must be all zeros for both nope and rope
    assert nope_512[3].abs().max().item() < 1e-6
    assert rope[3].abs().max().item() < 1e-6


def test_converter_input_validation() -> None:
    device = require_sm120()
    vllm_cache = torch.zeros((2, 32 * 584), dtype=torch.uint8, device=device)
    page_table = torch.zeros((1, 4), dtype=torch.int32, device=device)

    # Wrong dtype on cache
    with pytest.raises(TypeError, match="uint8"):
        convert_fp8ds_to_b12x_gathered(vllm_cache.float(), page_table, block_size=32)

    # Wrong block_size (584 bytes per block != block_size * 584)
    with pytest.raises(ValueError, match="block_size"):
        convert_fp8ds_to_b12x_gathered(vllm_cache, page_table, block_size=64)

    # Wrong page_table dtype
    with pytest.raises(TypeError, match="int32"):
        convert_fp8ds_to_b12x_gathered(vllm_cache, page_table.long(), block_size=32)

    # Wrong page_table rank
    with pytest.raises(ValueError, match="rank-2"):
        convert_fp8ds_to_b12x_gathered(vllm_cache, page_table.view(-1), block_size=32)
