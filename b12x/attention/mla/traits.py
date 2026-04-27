"""Trait selection for sparse MLA kernels — supports GLM-5.1 and DSV4-Flash."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_MLA_PACKED_WIDTH = 656
_MLA_ROPE_DIM = 64
_MLA_GROUP_SIZE = 128

# Supported (head_dim, v_head_dim) pairs.  head_dim = nope_logical + rope.
# nope_logical must be a multiple of _MLA_GROUP_SIZE for full groups, OR
# equal to k*128 + 64 for a single half-group (DSV4: 448 = 3*128 + 64).
#
# DSV4-Flash has TWO supported v_head_dim values:
#   * v=448: V = K_nope only (legacy / pre-absorption convention; matches
#     dense_mla_reference). Output dims 0..447.
#   * v=512: V = K_full = K_nope ⊕ K_rope (vLLM absorbed-MLA convention).
#     Output dims 0..511, where dims 448..511 = P @ K_rope (BF16 PV-rope MMA).
#     Selected automatically when v_head_dim > nope_logical_dim — the kernel
#     adds an extra rope-PV pass into o_frag3[mma_d 4..7].
_SUPPORTED_SHAPES: dict[tuple[int, int], int] = {
    # (head_dim, v_head_dim) -> nope_logical_dim
    (576, 512): 512,  # GLM-5.1
    (512, 448): 448,  # DSV4-Flash, V = K_nope only
    (512, 512): 448,  # DSV4-Flash, V = K_full (kernel folds rope into V via BF16 PV-rope)
}


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.uint8:
        return 1
    raise TypeError(f"unsupported dtype {dtype}")


@dataclass(frozen=True)
class SparseMLATraits:
    heads_per_cta: int
    num_threads: int
    head_dim: int
    v_head_dim: int
    nope_logical_dim: int
    rope_dim: int
    num_q_heads: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    o_dtype: torch.dtype
    q_smem_bytes: int
    kv_stage_bytes: int
    q_register_elements_per_lane: int
    shared_storage_bytes: int


def select_sparse_mla_traits(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    output_dtype: torch.dtype,
    v_head_dim: int,
) -> SparseMLATraits | None:
    if q_all.device.type != "cuda" or kv_cache.device.type != "cuda":
        return None
    if page_table_1.device != q_all.device:
        return None
    if q_all.ndim != 3 or kv_cache.ndim != 3 or page_table_1.ndim != 2:
        return None
    if q_all.dtype != torch.bfloat16 or output_dtype != torch.bfloat16:
        return None
    if kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return None
    if q_all.shape[0] != page_table_1.shape[0]:
        return None
    if q_all.shape[1] <= 0:
        return None
    if kv_cache.shape[1:] != (1, _MLA_PACKED_WIDTH):
        return None

    head_dim = int(q_all.shape[2])
    v_head_dim_int = int(v_head_dim)
    nope_logical_dim = _SUPPORTED_SHAPES.get((head_dim, v_head_dim_int))
    if nope_logical_dim is None:
        return None
    if nope_logical_dim + _MLA_ROPE_DIM != head_dim:
        return None

    nope_groups = (nope_logical_dim + _MLA_GROUP_SIZE - 1) // _MLA_GROUP_SIZE
    heads_per_cta = 1
    kv_stage_bytes = 0
    q_register_elements_per_lane = nope_groups * 4 + 2
    shared_storage_bytes = 0
    return SparseMLATraits(
        heads_per_cta=heads_per_cta,
        num_threads=heads_per_cta * 32,
        head_dim=head_dim,
        v_head_dim=v_head_dim_int,
        nope_logical_dim=nope_logical_dim,
        rope_dim=_MLA_ROPE_DIM,
        num_q_heads=int(q_all.shape[1]),
        q_dtype=q_all.dtype,
        kv_dtype=kv_cache.dtype,
        o_dtype=output_dtype,
        q_smem_bytes=0,
        kv_stage_bytes=kv_stage_bytes,
        q_register_elements_per_lane=q_register_elements_per_lane,
        shared_storage_bytes=shared_storage_bytes,
    )
