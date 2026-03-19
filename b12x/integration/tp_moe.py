"""Tensor-parallel MoE entrypoints backed by fused CuTe DSL kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch

from b12x.cute.fp4 import align_up, as_grouped_scale_view
from b12x.cute.utils import current_cuda_stream, get_max_active_clusters, get_num_sm, make_ptr
from b12x.moe.fused import MoEDynamicKernel, MoEStaticKernel

_NVFP4_BLOCK_SIZE = 16
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
_DYNAMIC_SLICE_CHUNK = 2


@dataclass
class _TPMoEState:
    """Pre-allocated work buffers for `b12x_moe_fp4`."""
    # Expert maps
    expert_counts: torch.Tensor       # [E] int32
    active_experts: torch.Tensor      # [E] int32 active local expert ids in scheduler order
    active_expert_count: torch.Tensor  # [1] int32 active expert count
    active_row_counts: torch.Tensor   # [E] int32 compact row counts aligned with active_experts
    weight_expert_ids: torch.Tensor   # [E] int32 local expert id -> global weight expert id
    global_to_local_expert: torch.Tensor  # [weight_E] int32 global -> local id map
    token_map: torch.Tensor           # [E, max_rows] int32
    token_weights_map: torch.Tensor   # [E, max_rows] float32
    # Packed FP4 activation buffers consumed by static compute
    packed_input: torch.Tensor        # [E, max_rows, k//2] uint8
    packed_input_scale: torch.Tensor  # [E, rows_pad, cols_pad] uint8
    # Dummy output tensor used only to define the grouped scheduler shape.
    scheduler_out: torch.Tensor       # [max_rows, n, E] bf16
    scatter_output: torch.Tensor      # [m, k] bf16
    # Pre-expanded scale tensors (filled at allocation time)
    input_gs: torch.Tensor            # [E] float32
    down_input_scale: torch.Tensor    # [E] float32
    barrier_count: torch.Tensor       # [1] int32 — static resident-grid barrier
    barrier_epoch: torch.Tensor       # [1] int32 — static resident-grid barrier
    pair_head: torch.Tensor           # [1] int32 — dynamic routed-pair allocator
    producers_done_count: torch.Tensor  # [1] int32 — dynamic producer completion
    all_work_published: torch.Tensor  # [1] int32 — dynamic terminal publication flag
    task_head: torch.Tensor           # [1] int32 — dynamic task consumer head
    task_tail: torch.Tensor           # [1] int32 — dynamic task producer tail
    task_ready: torch.Tensor          # [max_tasks] int32 — per-task publication flags
    task_expert: torch.Tensor         # [max_tasks] int32
    task_m_tile: torch.Tensor         # [max_tasks] int32
    task_slice_begin: torch.Tensor    # [max_tasks] int32
    task_slice_count: torch.Tensor    # [max_tasks] int32
    task_valid_rows: torch.Tensor     # [max_tasks] int32
    tile_write_count: torch.Tensor    # [E * max_m_tiles] int32
    # Pre-computed views (cached to avoid per-call Python overhead)
    packed_a_view: object = None      # packed_input permuted + fp4 view
    packed_a_flat: object = None      # packed_input.view(-1)
    scale_flat: object = None         # packed_input_scale.view(-1)
    sfa_ptr: object = None            # CuTe pointer for scale factors
    input_gs_src_ptr: int = 0
    down_input_scale_src_ptr: int = 0


@dataclass
class _WeightViews:
    """Cached weight views for the concatenated expert-weight layout."""
    w13: torch.Tensor        # [2*n, k//2, E] uint8 (permuted view, no copy)
    down: torch.Tensor       # [k, n//2, E] uint8 (permuted view, no copy)
    w13_sf: torch.Tensor     # 6D MMA view for concatenated w13 scale factors
    down_sf: torch.Tensor    # [E, down_sf_rows, sf_cols] uint8 (view)
    w1_alpha: torch.Tensor   # [E] float32 contiguous tensor in plain CUDA storage
    w2_alpha: torch.Tensor   # [E] float32 contiguous tensor in plain CUDA storage
    # Pre-computed fp4 views and CuTe pointers
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None

_STATE_CACHE: Dict[Tuple, _TPMoEState] = {}
_WEIGHT_CACHE: Dict[Tuple[int, int, int], _WeightViews] = {}
_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_MAC_CACHE: Dict[Tuple[int, str], int] = {}  # (device_idx, impl) → max_active_clusters
_PLAIN_PARAM_CACHE: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.dtype, int], torch.Tensor] = {}
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = _LEVEL_TILE_M
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: int | None = None
_DYNAMIC_MULTICTA_CACHE: bool | None = None
# Fast path: cache last-used state/weights/kernel to avoid dict lookup
_LAST_STATE: Tuple = (None, None)  # (cache_key, state)
_LAST_WEIGHTS: Tuple = (None, None)  # (cache_key, views)
_LAST_KERNEL: Tuple = (None, None)  # (cache_key, (compiled, mac))
_LAST_STATIC_EXECUTION_ARGS: Tuple = (None, None, None)  # (cache_key, exe_args, adapted_args)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("", "0", "false", "False")


_FAST_MATH_DEFAULT = _env_flag("B12X_FAST_MATH", default=True)


def _get_static_compact_cutover_pairs() -> int:
    global _STATIC_COMPACT_CUTOVER_PAIRS_CACHE
    if _STATIC_COMPACT_CUTOVER_PAIRS_CACHE is None:
        cutover = os.environ.get("B12X_STATIC_COMPACT_CUTOVER_PAIRS")
        if cutover is None:
            cutover = os.environ.get("B12X_DYNAMIC_STATIC_CUTOVER_PAIRS")
        if cutover is None:
            cutover = os.environ.get("B12X_LEVEL10_STATIC_CUTOVER_PAIRS")
        if cutover is None:
            _STATIC_COMPACT_CUTOVER_PAIRS_CACHE = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
        else:
            _STATIC_COMPACT_CUTOVER_PAIRS_CACHE = max(0, int(cutover))
    return _STATIC_COMPACT_CUTOVER_PAIRS_CACHE


def _dynamic_multicta_enabled() -> bool:
    global _DYNAMIC_MULTICTA_CACHE
    if _DYNAMIC_MULTICTA_CACHE is None:
        multicta_env = os.environ.get("B12X_DYNAMIC_ENABLE_MULTICTA")
        if multicta_env is None:
            multicta_env = os.environ.get("B12X_LEVEL10_ENABLE_MULTICTA", "1")
        _DYNAMIC_MULTICTA_CACHE = multicta_env == "1"
    return _DYNAMIC_MULTICTA_CACHE


def _tensor_arg_key(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _flatten_routing_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    flat_ids = topk_ids.view(-1)
    if flat_ids.dtype not in (torch.int32, torch.int64):
        return flat_ids.to(torch.int32)
    if not flat_ids.is_contiguous():
        return flat_ids.contiguous()
    return flat_ids


def _flatten_routing_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    flat_weights = topk_weights.view(-1)
    if flat_weights.dtype != torch.float32:
        return flat_weights.to(torch.float32)
    if not flat_weights.is_contiguous():
        return flat_weights.contiguous()
    return flat_weights


def _prepare_expert_scale(scale: torch.Tensor, weight_E: int) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.expand(weight_E).to(torch.float32).contiguous()
    if scale.numel() != weight_E:
        raise ValueError(f"expected expert scale with {weight_E} elements, got {scale.numel()}")
    return _get_plain_cuda_tensor(scale, dtype=torch.float32)


def _get_plain_cuda_tensor(t: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    target_dtype = t.dtype if dtype is None else dtype
    key = (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        target_dtype,
        int(t._version),
    )
    cached = _PLAIN_PARAM_CACHE.get(key)
    if cached is not None:
        return cached
    plain = torch.empty(tuple(t.shape), dtype=target_dtype, device=t.device)
    plain.copy_(t.to(target_dtype) if t.dtype != target_dtype else t)
    _PLAIN_PARAM_CACHE[key] = plain
    return plain


def _get_cached_static_execution_args(
    compiled,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    s: _TPMoEState,
    wv: _WeightViews,
    input_gs: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    mac: int,
    stream,
):
    global _LAST_STATIC_EXECUTION_ARGS
    cache_key = (
        id(compiled),
        _tensor_arg_key(a),
        _tensor_arg_key(flat_ids),
        _tensor_arg_key(flat_weights),
        id(s),
        id(wv),
        None if s.global_to_local_expert is None else _tensor_arg_key(s.global_to_local_expert),
        _tensor_arg_key(input_gs),
        _tensor_arg_key(w1_alpha),
        _tensor_arg_key(w2_alpha),
        _tensor_arg_key(down_input_scale),
        _tensor_arg_key(scatter_output),
        mac,
        int(stream),
    )
    last_key, last_exe_args, last_adapted_args = _LAST_STATIC_EXECUTION_ARGS
    if last_key == cache_key:
        return last_exe_args, last_adapted_args

    exe_args, adapted_args = compiled.generate_execution_args(
        a, flat_ids, flat_weights,
        s.packed_a_view, s.sfa_ptr,
        s.packed_a_flat, s.scale_flat,
        s.barrier_count, s.barrier_epoch,
        wv.w13_fp4, wv.sfb_w13_ptr,
        wv.down_fp4, wv.sfb_down_ptr,
        s.scheduler_out, s.expert_counts, s.active_experts, s.active_expert_count, s.weight_expert_ids, s.global_to_local_expert, s.active_row_counts,
        input_gs, w1_alpha, w2_alpha, down_input_scale,
        scatter_output, s.token_map, s.token_weights_map,
        mac, stream,
    )
    _LAST_STATIC_EXECUTION_ARGS = (cache_key, exe_args, adapted_args)
    return exe_args, adapted_args


def _alloc_unique_ld_tensor(
    shape: Tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a 3D tensor with exactly one stride-1 dimension.

    CuTe's dynamic-layout inference fails when a compact tensor has multiple
    stride-1 dimensions, which happens for shapes like [M, N, 1]. Padding the
    size-1 expert stride avoids the ambiguity while preserving the logical
    M-major [M, N, E] layout.
    """
    m, n, e = shape
    if e != 1:
        return torch.empty(shape, dtype=dtype, device=device)
    storage = torch.empty(m * n, dtype=dtype, device=device)
    return torch.as_strided(storage, shape, (n, 1, m * n))


def _safe_max_rows_per_launch(E: int, k: int, n: int) -> int:
    """Largest padded row count that fits within CuTe runtime memref limits."""
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    limits = [
        _RUNTIME_MEMREF_LIMIT // max(1, E * (k // 2)),
        _RUNTIME_MEMREF_LIMIT // max(1, E * cols_pad_k),
        _RUNTIME_MEMREF_LIMIT // max(1, E * n),
        _RUNTIME_MEMREF_LIMIT // max(1, E),
    ]
    max_rows = min(limits)
    return max_rows - (max_rows % 128)


def _safe_token_chunk(E: int, k: int, n: int, num_topk: int) -> int:
    """Largest token chunk that keeps all per-launch work buffers in range."""
    safe_rows = _safe_max_rows_per_launch(E, k, n)
    if safe_rows <= 0:
        return 1
    max_tokens = max(1, safe_rows // max(1, num_topk))
    while max_tokens > 1 and align_up(max_tokens * num_topk, 128) > safe_rows:
        max_tokens -= 1
    return max_tokens


def _normalize_impl(implementation: str | None) -> str:
    impl = implementation or "static"
    if impl == "level9":
        return "static"
    if impl == "level10":
        return "dynamic"
    if impl in {"static", "dynamic"}:
        return impl
    raise ValueError(f"b12x_moe_fp4 only supports implementation='static' or 'dynamic', got {impl!r}")


def _dynamic_task_geometry(E: int, n: int, max_rows: int) -> tuple[int, int, int]:
    max_m_tiles = max(1, max_rows // _LEVEL_TILE_M)
    gate_tile_cnt = max(1, (n + _LEVEL_TILE_N - 1) // _LEVEL_TILE_N)
    slice_groups = max(1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK)
    max_tasks = E * max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def _get_state(
    E: int, m: int, k: int, n: int, num_topk: int,
    device: torch.device, dtype: torch.dtype,
    a1_gscale: torch.Tensor, a2_gscale: torch.Tensor,
    *,
    max_rows: int | None = None,
    input_scales_static: bool,
) -> _TPMoEState:
    global _LAST_STATE
    if max_rows is None:
        max_rows = ((m * num_topk + 127) // 128) * 128
    stream_key = int(torch.cuda.current_stream(device).cuda_stream)
    cache_key = (E, m, k, n, num_topk, max_rows, device.index or 0, dtype, stream_key)

    # Fast path: check last-used state first
    last_key, last_state = _LAST_STATE
    if last_key == cache_key:
        state = last_state
    else:
        state = _STATE_CACHE.get(cache_key)
    if state is not None:
        a1_src_ptr = a1_gscale.data_ptr()
        a2_src_ptr = a2_gscale.data_ptr()
        if (
            not input_scales_static
            or state.input_gs_src_ptr != a1_src_ptr
            or state.down_input_scale_src_ptr != a2_src_ptr
        ):
            state.input_gs.copy_(a1_gscale.expand(E))
            state.down_input_scale.copy_(a2_gscale.expand(E))
            state.input_gs_src_ptr = a1_src_ptr if input_scales_static else 0
            state.down_input_scale_src_ptr = a2_src_ptr if input_scales_static else 0
        return state

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    max_m_tiles, _, max_tasks = _dynamic_task_geometry(E, n, max_rows)

    state = _TPMoEState(
        expert_counts=torch.zeros(E, dtype=torch.int32, device=device),
        active_experts=torch.empty(E, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        active_row_counts=torch.zeros(E, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(E, dtype=torch.int32, device=device),
        token_map=torch.zeros(E, max_rows, dtype=torch.int32, device=device),
        token_weights_map=torch.zeros(E, max_rows, dtype=torch.float32, device=device),
        packed_input=torch.empty(E, max_rows, k // 2, dtype=torch.uint8, device=device),
        packed_input_scale=torch.empty(E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device),
        scheduler_out=_alloc_unique_ld_tensor((max_rows, n, E), dtype=dtype, device=device),
        scatter_output=torch.zeros(m, k, dtype=dtype, device=device),
        input_gs=torch.empty(E, dtype=torch.float32, device=device),
        down_input_scale=torch.empty(E, dtype=torch.float32, device=device),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        producers_done_count=torch.zeros(1, dtype=torch.int32, device=device),
        all_work_published=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_ready=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_m_tile=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_begin=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_count=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        tile_write_count=torch.zeros(E * max_m_tiles, dtype=torch.int32, device=device),
        input_gs_src_ptr=a1_gscale.data_ptr() if input_scales_static else 0,
        down_input_scale_src_ptr=a2_gscale.data_ptr() if input_scales_static else 0,
    )
    state.input_gs.copy_(a1_gscale.expand(E))
    state.down_input_scale.copy_(a2_gscale.expand(E))
    # Pre-compute constant views
    sf_dtype = cutlass.Float8E4M3FN
    state.packed_a_view = state.packed_input.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    state.packed_a_flat = state.packed_input.view(-1)
    state.scale_flat = state.packed_input_scale.view(-1)
    state.sfa_ptr = make_ptr(sf_dtype, state.packed_input_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    _STATE_CACHE[cache_key] = state
    _LAST_STATE = (cache_key, state)
    return state


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
) -> _WeightViews:
    """Create weight views from the concatenated expert-weight layout.

    The kernel accepts concatenated ``w13`` data with shape ``[2*n, k//2, E]``
    directly via a single TMA descriptor, so the large FP4 weight tensors stay
    as views. Only the small scale-factor tensors need compact contiguous
    storage.
    """
    global _LAST_WEIGHTS
    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_alphas.data_ptr(),
    )
    last_wkey, last_wval = _LAST_WEIGHTS
    if last_wkey == key:
        return last_wval
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _LAST_WEIGHTS = (key, cached)
        return cached

    # Permute [E, 2*n, k//2] → [2*n, k//2, E] (view, no copy!)
    w13 = w1_fp4.permute(1, 2, 0)     # [2*n, k//2, E]
    down = w2_fp4.permute(1, 2, 0)    # [k, n//2, E]

    # Concatenated w13 scale factors (6D MMA view, small contiguous copy)
    bs_u8 = w1_blockscale.view(torch.uint8)
    w13_sf = as_grouped_scale_view(bs_u8, 2 * n, k)
    down_sf = as_grouped_scale_view(w2_blockscale.view(torch.uint8), k, n)

    sf_dtype = cutlass.Float8E4M3FN
    views = _WeightViews(
        w13=w13, down=down,
        w13_sf=w13_sf, down_sf=down_sf,
        w1_alpha=_get_plain_cuda_tensor(w1_alphas),
        w2_alpha=_get_plain_cuda_tensor(w2_alphas),
    )
    views.w13_fp4 = w13.view(torch.float4_e2m1fn_x2)
    views.down_fp4 = down.view(torch.float4_e2m1fn_x2)
    views.sfb_w13_ptr = make_ptr(sf_dtype, w13_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    views.sfb_down_ptr = make_ptr(sf_dtype, down_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    _WEIGHT_CACHE[key] = views
    _LAST_WEIGHTS = (key, views)
    return views


def _ensure_compact_static_scratch(
    state: _TPMoEState,
    *,
    weight_E: int,
    device: torch.device,
) -> None:
    if (
        state.global_to_local_expert is None
        or tuple(state.global_to_local_expert.shape) != (weight_E,)
    ):
        state.global_to_local_expert = torch.empty(weight_E, dtype=torch.int32, device=device)


def _get_kernel_cache(impl: str) -> Dict[Tuple, Tuple]:
    if impl == "static":
        return _STATIC_KERNEL_CACHE
    if impl == "dynamic":
        return _DYNAMIC_KERNEL_CACHE
    raise ValueError(f"unsupported implementation {impl!r}")


def _get_impl_mac(impl: str) -> int:
    dev_idx = torch.cuda.current_device()
    key = (dev_idx, impl)
    mac = _MAC_CACHE.get(key)
    if mac is not None:
        return mac

    sm_count = get_num_sm(torch.device("cuda"))
    mac_limit = min(get_max_active_clusters(1), sm_count)
    override_name = f"B12X_{impl.upper()}_MAX_ACTIVE_CLUSTERS"
    mac_override = os.environ.get(override_name)
    if mac_override is None and impl == "dynamic":
        mac_override = os.environ.get("B12X_LEVEL10_MAX_ACTIVE_CLUSTERS")
    if mac_override is not None:
        mac = max(1, min(int(mac_override), mac_limit))
    else:
        mac = mac_limit
    _MAC_CACHE[key] = mac
    return mac


def _get_static_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    mac_override: int | None = None,
):
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("static")

    global _LAST_KERNEL
    cache_key = (
        "static", state_E, weight_E, m, k, n, num_topk, max_rows, mac, topk_ids_dtype,
        input_scales_are_reciprocal, fast_math,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = os.environ.get("B12X_STATIC_REUSE_COMPILED", "1") != "0"
    if reuse_compiled:
        cached = _STATIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEStaticKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(128, 128),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (2 * n, k, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    c_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (max_rows, n, state_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    active_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4,
    )
    active_row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake, topk_ids_fake, topk_weights_fake,
        packed_a_fake, sfa_fake,
        packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake,
        b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake,
        c_fake, row_counts_fake, active_experts_fake, active_expert_count_fake, weight_expert_ids_fake, global_to_local_expert_fake, active_row_counts_fake,
        input_gs_fake, alpha_fake, down_alpha_fake, global_scale_fake,
        scatter_fake, token_map_fake, token_weights_fake,
        mac, current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _STATIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    mac_override: int | None = None,
):
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("dynamic")

    global _LAST_KERNEL
    cache_key = (
        "dynamic", E, m, k, n, num_topk, max_rows, mac, topk_ids_dtype,
        input_scales_are_reciprocal, fast_math,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = os.environ.get("B12X_DYNAMIC_REUSE_COMPILED")
    if reuse_compiled is None:
        reuse_compiled = os.environ.get("B12X_LEVEL10_REUSE_COMPILED", "1")
    reuse_compiled = reuse_compiled != "0"
    if reuse_compiled:
        cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32
    max_m_tiles, _, max_tasks = _dynamic_task_geometry(E, n, max_rows)

    kernel = MoEDynamicKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(_LEVEL_TILE_M, _LEVEL_TILE_N),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (E * max_rows * (k // 2),), assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (E * rows_pad_k * cols_pad_k,), assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    task_ready_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    task_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    task_m_tile_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    task_slice_begin_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    task_slice_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    task_valid_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_tasks,), assumed_align=4,
    )
    tile_write_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E * max_m_tiles,), assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (2 * n, k, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    c_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (max_rows, n, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E, max_rows), stride_order=(1, 0), assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E, max_rows), stride_order=(1, 0), assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake, topk_ids_fake, topk_weights_fake,
        packed_a_fake, sfa_fake,
        packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake,
        pair_head_fake, producers_done_count_fake, all_work_published_fake,
        task_head_fake, task_tail_fake, task_ready_fake,
        task_expert_fake, task_m_tile_fake,
        task_slice_begin_fake, task_slice_count_fake, task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake,
        c_fake, row_counts_fake,
        input_gs_fake, alpha_fake, down_alpha_fake, global_scale_fake,
        scatter_fake, token_map_fake, token_weights_fake,
        mac, current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _DYNAMIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def b12x_moe_fp4(
    a: torch.Tensor,           # [m, k] bf16 activations
    a1_gscale: torch.Tensor,   # [E] or scalar — input quant scale
    w1_fp4: torch.Tensor,      # [E, 2*n, k//2] uint8
    w1_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w1_alphas: torch.Tensor,   # [E] float32
    a2_gscale: torch.Tensor,   # [E] or scalar — intermediate quant scale
    w2_fp4: torch.Tensor,      # [E, k, n//2] uint8
    w2_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w2_alphas: torch.Tensor,   # [E] float32
    topk_weights: torch.Tensor,  # [m, topk] float
    topk_ids: torch.Tensor,    # [m, topk] int
    apply_router_weight_on_input: bool = False,
    implementation: str | None = None,
    *,
    output: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool = False,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
) -> torch.Tensor:
    """MoE with the fused static or dynamic kernels.

    Uses the selected fused kernel, chunking large token batches when the CuTe
    runtime cannot describe the required work buffers in a single launch.
    """
    m, k = a.shape
    E = w1_fp4.shape[0]
    weight_E = E
    n = w2_fp4.shape[2] * 2  # intermediate_size
    num_topk = topk_ids.shape[1]
    routed_rows = m * num_topk
    device = a.device
    requested_impl = _normalize_impl(implementation)

    if apply_router_weight_on_input:
        raise NotImplementedError("apply_router_weight_on_input is not implemented in b12x_moe_fp4")
    if fast_math is None:
        fast_math = _FAST_MATH_DEFAULT
    # Shared scalar input scales are weight-side constants in the benchmarked
    # path, so treat them as static and avoid re-expanding them every launch.
    effective_input_scales_static = (
        input_scales_static
        or (a1_gscale.numel() == 1 and a2_gscale.numel() == 1)
    )

    impl = requested_impl
    use_compact_static = requested_impl == "static" and routed_rows < _get_static_compact_cutover_pairs()

    max_rows = ((routed_rows + 127) // 128) * 128
    state_max_rows = max_rows
    if use_compact_static:
        state_max_rows = max(1, routed_rows)
    max_tokens_per_launch = _safe_token_chunk(E, k, n, num_topk)
    if m > max_tokens_per_launch:
        chunk_output = output
        if chunk_output is None:
            chunk_output = torch.empty(m, k, dtype=a.dtype, device=device)
        for start in range(0, m, max_tokens_per_launch):
            end = min(start + max_tokens_per_launch, m)
            b12x_moe_fp4(
                a[start:end],
                a1_gscale,
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                a2_gscale,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                topk_weights[start:end],
                topk_ids[start:end],
                apply_router_weight_on_input=apply_router_weight_on_input,
                implementation=impl,
                output=chunk_output[start:end],
                input_scales_are_reciprocal=input_scales_are_reciprocal,
                input_scales_static=effective_input_scales_static,
                fast_math=fast_math,
            )
        return chunk_output

    compact_static_path = impl == "static" and use_compact_static
    state_E = E
    state_a1_seed = a1_gscale
    state_a2_seed = a2_gscale
    state_input_scales_static = effective_input_scales_static
    if compact_static_path:
        state_E = max(1, routed_rows)
        if a1_gscale.numel() > 1:
            state_a1_seed = a1_gscale[:1]
        if a2_gscale.numel() > 1:
            state_a2_seed = a2_gscale[:1]
        state_input_scales_static = True

    s = _get_state(
        state_E, m, k, n, num_topk, device, a.dtype, state_a1_seed, state_a2_seed,
        max_rows=state_max_rows,
        input_scales_static=state_input_scales_static,
    )

    # CUDA graph capture may run on a non-default stream, so the launch stream
    # must be fetched per-call rather than cached per-device.
    stream = current_cuda_stream()

    if compact_static_path:
        _ensure_compact_static_scratch(
            s,
            weight_E=weight_E,
            device=device,
        )
        if s.global_to_local_expert is None:
            raise RuntimeError("compact static scratch allocation failed")

        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

        wv = _get_weight_views(
            w1_fp4,
            w1_blockscale,
            w2_fp4,
            w2_blockscale,
            w1_alphas,
            w2_alphas,
            n,
            k,
        )
        input_gs = _prepare_expert_scale(a1_gscale, weight_E)
        w1_alpha = wv.w1_alpha
        w2_alpha = wv.w2_alpha
        down_input_scale = _prepare_expert_scale(a2_gscale, weight_E)
    else:
        wv = _get_weight_views(
            w1_fp4,
            w1_blockscale,
            w2_fp4,
            w2_blockscale,
            w1_alphas,
            w2_alphas,
            n,
            k,
        )
        input_gs = s.input_gs
        w1_alpha = wv.w1_alpha
        w2_alpha = wv.w2_alpha
        down_input_scale = s.down_input_scale
        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

    scatter_output = s.scatter_output if output is None else output
    if scatter_output.shape != (m, k):
        raise ValueError(f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}")
    if scatter_output.dtype != a.dtype:
        raise ValueError(f"output must have dtype {a.dtype}, got {scatter_output.dtype}")
    if scatter_output.device != device:
        raise ValueError(f"output must be on device {device}, got {scatter_output.device}")
    if not scatter_output.is_contiguous():
        raise ValueError("output must be contiguous")

    if impl == "dynamic":
        effective_mac = _get_impl_mac("dynamic")
        if not _dynamic_multicta_enabled():
            effective_mac = 1
        compiled, mac = _get_dynamic_kernel(
            E, m, k, n, num_topk, max_rows,
            topk_ids_dtype=flat_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=effective_mac,
        )
        compiled(
            a, flat_ids, flat_weights,
            s.packed_a_view, s.sfa_ptr,
            s.packed_a_flat, s.scale_flat,
            s.barrier_count, s.barrier_epoch,
            s.pair_head, s.producers_done_count, s.all_work_published,
            s.task_head, s.task_tail, s.task_ready,
            s.task_expert, s.task_m_tile,
            s.task_slice_begin, s.task_slice_count, s.task_valid_rows,
            s.tile_write_count,
            wv.w13_fp4, wv.sfb_w13_ptr,
            wv.down_fp4, wv.sfb_down_ptr,
            s.scheduler_out, s.expert_counts,
            s.input_gs, wv.w1_alpha, wv.w2_alpha, s.down_input_scale,
            scatter_output, s.token_map, s.token_weights_map,
            mac, stream,
        )
    else:
        static_mac = None
        if use_compact_static and routed_rows < 40:
            # Tiny compact launches have very little FC2 tile work, so capping
            # resident clusters avoids idle CTA participation in the barrier phases.
            static_mac = min(_get_impl_mac("static"), 64)
        compiled, mac = _get_static_kernel(
            state_E, weight_E, m, k, n, num_topk, state_max_rows,
            topk_ids_dtype=flat_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=static_mac,
        )
        exe_args, adapted_args = _get_cached_static_execution_args(
            compiled,
            a,
            flat_ids,
            flat_weights,
            s,
            wv,
            input_gs,
            w1_alpha,
            w2_alpha,
            down_input_scale,
            scatter_output,
            mac,
            stream,
        )
        _ = adapted_args
        compiled.run_compiled_program(exe_args)
    return scatter_output
