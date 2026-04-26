#!/usr/bin/env python3
"""Microbenchmark for DSV4-Flash sparse MLA decode on b12x kernels.

Designed for both standalone use (self-times via CUDA events) and for
profiling under nsys/ncu (the inner loop is just kernel calls).

Usage:
    python benchmarks/bench_dsv4_decode.py
    nsys profile -o dsv4_decode python benchmarks/bench_dsv4_decode.py --nvtx
    ncu --kernel-name-base function --launch-skip 5 --launch-count 10 \\
        --metrics smsp__sass_thread_inst_executed.sum,gpu__time_duration.sum,\\
                  dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
                  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
                  l1tex__throughput.avg.pct_of_peak_sustained_elapsed \\
        python benchmarks/bench_dsv4_decode.py --ncu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

from b12x.attention.mla.kernel import run_sparse_mla_kernel
from b12x.attention.mla.split import (
    run_sparse_mla_split_decode,
    default_sparse_mla_split_decode_config_for_width,
)
from b12x.attention.mla.reference import pack_mla_kv_cache_reference


_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_DSV4_HEAD_DIM = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM   # 512
_DSV4_V_HEAD_DIM = 448
_DSV4_SM_SCALE = _DSV4_HEAD_DIM ** -0.5
_DSV4_NUM_HEADS = 8  # one TP rank for an 8-way TP setup
_PACKED_BYTES_PER_TOKEN = 656


@dataclass(frozen=True)
class BenchCase:
    label: str
    cache_len: int
    num_q: int
    use_split: bool


def _build_inputs(case: BenchCase, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q_all = (
        torch.randn(
            (case.num_q, _DSV4_NUM_HEADS, _DSV4_HEAD_DIM),
            generator=g, dtype=torch.bfloat16,
        ).to(device) * 0.02
    )
    k_nope = (
        torch.randn(
            (case.cache_len, 1, _DSV4_NOPE_DIM),
            generator=g, dtype=torch.bfloat16,
        ).to(device) * 0.02
    )
    k_rope = (
        torch.randn(
            (case.cache_len, 1, _DSV4_ROPE_DIM),
            generator=g, dtype=torch.bfloat16,
        ).to(device) * 0.02
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope, nope_logical_dim=_DSV4_NOPE_DIM)
    page_table = torch.arange(case.cache_len, dtype=torch.int32, device=device).repeat(case.num_q, 1)
    active_counts = torch.full((case.num_q,), case.cache_len, dtype=torch.int32, device=device)
    sm_scale = torch.tensor([_DSV4_SM_SCALE], dtype=torch.float32, device=device)
    output = torch.empty(case.num_q, _DSV4_NUM_HEADS, _DSV4_V_HEAD_DIM,
                         dtype=torch.bfloat16, device=device)
    return q_all, packed, page_table, active_counts, sm_scale, output


def _build_split_extras(case: BenchCase, device: torch.device):
    cfg = default_sparse_mla_split_decode_config_for_width(case.cache_len)
    if cfg is None:
        raise RuntimeError(f"no split config for cache_len={case.cache_len}")
    chunk_size = cfg.chunk_size
    num_chunks = cfg.num_chunks
    tmp_output = torch.empty(case.num_q, _DSV4_NUM_HEADS, num_chunks, _DSV4_V_HEAD_DIM,
                             dtype=torch.bfloat16, device=device)
    tmp_lse = torch.full((case.num_q, _DSV4_NUM_HEADS, num_chunks), float("-inf"),
                        dtype=torch.float32, device=device)
    kv_chunk_size_ptr = torch.tensor([chunk_size], dtype=torch.int32, device=device)
    num_chunks_ptr = torch.tensor([num_chunks], dtype=torch.int32, device=device)
    return tmp_output, tmp_lse, kv_chunk_size_ptr, num_chunks_ptr, num_chunks


def _run_one(case: BenchCase, inputs, split_extras=None) -> None:
    q_all, packed, page_table, active_counts, sm_scale, output = inputs
    if case.use_split:
        tmp_output, tmp_lse, kv_chunk_size_ptr, num_chunks_ptr, num_chunks = split_extras
        run_sparse_mla_split_decode(
            q_all=q_all, kv_cache=packed, page_table_1=page_table,
            active_token_counts=active_counts, sm_scale=sm_scale,
            kv_chunk_size_ptr=kv_chunk_size_ptr, num_chunks_ptr=num_chunks_ptr,
            tmp_output=tmp_output, tmp_lse=tmp_lse, output=output,
            launch_num_chunks=num_chunks,
        )
    else:
        run_sparse_mla_kernel(
            q_all=q_all, kv_cache=packed, page_table_1=page_table,
            active_token_counts=active_counts, sm_scale=sm_scale, output=output,
        )


def _self_time(case: BenchCase, device: torch.device, warmup: int, iters: int) -> tuple[float, float]:
    inputs = _build_inputs(case, device)
    split_extras = _build_split_extras(case, device) if case.use_split else None

    # JIT compile + warm caches
    for _ in range(warmup):
        _run_one(case, inputs, split_extras)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _run_one(case, inputs, split_extras)
    end.record()
    torch.cuda.synchronize(device)

    total_ms = start.elapsed_time(end)
    per_iter_us = (total_ms / iters) * 1_000.0

    # bytes read per iter: KV cache (packed) + Q + page_table + writes
    kv_bytes = case.cache_len * _PACKED_BYTES_PER_TOKEN
    q_bytes = case.num_q * _DSV4_NUM_HEADS * _DSV4_HEAD_DIM * 2  # bf16
    out_bytes = case.num_q * _DSV4_NUM_HEADS * _DSV4_V_HEAD_DIM * 2
    pt_bytes = case.num_q * case.cache_len * 4  # int32
    total_bytes = kv_bytes + q_bytes + out_bytes + pt_bytes
    achieved_gbs = (total_bytes / 1e9) / (per_iter_us / 1e6)

    return per_iter_us, achieved_gbs


def _make_cases() -> list[BenchCase]:
    return [
        # Scaling curve, single-tile path
        BenchCase("decode_1k",        cache_len=1024,  num_q=1, use_split=False),
        BenchCase("decode_2k",        cache_len=2048,  num_q=1, use_split=False),
        BenchCase("decode_4k",        cache_len=4096,  num_q=1, use_split=False),
        BenchCase("decode_8k",        cache_len=8192,  num_q=1, use_split=False),
        BenchCase("decode_16k",       cache_len=16384, num_q=1, use_split=False),
        BenchCase("decode_32k",       cache_len=32768, num_q=1, use_split=False),
        # Split-kernel comparison at the widths it supports
        BenchCase("decode_1k_split",  cache_len=1024,  num_q=1, use_split=True),
        BenchCase("decode_2k_split",  cache_len=2048,  num_q=1, use_split=True),
        # MTP cases (the actual unblock value-prop)
        BenchCase("mtp2_decode_4k",   cache_len=4096,  num_q=2, use_split=False),
        BenchCase("mtp4_decode_4k",   cache_len=4096,  num_q=4, use_split=False),
        BenchCase("mtp4_decode_8k",   cache_len=8192,  num_q=4, use_split=False),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iters per case (covers JIT compile)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Timed iters per case")
    parser.add_argument("--nvtx", action="store_true",
                        help="Wrap each case in NVTX range for nsys")
    parser.add_argument("--ncu", action="store_true",
                        help="Reduce iters for ncu (it does its own re-runs per metric)")
    parser.add_argument("--case", default=None,
                        help="Run only this case label")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"# device: {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    print(f"# torch={torch.__version__}")

    if args.ncu:
        args.warmup = 5
        args.iters = 10

    cases = _make_cases()
    if args.case:
        cases = [c for c in cases if c.label == args.case]
        if not cases:
            print(f"unknown case: {args.case}", file=sys.stderr)
            return 1

    print(f"# warmup={args.warmup} iters={args.iters}")
    header = f"{'case':<18} {'cache_len':>9} {'num_q':>5} {'kernel':>8} {'us/iter':>10} {'tok/s':>10} {'GB/s':>8}"
    print(header)
    print("-" * len(header))

    if args.nvtx:
        torch.cuda.nvtx.range_push("dsv4_bench")

    for case in cases:
        if args.nvtx:
            torch.cuda.nvtx.range_push(case.label)
        try:
            per_iter_us, gbs = _self_time(case, device, args.warmup, args.iters)
        except Exception as e:
            print(f"{case.label:<18} ERROR: {e}", file=sys.stderr)
            if args.nvtx:
                torch.cuda.nvtx.range_pop()
            continue
        kernel_kind = "split" if case.use_split else "single"
        # tok/s: each iter "produces" num_q tokens (decode), so steady-state
        # throughput per request is 1e6 / per_iter_us tok/s for a single decoder.
        toks_per_s = (case.num_q * 1_000_000.0) / per_iter_us
        print(f"{case.label:<18} {case.cache_len:>9d} {case.num_q:>5d} {kernel_kind:>8} "
              f"{per_iter_us:>10.2f} {toks_per_s:>10.1f} {gbs:>8.1f}")
        if args.nvtx:
            torch.cuda.nvtx.range_pop()

    if args.nvtx:
        torch.cuda.nvtx.range_pop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
