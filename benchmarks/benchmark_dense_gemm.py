#!/usr/bin/env python3
"""Benchmark: b12x dense_gemm vs FlashInfer-CUTLASS with CUDA graph replay.

Compares block-scaled FP4 dense GEMM performance on the Nemotron 3 Super
shared-expert down-projection shape `[M, 5376] x [5376, 4096]` across small
decode-style batch sizes.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from typing import Callable, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout, get_hardware_info
from b12x.gemm.dense import dense_gemm

from flashinfer.gemm import mm_fp4


# Nemotron 3 Super shared expert down projection from the released NVFP4
# checkpoint:
#   down: [M, 5376] x [5376, 4096]
NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
NEMOTRON_HIDDEN_SIZE = 4096

GEMM_SPECS = [
    # (name, K, N, note)
    (
        "Nemotron shared expert down",
        NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE,
        NEMOTRON_HIDDEN_SIZE,
        "NVIDIA Nemotron 3 Super shared_experts.down_proj",
    ),
]

BATCH_SIZES = [2, 4, 8]
REFERENCE_BACKEND = "cutlass"
REFERENCE_LABEL = "FlashInfer CUTLASS"
COSINE_THRESHOLD = 0.999999
_L2_FLUSH_BUFFER_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_AUTO_L2_FLUSH_MULTIPLIER = 2
_FALLBACK_L2_FLUSH_BYTES = 32 << 20


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def resolve_l2_flush_bytes(bytes_hint: int) -> int:
    if bytes_hint < 0:
        raise ValueError(f"l2 flush bytes must be non-negative, got {bytes_hint}")
    if bytes_hint > 0:
        return int(bytes_hint)
    try:
        l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    except Exception:
        l2_bytes = 0
    if l2_bytes > 0:
        return l2_bytes * _AUTO_L2_FLUSH_MULTIPLIER
    return _FALLBACK_L2_FLUSH_BYTES


def make_l2_flush_fn(
    *,
    enabled: bool,
    bytes_hint: int = 0,
) -> Callable[[], None] | None:
    if not enabled:
        return None
    flush_bytes = resolve_l2_flush_bytes(bytes_hint)
    device_idx = torch.cuda.current_device()
    key = (device_idx, flush_bytes)
    buffer = _L2_FLUSH_BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=f"cuda:{device_idx}")
        _L2_FLUSH_BUFFER_CACHE[key] = buffer

    def flush(cache_buffer: torch.Tensor = buffer) -> None:
        cache_buffer.bitwise_not_()

    return flush


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None] | None = None,
) -> List[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    cosine_threshold: float,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected during correctness check vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    cos = cosine_similarity(candidate, reference)
    print(
        f"    check vs {label}: max_abs={max_abs:.8f} "
        f"rmse={rmse:.8f} cos={cos:.10f}"
    )
    if not math.isfinite(cos):
        raise CorrectnessError(
            f"cosine similarity vs {label} is non-finite: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )
    if cos < cosine_threshold:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below threshold "
            f"{cosine_threshold:.6f}: got {cos:.10f}"
        )


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    # Warm eager launch state before capture so compile/cache work does not leak
    # into the replay measurement.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return replay


def make_quantized_operand(M: int, K: int):
    source = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) / 4
    row_counts = torch.full((1,), M, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32, device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def bench_one(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    """Benchmark one (M,N,K) problem with CUDA graph replay timing."""
    torch.manual_seed(42)
    a_packed, a_sf, a_gs = make_quantized_operand(M, K)
    b_packed, b_sf, b_gs = make_quantized_operand(N, K)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)

    a_fp4_2d = a_packed[:, :, 0].contiguous()
    b_fp4_2d = b_packed[:, :, 0].contiguous()
    a_sf_2d = convert_sf_from_mma_layout(a_sf, m=M, k=K, num_groups=1)
    b_sf_2d = convert_sf_from_mma_layout(b_sf, m=N, k=K, num_groups=1)

    results = {}

    # b12x
    try:
        b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

        def b12x_launch():
            dense_gemm(
                (a_packed, a_sf), (b_packed, b_sf), alpha=alpha,
                ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16", sf_vec_size=16, out=b12x_out,
            )
        b12x_replay = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = b12x_out
        results["b12x"] = bench_events(
            b12x_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results["b12x"] = None
        print(f"      b12x FAILED: {exc}")

    # FlashInfer CUTLASS reference
    try:
        ref_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        def cutlass_launch():
            mm_fp4(
                a_fp4_2d, b_fp4_2d.T, a_sf_2d, b_sf_2d.T,
                alpha, torch.bfloat16, ref_out, block_size=16,
                use_8x4_sf_layout=False, backend=REFERENCE_BACKEND, use_nvfp4=True,
            )
        ref_replay = capture_graph_replay(cutlass_launch)
        results["ref_replay"] = ref_replay
        results["ref_out"] = ref_out
        results[REFERENCE_LABEL] = bench_events(
            ref_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[REFERENCE_LABEL] = None
        print(f"      {REFERENCE_LABEL} FAILED: {exc}")

    if check:
        if results.get("b12x_replay") is None or results.get("ref_replay") is None:
            raise BenchmarkAbort(
                "correctness check requires both b12x and reference replays"
            )
        results["b12x_replay"]()
        results["ref_replay"]()
        torch.cuda.synchronize()
        check_outputs(
            results["b12x_out"][:, :, 0],
            results["ref_out"],
            label=REFERENCE_LABEL,
            cosine_threshold=COSINE_THRESHOLD,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict GPU L2 before each warmup and timed launch (default: enabled).",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    parser.set_defaults(check=True)
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Run correctness checks against FlashInfer CUTLASS and fail hard when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()

    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"Requires sm_120 or sm_121, got sm_{major}{minor}")
    torch.empty(1, device="cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    print(f"Dense FP4 GEMM: b12x vs {REFERENCE_LABEL}")
    print("NVIDIA Nemotron 3 Super shared-expert down-proj")
    print("Timing mode: CUDA graph replay")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    if args.check:
        print(f"Correctness check: on (cos >= {COSINE_THRESHOLD:.6f})")
    else:
        print("Correctness check: off")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    # Collect all results for summary
    all_results = []  # (name, bs, M, N, K, b12x_med, ref_med)

    for name, K, N, note in GEMM_SPECS:
        print(f"{'=' * 75}")
        print(f"  {name}  K={K} N={N}  [{note}]")
        print(f"{'=' * 75}")

        for bs in args.batch_sizes:
            M = bs
            try:
                results = bench_one(
                    M,
                    N,
                    K,
                    warmup=args.warmup,
                    iters=args.iters,
                    check=args.check,
                    l2_flush=l2_flush,
                )
            except BenchmarkAbort as exc:
                print(
                    f"ERROR: benchmark aborted for {name} "
                    f"(bs={bs}, M={M}, N={N}, K={K}): {exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            b12x_med = statistics.median(results["b12x"]) * 1000 if results.get("b12x") else None
            ref_med = statistics.median(results[REFERENCE_LABEL]) * 1000 if results.get(REFERENCE_LABEL) else None

            parts = [f"  bs={bs:<3} (M={M:>2})"]
            if b12x_med is not None:
                parts.append(f"b12x={b12x_med:6.1f}")
            if ref_med is not None:
                parts.append(f"CUTLASS={ref_med:6.1f}")

            ratios = []
            if b12x_med and ref_med:
                r = b12x_med / ref_med
                ratios.append(f"b12x/flashinfer-cutlass={r:.2f}x")

            print("  ".join(parts) + "  " + "  ".join(ratios) + "  (graph us)")

            all_results.append((name, bs, M, N, K, b12x_med, ref_med))

        print()

    print(f"\n{'=' * 75}")
    print(f"  SUMMARY: b12x / {REFERENCE_LABEL} (CUDA graph replay, lower = b12x faster)")
    print(f"{'=' * 75}")
    header = f"  {'GEMM':<30}"
    for bs in args.batch_sizes:
        header += f"  M={bs:<5}"
    print(header)
    print("  " + "-" * 70)

    ref_ratios = []
    for name, K, N, note in GEMM_SPECS:
        row = f"  {name:<30}"
        for bs in args.batch_sizes:
            match = [r for r in all_results if r[0] == name and r[1] == bs]
            if match and match[0][5] and match[0][6]:
                ratio = match[0][5] / match[0][6]
                row += f"  {ratio:.2f}x "
                ref_ratios.append(ratio)
            else:
                row += f"  {'n/a':>6}"
        print(row)

    if ref_ratios:
        geo = 1.0
        for r in ref_ratios:
            geo *= r
        geo **= 1.0 / len(ref_ratios)
        print(f"\n  geo mean: {geo:.2f}x")


if __name__ == "__main__":
    main()
