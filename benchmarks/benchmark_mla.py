#!/usr/bin/env python3
"""Benchmark graph-replayed GLM-5.1 TP8 decode public APIs: NSA indexer + sparse MLA."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.mla.split import select_sparse_mla_split_decode_config
from b12x.attention.nsa_indexer.reference import sparse_nsa_index_reference
from b12x.integration.mla import (
    MLASparseDecodeMetadata,
    MLAWorkspace,
    clear_mla_caches,
    dense_mla_reference,
    pack_mla_kv_cache_reference,
    sparse_mla_decode_forward,
)
from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
)

from benchmarks.common import require_sm120


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_CACHE_LENS = (1024, 8192, 16384, 32768, 65536)
DEFAULT_TP_SIZE = 8
DEFAULT_TP_RANK = 0
MLA_MAX_ABS_TOL = 0.10
MLA_RMSE_TOL = 0.005
MLA_COS_TOL = 0.9995


@dataclass(frozen=True)
class GLMDecodeContractConfig:
    num_attention_heads: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    tp_size: int
    tp_rank: int
    page_size: int = 64

    @property
    def num_local_heads(self) -> int:
        return self.num_attention_heads // self.tp_size

    @property
    def q_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def sm_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


@dataclass(frozen=True)
class DecodeCase:
    batch_size: int
    cache_len: int
    topk: int


@dataclass(frozen=True)
class SanityMetrics:
    max_abs: float
    rmse: float
    cos: float


@dataclass(frozen=True)
class CaseReport:
    case: DecodeCase
    indexer_us: float
    mla_us: float
    split_enabled: bool
    chunk_size: int
    num_chunks: int
    mla_sanity: SanityMetrics

    @property
    def total_us(self) -> float:
        return self.indexer_us + self.mla_us


class BenchmarkFailure(RuntimeError):
    def __init__(self, case: DecodeCase, message: str):
        super().__init__(f"bs={case.batch_size} ctx={case.cache_len} topk={case.topk}: {message}")
        self.case = case


def _require_glm_config() -> pathlib.Path:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        raise SystemExit(f"GLM-5.1 config not found at {config_path}")
    return config_path


def _load_glm_contract_config(*, tp_size: int, tp_rank: int) -> GLMDecodeContractConfig:
    config = json.loads(_require_glm_config().read_text())
    num_attention_heads = int(config["num_attention_heads"])
    if num_attention_heads % tp_size != 0:
        raise SystemExit(
            f"num_attention_heads={num_attention_heads} is not divisible by tp_size={tp_size}"
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise SystemExit(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    return GLMDecodeContractConfig(
        num_attention_heads=num_attention_heads,
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _resolve_topk(*, cache_len: int, topk_cap: int) -> int:
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if topk_cap <= 0:
        raise ValueError(f"topk_cap must be positive, got {topk_cap}")
    return min(cache_len, topk_cap)


def _build_decode_cases(
    *,
    batch_sizes: list[int],
    cache_lens: list[int],
    topk_cap: int,
) -> list[DecodeCase]:
    cases: list[DecodeCase] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"batch sizes must be positive, got {batch_size}")
        for cache_len in cache_lens:
            cases.append(
                DecodeCase(
                    batch_size=batch_size,
                    cache_len=cache_len,
                    topk=_resolve_topk(cache_len=cache_len, topk_cap=topk_cap),
                )
            )
    return cases


def _capture_graph(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _bench_graph(graph: torch.cuda.CUDAGraph, *, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    return statistics.geometric_mean(values)


def _compare(a: torch.Tensor, b: torch.Tensor) -> SanityMetrics:
    diff = (a - b).to(torch.float32)
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return SanityMetrics(
        max_abs=diff.abs().max().item(),
        rmse=torch.sqrt(diff.square().mean()).item(),
        cos=cos,
    )


def _make_dense_candidate_page_table(
    *,
    batch_size: int,
    cache_len: int,
    device: torch.device,
) -> torch.Tensor:
    row = torch.arange(cache_len, dtype=torch.int32, device=device)
    return row.unsqueeze(0).repeat(batch_size, 1)


def _make_indexer_inputs(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_fp8 = (
        torch.randn(
            (case.batch_size, cfg.index_n_heads, cfg.index_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(2.0)
    ).to(torch.float8_e4m3fn)
    weights = (
        torch.randn(
            (case.batch_size, cfg.index_n_heads, 1),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(cfg.index_n_heads**0.5)
    )
    k = (
        torch.randn(
            (case.cache_len, cfg.index_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(3.0)
    )
    return q_fp8, weights, pack_nsa_index_k_cache_reference(k, page_size=cfg.page_size)


def _make_mla_inputs(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_all = (
        torch.randn(
            (case.batch_size, cfg.num_local_heads, cfg.q_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_nope = (
        torch.randn(
            (case.cache_len, 1, cfg.kv_lora_rank),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_rope = (
        torch.randn(
            (case.cache_len, 1, cfg.qk_rope_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    return q_all, k_nope, k_rope, kv_cache


def _make_mla_workspace(
    *,
    cfg: GLMDecodeContractConfig,
    case: DecodeCase,
    device: torch.device,
) -> MLAWorkspace:
    return MLAWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=cfg.num_local_heads,
        head_dim=cfg.q_head_dim,
        v_head_dim=cfg.kv_lora_rank,
        topk=case.topk,
        max_total_q=case.batch_size,
        max_batch=case.batch_size,
    )


def _run_decode_case(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
) -> CaseReport:
    q_fp8, weights, index_k_cache = _make_indexer_inputs(
        case=case,
        cfg=cfg,
        seed=seed,
        device=device,
    )
    q_all, k_nope, k_rope, kv_cache = _make_mla_inputs(
        case=case,
        cfg=cfg,
        seed=seed + 1,
        device=device,
    )
    candidate_page_table = _make_dense_candidate_page_table(
        batch_size=case.batch_size,
        cache_len=case.cache_len,
        device=device,
    )
    full_cache_seqlens = torch.full(
        (case.batch_size,),
        case.cache_len,
        dtype=torch.int32,
        device=device,
    )
    indexer_metadata = NSAIndexerDecodeMetadata(
        page_table_1=candidate_page_table,
        cache_seqlens_int32=full_cache_seqlens,
    )

    def run_indexer():
        return sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            topk=case.topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual_topk = run_indexer()
    expected_topk = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=candidate_page_table,
        query_row_to_batch=torch.arange(case.batch_size, dtype=torch.int32, device=device),
        seqlens_per_query=full_cache_seqlens,
        topk=case.topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    if not torch.equal(actual_topk, expected_topk):
        mismatch = int((actual_topk != expected_topk).sum().item())
        raise BenchmarkFailure(
            case,
            f"indexer correctness mismatch: {mismatch} differing entries",
        )

    nsa_cache_seqlens = torch.full(
        (case.batch_size,),
        case.topk,
        dtype=torch.int32,
        device=device,
    )
    mla_metadata = MLASparseDecodeMetadata(
        page_table_1=actual_topk,
        cache_seqlens_int32=full_cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        max_seq_len_k=case.cache_len,
    )
    mla_workspace = _make_mla_workspace(cfg=cfg, case=case, device=device)
    split_cfg = select_sparse_mla_split_decode_config(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=actual_topk,
        output_dtype=q_all.dtype,
        v_head_dim=cfg.kv_lora_rank,
    )

    def run_mla():
        return sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=mla_metadata,
            workspace=mla_workspace,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    clear_mla_caches()
    actual_output = run_mla()
    expected_output = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=actual_topk,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize()
    mla_sanity = _compare(actual_output, expected_output)
    if mla_sanity.max_abs > MLA_MAX_ABS_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA max_abs {mla_sanity.max_abs:.6f} exceeded {MLA_MAX_ABS_TOL:.6f}",
        )
    if mla_sanity.rmse > MLA_RMSE_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA rmse {mla_sanity.rmse:.6f} exceeded {MLA_RMSE_TOL:.6f}",
        )
    if mla_sanity.cos < MLA_COS_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA cos {mla_sanity.cos:.6f} fell below {MLA_COS_TOL:.6f}",
        )

    clear_nsa_indexer_caches()
    indexer_graph = _capture_graph(run_indexer, warmup=warmup)
    indexer_us = statistics.median(_bench_graph(indexer_graph, replays=replays))

    clear_mla_caches()
    mla_graph = _capture_graph(run_mla, warmup=warmup)
    mla_us = statistics.median(_bench_graph(mla_graph, replays=replays))

    return CaseReport(
        case=case,
        indexer_us=indexer_us,
        mla_us=mla_us,
        split_enabled=split_cfg is not None,
        chunk_size=0 if split_cfg is None else split_cfg.chunk_size,
        num_chunks=0 if split_cfg is None else split_cfg.num_chunks,
        mla_sanity=mla_sanity,
    )


def collect_case_reports(
    args: argparse.Namespace,
    *,
    device: torch.device | None = None,
) -> list[CaseReport]:
    cfg = _load_glm_contract_config(tp_size=args.tp_size, tp_rank=args.tp_rank)
    device = require_sm120() if device is None else device
    cases = _build_decode_cases(
        batch_sizes=_parse_csv_ints(args.batch_sizes),
        cache_lens=_parse_csv_ints(args.cache_lens),
        topk_cap=min(args.topk_cap, cfg.index_topk),
    )
    reports: list[CaseReport] = []
    case_seed = args.seed
    for case in cases:
        reports.append(
            _run_decode_case(
                case=case,
                cfg=cfg,
                warmup=args.warmup,
                replays=args.replays,
                seed=case_seed,
                device=device,
            )
        )
        case_seed += 17
    return reports


def _render_case_line(report: CaseReport) -> str:
    split_flag = "on" if report.split_enabled else "off"
    return (
        f"glm51-decode tp8 bs={report.case.batch_size:2d} ctx={report.case.cache_len:6d} "
        f"topk={report.case.topk:4d} split={split_flag:>3s} "
        f"chunk={report.chunk_size:3d} nchunks={report.num_chunks:d} | "
        f"total={report.total_us:8.2f} us | "
        f"indexer={report.indexer_us:8.2f} us | "
        f"mla={report.mla_us:8.2f} us"
    )


def _render_summary_lines(reports: list[CaseReport]) -> list[str]:
    total_geo = _geomean([report.total_us for report in reports])
    indexer_geo = _geomean([report.indexer_us for report in reports])
    mla_geo = _geomean([report.mla_us for report in reports])
    return [
        "Summary",
        f"  cases: {len(reports)}",
        f"  total geo:   {total_geo:.2f} us",
        f"  indexer geo: {indexer_geo:.2f} us",
        f"  mla geo:     {mla_geo:.2f} us",
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help=f"decode batch sizes, default {','.join(str(v) for v in DEFAULT_BATCH_SIZES)}",
    )
    parser.add_argument(
        "--cache-lens",
        default="1024,8192,16384,32768,65536",
        help=f"decode cache lengths, default {','.join(str(v) for v in DEFAULT_CACHE_LENS)}",
    )
    parser.add_argument("--topk-cap", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--tp-rank", type=int, default=DEFAULT_TP_RANK)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=70_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reports = collect_case_reports(args)
    except BenchmarkFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for report in reports:
        print(_render_case_line(report))
    for line in _render_summary_lines(reports):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
