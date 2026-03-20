from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x.attention import layout_utils
from b12x.attention import utils
from b12x.attention.forward import SM120ForwardKernel, copy_flattened
from b12x.cute.fp4 import byte_perm


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


def _selector_from_perm(perm: tuple[int, int, int, int]) -> int:
    selector = 0
    for idx, src in enumerate(perm):
        selector |= int(src) << (4 * idx)
    return selector


def _default_selectors() -> list[int]:
    return [_selector_from_perm(perm) for perm in itertools.permutations(range(4))]


@dataclass(frozen=True)
class ProbeConfig:
    source_mode: str
    copy_tiling: str


class Fp8PvLayoutProbeKernel:
    tile_m = 48
    tile_n = 64
    head_dim = 256
    num_compute_warps = 3
    num_threads = num_compute_warps * 32

    def __init__(self, *, source_mode: str, copy_tiling: str):
        if source_mode not in {"word_direct", "word_transpose"}:
            raise ValueError(f"unsupported source_mode={source_mode}")
        if copy_tiling not in {"A", "B"}:
            raise ValueError(f"unsupported copy_tiling={copy_tiling}")
        self.source_mode = source_mode
        self.copy_tiling = copy_tiling
        self.kernel_spec = SM120ForwardKernel(
            cutlass.BFloat16,
            self.head_dim,
            kv_dtype=cutlass.Float8E4M3FN,
            head_dim_v=self.head_dim,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            num_threads=(self.num_compute_warps + 1) * 32,
            num_compute_warps=self.num_compute_warps,
            Q_in_regs=False,
        )

    @cute.jit
    def __call__(
        self,
        mVRaw: cute.Tensor,
        mMismatch: cute.Tensor,
        mSelector: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel_spec._setup_attributes()
        shared_storage = self.kernel_spec._get_shared_storage_cls()
        _, tiled_mma_pv = self.kernel_spec._get_tiled_mma()
        self.kernel(
            mVRaw,
            mMismatch,
            mSelector,
            shared_storage,
            self.kernel_spec.sV_layout,
            self.kernel_spec.sV_raw_layout,
            tiled_mma_pv,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVRaw: cute.Tensor,
        mMismatch: cute.Tensor,
        mSelector: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
        sV_layout: cutlass.Constexpr,
        sV_raw_layout: cutlass.Constexpr,
        tiled_mma_pv: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVRaw = storage.sV_raw.get_tensor(
            cute.make_layout(
                (self.tile_n, self.head_dim, 1),
                stride=(self.head_dim, 1, self.tile_n * self.head_dim),
            )
        )
        total_elems = self.tile_n * self.head_dim
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_threads)):
            linear_idx = tidx + idx_iter * self.num_threads
            if linear_idx < total_elems:
                row = linear_idx // self.head_dim
                col = linear_idx - row * self.head_dim
                sVRaw[row, col, 0] = mVRaw[row, col]
        cute.arch.sync_threads()

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        sVt = layout_utils.transpose_view(sV)
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))

        # Known-good byte path from the current kernel.
        sVtRaw = layout_utils.transpose_view(sVRaw)
        sVtRawU8 = cute.recast_tensor(sVtRaw, cutlass.Uint8)
        ref_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint8)
        ref_copy = utils.make_tiled_copy_B(ref_copy_atom, tiled_mma_pv).get_slice(tidx)
        tRefRaw = cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint8), cutlass.Uint8)
        tRefCopyView = ref_copy.retile(tRefRaw)
        tRefSrc = ref_copy.partition_S(sVtRawU8)
        copy_flattened(tRefSrc[None, None, 0], tRefCopyView[None, None, 0])

        # Candidate word path to be fixed with per-word prmt.
        sVRawU32 = cute.recast_tensor(sVRaw, cutlass.Uint32)
        if const_expr(self.source_mode == "word_transpose"):
            sCand = layout_utils.transpose_view(sVRawU32)
        else:
            sCand = sVRawU32
        cand_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint32)
        if const_expr(self.copy_tiling == "A"):
            cand_copy = utils.make_tiled_copy_A(cand_copy_atom, tiled_mma_pv).get_slice(tidx)
        else:
            cand_copy = utils.make_tiled_copy_B(cand_copy_atom, tiled_mma_pv).get_slice(tidx)
        tCandRaw = cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint32), cutlass.Uint32)
        tCandCopyView = cand_copy.retile(tCandRaw)
        tCandSrc = cand_copy.partition_S(sCand)
        copy_flattened(tCandSrc[None, None, 0], tCandCopyView[None, None, 0])

        selector = Int32(mSelector[0])
        cand_words = cute.flatten(tCandRaw)
        cand_bytes = cute.flatten(cute.recast_tensor(tCandRaw, cutlass.Uint8))
        ref_bytes = cute.flatten(tRefRaw)

        for word_idx in cutlass.range_constexpr(cute.size(cand_words.shape)):
            cand_words[word_idx] = byte_perm(cand_words[word_idx], cand_words[word_idx], selector)

        local_mismatch = Int32(0)
        for idx in cutlass.range_constexpr(cute.size(ref_bytes.shape)):
            if cand_bytes[idx] != ref_bytes[idx]:
                local_mismatch += Int32(1)
        mMismatch[tidx] = local_mismatch


def run_probe(config: ProbeConfig, selectors: list[int]) -> list[tuple[int, int]]:
    device = torch.device("cuda")
    raw = torch.arange(64 * 256, device=device, dtype=torch.uint8).view(64, 256)
    v_raw = raw.view(torch.float8_e4m3fn)
    mismatch = torch.empty(96, device=device, dtype=torch.int32)
    selector_buf = torch.empty(1, device=device, dtype=torch.int32)
    kernel = Fp8PvLayoutProbeKernel(
        source_mode=config.source_mode,
        copy_tiling=config.copy_tiling,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(v_raw, cutlass.Float8E4M3FN),
        _to_cute_tensor(mismatch, cutlass.Int32),
        _to_cute_tensor(selector_buf, cutlass.Int32),
        stream,
    )

    results: list[tuple[int, int]] = []
    for selector in selectors:
        selector_buf.fill_(selector)
        compiled(
            _to_cute_tensor(v_raw, cutlass.Float8E4M3FN),
            _to_cute_tensor(mismatch, cutlass.Int32),
            _to_cute_tensor(selector_buf, cutlass.Int32),
            stream,
        )
        torch.cuda.synchronize()
        results.append((selector, int(mismatch.sum().item())))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FP8 PV raw-fragment byte layouts.")
    parser.add_argument(
        "--source-mode",
        choices=["word_direct", "word_transpose", "all"],
        default="all",
    )
    parser.add_argument(
        "--copy-tiling",
        choices=["A", "B", "all"],
        default="all",
    )
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    torch.cuda.init()
    selectors = _default_selectors()
    configs: list[ProbeConfig] = []
    source_modes = ["word_direct", "word_transpose"] if args.source_mode == "all" else [args.source_mode]
    copy_tilings = ["A", "B"] if args.copy_tiling == "all" else [args.copy_tiling]
    for source_mode in source_modes:
        for copy_tiling in copy_tilings:
            configs.append(ProbeConfig(source_mode=source_mode, copy_tiling=copy_tiling))

    for config in configs:
        results = sorted(run_probe(config, selectors), key=lambda item: item[1])
        print(f"== source={config.source_mode} copy={config.copy_tiling} ==")
        for selector, mismatches in results[: args.topk]:
            print(f"selector=0x{selector:04x} mismatches={mismatches}")


if __name__ == "__main__":
    main()
