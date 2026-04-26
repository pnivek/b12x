"""Tensor-capture shims for vLLM's DeepseekV4 sparse MLA reference paths.

Used to record the inputs/outputs of the existing Triton-based reference
attention paths so we can build an offline parity oracle for the b12x kernel
without needing vLLM in the loop on every test.

Usage from a vLLM driver script (run inside the container):

    from b12x.integration.vllm_capture import install_capture_shims
    install_capture_shims(out_dir="/workspace/captures", capture_n=1)
    # ... then run vllm.LLM(...).generate(...) on a small prompt

Each captured call writes a single ``.pt`` file with all tensors needed to
reproduce the call offline.  The shim auto-disables itself after ``capture_n``
captures per phase to keep file count small.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch


_LOCK = threading.Lock()
_CAPTURE_COUNTERS: dict[str, int] = {}


def _safe_clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().cpu()
    return x


def _dump_payload(out_dir: Path, phase: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    seq = _CAPTURE_COUNTERS.get(phase, 0)
    path = out_dir / f"{phase}_{timestamp}_{seq:03d}.pt"
    torch.save(payload, path)
    return path


def install_capture_shims(
    out_dir: str | Path = "/workspace/captures",
    capture_n: int = 1,
) -> None:
    """Monkey-patch vLLM's DeepseekV4 reference paths to dump tensors.

    Args:
        out_dir: directory where ``.pt`` capture files are written.
        capture_n: max captures to write per phase (swa, compressed, mtp).
            After this count is reached, the shim becomes a no-op for that phase.
    """
    out_dir = Path(out_dir)

    from vllm.model_executor.layers import deepseek_v4_attention as dsv4_mod

    target_cls = dsv4_mod.DeepseekV4MLAAttention

    swa_orig = target_cls._forward_sparse_mla_swa_decode_reference
    compressed_orig = target_cls._forward_sparse_mla_compressed_decode_reference

    def _maybe_capture(phase: str, payload_fn) -> None:
        with _LOCK:
            n = _CAPTURE_COUNTERS.get(phase, 0)
            if n >= capture_n:
                return
            _CAPTURE_COUNTERS[phase] = n + 1
        path = _dump_payload(out_dir, phase, payload_fn())
        # Use eprint so vllm logger doesn't drop it
        import sys
        print(f"[b12x_capture] wrote {phase} -> {path}", file=sys.stderr, flush=True)

    def swa_shim(self, *, q, swa_k_cache, swa_metadata, output):
        # Run the real path first so we capture the EXPECTED output too
        swa_orig(self, q=q, swa_k_cache=swa_k_cache, swa_metadata=swa_metadata, output=output)
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_decodes = swa_metadata.num_decodes
        sub_phase = "mtp" if num_decode_tokens != num_decodes else "single"
        phase = f"swa_{sub_phase}"

        def payload():
            return {
                "phase": phase,
                "q": _safe_clone(q),
                "swa_k_cache": _safe_clone(swa_k_cache),
                "swa_indices": _safe_clone(swa_metadata.decode_swa_indices[:num_decode_tokens]),
                "swa_lens": _safe_clone(swa_metadata.decode_swa_lens[:num_decode_tokens]),
                "seq_lens": _safe_clone(swa_metadata.seq_lens[:num_decodes]),
                "block_table": _safe_clone(swa_metadata.block_table[:num_decodes]),
                "block_size": int(swa_metadata.block_size),
                "num_decodes": int(num_decodes),
                "num_decode_tokens": int(num_decode_tokens),
                "scale": float(self.scale),
                "attn_sink": _safe_clone(self.attn_sink),
                "padded_heads": int(self.padded_heads),
                "num_heads": int(self.num_heads),
                "head_dim": int(self.head_dim),
                "v_head_dim": int(self.v_head_dim),
                "expected_output": _safe_clone(output),
            }

        _maybe_capture(phase, payload)

    def compressed_shim(
        self,
        *,
        q,
        compressed_k_cache,
        swa_k_cache,
        topk_indices,
        topk_lens,
        swa_metadata,
        attn_metadata,
        output,
    ):
        compressed_orig(
            self,
            q=q,
            compressed_k_cache=compressed_k_cache,
            swa_k_cache=swa_k_cache,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            output=output,
        )
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_decodes = swa_metadata.num_decodes
        sub_phase = "mtp" if num_decode_tokens != num_decodes else "single"
        phase = f"compressed{self.compress_ratio}_{sub_phase}"

        def payload():
            return {
                "phase": phase,
                "compress_ratio": int(self.compress_ratio),
                "q": _safe_clone(q),
                "compressed_k_cache": _safe_clone(compressed_k_cache),
                "swa_k_cache": _safe_clone(swa_k_cache),
                "topk_indices": _safe_clone(topk_indices),
                "topk_lens": _safe_clone(topk_lens),
                "swa_indices": _safe_clone(swa_metadata.decode_swa_indices[:num_decode_tokens]),
                "swa_lens": _safe_clone(swa_metadata.decode_swa_lens[:num_decode_tokens]),
                "seq_lens": _safe_clone(swa_metadata.seq_lens[:num_decodes]),
                "block_table": _safe_clone(swa_metadata.block_table[:num_decodes]),
                "block_size": int(swa_metadata.block_size),
                "num_decodes": int(num_decodes),
                "num_decode_tokens": int(num_decode_tokens),
                "scale": float(self.scale),
                "attn_sink": _safe_clone(self.attn_sink),
                "padded_heads": int(self.padded_heads),
                "num_heads": int(self.num_heads),
                "head_dim": int(self.head_dim),
                "v_head_dim": int(self.v_head_dim),
                "expected_output": _safe_clone(output),
            }

        _maybe_capture(phase, payload)

    target_cls._forward_sparse_mla_swa_decode_reference = swa_shim
    target_cls._forward_sparse_mla_compressed_decode_reference = compressed_shim

    # Force the reference path on, otherwise FlashMLA path runs and we capture nothing
    os.environ.setdefault("VLLM_TRITON_MLA_SPARSE", "1")

    import sys
    print(
        f"[b12x_capture] shims installed (out_dir={out_dir}, capture_n={capture_n}); "
        f"VLLM_TRITON_MLA_SPARSE={os.environ.get('VLLM_TRITON_MLA_SPARSE')}",
        file=sys.stderr, flush=True,
    )
