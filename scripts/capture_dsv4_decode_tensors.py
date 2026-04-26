"""Run vLLM with the DSV4-Flash model and capture sparse-MLA decode tensors.

Designed to be run inside the b12x dev container (which has vLLM, the model
weights mounted, and the b12x package installed).  It installs the capture
shims from ``b12x.integration.vllm_capture`` and then drives vLLM with a few
short prompts to trigger SWA-only and (optionally) MTP decode paths.

Usage:
    docker exec -w /workspace -e VLLM_USE_V1=1 vllm-b12x-dev \
        python3 -m scripts.capture_dsv4_decode_tensors \
            --out-dir /workspace/captures \
            --capture-n 1 \
            --num-spec 0
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

# Must set spawn BEFORE any CUDA-touching import so vLLM's engine subprocess
# can re-initialize CUDA cleanly.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash",
                        help="HF model id (must already be cached locally).")
    parser.add_argument("--out-dir", default="/workspace/captures",
                        help="Where to write captured .pt files.")
    parser.add_argument("--capture-n", type=int, default=1,
                        help="Captures per phase before the shim becomes a no-op.")
    parser.add_argument("--num-spec", type=int, default=0,
                        help="Number of speculative tokens (0 = single-token decode only).")
    parser.add_argument("--max-tokens", type=int, default=4,
                        help="Generation length per prompt.")
    parser.add_argument("--gpu-mem", type=float, default=0.85,
                        help="gpu_memory_utilization for vllm.LLM.")
    args = parser.parse_args()

    # Force the Triton sparse MLA reference path so the shims actually fire
    os.environ.setdefault("VLLM_TRITON_MLA_SPARSE", "1")
    # Use V0 in-process engine so our monkey-patches actually run in the worker
    # (V1 spawns a separate engine subprocess; patches in the parent wouldn't apply).
    os.environ["VLLM_USE_V1"] = "0"

    from b12x.integration.vllm_capture import install_capture_shims

    install_capture_shims(out_dir=args.out_dir, capture_n=args.capture_n)

    print(f"[capture] importing vLLM and loading {args.model} ...", file=sys.stderr, flush=True)
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,  # avoid CUDA graph capture during the shimmed run
        max_model_len=2048,
    )
    if args.num_spec > 0:
        llm_kwargs["speculative_config"] = {"num_speculative_tokens": args.num_spec}

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = [
        "The capital of France is",
        "Quantum computing relies on",
    ]
    print(f"[capture] generating {args.max_tokens} tokens for {len(prompts)} prompts ...",
          file=sys.stderr, flush=True)
    outputs = llm.generate(prompts, sampling)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip().replace("\n", " ")
        print(f"[capture] prompt {i}: '{prompts[i]}' -> '{text}'", file=sys.stderr, flush=True)

    captures = sorted(Path(args.out_dir).glob("*.pt"))
    print(f"[capture] {len(captures)} capture file(s) written to {args.out_dir}",
          file=sys.stderr, flush=True)
    for p in captures:
        print(f"  - {p.name}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
