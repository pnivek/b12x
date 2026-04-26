"""Probe DSV4 attn_sink magnitude from a captured fixture.

Decides Phase 3 of the b12x-vllm integration plan:
  - if max sink contribution to softmax < ~1% of dominant logit, we can ship v1
    without sink support (Option 3)
  - else, we must extend the b12x kernel with a sink parameter (Option 1)

Usage:
    python3 -m scripts.probe_dsv4_attn_sink path/to/capture.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path,
                        help=".pt file written by b12x.integration.vllm_capture")
    args = parser.parse_args()

    payload = torch.load(args.fixture, map_location="cpu", weights_only=False)
    if "attn_sink" not in payload:
        print(f"ERROR: no attn_sink in {args.fixture}", file=sys.stderr)
        return 1

    sink = payload["attn_sink"]
    if sink is None:
        print("attn_sink is None (model has no sink) -> Option 3 (skip) is safe.")
        return 0

    sink_f = sink.float()
    print(f"attn_sink shape: {tuple(sink.shape)}, dtype: {sink.dtype}")
    print(f"  abs.max  = {sink_f.abs().max().item():.4f}")
    print(f"  abs.mean = {sink_f.abs().mean().item():.4f}")
    print(f"  count(-inf padding) = {torch.isinf(sink_f).sum().item()}")

    finite = sink_f[~torch.isinf(sink_f)]
    if finite.numel() == 0:
        print("All sink values are -inf padding -> sink contribution is 0.")
        print("Decision: Option 3 (skip sink) is safe.")
        return 0

    print(f"Finite sink values ({finite.numel()}):")
    print(f"  min  = {finite.min().item():.4f}")
    print(f"  max  = {finite.max().item():.4f}")
    print(f"  mean = {finite.mean().item():.4f}")
    print(f"  std  = {finite.std().item():.4f}")

    # Heuristic: if sink magnitudes are << typical softmax row max (logits ~10-50
    # after sm_scale), the sink contribution to denominator is exp(sink-max)
    # which is tiny.  Compare to a ballpark logit max from the captured q.
    if "expected_output" in payload:
        out = payload["expected_output"].float().reshape(-1)
        print(f"  ref output magnitude: max={out.abs().max().item():.4f}, "
              f"mean={out.abs().mean().item():.4f}")

    if finite.abs().max().item() < 5.0:
        print("\nDecision: sink magnitudes look modest (<5). "
              "Option 3 (skip) likely viable; verify via parity test tolerance.")
    else:
        print("\nDecision: sink magnitudes large (>=5). "
              "Option 1 (extend b12x kernel with sink param) likely required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
