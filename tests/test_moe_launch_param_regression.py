"""Regression tests for Parameter-backed tp_moe launch arguments."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"Requires SM120, got sm_{major}{minor}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _run_parameter_launch_case(case: str) -> subprocess.CompletedProcess[str]:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    spec = _make_spec()
    script = textwrap.dedent(
        f"""
        import pathlib
        import sys

        import torch
        from torch.nn import Parameter

        sys.path.insert(0, {str(repo_root)!r})

        from benchmarks.benchmark_moe import ModelSpec, load_expert_weights, make_routed_inputs
        from b12x.integration.tp_moe import (
            allocate_tp_moe_workspace,
            b12x_moe_fp4,
            clear_tp_moe_caches,
        )

        case = {case!r}
        clear_tp_moe_caches()

        device = torch.device("cuda")
        spec = ModelSpec(
            hidden_size={spec.hidden_size},
            intermediate_size={spec.intermediate_size},
            num_experts={spec.num_experts},
            top_k={spec.top_k},
            tp_size={spec.tp_size},
            tp_rank={spec.tp_rank},
        )
        weights = load_expert_weights(pathlib.Path({str(MODEL_PATH)!r}), spec)
        x, topk_ids, topk_weights = make_routed_inputs(spec, 8, seed=123, device=device)

        a1_gscale = weights.w13_input_scale_per_expert.clone()
        a2_gscale = weights.w2_input_scale_per_expert.clone()
        w1_alphas = weights.g1_alphas_per_expert.clone()
        w2_alphas = weights.g2_alphas_per_expert.clone()

        if case in ("scales", "all"):
            a1_gscale = Parameter(a1_gscale, requires_grad=False)
            a2_gscale = Parameter(a2_gscale, requires_grad=False)
        if case in ("alphas", "all"):
            w1_alphas = Parameter(w1_alphas, requires_grad=False)
            w2_alphas = Parameter(w2_alphas, requires_grad=False)

        out = torch.empty_like(x)
        workspace = allocate_tp_moe_workspace(
            x,
            a1_gscale,
            weights.w13_weight,
            a2_gscale,
            weights.w2_weight,
            topk_ids,
            implementation="static",
            input_scales_static=True,
        )
        print(f"case={{case}} start", flush=True)
        b12x_moe_fp4(
            x,
            a1_gscale,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            w1_alphas,
            a2_gscale,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            w2_alphas,
            topk_weights,
            topk_ids,
            implementation="static",
            workspace=workspace,
            output=out,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
        )
        torch.cuda.synchronize()
        print(f"case={{case}} ok", flush=True)
        """
    )
    env = os.environ.copy()
    env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("case", ["alphas", "scales"])
def test_b12x_moe_accepts_parameter_backed_launch_args(case: str) -> None:
    """The static path should not segfault on Parameter-backed scale tensors."""
    _skip_if_unavailable()

    result = _run_parameter_launch_case(case)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"child process failed for case={case} with rc={result.returncode}\n{output}"
    )
