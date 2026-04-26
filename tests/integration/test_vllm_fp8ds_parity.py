"""Parity oracle: b12x decode vs vLLM's Triton sparse MLA reference output.

Loads ``.pt`` capture files written by ``b12x.integration.vllm_capture`` and
runs the b12x decode kernel on the same inputs, asserting the output matches
the captured ``expected_output`` from the Triton path.

Captures are produced by running ``scripts/capture_dsv4_decode_tensors.py``
inside a vLLM container.  Fixture path can be overridden via the
``B12X_PARITY_FIXTURE_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from b12x.integration.mla import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspaceContract,
    MLASparseDecodeMetadata,
    sparse_mla_decode_forward,
)
from b12x.integration.vllm_kv_converter import convert_fp8ds_to_b12x_gathered

from tests.helpers import require_sm120


_DEFAULT_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture_dir() -> Path:
    env = os.environ.get("B12X_PARITY_FIXTURE_DIR")
    return Path(env) if env else _DEFAULT_FIXTURE_DIR


def _list_swa_fixtures() -> list[Path]:
    d = _fixture_dir()
    return sorted(d.glob("swa_*.pt")) if d.exists() else []


def _gather_swa_page_table(
    swa_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_decodes: int,
    num_decode_tokens: int,
) -> torch.Tensor:
    """Translate vLLM's per-block SWA gather indices into global slot ids.

    swa_indices: (num_decode_tokens, max_swa_len) int32 — gather indices into
        the per-request SWA window (relative offsets, not global).  vLLM uses
        these together with a per-request block_table to address tokens.
    """
    if swa_indices.shape[0] != num_decode_tokens:
        raise ValueError(
            f"swa_indices rows {swa_indices.shape[0]} != num_decode_tokens {num_decode_tokens}"
        )
    return swa_indices.to(torch.int32)


@pytest.mark.parametrize("fixture", _list_swa_fixtures(),
                         ids=lambda p: p.name)
def test_swa_decode_parity(fixture: Path) -> None:
    """b12x SWA-only decode matches vLLM's Triton reference on captured tensors."""
    if not fixture.exists():
        pytest.skip(f"no fixture at {fixture} — run scripts/capture_dsv4_decode_tensors.py")

    device = require_sm120()
    payload = torch.load(fixture, map_location=device, weights_only=False)

    q = payload["q"].to(device)                          # (N, 1, padded_heads, head_dim) bf16
    swa_k_cache = payload["swa_k_cache"].to(device)
    expected = payload["expected_output"].to(device)
    num_decode_tokens = payload["num_decode_tokens"]
    num_decodes = payload["num_decodes"]
    num_heads = payload["num_heads"]
    v_head_dim = payload["v_head_dim"]
    head_dim = payload["head_dim"]
    block_size = payload["block_size"]
    scale = payload["scale"]
    swa_indices = payload["swa_indices"].to(device)      # (N, max_swa_len) int32
    swa_lens = payload["swa_lens"].to(device).to(torch.int32)

    # 1. Slice q from padded_heads -> num_heads, drop the dummy '1' decode dim
    q_b12x = q.view(num_decode_tokens, q.shape[-2], q.shape[-1])[:, :num_heads, :].contiguous()
    assert q_b12x.shape == (num_decode_tokens, num_heads, head_dim), q_b12x.shape

    # 2. Convert KV cache (gather only the rows we need)
    page_table = swa_indices.to(torch.int32)
    b12x_cache, new_page_table = convert_fp8ds_to_b12x_gathered(
        swa_k_cache, page_table, block_size=block_size,
    )

    # 3. Build b12x workspace sized for this single call
    topk = int(new_page_table.shape[-1])
    cache_seqlens_int32 = swa_lens.clone().to(torch.int32)
    nsa_cache_seqlens_int32 = swa_lens.clone().to(torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=new_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        max_seq_len_k=int(swa_lens.max().item()) if swa_lens.numel() else 0,
    )

    caps = B12XAttentionArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=head_dim,
        max_v_head_dim=v_head_dim,
        topk=max(topk, 64),
        max_page_table_width=max(topk, 64),
        extend_max_total_q=num_decode_tokens,
        extend_max_batch=num_decodes,
        extend_max_kv_rows=int(b12x_cache.shape[0]),
        paged_max_q_rows=num_decode_tokens,
        paged_max_batch=num_decodes,
        page_size=block_size,
        padded_heads=num_heads,
    )
    arena = B12XAttentionArena.allocate(caps)
    workspace = arena.make_workspace(B12XAttentionWorkspaceContract(
        mode="decode",
        max_total_q=num_decode_tokens,
        max_batch=num_decodes,
        max_paged_q_rows=num_decode_tokens,
        max_kv_rows=int(b12x_cache.shape[0]),
        v_head_dim=v_head_dim,
        indexer_num_q_heads=num_heads,
        max_page_table_width=max(topk, 64),
    ))

    # 4. Run b12x decode
    output_b12x = sparse_mla_decode_forward(
        q_all=q_b12x,
        kv_cache=b12x_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=scale,
        v_head_dim=v_head_dim,
    )
    assert output_b12x.shape == (num_decode_tokens, num_heads, v_head_dim)

    # 5. Compare to expected (slice padded_heads -> num_heads, drop dummy dim)
    expected_sliced = expected.view(num_decode_tokens, expected.shape[-2], expected.shape[-1])
    expected_sliced = expected_sliced[:, :num_heads, :v_head_dim]

    diff = (output_b12x.float() - expected_sliced.float()).abs()
    max_abs = diff.max().item()
    rmse = (diff ** 2).mean().sqrt().item()
    cos = torch.nn.functional.cosine_similarity(
        output_b12x.float().reshape(-1), expected_sliced.float().reshape(-1), dim=0,
    ).item()
    print(f"\n[{fixture.name}] max_abs={max_abs:.4f} rmse={rmse:.4f} cos={cos:.6f}")

    # Tolerances: FP8 round-trip + per-128 vs per-64 scale combine + (no sink yet)
    # If sink is non-trivial, this WILL fail loudly — that's the signal.
    assert cos >= 0.95, f"{fixture.name}: cosine similarity {cos:.4f} too low (sink?)"
    assert max_abs <= 0.20, f"{fixture.name}: max_abs={max_abs:.4f}"


def test_at_least_one_swa_fixture_exists() -> None:
    fixtures = _list_swa_fixtures()
    if not fixtures:
        pytest.skip(
            f"no SWA capture fixtures in {_fixture_dir()} — "
            "run scripts/capture_dsv4_decode_tensors.py to populate"
        )
    assert len(fixtures) >= 1
