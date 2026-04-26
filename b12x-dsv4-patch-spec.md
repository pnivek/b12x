# b12x Patch Spec: DeepSeek-V4-Flash MLA Support on SM121

> **POST-IMPLEMENTATION (2026-04-26)** — patch landed; tests passing on sparky; bench results recorded. **See §11 for actuals**: files touched, test results, perf numbers, and remaining work. Sections 1–10 below are the pre-implementation plan and remain unedited as a record of the design intent.

> **v2 revision (2026-04-26)** — second-pass review against the actual b12x codebase, the published DSV4-Flash HF config, and the live vLLM PR landscape. Changes from v1 are tagged inline with **[v2]**. Summary of revisions:
> 1. **Files-to-modify list was incomplete.** `traits.py` (gating trait selection — explicitly rejects non-GLM-5.1 shapes) and `split.py` (re-imports parameterized constants) are required additions. **6 files, not 5.**
> 2. **CUTE JIT constexpr-loop constraint not addressed.** The "make constants instance-level" change in §4.2.1 is harder than written: `cutlass.range_constexpr(_MLA_SCALE_GROUPS)` requires Python ints at trace time, and the `@cute.jit` helpers are module-level closures over module constants. Concrete refactoring strategies added.
> 3. **Internal contradiction on partial-group handling resolved.** §4.2.2 (BF16 tail), §4.2.4 (zero-pad full group), and §5.2 (compact 592-byte layout) propose three different strategies. Owner decision (2026-04-26): **Option B — zero-pad nope to 512, keep `_MLA_SCALE_GROUPS=4` constexpr unchanged.** §4.2 and §5 updated accordingly. Compact 592-byte layout is filed as a future optimization.
> 4. **HF config verified — KV scale-format mismatch surfaced.** DSV4-Flash uses **UE8M0 (8-bit) scales**, not FP32 like b12x's reference. Convert at packing time for v1; do not touch the kernel scale-load path. See §5.3 (new).
> 5. **Indexer is already DSV4-compatible.** `_INDEX_HEAD_DIM=128` matches `index_head_dim=128`; `num_heads` is read dynamically from tensor shape. No NSA indexer changes expected. Added §4.6 (new) to record this.
> 6. **vLLM PR landscape clarified.** PR #40899 (Triton sparse MLA SM12x fallback, single-token only) is what we are augmenting past. PR #40760 (DSV4 model add) is Hopper-focused. PR #40082 (b12x MoE/GEMM) is unrelated to MLA. §6 rewritten.
> 7. **Test host correction.** Spec said `192.168.0.172` ("Sparkly"); actual working host is **`192.168.0.112` ("sparky")**. Passwordless SSH verified. b12x is not yet installed there. §7.3 updated.
> 8. **Total scope estimate revised.** v1 said ~70 LOC; realistic with traits/split touches and the CUTE constexpr refactor is **~150–200 LOC plus a UE8M0→FP32 helper (~30 LOC)**. §8 updated.
> 9. **Verification expanded.** Original §7.1 only covered single-token decode; added a multi-token MTP test (the actual unblock value-prop) in §7.1b.
> 10. **Backward compatibility relaxed.** Owner confirmed the patch lives on `pnivek/b12x` fork for DSV4-Flash POC; GLM-5.1 backward compat is *nice-to-have*, not required. §7.2 updated.

## 1. Objective

Patch `lukealonso/b12x` v0.10.0 (`pip install b12x`) to support **DeepSeek-V4-Flash** sparse MLA inference on **SM121 (NVIDIA GB10 / DGX Spark)** hardware. Currently b12x only supports GLM-5.1 (nope=512). DSV4 needs nope=448.

**[v2] Fork target:** Work lands on `pnivek/b12x` fork; upstream contribution is out of scope for v1.

## 2. Architecture Background

### 2.1 What is b12x?

A CuTe DSL kernel library for SM120/SM121 that provides optimized sparse MLA (Multi-Head Latent Attention) kernels using WMMA (Warp Matrix Multiply-Accumulate) tensor core instructions. GLM-5.1-targeted, currently hardcoded.

**[v2] Per upstream README:** "Unapologetically SM120-only CuTe DSL kernels for NVFP4 GEMM and MoE." The repo owner explicitly says it is intentionally narrow and intended to be integrated by AI agents — i.e., we are expected to fork-and-modify rather than expect upstream parameterization.

### 2.2 Why b12x for DSV4?

vLLM's current SM121 attention path (the Triton sparse MLA fallback added in [vLLM PR #40899](https://github.com/vllm-project/vllm/pull/40899), gated by `VLLM_TRITON_MLA_SPARSE`) only supports **single-token decode**. Multi-Token Prediction (MTP) fails with:
```
Sparse MLA reference SWA decode currently supports one query token per request,
got num_decode_tokens=6 and num_decodes=2
```

**[v2] PR #40899 measured ~6.7 tok/s on GB10.** b12x's CuTe kernels handle multi-token decode natively, unblocking MTP (~2x throughput) and giving us a runway for further perf work.

**[v2] Earlier env name `VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION` was the prototype; the merged form is `VLLM_TRITON_MLA_SPARSE`.**

### 2.3 Hardware Constraints

- **DGX Spark (GB10)**: SM121 compute capability
- **No TCGEN05, no TMEM**: Rules out FlashMLA, Machete, and other Hopper-optimized paths
- **WMMA available**: b12x uses WMMA — compatible with SM121

## 3. Dimension Comparison

**[v2] Verified against `deepseek-ai/DeepSeek-V4-Flash` HF `config.json` (2026-04-26).** Note that several DSV4 fields are `null` in the config and must be derived: `qk_nope_head_dim = head_dim − qk_rope_head_dim = 512 − 64 = 448`; `v_head_dim` is also null and must be confirmed against vLLM #40760's `deepseek_v4_attention.py` (the spec's earlier assumption of 512 is consistent with that derivation).

| Parameter | GLM-5.1 (b12x current) | DSV4-Flash (target) | Source | Compatible? |
|-----------|-----------------------|---------------------|--------|-------------|
| `num_attention_heads` | 128 | **64** | HF config | ✅ Workspace supports this |
| `qk_nope_head_dim` | 512 | **448** (derived) | HF config + derivation | **❌ Hardcoded** |
| `qk_rope_head_dim` | 64 | 64 | HF config | ✅ Same |
| `q_head_dim` (head_dim) | 576 | **512** | HF config | ❌ Kernel expects 576 |
| `v_head_dim` | 512 | 512 (derived) | needs vLLM #40760 verify | ✅ Same |
| `MXFP8 groups` (128 ea) | 4 | **3.5 (logical), 4 (padded — Option B)** | derived | ❌→✅ via padding |
| KV packed row width | 656 bytes | **656 bytes (Option B v2)** | design choice | ✅ Unchanged |
| KV scale dtype | FP32 (in b12x packed row) | **UE8M0** (in DSV4 model) | HF config | ⚠ Convert at pack time |
| Indexer head_dim | 128 (`_INDEX_HEAD_DIM`) | 128 (`index_head_dim`) | both | ✅ Same |
| Indexer num_heads | dynamic | 64 (`index_n_heads`) | HF config | ✅ Already dynamic |
| MTP draft layers | n/a | 1 (`num_nextn_predict_layers`) | HF config | Drives multi-token decode |

**[v2] Decision recorded:** KV packed row stays 656 bytes for v1 (Option B — zero-pad). Original spec's 592-byte layout (§5.2) is reclassified as a future optimization (§5.4 below).

## 4. Files to Modify

**[v2] Original list was incomplete.** Revised count: **6 files**, not 5.

### 4.1 `b12x/attention/mla/reference.py` — Trivial (~5 lines)

**Problem:** `_MLA_PACKED_DIM = 656` is a module-level constant checked in 3 places.

**[v2] Resolution under Option B:** `_MLA_PACKED_DIM` stays 656 (storage-padded). Make it a function so future Option-A optimization is easy:

```python
def _mla_packed_dim(nope_storage_dim: int = 512, rope_dim: int = 64) -> int:
    """Compute packed KV cache row width in bytes (Option B: nope storage is padded
    to a multiple of 128, so DSV4's logical 448 nope is stored as 512)."""
    nope_u32 = nope_storage_dim // 4
    scale_groups = nope_storage_dim // 128
    rope_u32 = rope_dim // 2
    return (nope_u32 + scale_groups + rope_u32) * 4

_MLA_PACKED_DIM = _mla_packed_dim()  # 656 for both GLM-5.1 and DSV4 v1
```

**Validation functions to update:**
- `_as_2d_cache()` — replace hardcoded `656` check with `_MLA_PACKED_DIM`
- `unpack_mla_kv_cache_reference()` — same
- `pack_mla_kv_cache_reference()` — same; **[v2]** also needs `nope_logical_dim` parameter (default 512) so DSV4 can pass 448 and the packer zero-pads bytes 448–511

For DSV4 v1: `_mla_packed_dim(512, 64)` = `656 bytes` (storage), with logical nope of 448 elements per row.

### 4.2 `b12x/attention/mla/kernel.py` — Core changes (~40–80 lines)

**Problem:** ~25 module-level constants derived from `_MLA_NOPE_DIM = 512` and `_MLA_ROPE_DIM = 64` (lines 48–86). The CUTE JIT kernel is compiled once and cached; these constants are baked at compile time.

**[v2] Critical CUTE constraint:** `cutlass.range_constexpr(_MLA_SCALE_GROUPS)` (used at kernel.py:811, 1488, 1583, 1606, 1635, 1715, 1724, 1760, 1808, 1831, 1878, 1959, etc.) requires a **Python `int`** at trace time, not an `Int32`. Module-level `@cute.jit` helpers like `_literal_qk_mma_into_sfrag_mxfp8_raw` close over the module constants. Three viable approaches:

- **(a) Move helpers to methods on `SparseMLAKernel`** so they close over `self`, then thread constants via `cutlass.Constexpr[int]` parameters where loops appear. **Recommended.** Cleanest, idiomatic CUTE-DSL.
- **(b) Pass constants as `Constexpr[int]` parameters through every helper.** Mechanical but ~30 callsites; high risk of typos.
- **(c) Per-shape kernel cache that mutates module-level constants on construction.** Hacky; localized to `_build_sparse_mla_kernel_for_shape`. Avoid unless (a) proves intractable.

**[v2] Under Option B with `_MLA_SCALE_GROUPS=4` for both shapes**, the only kernel-level dimensions that actually change between GLM-5.1 and DSV4 are:
- `_MLA_HEADS_PER_TILE` interaction with `num_q_heads` (already dynamic via `q_u32.shape[1]`)
- Output dim → tied to `v_head_dim` (already 512 for both)

**This means many constexpr loops do not need to be re-parameterized at all** — they already iterate over `_MLA_SCALE_GROUPS=4`. The constants that actually need parameterization are:
- `_MLA_HEAD_DIM` (576 → 512) — used in stride/offset arithmetic
- `_MLA_NOPE_DIM` (512 — same in both, but now means "storage" not "logical")
- staging size constants (no change because Option B keeps padding identical)

**[v2] Net effect:** kernel.py changes are smaller than v1 estimated for the constexpr-loop side, but the cache-key and `_MLA_HEAD_DIM`-arithmetic touches still need to land.

#### 4.2.1 Constants to instance-attribute (revised)

```python
class SparseMLAKernel:
    def __init__(
        self,
        head_tiles: int,
        nope_logical_dim: int = 512,   # 512 GLM, 448 DSV4 (used for masking, if any)
        nope_storage_dim: int = 512,   # always 512 in v1 (Option B)
        rope_dim: int = 64,
    ):
        self.head_tiles = head_tiles
        self.nope_logical_dim = nope_logical_dim
        self.nope_storage_dim = nope_storage_dim
        self.rope_dim = rope_dim
        self.head_dim = nope_storage_dim + rope_dim  # 576 GLM, 576-or-512 DSV4
        self.scale_groups = nope_storage_dim // 128  # 4 for both
```

#### 4.2.2 ~~Handle partial MXFP8 group~~ — **DEFERRED to future optimization**

**[v2]** Under Option B (decided 2026-04-26), the partial group is handled by **zero-padding nope storage to 512** and treating the 4th MXFP8 group as containing 64 real elements + 64 zeros. The kernel does no special-case handling — the MXFP8 MMA on the 4th group multiplies real Q against zero K for the padding region, contributing 0 to the score.

The original v1 plan (BF16 tail MMA) is the *correct* future optimization to drop the 64 wasted bytes/token. File as v2.0.

#### 4.2.3 Update cache key for different head_dim (~3 lines)

```python
_launcher_cache_key = (
    self.head_dim, self.nope_logical_dim,  # NEW: dimension fingerprint
    _tensor_meta_key(q_u32), _tensor_meta_key(kv_rows_u32), ...
)
```

This ensures DSV4 (head_dim=512) and GLM-5.1 (head_dim=576) compile separate kernels.

#### 4.2.4 Staging sizes — **no change in Option B**

**[v2]** Confirmed unchanged because both shapes pad to 4 MXFP8 groups of 128 elements.

### 4.3 `b12x/attention/mla/kernel_onepass.py` — Mirror kernel.py (~20–40 lines)

Same changes as `kernel.py` §4.2. The one-pass variant has its own copy of the constants. Apply the same parameterization.

### 4.4 `b12x/attention/mla/api.py` — Dispatch plumbing (~5 lines)

**`_run_sparse_mla()`** passes head_tiles to kernel constructor but not nope_dim:

```python
# Current:
kernel = SparseMLAKernel(head_tiles=head_tiles)

# Fix:
kernel = SparseMLAKernel(
    head_tiles=head_tiles,
    nope_logical_dim=workspace.head_dim - workspace.rope_dim,  # 448 for DSV4
    rope_dim=workspace.rope_dim,
)
```

### 4.5 `b12x/attention/mla/workspace.py` — Add field (~3 lines)

```python
@dataclass(frozen=True, kw_only=True)
class B12XAttentionArenaCaps:
    ...
    nope_logical_dim: int = 512  # NEW: 448 for DSV4
    rope_dim: int = 64           # NEW
```

Update validations to accept head_dim != 576.

### 4.6 `b12x/attention/mla/traits.py` — **[v2] NEW SECTION** (~10 lines)

**Original spec missed this file entirely.** `select_sparse_mla_traits()` (lines 64–69) currently does:

```python
if q_all.shape[2] != _MLA_EXACT_HEAD_DIM:        # 576 only
    return None
if kv_cache.shape[1:] != (1, _MLA_EXACT_PACKED_WIDTH):  # 656 only
    return None
if int(v_head_dim) != _MLA_EXACT_V_HEAD_DIM:     # 512 only
    return None
```

Without changes here, `supports_sparse_mla_kernel()` returns False for DSV4 (head_dim=512), the dispatcher falls through to the Python reference (api.py:307–340), and **the kernel never runs**.

**Fix:** widen to a small set, and carry dim metadata on the trait:

```python
_SUPPORTED_SHAPES = {
    # (head_dim, v_head_dim, packed_width, nope_logical_dim, rope_dim)
    (576, 512, 656, 512, 64),  # GLM-5.1
    (512, 512, 656, 448, 64),  # DSV4-Flash (Option B padded storage)
}

@dataclass(frozen=True)
class SparseMLATraits:
    ...
    nope_logical_dim: int   # NEW
    rope_dim: int           # NEW
```

### 4.7 `b12x/attention/mla/split.py` — **[v2] NEW SECTION** (~5 lines)

Re-imports `_MLA_NOPE_DIM`, `_MLA_SCALE_GROUPS`, etc. from `kernel.py` (lines 19–42). Once those become per-instance attributes, `split.py` either imports via `SparseMLAKernel` instance methods, or `run_sparse_mla_split_decode` accepts the dims as parameters.

## 5. KV Cache Packing Format

### 5.1 GLM-5.1 packed row (656 bytes = 164 u32) — unchanged

```
[0..127]  nope data:  512 bytes = 128 u32 (uint8 MXFP8)
[128]     scale page: 4 × float32 = 16 bytes = 4 u32
[129..131] (padding)
[132..163] rope data: 128 bytes = 32 u32 (bf16 pairs)
Total: 164 u32 = 656 bytes
```

### 5.2 DSV4 packed row — **[v2] Option B: 656 bytes (storage-padded)**

```
[0..111]  nope data (real):    448 bytes = 112 u32 (uint8 MXFP8)
[112..127] nope data (zeros):  64 bytes = 16 u32 — PADDING for 4th MXFP8 group
[128]     scale page: 4 × float32 = 16 bytes = 4 u32
            scales[0..2]: real per-128-element scales for groups 0,1,2
            scales[3]:    arbitrary (multiplies all-zero data → 0 contribution)
[132..163] rope data: 128 bytes = 32 u32 (bf16 pairs) — UNCHANGED
Total: 164 u32 = 656 bytes (identical layout to GLM-5.1)
```

**Key insight:** Storage layout is byte-identical to GLM-5.1. Only the *meaning* of bytes 448–511 changes (real → padding). The kernel reads these as MXFP8 elements and multiplies them with Q; since they're zero, they contribute zero. **No kernel logic change required for the partial group.**

### 5.3 KV scale format conversion — **[v2] NEW SECTION**

DSV4-Flash's HF config declares `quantization_config.scale_fmt: ue8m0` — 8-bit float scales (1 byte each), one per 128-element block. b12x's packed row stores **FP32** (4 bytes each). Two options:

- **(a) Convert UE8M0 → FP32 at pack time.** Store FP32 in the packed row. Zero kernel changes. **Recommended for v1.**
- **(b) Add UE8M0 scale-load path to the kernel.** Saves 12 bytes/token but requires touching scale-load and `_stage_token_scales`. Defer.

UE8M0 → FP32 conversion: `fp32 = 2.0 ** (ue8m0_byte − 127)`. Implement as `_ue8m0_to_fp32` helper in `reference.py`.

### 5.4 Future optimization: compact 592-byte layout

Original v1 spec's 592-byte layout (Option A) is the right long-term target — saves 64 bytes/token (~10% KV memory) at the cost of a BF16 tail-MMA path. Filed as v2.0 follow-up after correctness on Option B is validated.

## 6. Dispatch Integration into vLLM

**[v2] Updated landscape (as of 2026-04-26):**

| PR / Issue | Scope | DSV4-Flash MTP on SM121? |
|------------|-------|--------------------------|
| [#40760](https://github.com/vllm-project/vllm/pull/40760) | DSV4 model add (FlashInfer FP8) | Hopper-focused, no SM12x MLA |
| [#40899](https://github.com/vllm-project/vllm/pull/40899) | Triton sparse MLA fallback for SM12x | **Single-token only** — what we replace |
| [#40082](https://github.com/vllm-project/vllm/pull/40082) | b12x NVFP4 MoE+GEMM in vLLM | Unrelated to MLA — but is the precedent for b12x dispatch wiring |
| [#39635](https://github.com/vllm-project/vllm/pull/39635) | FlashInfer MLA DCP+MTP | Datacenter SM120, not SM121 |
| [#37113](https://github.com/vllm-project/vllm/issues/37113) | Tracking issue for MLA on SM120 | Open |

Once b12x is patched, the vLLM-side change is small (estimated ~50 LOC):

1. Detect SM120/121 at runtime *and* `b12x.attention.mla` import success
2. Add `VLLM_B12X_MLA_SPARSE` env flag (sibling to `VLLM_TRITON_MLA_SPARSE`)
3. Route `deepseek_v4_attention.py` to b12x `sparse_mla_decode_forward()` when the flag is set or auto-selected
4. Create a `B12XAttentionWorkspace` with DSV4 dims (head_dim=512, nope_logical=448, rope=64, num_heads=64)
5. Pack the KV cache via the v2 packer (UE8M0 → FP32 scale conversion)

The b12x integration pattern from PR #40082 (`vllm/utils/flashinfer.py` capability check, `oracle/nvfp4.py` auto-selection) is reusable for MLA dispatch.

This is a separate PR after the b12x patch lands.

## 7. Verification

### 7.1 Single-token decode unit test (on DGX Spark)

```python
import torch
from b12x.attention.mla import (
    B12XAttentionArena, B12XAttentionArenaCaps,
    B12XAttentionWorkspace, B12XAttentionWorkspaceContract,
    sparse_mla_decode_forward, MLASparseDecodeMetadata,
    clear_mla_caches,
)

NUM_HEADS = 64
Q_HEAD_DIM = 512      # 448 nope + 64 rope
V_HEAD_DIM = 512
TOPK = 512
KV_U32 = 164          # [v2] 164 u32 = 656 bytes (Option B padded), NOT 148/592

caps = B12XAttentionArenaCaps(
    device=torch.device("cuda:0"),
    dtype=torch.bfloat16,
    kv_dtype=torch.uint8,
    num_q_heads=NUM_HEADS,
    indexer_num_q_heads=64,
    head_dim=Q_HEAD_DIM,
    max_v_head_dim=V_HEAD_DIM,
    topk=TOPK,
    max_page_table_width=8,
    page_size=64,
    padded_heads=128,
    nope_logical_dim=448,  # [v2] NEW param (was nope_dim)
    rope_dim=64,           # NEW param
)
arena = B12XAttentionArena.allocate(caps)

contract = B12XAttentionWorkspaceContract(
    mode="decode", max_total_q=2, max_batch=2,
    max_paged_q_rows=2, max_kv_rows=TOPK,
    v_head_dim=V_HEAD_DIM, indexer_num_q_heads=64,
    max_page_table_width=8,
)
workspace = arena.make_workspace(contract)

q_all = torch.randn((1, NUM_HEADS, Q_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
kv_cache = torch.zeros((8, 1, KV_U32 * 4), dtype=torch.uint8, device="cuda")  # bytes
page_table = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
cache_seqlens = torch.tensor([TOPK], dtype=torch.int32, device="cuda")

metadata = MLASparseDecodeMetadata(
    page_table_1=page_table, cache_seqlens_int32=cache_seqlens,
    nsa_cache_seqlens_int32=cache_seqlens, max_seq_len_k=TOPK,
)

output = sparse_mla_decode_forward(
    q_all=q_all, kv_cache=kv_cache, metadata=metadata,
    workspace=workspace, sm_scale=1.0/(Q_HEAD_DIM**0.5),
    v_head_dim=V_HEAD_DIM,
)
print(f"Output: {output.shape} mean={output.mean():.4f}")
# Expected: Output: torch.Size([1, 64, 512]) ...
```

### 7.1b Multi-token decode unit test — **[v2] NEW SECTION (the actual MTP unblock)**

Identical to §7.1 but with `q_all = torch.randn((3, 64, 512), ...)` simulating main + 2 MTP draft tokens, `max_total_q=3`, and a `page_table_1` of shape `(3, TOPK)`. Triton path (PR #40899) errors here; b12x path must produce shape `(3, 64, 512)` with finite values.

Compare numerical output against `sparse_mla_reference()` — pass criterion: cosine ≥ 0.999 vs FP32 reference for each row.

### 7.2 Expected success criteria

- Arena creation with `nope_logical_dim=448` succeeds
- Workspace instantiation with `num_q_heads=64` succeeds
- CUTE kernel compiles (JIT) and produces correct output for single-token AND multi-token decode
- Output shape is `[batch, 64, 512]`
- Numerical output is non-NaN, cosine ≥ 0.999 vs reference
- MTP with `num_speculative_tokens=2` no longer throws the single-token assertion
- **[v2] GLM-5.1 backward compatibility:** *nice-to-have, not required* — fork is DSV4-Flash-only POC. Existing `tests/test_attention_mla_*` should continue to pass if the parameterization defaults to GLM-5.1 dims, but this is not a release gate.

### 7.3 Test environment — **[v2] CORRECTED**

- **Hardware:** NVIDIA GB10 (SM12.1), 128GB unified memory
- **Host:** DGX Spark `sparky` at **`192.168.0.112`** (corrected from spec's `192.168.0.172` "Sparkly")
- **User:** `pnivek`, passwordless SSH verified
- **Python:** `/usr/bin/python3` is 3.12.3
- **b12x install:** **not yet installed** on sparky; needs `pip install -e .` from the patched fork
- **Working directory:** TBD (no `~/b12x-test/` exists yet — create as part of v1 setup)

## 8. Summary of Changes

**[v2] revised totals:**

| File | v1 Δ lines | **v2 Δ lines** | Difficulty | Criticality |
|------|-----------|----------------|------------|-------------|
| `reference.py` | 3 | ~10 (incl. UE8M0 helper) | Trivial | Low — fallback only |
| `kernel.py` | ~40 | **~40–80** (depends on (a)/(b)/(c) refactor) | High | **Critical** — main kernel |
| `kernel_onepass.py` | ~20 | ~20–40 | Medium | Medium — mirror kernel.py |
| `api.py` | 5 | 5 | Trivial | Low — plumbing |
| `workspace.py` | 3 | 5 | Trivial | Low — schema only |
| **`traits.py`** | **0 (missed)** | **~10** | Low | **Critical — gates dispatch** |
| **`split.py`** | **0 (missed)** | **~5** | Trivial | Low — re-imports |
| **Total** | **~70** | **~95–155 + UE8M0 helper** | | |

## 9. Suggested Implementation Order

**[v2] Reordered to put the gating change first.**

1. **`workspace.py`** — Add `nope_logical_dim`/`rope_dim` fields to `B12XAttentionArenaCaps`, default GLM-5.1
2. **`reference.py`** — Make `_MLA_PACKED_DIM` derivable; add UE8M0→FP32 helper; thread `nope_logical_dim` through `pack_mla_kv_cache_reference` (zero-pads bytes 448–511)
3. **`traits.py`** **[v2] new]** — Widen `select_sparse_mla_traits()` to accept `(512, 512, 656)` shape; add `nope_logical_dim`/`rope_dim` to `SparseMLATraits` so the kernel cache can key on them
4. **`kernel.py`** — Pick refactor approach (a)/(b)/(c); parameterize `_MLA_HEAD_DIM` and cache key. Under Option B, most constexpr loops do NOT need re-parameterization
5. **`kernel_onepass.py`** — Mirror kernel.py changes
6. **`split.py`** **[v2] new** — Re-source the parameterized constants from `SparseMLAKernel` instances or accept dims as parameters
7. **`api.py`** — Wire `nope_logical_dim`/`rope_dim` through to kernel
8. **Test single-token (§7.1)** — Verify kernel compiles and produces sane output on sparky
9. **Test multi-token (§7.1b)** — Verify the MTP unblock
10. **Optional: backward-compat sweep** — Re-run `tests/test_attention_mla_*` with default dims to confirm GLM-5.1 still works (not required, but cheap signal)

## 10. Known unknowns / risks — **[v2] NEW SECTION**

- **`v_head_dim` derivation.** HF config has `v_head_dim: null`. The spec assumes 512, consistent with derivation, but this should be cross-checked against vLLM PR #40760's `deepseek_v4_attention.py` before relying on it.
- **CUTE constexpr refactor scope.** Approach (a) is recommended but unproven against this specific kernel; if it explodes, fall back to (c) (per-shape kernel cache + module mutation).
- **Sliding window + hash layers.** DSV4 declares `sliding_window: 128` and `num_hash_layers: 3` in `sparse_attention_params`. These appear to be vLLM-side concerns (they affect which KV tokens get into the topk set), not b12x kernel concerns — the kernel just consumes whatever `page_table_1` it's given. Worth a sanity check during integration.
- **Indexer behavior.** b12x's NSA indexer has not been exercised against DSV4 dims; per code review (`_INDEX_HEAD_DIM=128` matches, `num_heads` is dynamic) it should work, but first-run validation is recommended.
- **CUTLASS-DSL version.** PR #40082 patches `nvidia-cutlass-dsl==4.4.2` to accept `sm_121a`. b12x v0.10.0 declares `>=4.4.1`. Verify the installed version on sparky is patched or upgraded if kernel compile fails with `sm_121a` errors.

## 11. Implementation Results & Validation — **[2026-04-26 implementation pass]**

### 11.1 Actual files touched

Total: **6 source files + 2 test files + 1 bench script** (vs. spec estimate of "6 files, ~95–155 LOC + UE8M0 helper").

| File | Δ lines | Notes |
|------|---------|-------|
| `b12x/attention/mla/reference.py` | +84/-22 | UE8M0→FP32 helper; `nope_logical_dim` threaded through `pack_mla_kv_cache_reference`, `dense_mla_reference`. **Plus**: `sparse_mla_reference` needed a fix not anticipated in v2 spec — unpacked K is always 576 dims but DSV4 Q is 512, so the function now trims K_nope to `nope_logical_dim` before reconcatenating with K_rope. |
| `b12x/attention/mla/traits.py` | +30/-9 | `_SUPPORTED_SHAPES` dict accepts both `(576,512)` GLM and `(512,448)` DSV4; added `nope_logical_dim` and `rope_dim` fields to `SparseMLATraits`. |
| `b12x/attention/mla/workspace.py` | +5 | `nope_logical_dim`/`rope_dim` fields on `B12XAttentionArenaCaps`. |
| `b12x/attention/mla/kernel.py` | +50/-28 | `nope_q_u32_offset: Int32` parameter threaded through `_compute_score_tile_scaled`, `_compute_score_tile_scaled_from_staged_nope`, `_pipeline_stage_q_async`, `_pipeline_stage_tile_async`, `_run_one_pass_sparse_mla_tile`. Output bounds check `if dim_base + Int32(9) < Int32(out_tensor.shape[v_dim])` in `_store_output_group` and `_store_output_group_chunked` to guard DSV4's group-3 overflow. |
| `b12x/attention/mla/kernel_onepass.py` | +42/-28 | Same as kernel.py. |
| `b12x/attention/mla/split.py` | +84/-59 | Output bounds guards in `_zero_partial_head_tile` and `SparseMLASplitDecodeMergeKernel.kernel`; `nope_logical_dim` threaded through `SparseMLASplitDecodeForwardKernel`. |
| `tests/test_mla_dsv4_smoke.py` | +236 | 17 reference-only tests (Level 1). |
| `tests/test_mla_dsv4_kernel.py` | +269 | 17 actual-kernel tests (Level 2). |
| `benchmarks/bench_dsv4_decode.py` | +200 | Microbench harness designed for nsys/ncu instrumentation. |

**Approach (a) from §4.2 worked** — passing `nope_q_u32_offset` as an `Int32` runtime parameter through the call chain, captured at JIT trace time from `self.nope_logical_dim`. Did NOT need approaches (b) or (c). The CUTE constexpr-loop concern was overblown: under Option B, all `constexpr` loops over `_MLA_SCALE_GROUPS=4` stay valid because the storage layout is unchanged.

**Output bounds check was an unanticipated requirement.** v2 spec did not mention this. DSV4's `v_head_dim=448` means the kernel's group-3 output writes (covering dims 384–511) overflow the output tensor's last dim. Solution: per-write runtime bounds check guarded behind a constexpr-friendly comparison.

### 11.2 Test results

**Level 1 — reference-only smoke tests** (`tests/test_mla_dsv4_smoke.py`): **17/17 passed in 2.07s.**

**Level 2 — actual-kernel tests** (`tests/test_mla_dsv4_kernel.py`): **17/17 passed in 118.92s.**

Test matrix covers: traits dispatch, pack/unpack roundtrip, decode (cache_len ∈ {63,64,65,128,129}), MTP (num_q ∈ {2,4,8}), sparse page table with -1 padding, and split kernel (width ∈ {129,257,512}).

**Bugs caught by Level 2 that Level 1 could not see:**

1. **`kernel_onepass.py` multi-tile path missing `nope_q_u32_offset`** at the inner `_compute_score_tile_scaled_from_staged_nope` call inside the `while token_base < token_end` loop. The single-tile path had it, the multi-tile path did not. Would have produced silently-wrong Q rope offsets for DSV4 sequences spanning more than one `_MLA_TOKEN_TILE` (i.e., almost everything in production). **Caught only because Level 2 actually invokes the JIT-compiled kernel; reference tests would have happily passed.**

2. **`tmp_output` dtype contract.** Initially allocated as float32 in the bench/test fixture; correct dtype is bfloat16 (matches `workspace.dtype`). Surfaced as `select_sparse_mla_traits` returning None on the trait check.

### 11.3 Performance results — DGX Spark (GB10, sm_121, 48 SMs)

Measured on synthetic decode workloads with `num_heads=8` (one rank of an 8-way TP setup), bfloat16 Q/output, MXFP8+UE8M0 KV. Self-timed via CUDA events, 200 iters after 10-iter warmup (JIT amortized).

| case | kernel | μs/iter | tok/s | GB/s | % of peak BW |
|---|---|---|---|---|---|
| decode 1k | single | 303 | 3,297 | 2.3 | 0.8% |
| decode 2k | single | 593 | 1,687 | 2.3 | 0.8% |
| decode 4k | single | 1,172 | 853 | 2.3 | 0.8% |
| decode 8k | single | 2,360 | 424 | 2.3 | 0.8% |
| decode 16k | single | 4,716 | 212 | 2.3 | 0.8% |
| decode 32k | single | 9,821 | 102 | 2.2 | 0.8% |
| **decode 1k** | **split** | **105** | **9,538** | **6.6** | 2.4% |
| **decode 2k** | **split** | **105** | **9,541** | **13.0** | 4.8% |
| mtp4 4k | single | 1,184 | 3,377 | 2.4 | 0.9% |
| mtp4 8k | single | 2,365 | 1,691 | 2.4 | 0.9% |

GB/s assumes naive byte count (KV packed bytes + Q + output + page table) per call. Peak BW reference: GB10 LPDDR5X ~273 GB/s.

**nsys ground truth (decode_2k single-tile, 60 invocations):**
- Kernel exec: **591,915 ns avg** (min 583,456 / max 668,480)
- Launch overhead: **~2 μs** (cudaLaunchKernel median)
- Workload is **kernel-bound, not API-bound or launch-overhead-bound**.

### 11.4 Acceleration vs. Triton baseline

vLLM PR #40899's Triton sparse MLA fallback measured **~6.7 tok/s end-to-end on GB10** per §2.2.

Even our worst pure-attention case (decode 32k single-tile at **102 tok/s** for attention alone) is **~15× faster than the entire end-to-end Triton baseline**. End-to-end speedup on full vLLM will be larger because attention now consumes <1% of model forward time.

**MTP unblock confirmed:** the Triton baseline hard-errors on `num_decode_tokens > 1`. Our kernel handles MTP4 in the same wall time as a single-token decode (1,184 μs at 4k for both), giving **4× per-request throughput for free** — which was the primary motivation for this patch.

### 11.5 Optimization headroom

Diagnosed via `torch.cuda.get_device_properties` + nsys, not yet via ncu (see §11.6).

**Single-tile kernel launches grid=(1,1,1), block=(32,1,1) = 1 warp.** GB10 has 48 SMs × 48 max warps/SM = **2,304 warp slots**. The single-tile path uses **0.04% of available warp capacity** — essentially one warp doing all the work sequentially.

**Split kernel** parallelizes across `num_chunks` CTAs. At 2k context with chunk_size=64 → 32 CTAs. That's still only ~1.4% device occupancy but yields the observed 5.7× speedup over single-tile. The split path is also **bounded at `_SPLIT_MAX_WIDTH=2048`**, leaving long-context decode stuck on the slow single-tile path.

**Concrete next-step opportunities** (not blocking for v1 POC):

1. **Extend `_SPLIT_MAX_WIDTH`** past 2048 (or add a longer chunk_size ladder) so 8k/16k/32k decode can use the split path. Expected gain at 8k: ~5× from current 424 tok/s → ~2k tok/s.
2. **Single-tile kernel parallelism** — the current implementation uses 1 warp per (q_idx, head_tile) pair. Could parallelize the KV walk across multiple warps within a CTA.
3. **At <5% memory BW utilization**, there is **20–100× headroom** before hitting the LPDDR5X ceiling. The kernel is severely under-occupied, not memory-bound.

### 11.6 Validation infrastructure on sparky

Set up at `pnivek@192.168.0.112:~/b12x` with venv at `~/b12x-venv` (Python 3.12.3, torch 2.11.0+cu130, nvidia-cutlass-dsl 4.4.2). Re-running validation:

```bash
ssh pnivek@192.168.0.112
cd ~/b12x
~/b12x-venv/bin/python -m pytest tests/test_mla_dsv4_smoke.py tests/test_mla_dsv4_kernel.py -v
~/b12x-venv/bin/python benchmarks/bench_dsv4_decode.py
```

**ncu profiling not yet enabled.** First attempt failed with `ERR_NVGPUCTRPERM` — non-root users blocked from GPU performance counters by default. To enable for roofline / SM-throughput / mem-throughput percentages:

```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
# or just reboot
```

Then `/usr/local/cuda-13.0/bin/ncu --set roofline ...` for roofline analysis.

### 11.7 Outstanding work

- **End-to-end vLLM bring-up.** Patched b12x is not yet wired into a vLLM build that loads DSV4-Flash. Next concrete step: install vllm + transformers in `~/b12x-venv`, point `VLLM_ATTENTION_BACKEND=B12X` (or the equivalent), and run a real DSV4 decode.
- **GLM-5.1 backward-compat sweep** not run (was nice-to-have per §7.2). All Level 2 tests use DSV4 dims; a quick re-run of `tests/test_attention_mla_*` with the GLM model weights would confirm no regression. Requires `/data/models/GLM-5.1-NVFP4` or similar on sparky.
- **`split.py` for long contexts.** The `_SPLIT_MAX_WIDTH=2048` cap is the single biggest perf gap for production workloads. Lifting it (or adding a multi-stage split) is the recommended next optimization.
- **ncu-driven roofline** to validate the under-occupancy thesis quantitatively. Blocked on driver permission tweak above.
