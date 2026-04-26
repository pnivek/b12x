# b12x-vLLM Integration Spec: DeepSeek-V4-Flash MLA on SM121

## 1. Objective

Wire the patched b12x `sparse_mla_decode_forward()` / `sparse_mla_extend_forward()` into
vLLM's `deepseek_v4_attention.py` so that SM120/121 decode and prefill paths use
the optimized CuTe WMMA kernels instead of the Triton sparse MLA fallback.

**Goal:** Unblock MTP (multi-token decode) on DGX Spark, achieve ~2-4x decode throughput.

---

## 2. Architecture Overview

### 2.1 Current Flow (Triton Fallback)

```
DeepseekV4MLAAttention.forward()
  ├── split prefill / decode
  ├── _forward_prefill() → flash_mla_sparse_fwd() [via TileLang/FlashMLA]
  └── _forward_decode() → flash_mla_with_kvcache() [via TileLang/FlashMLA]
       └── fallback when is_sparse_mla_reference_attention_enabled():
            ├── _forward_sparse_mla_swa_decode_reference()     [SWA-only]
            └── _forward_sparse_mla_compressed_decode_reference() [C4A/C128A]
                 └── calls vllm Triton kernels: fp8ds_paged/global_sparse_mla_attention...
                      └── FAILS on MTP: "supports one query token per request"
```

### 2.2 Target Flow (b12x)

```
DeepseekV4MLAAttention.forward()
  ├── split prefill / decode
  ├── _forward_prefill() → b12x.sparse_mla_extend_forward()  [SM120/121]
  └── _forward_decode() → b12x.sparse_mla_decode_forward()   [SM120/121]
       └── supports MTP natively (multi-token decode)
```

### 2.3 b12x Public API

**Decode:**
```python
from b12x.attention.mla import (
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    B12XAttentionWorkspace,
    B12XAttentionArena,
    B12XAttentionArenaCaps,
)
```

**`sparse_mla_decode_forward(q_all, kv_cache, metadata, workspace, sm_scale, v_head_dim)`**
- `q_all`: `(num_tokens, num_heads, head_dim)` BF16 — **pre-padded** to 128 heads, NoPE+RoPE concatenated
- `kv_cache`: `(num_kv_rows, 1, packed_dim)` uint8 — packed MLA cache in b12x format
- `metadata`: `MLASparseDecodeMetadata(page_table_1, cache_seqlens_int32, nsa_cache_seqlens_int32, max_seq_len_k)`
- `workspace`: `B12XAttentionWorkspace` — pre-allocated arena with fixed capacity
- `sm_scale`: float — attention scaling factor
- `v_head_dim`: int — output head dimension for V
- Returns: `(num_tokens, num_heads, v_head_dim)` BF16

**`sparse_mla_extend_forward(q_all, kv_cache, metadata, workspace, sm_scale, v_head_dim)`**
- Same signatures but with `MLASparseExtendMetadata`

**Metadata classes:**
```python
@dataclass(frozen=True)
class MLASparseDecodeMetadata:
    page_table_1: torch.Tensor          # (num_tokens, topk) int32 — token indices per query
    cache_seqlens_int32: torch.Tensor   # (num_tokens,) int32 — KV length per query token
    nsa_cache_seqlens_int32: torch.Tensor  # (num_tokens,) int32 — active token count per query
    max_seq_len_k: int                  # max KV length in batch
```

**Workspace construction:**
```python
caps = B12XAttentionArenaCaps(
    device=torch.device("cuda:0"),
    dtype=torch.bfloat16,       # compute dtype
    kv_dtype=torch.uint8,       # cache dtype
    num_q_heads=64,             # DSV4: 64 attention heads
    indexer_num_q_heads=64,     # same
    head_dim=576,               # 448 NoPE + 64 RoPE + 64 zero-pad = 576
    max_v_head_dim=512,         # v_head_dim
    topk=64,                    # max compressed candidates per query
    max_page_table_width=64,    # max page-table width
    extend_max_total_q=16384,   # max prefill tokens
    extend_max_batch=8,         # max prefill batch
    extend_max_kv_rows=65536,   # max KV rows in prefill workspace
    paged_max_q_rows=2048,      # max decode query rows
    paged_max_batch=8,          # max decode batch
    page_size=256,              # block size
    padded_heads=128,           # FlashMLA/padding requirement
)
arena = B12XAttentionArena.allocate(caps)
workspace = arena.make_workspace(B12XAttentionWorkspaceContract(
    mode="decode",
    max_total_q=2048,
    max_batch=8,
    max_paged_q_rows=2048,
    max_kv_rows=65536,
    v_head_dim=512,
    indexer_num_q_heads=64,
    max_page_table_width=64,
))
```

---

## 3. File Changes

### File 1: `vllm/model_executor/layers/deepseek_v4_attention.py`

#### 3.1 New imports (top of file, after existing b12x-compatible imports)

```python
# === b12x MLA integration ===
try:
    from b12x.attention.mla import (
        B12XAttentionArena,
        B12XAttentionArenaCaps,
        B12XAttentionWorkspace,
        B12XAttentionWorkspaceContract,
        MLASparseDecodeMetadata,
        MLASparseExtendMetadata,
        sparse_mla_decode_forward,
        sparse_mla_extend_forward,
    )
    _B12X_AVAILABLE = True
except ImportError:
    _B12X_AVAILABLE = False
```

#### 3.2 Environment flag (with other env vars or in `sparse_mla_env.py`)

```python
def is_b12x_enabled(device: torch.device) -> bool:
    if not _B12X_AVAILABLE:
        return False
    capability = current_platform.get_device_capability()
    if capability is None:
        return False
    # b12x supports SM120 and SM121
    return capability.major == 12 and capability.minor in (0, 1)
```

#### 3.3 Extended `DeepseekV4MLAAttention.__init__`

Add these fields after existing init:

```python
self.b12x_enabled = is_b12x_enabled(torch.device("cuda"))
self.b12x_workspace: B12XAttentionWorkspace | None = None

if self.b12x_enabled:
    max_model_len = vllm_config.model_config.max_model_len
    max_batch = vllm_config.scheduler_config.max_num_seqs
    max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    caps = B12XAttentionArenaCaps(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=self.num_heads,
        indexer_num_q_heads=self.num_heads,
        head_dim=self.head_dim,                      # 576 (448+64+64pad)
        max_v_head_dim=self.v_head_dim,              # 512
        topk=max(64, self.compress_ratio * 16),      # generous upper bound
        max_page_table_width=max(64, self.compress_ratio * 16),
        extend_max_total_q=max_batched_tokens,
        extend_max_batch=max_batch,
        extend_max_kv_rows=max_model_len,            # full KV cache capacity
        paged_max_q_rows=max_batched_tokens,
        paged_max_batch=max_batch,
        page_size=vllm_config.cache_config.block_size,
        padded_heads=self.padded_heads,
    )
    arena = B12XAttentionArena.allocate(caps)
    self.b12x_workspace = arena.make_workspace(
        B12XAttentionWorkspaceContract(
            mode="decode" if self.compress_ratio <= 1 else "decode",
            max_total_q=max_batched_tokens,
            max_batch=max_batch,
            max_paged_q_rows=max_batched_tokens,
            max_kv_rows=max_model_len,
            v_head_dim=self.v_head_dim,
            indexer_num_q_heads=self.num_heads,
            max_page_table_width=max(64, self.compress_ratio * 16),
        ),
        use_cuda_graph=True,
    )
```

#### 3.4 New KV cache format converter

Add helper to convert vLLM `fp8_ds_mla` → b12x packed format:

```python
def _convert_fp8ds_to_b12x_kv_cache(
    vllm_cache: torch.Tensor,       # (num_blocks, block_size, 1, 584) uint8
    nope_dim: int = 448,            # DSV4 NoPE dimension
    b12x_nope_dim: int = 512,       # b12x padded NoPE dimension
    rope_dim: int = 64,             # RoPE dim (128 bytes BF16)
    num_scales: int = 7,            # DSV4 scale count (448/64)
    b12x_num_scales: int = 8,       # b12x scale count (512/128*4 bytes)
    block_size: int = 256,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert vLLM fp8_ds_mla format to b12x packed format.

    vLLM layout (584 bytes/row):
      [0..447]    FP8 E4M3 NoPE (448 bytes)
      [448..575]  BF16 RoPE (128 bytes)
      [576..583]  UE8M0 scales (7 bytes + 1 pad)

    b12x layout (656 bytes/row):
      [0..511]    FP8 E4M3 NoPE (512 bytes, last 64 zero-padded)
      [512..543]  FP32 scales (32 bytes, 4 bytes × 8 groups)
      [544..655]  BF16 RoPE (128 bytes)
    """
    # Flatten to 2D for processing
    num_blocks = vllm_cache.shape[0]
    num_rows = num_blocks * block_size

    # Create output buffer: (num_rows, 1, 656) uint8
    out = torch.zeros(num_rows, 1, b12x_nope_dim + b12x_num_scales * 4 + rope_dim * 2,
                      dtype=torch.uint8, device=device)

    flat_vllm = vllm_cache.view(num_rows, -1)      # (num_rows, 584)
    flat_out = out.view(num_rows, -1)               # (num_rows, 656)

    # Copy NoPE (first 448 bytes)
    flat_out[:, :nope_dim] = flat_vllm[:, :nope_dim]

    # Copy RoPE (bytes 448-575 in vllm → bytes 544-655 in b12x)
    rope_start_vllm = nope_dim     # 448
    rope_start_b12x = b12x_nope_dim + b12x_num_scales * 4  # 512 + 32 = 544
    flat_out[:, rope_start_b12x:rope_start_b12x + rope_dim * 2] = \
        flat_vllm[:, rope_start_vllm:rope_start_vllm + rope_dim * 2]

    # Convert UE8M0 scales to FP32
    # vLLM UE8M0: 7 bytes (exponent+127), one per 64 elements
    # b12x FP32: 4 bytes per group of 128 elements
    scales_vllm = flat_vllm[:, nope_dim + rope_dim * 2: nope_dim + rope_dim * 2 + num_scales].float()
    # Expand to 8 groups (group 8 = 1.0 for zero-padded 64 nope bytes)
    scales_b12x = torch.zeros(num_rows, b12x_num_scales, dtype=torch.float32, device=device)
    # DSV4 scale: stored as E8M0 exponent byte → val = 2^(exp-127)
    # So scale = 2^(uint8 - 127)
    scales_val = 2.0 ** (scales_vllm - 127.0)
    scales_b12x[:, :num_scales] = scales_val       # groups 0-6
    scales_b12x[:, num_scales] = 1.0                # group 7 (zero-padded)
    flat_out[:, b12x_nope_dim:b12x_nope_dim + b12x_num_scales * 4] = \
        scales_b12x.view(torch.uint8).view(num_rows, -1)

    # Zero-pad the extra 64 bytes of NoPE (slots 448-511)
    # Already zero from torch.zeros allocation

    return out
```

**NOTE:** The conversion above is done per-token at decode time via a simple elementwise kernel. For production, fuse this into the KV cache insert path to avoid O(num_tokens * 584) copy overhead. See §6 Optimizations.

#### 3.5 Extended `_forward_decode` — add b12x dispatch

Replace the reference-fallback guard with b12x dispatch. Insert this block **before** the `is_sparse_mla_reference_attention_enabled()` check (around line 1260):

```python
# === b12x accelerated decode path ===
if self.b12x_enabled and self.b12x_workspace is not None:
    # Build b12x metadata from swa_metadata + flashmla_metadata
    if swa_only:
        # SWA-only: page_table_1 = swa_indices, seq_lens from swa_metadata
        active_tokens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        page_table = swa_metadata.decode_swa_indices[:num_decode_tokens]
    else:
        # C4A/C128A: combine swa + compressed indices
        # page_table_1 = concat(compressed_slot_ids, swa_indices) along last dim
        combined_indices = torch.cat([
            compressed_slot_ids,    # (num_decode_tokens, compressed_topk)
            swa_indices,            # (num_decode_tokens, max_swa_len)
        ], dim=-1).to(torch.int32)
        active_tokens = topk_lens + swa_lens
        page_table = combined_indices

    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table,        # (num_decode_tokens, total_candidates) int32
        cache_seqlens_int32=swa_metadata.seq_lens[:num_decodes].int(),
        nsa_cache_seqlens_int32=active_tokens.int(),
        max_seq_len_k=int(swa_metadata.seq_lens[:num_decodes].max().item()),
    )

    # Convert KV cache format (vLLM fp8_ds_mla → b12x packed)
    b12x_cache = _convert_fp8ds_to_b12x_kv_cache(
        swa_cache.squeeze(-2),          # Remove dummy dim
        device=q.device,
    )

    if not swa_only and compressed_k_cache is not None:
        # Add compressed KV rows to b12x cache
        comp_b12x = _convert_fp8ds_to_b12x_kv_cache(
            compressed_k_cache.squeeze(-2),
            device=q.device,
        )
        b12x_cache = torch.cat([b12x_cache, comp_b12x], dim=0)

    # Run b12x decode
    output_b12x = sparse_mla_decode_forward(
        q_all=q.to(torch.bfloat16),     # q is (num_decode_tokens, 1, padded_heads, head_dim)
        kv_cache=b12x_cache,
        metadata=metadata,
        workspace=self.b12x_workspace,
        sm_scale=self.scale,
        v_head_dim=self.v_head_dim,
    )
    # output_b12x: (num_decode_tokens, num_heads, v_head_dim)
    output[:num_decode_tokens, :self.num_heads, :self.v_head_dim].copy_(output_b12x)
    return
```

#### 3.6 Extended `_forward_prefill` — add b12x dispatch

Insert a similar guard at the start of `_forward_prefill`:

```python
# === b12x accelerated prefill path ===
if self.b12x_enabled and self.b12x_workspace is not None:
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=...,        # Combined compressed + SWA gather indices
        cache_seqlens_int32=...,           # KV sequence lengths
        nsa_cache_seqlens_int32=...,       # Active token counts
        nsa_cu_seqlens_q=...,             # CUDA seqlens for Q
        nsa_cu_seqlens_k=...,             # CUDA seqlens for K
        max_seq_len_q=int(seq_lens.max()),
        max_seq_len_k=int(seq_lens.max()),
        mode="extend",
    )
    output = sparse_mla_extend_forward(
        q_all=q.to(torch.bfloat16),
        kv_cache=b12x_cache,
        metadata=metadata,
        workspace=self.b12x_workspace,
        sm_scale=self.scale,
        v_head_dim=self.v_head_dim,
    )
    output_prefill.copy_(output)
    return
```

**NOTE:** Prefill integration is less critical for the first iteration — the Triton prefill path (`flash_mla_sparse_fwd`) already works. Focus on decode first.

### File 2: Dockerfile (add `pip install b12x`)

```dockerfile
# Add b12x wheel — pre-built or build from source
RUN uv pip install b12x==0.10.0 \
    && (pip show b12x || echo "b12x fallback: build from source")
```

**Important:** The `pip install b12x` must happen **after** the flashinfer install (b12x's CUTLASS/CUTE deps need non-conflicting CUDA runtime). The b12x fork with the DSV4 patch (commit `90c20f9`) needs to be published to PyPI or installed from a GitHub ref:

```dockerfile
RUN uv pip install "b12x @ git+https://github.com/pnivek/b12x.git@dsv4-support"
```

If publishing to PyPI isn't possible, the pre_build patch script should:
1. `pip install b12x`
2. `sed` in the DSV4 dimension changes to the installed package files
3. Version-pin with a date suffix

---

## 4. KV Cache Format Translation

### 4.1 Why Translation Is Needed

| Field | vLLM `fp8_ds_mla` (584B/row) | b12x packed (656B/row) |
|-------|------|------|
| NoPE quantized | 448 bytes FP8 E4M3 | 512 bytes FP8 E4M3 (last 64 zero) |
| Scales | 7 bytes, UE8M0 (1B/64elem) | 32 bytes, FP32 (4B/128elem) |
| RoPE | 128 bytes BF16 | 128 bytes BF16 |
| **Total** | **584** | **656** |

### 4.2 Scale Conversion

**UE8M0 → FP32:**
```
UE8M0 byte = uint8 exponent with bias 127
  scale = 2^(byte - 127)
```

**FP32 → UE8M0 (for write-back):**
```
UE8M0 byte = clamp(round(log2(scale)) + 127, 0, 255)
```

### 4.3 When to Convert

**Option A (preferred for v1):** Convert at attention time, right before calling b12x. Allocate a second cache buffer (temporary) in b12x format. This is simplest but doubles cache memory for the attention window.

**Option B (preferred for production):** Fuse the conversion into the KV cache insert path (`_fused_qnorm_rope_kv_insert`). Replace the `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` kernel with a version that writes b12x format directly. This requires modifying the CUDA kernel (the `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`).

**Recommendation:** Start with Option A for correctness, migrate to Option B once the attention path is validated.

---

## 5. Integration Sequence

### Phase 1: Baseline Validation (already done)
- ✅ b12x workspace/arena creation with DSV4 dims
- ✅ DSV4 kernel patch (commit `90c20f9`) — zero-pad nope to 512
- ✅ Test suite 34/34 passes

### Phase 2: vLLM Integration (this spec)
1. [ ] Build a custom vLLM image with `pip install b12x`
2. [ ] Add the import guard and workspace initialization in `__init__`
3. [ ] Add the KV cache converter function
4. [ ] Replace the decode reference path with `sparse_mla_decode_forward` for SWA-only
5. [ ] Extend to C4A/C128A decode path
6. [ ] Enable MTP — verify multi-token decode no longer fails
7. [ ] Add prefill path (lower priority)

### Phase 3: Tuning
8. [ ] Measure decode throughput vs Triton baseline
9. [ ] Fuse KV cache conversion into insert kernel (Option B)
10. [ ] Tune `topk`, `head_block_size` env overrides
11. [ ] CUDA graph capture — ensure b12x kernel is graph-compatible
12. [ ] Test with `VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1`

---

## 6. Edge Cases & Caveats

### 6.1 CUDA Graph Compatibility
b12x uses CUTLASS host launchers that compile dynamically. For CUDA graph capture:
- The `fixed_capacity=True` + `use_cuda_graph=True` flags tell b12x to pre-allocate all runtime metadata
- Workspace size must be over-provisioned (max capacity, not current batch)
- `nope_q_u32_offset` must be runtime-parameterized, not compile-time
- Patch commit `90c20f9` already handles this with the Int32 param

### 6.2 KV Cache Memory
Option A doubles KV cache memory for the attention buffer (584→656 bytes/row = +12.3%). For DGX Spark's 48GB, this is ~5.9GB extra for 10K tokens — acceptable.

### 6.3 MTP Path
b12x handles multi-token decode in the same kernel call. The `num_decode_tokens != num_decodes` check in the reference path is bypassed entirely. Just pass the full `q[:num_decode_tokens]` tensor to `sparse_mla_decode_forward`.

### 6.4 NCCL / TP=2
b12x is a single-GPU kernel. TP splits are handled by vLLM's model-level sharding (pre-kernel). b12x sees only the local shard's heads (32 heads per GPU with TP=2). Workspace must reflect the local head count, not the global count.

### 6.5 Cleanup
On model unload or workspace reset, call `b12x.attention.mla.clear_mla_caches()` to free compiled kernel cache.

---

## 7. Files to Modify

| File | Change |
|------|--------|
| `vllm/model_executor/layers/deepseek_v4_attention.py` | +import guard, +workspace init, +cache converter, +b12x decode dispatch, +b12x prefill dispatch |
| `vllm/v1/attention/backends/mla/sparse_mla_env.py` (or local) | +`is_b12x_enabled()` helper |
| Dockerfile (in pre_build patch) | +`pip install b12x @ git+https://github.com/pnivek/b12x.git@dsv4-support` |

### Total approximate code to write:
- **Workspace init:** ~40 lines
- **Cache converter:** ~35 lines
- **Decode dispatch insertion:** ~30 lines
- **Prefill dispatch insertion:** ~30 lines
- **Import guard + helper:** ~10 lines
- **Docker mod:** 1 line
- **Total:** ~146 lines of new code, ~5 files touched

---

## 8. Verification

After integration, verify:

1. **Single-token decode works** — `/v1/chat/completions` returns correct output
2. **MTP decode works** — enable `--num-speculative-tokens 4`, verify no "supports one query token" error
3. **Throughput improvement** — benchmark: 1024 output tokens, MTP=4 vs MTP=0
4. **CUDA graph capture** — verify successful capture on warm restart
5. **KV cache correctness** — compare output tokens between Triton and b12x paths (set `B12X_MLA_FORCE_REFERENCE=1` vs kernel path)
6. **Memory usage** — verify no OOM with `gpu_memory_utilization=0.90`

---

## 9. Future Work (beyond this spec)

- **Fused KV cache insert** — Modify the fused Q-norm + RoPE + KV insert CUDA kernel to write b12x format directly, eliminating the format conversion step
- **FlashMLA backend via b12x** — Register b12x as a FlashMLASparseBackend variant so it's selected automatically by `get_attn_backend()`
- **Multi-node MTP** — NCCL sync for MTP draft model outputs across TP=2