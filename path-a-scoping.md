# Path A — cuTe Sparse MLA Kernel Native DSV4-Flash Support: Scoping

**Status:** scoping (no code changes yet)
**Author:** ds4 integration follow-up
**Baseline being beaten:** 23.5 tok/s (build #8 image, vectorized BF16 reference path with bypass) — already +59% over Triton+MTP=14.8.
**Target:** 35–40 tok/s by eliminating Python/sync overhead in the dispatch loop and running the cuTe kernel natively.

---

## 1. The mismatch (root cause)

vLLM's DSV4-Flash contract requires the attention kernel to return shape
`(num_decode_tokens, num_heads, kv_lora_rank=512)` — i.e. V is the **full latent K = nope(448) ⊕ rope(64) = 512 dims**, the post-absorption MLA convention. Confirmed by:

- `deepseek_v4_attention.py:887,1252` — `swa_acc` and the b12x output assert both use `kv_lora_rank=512`.
- The Triton fallback path (`_forward_sparse_mla_swa_decode_reference`) sizes its V workspace as `q.shape[-1]=512`.
- The current b12x bypass path explicitly does `v_all = k_all = concat(k_nope, k_rope)` (`deepseek_v4_attention.py:1231`).

The b12x cuTe kernel as currently shipped supports DSV4 only via:

```python
# traits.py:17
_SUPPORTED_SHAPES: dict[tuple[int, int], int] = {
    (576, 512): 512,  # GLM-5.1
    (512, 448): 448,  # DSV4-Flash   ← v_head_dim=448, NOT 512
}
```

The existing `(512, 448)` entry treats V as the **nope portion only** — rope is stored in cache and used in QK score, but never loaded into the PV MMA. Output is 448 dims, then write guards (`kernel.py:1333,1374`) silently drop dims ≥ v_head_dim.

**Conclusion:** the cuTe kernel today **cannot** be dropped into the existing vLLM dispatch unchanged — its output is 448-wide, vLLM's contract is 512-wide. The current bypass works precisely because it computes the rope contribution to V on the CPU/Python side via `_sparse_attention_reference_vectorized`. To fire the cuTe kernel, we either

- **(A1)** extend the kernel to compute PV over rope as well, or
- **(A2)** change vLLM's contract to V=448 (drop rope from V), accepting the numerical delta.

A2 would change model output (different mathematical operation than Triton baseline). Not viable for parity. **Path A means A1.**

---

## 2. Kernel architecture as it stands

### 2.1 Storage and tile constants (`kernel.py:48-85`)

```
_MLA_NOPE_DIM         = 512           # storage, always 512 (zero-padded for DSV4)
_MLA_ROPE_DIM         = 64
_MLA_HEAD_DIM         = 576           # nope_storage + rope (storage-side)
_MLA_GROUP_SIZE       = 128
_MLA_SCALE_GROUPS     = 4             # _MLA_NOPE_DIM // _MLA_GROUP_SIZE
_MLA_OUTPUT_DIM       = _MLA_NOPE_DIM # 512  ← this is where rope-as-V is excluded
_MLA_VO_NUM_MMA_D     = 8             # _MLA_NOPE_GROUP_ELEMS // 16, per-group MMAs
_MLA_TOKEN_TILE       = 32
_MLA_HEADS_PER_TILE   = 16
```

### 2.2 The PV MMA loop (`kernel.py:1479` `_accumulate_pv_groups_from_p_frag`)

```
for block_offset in 0..3:                # 4 nope groups (each 128 dims)
    stage scales for group
    stage 32×128 byte tile of V (=K_nope[group])      ← FP8 e4m3
    issue mxfp8 PV MMAs into tile_o_frag
    accumulate into o_frag{0..3}                       ← 4 separate accumulators
```

Each `o_frag{N}` is `(1, _MLA_VO_NUM_MMA_D=8, 8)` Float32 — covers 128 output dims. Final layout: `output[q, head, 0..127]`, `output[q, head, 128..255]`, etc.

**Rope is not in this loop.** Rope is loaded only for the QK score compute (`_compute_score_tile_scaled_from_staged_nope`).

### 2.3 Output store (`_store_output_group`, `kernel.py:1308`)

```
dim_base = group_idx * 128 + mma_d * 16 + lane_pair_base
if dim_base + 9 < out_tensor.shape[2]:    # guard, today truncates DSV4 to 448
    out_tensor[..., dim_base + {0,1,8,9}] = o_frag[mma_d, ...] * inv_d
```

### 2.4 Cache layout (b12x packed format, `_MLA_PACKED_DIM=656`)

```
[0..511]    FP8 nope (storage 512 wide; DSV4 uses [0..447], pads [448..511]=0)
[512..527]  FP32 scales (4 groups × 4 bytes)
[528..655]  BF16 rope (64 elements × 2 bytes)
```

Rope lives at byte offset 528, separate from nope's FP8 region.

### 2.5 Workspace coupling (`workspace.py`)

Sizes scratch buffers by `v_head_dim`:
- `(max_total_q, num_q_heads, v_head_dim)` BF16 final output
- `(max_total_q, num_q_heads, max_chunks, v_head_dim)` BF16 split tmp output

Going from v=448 → v=512 grows these buffers by ~14% (448→512). Trivial.

### 2.6 Split-decode (`split.py`)

The chunked variant (`split.py:213` `_run_one_pass_split_sparse_mla_tile`) calls into the same `_accumulate_pv_groups_from_p_frag` machinery. Same fix surface as single-pass — no separate kernel-side V handling.

---

## 3. The actual work — diff plan

### Step 1 — `_SUPPORTED_SHAPES`: add the new shape

```python
# traits.py
_SUPPORTED_SHAPES = {
    (576, 512): 512,  # GLM-5.1
    (512, 448): 448,  # DSV4-Flash, V=nope only (legacy / uncommon)
    (512, 512): 448,  # DSV4-Flash, V=K=nope+rope (vLLM absorbed contract)
}
```

`nope_logical_dim` stays 448 (controls QK score path); `v_head_dim=512` triggers the rope-as-V branch.

### Step 2 — kernel: PV over rope as a 5th output group

Add a rope MMA after the nope-group loop in `_accumulate_pv_groups_from_p_frag`. Two design decisions:

**(a) MMA dtype.** Rope is BF16 in cache (no FP8, no UE8M0 scales). Reuse the existing BF16 PV path machinery — there is a `_literal_pv_mma_into_ofrag_bf16` style helper for the QK side; PV-side BF16 MMA needs to be added or composed from CUTLASS BF16 m16n8k16 instructions. **~150 lines of new cuTe DSL.**

**(b) Tile shape.** Rope is 64 dims = half of a nope group. Two options:
- *Option B1:* one MMA tile of 64 dims + masked output store (`mma_d` runs 0..3 instead of 0..7). Cheaper.
- *Option B2:* zero-pad rope to 128 in shared memory and reuse the existing 128-wide store. Wastes 64 dims of write but reuses code.

Go with **B1** — it's the existing pattern used for `_MLA_VO_NUM_MMA_D` ranged loops, and avoids a useless half-tile of work.

```python
# pseudocode for the addition to _accumulate_pv_groups_from_p_frag
if cutlass.const_expr(v_head_dim_includes_rope):
    # 5th group: rope segment, dims [448..511] → output [448..511]
    _stage_kv_bf16_block_async(  # already exists for QK side
        kv_rows_u32, sTokenIdx,
        rope_u32_offset, rope_vecs, rope_vecs,
        kv_base_addr, num_kv, lane,
    )
    cute.arch.sync_threads()
    rope_o_frag = make_rmem_tensor((1, 4, 8))  # 4 mma_d × 8 regs = 64 dims/lane-pair
    _zero_output_frag(rope_o_frag)
    _literal_pv_mma_into_ofrag_bf16(   # NEW helper; mirror fp8 version
        rope_o_frag, p_frag, kv_base_addr, lane,
    )
    _accumulate_scaled_output_frag(o_frag_rope, rope_o_frag, Float32(1.0))
```

Then a 5th output store with `dim_base = 448 + mma_d*16 + lane_pair_base`, guarded `< 512`.

### Step 3 — output frag plumbing

`_run_one_pass_sparse_mla_tile` currently allocates `o_frag0..3`. Add `o_frag_rope` of layout `(1, 4, 8)`:

```python
rope_o_layout = cute.make_layout((1, 4, 8), stride=(32, 8, 1))
o_frag_rope = cute.make_rmem_tensor(rope_o_layout, Float32)
_zero_output_frag_4(o_frag_rope)  # NEW; 4 mma_d instead of 8
```

Thread it through `_accumulate_pv_groups_from_p_frag` and the final softmax denom division + store-output stages. **~40 lines of plumbing across `kernel.py:2244..2350` and split.py merge.**

### Step 4 — softmax denom: rope group shares the same `d_frag`

The softmax denominator (`d_frag`) is computed once over all KV tokens — it is independent of the output dim split. Rope output divides by the same `inv_d` used for nope groups. **Zero kernel-math change** — just thread `d_frag` into the new store.

### Step 5 — split-decode merge (`split.py` ~lines 280-320)

The merge kernel reads partial outputs of shape `(num_chunks, v_head_dim)` per (q, head). When `v_head_dim=512`, the merge naturally widens. Walk through `split.py:319` (`Int32(self.nope_logical_dim // 2)`) — that's a Q-staging offset, unrelated to V-side dim. **Likely no change** but verify with a parity test.

### Step 6 — `_MLA_OUTPUT_DIM` becomes parameterized

Currently `_MLA_OUTPUT_DIM = _MLA_NOPE_DIM = 512` (the storage dim, not the logical). For (512, 448) DSV4 today, output dim is effectively 448 because writes ≥448 are gated. For (512, 512) DSV4-vLLM, we want all 512 dims written.

The cleanest refactor: make `output_dim` a kernel constructor arg derived from traits (`v_head_dim`). The `_MLA_VO_NUM_MMA_D` constant stays 8 (it's per nope-group). The new constant is `_MLA_VO_NUM_ROPE_MMA_D = 4`.

### Step 7 — workspace (`workspace.py:223,697,1348,1354`)

Already sized by `v_head_dim`. No code change — passing `v_head_dim=512` from vLLM grows the buffer correctly. Verify on sparky that GB10's 48 GB doesn't OOM at the larger workspace (delta is small).

### Step 8 — vLLM dispatch (`deepseek_v4_attention.py`)

Replace the bypass branch (currently calling `_sparse_attention_reference_vectorized`) with the cuTe path:

```python
# delete the bypass; restore native sparse_mla_decode_forward call
output_b12x = sparse_mla_decode_forward(
    q_all=q_b12x, kv_cache=b12x_cache,
    metadata=MLASparseDecodeMetadata(...),
    workspace=self.b12x_workspace,
    sm_scale=self.scale, v_head_dim=self.kv_lora_rank,  # 512
    nope_logical_dim=448,
    attn_sink=self.attn_sink,  # already wired through reference path
)
```

Keep the bypass path behind `B12X_MLA_REFERENCE=1` for parity comparison.

### Step 9 — `attn_sink` in the cuTe kernel

The reference path (`reference.py`) already accepts `attn_sink` and folds it into the softmax `m`/`d` accumulators. The cuTe kernel does **not** today. Add it to the softmax epilogue:

- Locate softmax finalize in `_run_one_pass_sparse_mla_tile` (where `m_frag`, `d_frag` are produced).
- After the loop, before division: `m_new = max(m_frag, sink); d_new = d_frag * exp2(m_frag - m_new) + exp2(sink - m_new)`. Same accumulator pattern as chunk merges.
- Pass `attn_sink: cute.Tensor | None` (shape `(num_q_heads,)` BF16) into the kernel signature.

**~80 lines** across `kernel.py`, `kernel_onepass.py` (if also used), `split.py` merge, and the `api.py` thread-through.

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| BF16 PV MMA path doesn't exist; need new helper | High | Medium | Mirror existing FP8 fp8 helpers (`_literal_pv_mma_into_ofrag_fp8_raw_scaled`); BF16 m16n8k16 PTX is well-trodden |
| Split-decode merge silently truncates at 448 | Medium | High (silent num bug) | Parity test with split-decode forced (small chunk size) before/after |
| `attn_sink` integration interacts with split-decode chunk merge weights | Medium | High | Sink contributes to *final* softmax only — must be applied **after** chunk merge, not per-chunk. Plumb it into split.py merge kernel only, not per-chunk computation |
| Workspace bumping max_v_head_dim invalidates baked image's compiled JIT cache | Low | Low | Acceptable — first call after deploy recompiles |
| Existing `(512, 448): 448` shape silently selected over new `(512, 512): 448` because traits dispatch picks first match | Medium | Medium | Make `select_sparse_mla_traits` strict on `v_head_dim` exact match (it already is — keyed on `(head_dim, v_head_dim)` tuple) |
| Rope is BF16 in storage but FP8 PV MMAs are tuned for E4M3 — mixed-dtype PV needs separate code path | Certain | Medium | Plan accounts for it (Step 2a); do not try to quantize rope |
| Attention output non-zero in slots 64..127 (padded heads) breaks downstream W_O projection | Low | High (silent num bug) | Existing dispatch zeros padded heads (`output[:, num_heads:].zero_()`); preserve that |

---

## 5. Effort estimate

| Step | Effort | Notes |
|---|---|---|
| 1. Traits/`_SUPPORTED_SHAPES` | 0.5 hr | Trivial |
| 2. BF16 PV MMA helper | 0.5 day | Compose existing `bf16_mma_m16n16k16_f32` (`b12x/cute/fp4.py:1163`); mirror QK helper at `kernel.py:682` |
| 3. Output frag plumbing | 0.5 day | Mechanical |
| 4. Softmax denom — no math change | — | Zero |
| 5. Split-decode merge audit + fix | 0.5 day | Likely small but parity-test heavy |
| 6. `_MLA_OUTPUT_DIM` parameterization | 0.5 day | Mechanical refactor |
| 7. Workspace — pass-through | 0.5 hr | No code |
| 8. vLLM dispatch swap (cuTe over bypass) | 0.5 day | Already integrated; minor edits |
| 9. `attn_sink` in cuTe softmax | 1 day | New kernel epilogue + plumbing |
| Parity validation (existing 34-test harness + new 512 cases) | 1 day | Run on sparky against the captured fixtures from Phase 1 |
| End-to-end perf measurement on baked image | 0.5 day | Build+deploy via Komodo |
| **Total** | **~4.5–5 days** | Single engineer, sparky access required (revised after pre-kickoff #1) |

The longest poles are (2) the BF16 PV helper and (9) attn_sink — both touch the cuTe core. Steps 1, 3, 6, 7, 8 are mechanical.

---

## 6. Parity-first validation plan (carry over from Phase 1)

The capture fixtures from the original integration plan are still on disk:
- `swa_decode_capture.pt`
- `compressed_decode_capture.pt`
- `mtp_decode_capture.pt`

After Path A lands, before re-baking the image:

```python
# tests/integration/test_dsv4_native_kernel.py (NEW)
for fixture in fixtures:
    q, k_cache, page_table, lens, attn_sink, expected = load(fixture)
    # bypass (current production)
    out_bypass = _sparse_attention_reference_vectorized(...)
    # native cuTe with v_head_dim=512
    out_cute   = sparse_mla_decode_forward(..., v_head_dim=512, attn_sink=...)
    assert allclose(out_bypass, out_cute, rtol=1e-2, atol=1e-2)
    assert allclose(out_cute, expected, rtol=1e-2, atol=1e-2)  # vs Triton ground truth
```

Both comparisons must pass before any vLLM E2E perf run. The `expected` half is the harder bar — bypass-vs-cuTe parity catches kernel bugs, but doesn't catch the case where the bypass *itself* drifted from the Triton reference.

---

## 7. Alternative considered & rejected

**Path A0 — keep V=448 in the kernel, post-multiply rope contribution on the host.** Out: would need an extra GEMM `P @ K_rope` per call, eating exactly the gain we're chasing. The rope contribution is small (64/512 = 12.5% of V), but a separate launch + sync per layer per token is lethal at our token rate.

**Path A2 — use the existing (512, 448): 448 entry, drop rope from V in vLLM.** Out: numerically incorrect. DSV4-Flash's absorbed-MLA `W_O` was trained with V=full latent; truncating to nope-only changes model behavior.

---

## 8. Decision points for execution kickoff

### Pre-kickoff resolved

1. **BF16 PV MMA primitives already exist.** `bf16_mma_m16n8k16_f32` and `bf16_mma_m16n16k16_f32` ship in `b12x/cute/fp4.py:1065,1163`. A working composition pattern is in `kernel.py:682` (`_literal_qk_mma_into_sfrag_bf16`). Rope is **already staged into SMEM as BF16** (`_stage_kv_bf16_block`) for QK score computation today — the same SMEM block can be reused for PV-rope. **Step 2 drops from 1 day / ~150 lines to 0.5 day / ~80 lines** — compose existing primitives rather than write new MMA wrappers.

2. **`attn_sink` IS load-bearing — Step 9 is required, not optional.** Probed real DSV4-Flash checkpoint weights: 44 layers each with `attn_sink: (num_heads=64,)`. Absolute max per layer ranges **0.9 to 2.2**; absolute mean **0.34 to 0.80**. `exp(sink)` ≈ 2.7–7.4 added directly to softmax denominator. Empirically, the "DeepSeek-R1 articles" output-loop bug seen during integration was the symptom of skipping sink — confirming sink magnitudes drive >1% logit shift, well above any reasonable parity tolerance.

3. **Bypass stays as fallback** behind `B12X_MLA_REFERENCE=1`. Env flag mechanism already wired (`b12x.attention.mla.kernel.supports_sparse_mla_kernel` reads `B12X_MLA_FORCE_REFERENCE`).

### Updated total estimate

With Step 2 reduced by 0.5 day, total Path A is **~4.5–5 days** (down from 5–6).

---

## 9. Critical files (concrete pointers)

| Concern | File | Line |
|---|---|---|
| Shape gate | `b12x/attention/mla/traits.py` | 17, 79 |
| Output dim constant | `b12x/attention/mla/kernel.py` | 54 |
| PV MMA loop | `b12x/attention/mla/kernel.py` | 1479 (`_accumulate_pv_groups_from_p_frag`) |
| Output store | `b12x/attention/mla/kernel.py` | 1308 (`_store_output_group`), 1349 (chunked) |
| FP8 PV helper to mirror for BF16 | `b12x/attention/mla/kernel.py` | 1212 (`_literal_pv_mma_into_ofrag_fp8_raw_scaled`) |
| Top-level kernel runner | `b12x/attention/mla/kernel.py` | 2355 (`SparseMLAKernel`) |
| Split-decode entry | `b12x/attention/mla/split.py` | 213 (`_run_one_pass_split_sparse_mla_tile`) |
| Workspace dim | `b12x/attention/mla/workspace.py` | 223, 697, 1348 |
| vLLM dispatch (bypass to remove) | `vllm/model_executor/layers/deepseek_v4_attention.py` | 1239 (`_sparse_attention_reference_vectorized` call) |
| vLLM contract assert | `vllm/model_executor/layers/deepseek_v4_attention.py` | 1252 (`output_b12x.shape != (..., kv_lora_rank=512)`) |
| Reference sink machinery (port pattern from) | `b12x/attention/mla/reference.py` | softmax block in `_sparse_attention_reference_vectorized` |
