# SM120 Paged Attention: Planner Roadmap

## Goal

Build a serving-grade SM120 paged-attention backend in `b12x` for the real Qwen full-attention path:

- causal self-attention
- `page_size=64`
- `8q:1kv` GQA
- `d=256`
- `bf16` Q/K/V first
- `fp8` KV later
- CUDA-graph-first

This backend should be shaped like a serving system, not like a generic attention library API.

## Current State

The current `b12x` paged path is mathematically correct and already has a working split-KV prototype, but it is still materially behind FlashInfer `fa2` on the long-context Qwen matrix.

Current best observed policy from the benchmark:

- use `num_splits=1` for `k <= 512`
- use `num_splits=4` for `k >= 2048`

Even with that oracle policy, `b12x` is still about `2.09x` slower than `fa2` geomean-wise on the graph-captured Qwen-like benchmark.

That means the architecture is pointed in the right direction, but the runtime/kernel contract is still one generation behind a mature serving backend.

## Synthesis

The correct course is:

- take FA2 as the runtime donor
- take FA4 as the kernel-structure donor
- follow the researcher’s implementation order

### What FA2 Contributes

FA2, as exposed through FlashInfer, provides the right serving shape:

- explicit planner / wrapper state
- exact workspace ownership
- separate decode and prefill-or-extend surfaces
- split-KV as a first-class serving knob
- CUDA-graph-aware capture buckets

That is the right runtime model for `b12x`.

### What FA4 Contributes

FA4 contributes the right internal kernel organization:

- packed GQA as an internal layout
- in-kernel paged KV addressing
- factored scheduler / block-range / masking logic
- partial-output plus combine-kernel structure for split attention

That is the right kernel model for `b12x`.

Important nuance:

- SM90 FA4 is the donor for general kernel organization and packed GQA
- SM100 FA4 is the better donor for split-KV plus combine behavior
- the generic FA4 `interface.py` is not the runtime model to copy

### What The Research Guidance Contributes

The researcher’s order of operations is the right one:

1. split-KV first
2. split-bucket planner second
3. decode-specialized kernel third
4. constant-factor retuning after that

That ordering should be preserved.

## SM120 Interpretation

The runtime must respect what SM120 is actually good at:

- use TMA as the normal transport mechanism for paged KV
- treat `page_size=64` as the primary serving contract
- preserve packed GQA inside the CTA so one CTA can reuse one K/V stream for the whole `8q:1kv` group
- use warp-level MMA and the existing producer / consumer kernel structure as the main compute template
- avoid host-side gather bridges and host-side control flow inside captured regions

The two important consequences are:

- the planner should own split policy and workspace shape
- the kernel should not infer launch semantics from packed tensor layout

## Decisions Already Made

These are now part of the intended design and should not be revisited casually:

- keep packed GQA as the intra-CTA representation
- do not recover parallelism by disabling packed GQA
- support `page_size=64` first and reject other page sizes in the primary path
- support `bf16` KV first and add `fp8` KV only after the serving architecture is stable
- use true in-kernel paged loads only
- use discrete split buckets, not one graph with masked-off work
- reject unrealistic corner cases instead of graph-hostile host workarounds

## Non-Goals

These are explicitly out of scope for the first serving-grade path:

- generic FlashAttention compatibility
- training and backward
- arbitrary page sizes
- dense non-causal attention as the primary optimization target
- local attention, block sparsity, `score_mod`, `mask_mod`, or learnable sinks
- split-KV only in eager mode
- page-gather fallback paths for graph mode

## Real Serving Contract

The runtime contract should be narrow and explicit:

- `q: [total_q, q_heads, d]`
- `k_cache: [num_pages, page_size, kv_heads, d]`
- `v_cache: [num_pages, page_size, kv_heads, d]`
- `page_table: [batch, max_pages_per_request]`
- `cache_seqlens: [batch]`
- `cu_seqlens_q: [batch + 1]`
- `workspace: exact-shape caller-owned buffers`
- `plan: exact-shape runtime policy and launch contract`

The primary shape buckets are:

- decode: `q_len = 1`
- extend: `q_len` small, with Qwen-like observed `q_len = 6`

## What A Planner Means Here

The planner turns a serving problem description into a reusable launch contract.

The planner should decide:

- which kernel family to use
- which tile family to use
- which split bucket to use
- which scratch buffers are required
- which metadata tensors must exist
- which launch dimensions are fixed for capture

The planner should not do per-token or per-request work inside the captured region. The planner should instead freeze structure once and leave only metadata updates for replay.

### Planner Inputs

- mode: decode or extend
- `bs`
- `max_q`
- `max_k`
- `num_q_heads`
- `num_kv_heads`
- `head_dim`
- `page_size`
- dtypes
- causal flag
- graph mode flag

### Planner Outputs

- kernel family identifier
- tile config
- split bucket
- exact output and LSE shapes
- exact partial-output and partial-LSE shapes
- exact reducer scratch shapes
- exact metadata buffer shapes
- compile key / cached executable selection

### Suggested `b12x` Objects

- `AttentionPlanKey`
- `AttentionPlan`
- `AttentionWorkspace`
- `AttentionWorkspacePool`
- `AttentionPlanCache`

The separation should mirror the mature MoE path:

- workspaces own buffers
- plans own policy and launch structure
- pools cache exact-shape workspaces
- runtime entrypoints resolve a plan plus a workspace and then run

## Proposed Runtime Shape

The planner-backed runtime should expose two serving surfaces:

- `paged_decode(...)`
- `paged_extend(...)`

Each surface should have:

- an eager path
- a graph-capture path
- a graph-replay path

Each surface should use the same underlying concepts:

- exact-shape workspaces
- split buckets
- compiled kernels cached by specialization key
- graph-stable metadata updates

## Proposed Kernel Families

### Family A: Main Paged Kernel

Purpose:

- extend
- decode fallback
- shared serving kernel for larger packed-M

Characteristics:

- TMA paged loads
- packed GQA
- split-KV support
- writes either final output or split partials

This is the direct evolution of the current `forward.py` kernel.

### Family B: Combine Kernel

Purpose:

- reduce split partial outputs

Characteristics:

- compiled kernel, not Python/Torch reduction
- consumes partial `O_i` and `LSE_i`
- produces final `O` and `LSE`

This should follow the FA4 split-combine idea, but be implemented in the `b12x` runtime style.

### Family C: Decode Micro-M Kernel

Purpose:

- optimize `q_len=1`

Characteristics:

- smaller effective M
- separate warp partitioning
- separate producer / consumer balance
- same paged-KV and split-KV planner contract

This should be added after split-KV and the planner are stable.

## Immediate Problems To Solve

### Problem 1: Plannerless Runtime

Right now the runtime still mixes:

- structural decisions
- workspace allocation logic
- split policy
- metadata handling
- launch

That is the main architectural gap relative to FA2.

### Problem 2: Split-KV Still Uses A Runtime-Level Reducer

The current split prototype proves the architecture, but the reduction still lives in the integration layer rather than as a compiled kernel. That is a useful prototype, not the end state.

### Problem 3: Decode Still Shares The Extend Kernel Family

This keeps the architecture simple, but it is not the final performance shape for `q_len=1`.

## Roadmap

## Phase 0: Stabilize The Current Baseline

Objective:

- lock the current benchmark, correctness tests, and split prototype in place

TODO:

- keep the FlashInfer `fa2` comparison in `benchmarks/benchmark_paged_attention.py`
- keep graph-capture plus `100x` replay as the benchmark method
- keep the Qwen-like matrix:
  - `bs=8`
  - `q in {1, 6}`
  - `k in {64, 512, 2048, 8192}`
  - `page_size=64`
  - `8q:1kv`
  - `d=256`
- fix any new regressions in the contiguous runtime path before attention work fans out further

Exit criteria:

- the benchmark remains stable and reproducible on `CUDA_VISIBLE_DEVICES=7`
- paged correctness and graph tests remain green

## Phase 1: Introduce The Planner Layer

Objective:

- separate structural decisions from per-call execution

TODO:

- define `AttentionPlanKey`, `AttentionPlan`, `AttentionWorkspace`, and `AttentionWorkspacePool`
- refactor `b12x/integration/attention.py` so entrypoints resolve:
  - plan
  - workspace
  - metadata views
  - compiled kernels
- make exact-shape workspaces explicit and caller-owned for graph mode
- mirror the local MoE ownership discipline instead of FA-style implicit allocations
- move split policy selection out of the raw launch path and into the plan

File touchpoints:

- `b12x/integration/attention.py`
- `b12x/integration/tp_moe.py` as the local donor
- tests under `tests/`

Exit criteria:

- workspace and plan are distinct objects
- graph capture no longer depends on implicit allocation paths
- launch code consumes a pre-resolved plan instead of deciding everything inline

## Phase 2: Finish The Control-Plane Cleanup

Objective:

- make scheduling use logical work dimensions, not packed storage layout

TODO:

- compute scheduler quantities from explicit logical values:
  - `num_kv_heads`
  - `qhead_per_kvhead`
  - `logical_q_rows`
  - `logical_num_m_blocks`
- audit every place where packed tensor shapes leak into launch semantics
- keep `pack_gqa_layout(...)` purely as a layout transform

File touchpoints:

- `b12x/attention/forward.py`
- `b12x/attention/pack_gqa.py`
- `b12x/attention/tile_scheduler.py`

Exit criteria:

- packed GQA no longer silently changes control-plane head cardinality
- split heuristics and decode-vs-extend logic can reason in logical units

## Phase 3: Convert Split-KV From Prototype To Planned Feature

Objective:

- make split-KV a first-class planner choice

TODO:

- keep the existing split partitioning logic in `block_info.py`
- define split buckets in the planner, starting with `{1, 2, 4, 8}`
- store split bucket in the plan
- make workspace sizing depend on split bucket
- ensure graph capture is done separately per split bucket
- remove any assumption that one workspace shape can cover all split counts in one graph

File touchpoints:

- `b12x/integration/attention.py`
- `b12x/attention/forward.py`
- `b12x/attention/block_info.py`

Exit criteria:

- split bucket is a plan property, not an ad hoc runtime override
- split and non-split paths share one serving contract
- graph capture works cleanly for each supported bucket

## Phase 4: Add A Compiled Combine Kernel

Objective:

- replace the runtime-level reduction with a true kernel stage

TODO:

- define a compiled combine kernel for partial `O_i` and `LSE_i`
- preserve the current per-split state contract if finalized `LSE` is sufficient
- otherwise expose the minimum additional state needed for exact reduction
- make combine part of the plan:
  - no combine if `num_splits == 1`
  - launch combine if `num_splits > 1`

File touchpoints:

- `b12x/attention/forward.py`
- `b12x/attention/softmax.py`
- `b12x/integration/attention.py`
- new combine-kernel file if needed

Exit criteria:

- no Python/Torch reduction remains on the critical path
- split combine is capture-safe
- split-KV becomes a fully compiled path

## Phase 5: Add Split-Bucket Heuristics

Objective:

- choose split count in a way that matches serving and CUDA graph requirements

TODO:

- implement a first page-count heuristic
- implement a second occupancy-oriented heuristic if needed
- choose the smallest power-of-two split count that creates enough CTAs while keeping enough pages per split
- keep the heuristic planner-only and graph-safe

Suggested first rule:

- small page counts stay on `1`
- medium page counts move to `2`
- long page counts move to `4`
- very long page counts can move to `8`

Exit criteria:

- short-context latency does not regress like forced split `4`
- long-context slope improves materially over split `1`
- split selection is deterministic from static inputs

## Phase 6: Add A Decode-Specialized Kernel Family

Objective:

- stop forcing `q_len=1` through a shared extend-oriented kernel shape

TODO:

- define a separate decode kernel family with smaller effective M
- keep the same planner contract and same workspace discipline
- let the planner choose between:
  - main paged kernel
  - decode micro-M kernel
- keep the combine contract shared if possible

Exit criteria:

- decode outperforms the main paged kernel on `q_len=1`
- extend remains on the main kernel unless data justifies a second extend family

## Phase 7: Tune Constants Only After The Architecture Stabilizes

Objective:

- reduce constants after the main slope problem is solved

TODO:

- test `Q_in_regs=True` on the serving path
- test `num_stages=2` for paged KV
- re-evaluate tile choices after split buckets and decode specialization are in place
- profile producer / consumer utilization with the new split policy

Exit criteria:

- improvements stack on top of the split-KV slope improvement
- no regression in graph capture or correctness

## Phase 8: Build The SGLang Backend Shim

Objective:

- connect the planner-backed `b12x` runtime to SGLang without waiting for generic API changes

TODO:

- implement a dedicated SGLang attention backend for `b12x`
- use SGLang’s existing backend lifecycle:
  - `init_cuda_graph_state(...)`
  - `init_forward_metadata(...)`
  - `init_forward_metadata_capture_cuda_graph(...)`
  - `init_forward_metadata_replay_cuda_graph(...)`
- let the `b12x` backend own its internal planner
- keep fallback to FlashInfer for unsupported cases

The key point:

- SGLang does not provide a generic planner abstraction
- `b12x` should therefore bring its own planner and expose it through SGLang’s metadata-oriented backend interface

Exit criteria:

- Qwen full-attention layers can run through `b12x`
- unsupported modes fall back cleanly
- graph capture uses pre-resolved plans and exact workspaces

## Phase 9: Add FP8 KV Cache Support

Objective:

- reach the production-relevant Qwen serving path

TODO:

- define mixed `bf16` Q and `fp8` KV contract
- define per-page or per-tensor descale handling
- thread descale metadata through the planner and workspace
- preserve the same split and graph model

Exit criteria:

- `fp8` KV path matches reference within acceptable tolerance
- graph capture still works
- benchmark reports `bf16` and `fp8` results side by side

## Validation Plan

Every phase should be validated against the same core matrix:

- decode `q=1`
- extend `q=6`
- `k in {64, 512, 2048, 8192}`
- `bs=8`
- `page_size=64`
- `8q:1kv`
- `d=256`
- graph capture plus at least `100x` replay timing
- FlashInfer `fa2` comparison

Correctness should always include:

- `max_abs`
- cosine similarity
- split-vs-nonsplit equivalence
- graph replay equivalence

## Success Criteria

The architecture is on track when all of the following become true:

- long-context latency is no longer close to linear in page count
- split bucket policy beats any single fixed split count
- decode and extend each have an intentional kernel family
- the runtime is planner-backed rather than launch-heuristic-driven
- no Python/Torch reduction remains in split-KV
- SGLang integration fits cleanly through its backend lifecycle

The architecture reaches milestone 1 when:

- Qwen full-attention `bf16` paged decode and extend run through `b12x`
- CUDA graph works on supported buckets
- the long-context gap to `fa2` is substantially narrowed

## Concrete Next Tasks

- [x] Refactor `b12x/integration/attention.py` to introduce explicit plan and workspace objects.
- [x] Fix any contiguous-path regression caused by the new split plumbing so both contiguous and paged paths have a consistent contract.
- [x] Finish the logical-dimension cleanup in `b12x/attention/forward.py`.
- [x] Make split bucket a planner property instead of a raw runtime argument.
- [x] Replace the current runtime split reducer with a compiled combine kernel.
- [x] Add split-bucket capture and replay tests.
- [x] Add a page-count heuristic for `{1, 2, 4, 8}`.
- [x] Re-run the FlashInfer comparison and record the new `fa2/b12x` ratios.
- [ ] Start the decode-kernel design once the split-bucket planner is stable.
