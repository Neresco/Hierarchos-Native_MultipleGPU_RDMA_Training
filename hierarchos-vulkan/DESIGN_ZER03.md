# ZeRO-3 for the Vulkan Transformer Training Path — Design

Goal: train 27B-class Hugging Face transformers (e.g. Qwen3.5, 54 GB BF16) across a
small cluster of 16 GB GPUs (e.g. 4x RX 9060 XT + RX 7600 XT + 8060S iGPU = 6 ranks),
using **VRAM + RAM + SSD** as a slow-but-functional memory hierarchy.

ZeRO-3 shards **parameters + gradients + optimizer state** across ranks (each rank
holds `1/world` of every tensor) and reconstructs full tensors on demand via
all-gather, then scatters gradients via reduce-scatter. State that does not fit in
VRAM spills to host RAM / mmap'd SSD through the existing `OffloadStore`.

This document is the implementation plan. It records the ground-truth code layout,
the design, and the phased work order. It is updated as phases land.

---

## 1. Ground truth (from structural analysis)

Files: `src/transformer.rs` (129,874 lines), `src/bin/transformer_train.rs` (1,453),
`src/shared_lm_head.rs` (644), `src/param_offload.rs` (345), `src/distributed.rs` (1,264).

### Training orchestration
- Entry: `train_step` (80581) → … → `train_step_with_context_and_roc_bert_ids`
  (80774) → core `…_and_layer_count` which contains **forward and backward in one
  recorded `ComputeBatch`**.
- **Forward is layer-by-layer** (81581):
  `for layer_index in 0..self.layers.len() { …layers[layer_index].record_forward_with_cross_attention(…) }`.
- **Backward is layer-by-layer in reverse** (82141):
  `for layer_index in (0..self.layers.len()).rev() { …layers[layer_index].record_backward_with_cross_attention(…) }`.
- Loss + `grad_logits` computed inline (81867); terminal (lm_head/norm) backwards
  before the reverse layer loop.
- One `ComputeBatch` is recorded for the whole step and submitted once.

### Parameter storage & access
- **No `get_param(name)` getter.** Weights are read via direct `&GpuBuffer` field
  references baked into each dispatch call.
- **Descriptors are bound at dispatch time** (`VulkanKernel::record_dispatch`,
  vulkan.rs:7579), not compiled into the shader. So **swapping a `TrainableParameter`'s
  `values` buffer handle is transparent to the recorded tape**. This is the key
  enabler for sharding: we gather a param's shards into a full buffer, point `values`
  at it for the layer's compute, then restore the shard handle.
- Concrete reads: `VulkanLinear::record_forward_mode` (27695) binds
  `&self.weight.values`; attention projections (59060–59093); `VulkanLayerNorm::record_forward`
  (28589) binds `&self.weight.values, &self.bias.values`.

### `TrainableParameter` (26825)
Fields: `device`, `values: GpuBuffer`, `grad: Option<GpuBuffer>`,
`exp_avg/exp_avg_sq: Option<GpuBuffer>`, `len`, `decay`, `grad_accumulate` (shared-use),
`step_owner`.
Methods: `ensure_training_state` (26873, allocates grad/moments),
`record_step` (26963, AdamW dispatch reading grad+exp_avg+exp_avg_sq, writing values),
`offload_moments`/`restore_moments` (27048/27064, moves grad+exp_avg+exp_avg_sq through
the `OffloadStore` keyed by caller name; **`values` is never offloaded today**).

### Gradient flatten interface (the reduce-scatter hook)
- `read_all_gradients` (64170) / `write_all_gradients` (64189) concatenate every
  trainable's grad in `collect_trainables` order (+ shared embedding last).
  Order is **deterministic across ranks** — the exact interface a reduce-scatter plugs
  into.

### `collect_trainables` (63988)
Flat `Vec<&mut TrainableParameter>` in deterministic order:
`position_embedding → lm_head_bias → relative_attention_bias → per_layer_relative_attention_bias
→ lm_head → embedding_project → output_project → embedding_norm → final_norm →
for layer in self.layers { … per-layer params … }`.
The **shared embedding is NOT in `collect_trainables`** (appended last in
read/write_all_gradients).

### Chunked optimizer (feature A) — the template
`run_chunked_optimizer` (64094): `store.reset_cursor()`; loop chunks; per chunk
`restore_moments` → `record_step` into a fresh `ComputeBatch` → `submit` → `offload_moments`.
`enable_chunked_optimizer_offload` (64056) rejects LoRA/PEFT, clamps chunk ≥16 MB.
CLI wiring in transformer_train.rs:1199–1204 (`CHUNK_BYTES = 1 GiB`); distributed uses
rank-dedicated SSD files `{stem}-rank{rank}{ext}`; requires `--cpu-offload`.

### Communication primitives (`distributed.rs`)
- `ring_allgather_sharded` (1106): all-gather an f32 buffer from **unequal per-rank
  segments**; every rank ends with the full buffer. **= the ZeRO-3 parameter gather.**
- `ring_allreduce_sum` (992): phase 1 is reduce-scatter, phase 2 is allgather; result
  is the element-wise world sum. (Full allreduce — used by current data-parallel mode.)
- `exchange_blocks` (850): chunked f32 RDMA exchange powering both.
- `RingTransport::connect` (910): RDMA send→successor / recv←predecessor + TCP control.
- **Gap:** no standalone exported reduce-scatter. Add one for gradient scattering.

---

## 2. Memory model for the target (27B / 6×16 GB)

| tensor | total | per-rank shard (÷6) |
|---|---|---|
| params (BF16) | 54 GB | 9 GB |
| grads (BF16) | 54 GB | 9 GB |
| exp_avg (FP32) | 108 GB | 18 GB |
| exp_avg_sq (FP32) | 108 GB | 18 GB |
| **state** | **324 GB** | **54 GB** |

Per rank: 16 GB VRAM. Steady-state plan:
- **VRAM**: param shards (~9 GB) + one gathered layer buffer (~1.7 GB) + activations
  (small for batch 1 / short seq) + grad shard for the layer being reduced.
- **RAM / SSD (OffloadStore)**: moment shards (exp_avg + exp_avg_sq, ~36 GB) and the
  grad shards between steps.

So the offload is **mandatory** — even the param+grad shards (18 GB) exceed 16 GB.
The design therefore keeps param shards in VRAM and spills grad + moment shards to the
OffloadStore, reusing the chunked-optimizer restore/step/evict pattern.

---

## 3. Design

### 3.1 Sharding layout
Parameters are sharded **flat and contiguous by `collect_trainables` order** (the same
order used by read/write_all_gradients). Every trainable is assigned a rank
`r = index % world_size` (round-robin) OR a contiguous block. Round-robin gives even
sizes; contiguous is simpler for gather segments. Decision: **flat by collect order,
each param split into `world_size` equal slices, slice `rank` owned by rank.** (A
param smaller than `world_size` is replicated instead of sharded.)

For each `TrainableParameter`:
- `shard_len = ceil(len / world_size)` (owned slice length), `shard_offset = rank * shard_len`.
- `shard_values: GpuBuffer` (resident) — this rank's slice of the weights.
- `grad`/`exp_avg`/`exp_avg_sq` — this rank's slice only (in VRAM or offload store).
- `full_values: Option<GpuBuffer>` — temporary, populated during gather.

### 3.2 Parameter gather (forward + backward)
Before recording a layer's forward (and again before its backward), for each param in
that layer:
1. Build `full_values` (len elements).
2. `ring_allgather_sharded` over the flat gather buffer: each rank contributes its
   `shard_values` slice; every rank receives the full param.
   (Gather is done for the whole model in `collect_trainables` order, then per-layer
   slices of the flat buffer are bound — or per-layer flat buffers. Decide per phase.)
3. Point the param's `values` handle at `full_values` for the compute (descriptor
   binds at dispatch time → transparent).
4. After the layer's backward, restore `values` to `shard_values` and free
   `full_values`.

### 3.3 Gradient reduce-scatter
After the layer's backward, each rank has the **full** gradient for that layer's
params (redundant compute). Reduce-scatter so rank `r` keeps only slice `r`:
- New primitive `ring_reduce_scatter(transport, rank, world, data, owned_offset, owned_len)`:
  sums the full buffer across ranks, then each rank keeps its slice.
- The kept slice is written into the param's `grad` shard.

### 3.4 Sharded optimizer
Reuse the chunked-optimizer pattern on the **shards**:
- Each rank updates only its owned slices.
- `record_step` currently assumes the param owns its full `values`. For sharded step,
  the AdamW dispatch runs over `shard_len` elements on the owned slice.
- Moment shards are offloaded to RAM/SSD via the OffloadStore (restore→step→evict per
  chunk), exactly like feature A but over the shard.

### 3.5 Broadcast for init
At startup, rank 0 loads the full model; then each rank keeps its shards and the
full model is freed. (Or every rank loads only its shards directly from safetensors —
preferred, avoids the 54 GB host spike that overflowed RAM.) **Decision: load only the
owned shards per rank** to keep host RAM bounded.

---

## 4. Phases (incremental, tested)

- **Phase 0** — Design doc (this file). ✅
- **Phase 1** — `ring_reduce_scatter` primitive in distributed.rs + unit test. ✅
  - Added `ring_reduce_scatter` (distributed.rs) — phase 1 of ring_allreduce_sum;
    rank r ends with the reduced block `reduce_scatter_owned_block(r) = (r+1) mod n`.
  - Unit test `reduce_scatter_rotation_reduces_owned_block` verifies the rotation
    in memory for n ∈ {2,3,4,7}.
  - **Refinement (important):** the forward/backward are recorded into ONE
    `ComputeBatch` and submitted once. A gathered full-param buffer bound by a
    dispatch must stay alive until that batch executes. Therefore the model
    cannot hold all gathered layers at once (54 GB). The forward must be
    decomposed into **chunks of layers**: gather chunk params (CPU RDMA) → record
    chunk forward → submit chunk → free chunk params → next chunk. Same for
    backward (reverse chunks). Chunk size is bounded by
    `(VRAM - resident_shards - activations - grad_buffer)`. This is the
    "slow but functional" part. Phase 3-5 target this chunked gather.
- **Phase 2a** — Shard math foundation. ✅
  - New module `src/zero3.rs`: `ShardLayout` (equal contiguous slices), `chunk_ranges`
    (partition the flat buffer into VRAM-bounded chunks), `slots_in_chunk`, and
    `ShardStore` (one rank's slice). Unit-tested: slice partitioning, chunk
    coverage, gather round-trip, reduce-scatter into slices.
  - `ring_reduce_scatter` + `reduce_scatter_owned_block` in distributed.rs (Phase 1).
- **Phase 2b** — GPU integration: build the flat values buffer from `collect_trainables`
  order; shard it (each rank keeps slice `rank`); free each param's `values` to a
  1-element stub (reusing the offload_moments stub-swap pattern); store shards in
  the `OffloadStore` (RAM/SSD). Test: 27B shards materialize on 2 GPUs without the
  54 GB host spike.
- **Phase 3** — Chunked forward gather. ✅ DONE (single-rank parity verified,
  bit-identical losses + params; multi-chunk verified). Keep-all-through-backward:
  the prefix is gathered before the embedding forward, layer chunks are gathered
  inside the layer loop and kept materialized (no reshard until after the
  optimizer), then `zero3_reshard_all()` runs post-optimizer.
  - **CRITICAL FINDING:** the prefix gather MUST happen BEFORE the embedding
    forward is recorded. The embedding forward reads `position_embedding` and
    `embedding_norm`, which live in the prefix and are sharded. If the gather
    runs after the embedding forward is recorded, the forward reads the 1-elem
    stubs left by the previous reshard, and step ≥1 diverges. (The forward
    descriptor binds the buffer handle at record time, so a later gather that
    swaps `t.values` does NOT retroactively update the recorded embedding
    dispatch.)
  - The non-layer prefix = `[0, non_layer)` where `non_layer =
    zero3_layout.total() - sum(zero3_layer_flat_ranges)`. It is gathered once
    before the embedding forward and kept live through the terminal logits.
  - Layer chunks: `zero3::layer_chunks(layer_sizes, chunk_bytes/4)` groups the
    layers; inside the layer loop, entering a new group triggers
    `zero3_gather_chunk` on that group's flat range. Chunks are kept materialized
    (keep-all-through-backward) so the backward re-binds them directly.
  - `token_embedding_source`, `layer0_standard_input`, and the deepseek/qwen4
    initial-input buffers were refactored to owned `GpuBuffer` (Arc clones) so no
    `&self` borrow persists into the layer loop's `&mut self` gather calls; a
    `layer0_input` helper closure serves both the forward and backward loops.
  - **Memory note:** keep-all-through-backward means peak VRAM still holds all
    params after the forward completes. True per-chunk memory savings (gather →
    submit → reshard between chunks) requires the backward to re-gather, which is
    Phase 5.
- **Phase 4 — COMPLETE.** Full forward parity on a small model (2 GPUs, loopback):
  chunked gather forward logits match the non-ZeRO-3 single-rank logits. Implemented
  the `RingExchange` trait + in-memory `LoopbackRingExchange` so `ring_allgather_sharded`
  can be driven for every rank in one process without RDMA; added
  `zero3_two_rank_loopback_forward_parity` to transformer.rs. A small chunk budget
  (1 KB) forces the 2 tiny layers across multiple ring gathers, and both ranks match
  the un-sharded baseline loss exactly.
- **Phase 5 — reduce-scatter COMPLETE, backward re-gather deferred.**
  - **Done:** `ring_reduce_scatter` refactored to take `&mut dyn RingExchange` and
    `&mut [f32]`; added `LoopbackReduceScatter` (pre-computes the full n-rank
    simulation so a single rank can be driven in one thread); added
    `zero3_reduce_scatter_gradients` which reduce-scatters the shardable prefix
    of the flat grad buffer and ÷world the owned block to match the AllReduce
    averaging convention. Grad parity test
    `zero3_two_rank_grad_reduce_scatter_parity` verifies each rank's reduce-scatter
    owned slice equals the corresponding block of a full AllReduce.
  - **Deferred (5b):** backward chunked gather — reshard weights after forward,
    re-gather per chunk during backward — requires splitting the single
    ComputeBatch into forward-submit + per-chunk backward-submit. This is the VRAM
    win for 27B (currently keep-all-through-backward holds all weights). High
    risk in the 130k-line train_step; do as a focused sub-task. The reduce-scatter
    primitive is wired into the train step by Phase 6 (sharded optimizer).
  - **Note:** `zero3_reduce_scatter_gradients` is a standalone primitive (only
    called by the parity test) — it is NOT yet wired into `train_step`, because
    the current optimizer steps all params and would use stale grads for
    non-owned blocks. Phase 6 wires it in alongside the sharded optimizer.
- **Phase 6 — COMPLETE.** Sharded optimizer over owned slices + moment offload;
  step parity vs single-rank AllReduce on a small model (`zero3_two_rank_step_parity`).
  `zero3_run_optimizer` sets `step_owner=false` for non-shard params and only steps the
  params overlapping `layout.slice_range(rank)` (~50% optimizer FLOP savings at world=2).
- **Phase 7 — COMPLETE.** Moment offload for non-owned ZeRO-3 params (`fd81c01`):
  `zero3_moment_store` + `enable_zero3_moment_offload`. After the first optimizer step,
  non-owned params' AdamW moments are evicted to the OffloadStore (RAM/SSD) via
  `offload_moments`, freeing VRAM for the owned params. `zero3_moment_store_stats()`
  exposes offloaded bytes/param count.
- **Phase 8 — COMPLETE.** 6-rank end-to-end smoke with SSD (mmap) moment offload and
  loss + checksum parity (`zero3_six_rank_loopback_end_to_end_smoke`). A scaled
  `medium_gpt2_vulkan_graph` (256 vocab, 64 hidden, 8 layers) drives the full
  train_step path (chunked forward gather, gradient reduce-scatter, sharded
  optimizer, SSD moment offload) across world=6 in a single loopback process, with a
  tiny 1 KB gather budget forcing multiple ring allgathers. Asserts: per-rank loss
  matches a single-rank baseline (< 1e-5), each rank's owned slice matches the
  baseline checksum, shard size equals the layout slice and is a strict subset of the
  full buffer, and the moment store holds non-zero offloaded bytes after step 1 (SSD
  path exercised). A real 6-GPU + 54 GB 27B hardware smoke is deferred to Phase 9
  (CLI wiring) — the loopback verifies the same sharding/ring/offload logic in-process.
  - **Loopback reduce-scatter sim fix:** `LoopbackTrainStepRing` previously inferred the
    reduce-scatter total from `world * first_send_block_len`, which only matches when the
    total divides evenly by world (true for the Phase 6 world=2 model, broken for the
    Phase 8 world=6 medium model → `recv len` mismatch). It now seeds `sim` from
    `self.full` (same length/structure as the gradient buffer). This is safe because in
    the redundant-backward loopback the only block the test verifies is the shard block
    (`slice_range(rank)`), which reduce-scatter never overwrites — the buggy value
    seeding only ever landed in non-shard blocks that are discarded by the sharded
    optimizer.
- **Phase 9** — CLI/WebUI wiring (`--zero3`, `--moment-offload`), README, cleanup,
  and a real-hardware 6-GPU 27B smoke.

Each phase builds and runs a parity test on a small model (Qwen3-0.6B) before moving
on, so regressions are caught early.

---

## 5. Risks / open questions
- **Activation tape** is resident for all layers during backward (prime recompute /
  offload target) — with short seq it is small; revisit for long seq.
- **Shared embedding** is outside `collect_trainables` — must be sharded/gathered
  separately (it is the lm_head when untied, or the token embedding when tied).
- **ALBERT canonical-grad zeroing** was not located — verify accumulation semantics
  before relying on them (Qwen3.5 is not ALBERT, so low priority).
- **Communication volume** (~2x param gather + grad reduce-scatter per step) makes
  27B slow on a 3 GB/s link — acceptable per the "slow but functional" requirement.
- **Descriptor bind-at-dispatch-time** must hold for every kernel that reads a weight
  (verified for the sampled kernels; re-verify for attention/norm/MLP in Phase 3).
