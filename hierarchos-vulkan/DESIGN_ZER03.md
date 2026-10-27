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
- **Phase 1** — `ring_reduce_scatter` primitive in distributed.rs + unit test on 2 ranks.
- **Phase 2** — Shard layout: extend `TrainableParameter` with shard fields; load
  only owned shards; `collect_trainables`-ordered flat shard buffer; expose
  `owned_slices`. Test: 27B shards load on 2 GPUs without RAM overflow.
- **Phase 3** — Parameter gather: `ring_allgather_sharded` + transparent `values`
  swap for one layer's forward. Test: forward logits match single-rank on a small model.
- **Phase 4** — Per-layer gather across the whole forward + activation tape; verify
  forward parity.
- **Phase 5** — Backward gather + gradient reduce-scatter; verify grad parity vs
  AllReduce.
- **Phase 6** — Sharded optimizer over owned slices + moment offload; verify step
  parity vs single-rank AllReduce on a small model.
- **Phase 7** — Offload grad/moment shards to RAM/SSD; VRAM accounting.
- **Phase 8** — 27B end-to-end smoke (6 GPUs, SSD offload), loss + checksum parity.
- **Phase 9** — CLI/WebUI wiring (`--zero3`), README, cleanup.

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
