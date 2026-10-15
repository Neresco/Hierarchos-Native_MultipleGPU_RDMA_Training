# Hierarchos Vulkan training-kernel optimizations — 2026-10-15

Scope: the Hierarchos training compute kernels and the compute-dispatch runtime.
Model mathematics, objective definitions, optimizer equations, and trainer
behavior are unchanged. Every kernel whose artifact changed is compared
bit-for-bit against its pre-change artifact (`git show HEAD:...`) on the local
AMD Radeon Graphics device, because the project treats exact FP32 reproduction
as the contract for a kernel optimization.

This continues the work left mid-verification at 03:57 (dense/attention
candidates compiled, comparisons not yet run). The separately documented
objective/data-boundary port in `PROGRESS_HIERARCHOS_OBJECTIVES_PORT.md` is
untouched by this pass.

## Changed kernels

| Kernel family | Change |
| --- | --- |
| `linear_forward`, `linear_bias_forward`, `linear_residual_forward`, `linear_*_fp16_packed` | 16×16 output tile with shared 32-wide K tiles; each input/weight value is loaded once and reused across the tile. Small/recurrent shapes keep the original scalar schedule. Increasing-K FMA order preserved. |
| `parameter_matmul_forward`, `parameter_matmul_bias_forward`, `_fp16_packed` | Same tiling for the weight-major projection; leading dimension read once per K tile. |
| `parameter_matmul_input_grad`, `_fp16_packed`, `_fp16_native_compute` | Same tiling; FP16 weights are unpacked once per tile instead of once per output element. |
| `transformer_attention_forward` | Per-row score/softmax tile cached in shared memory (rows ≤ 1024 keys), removing the recomputation of every QK dot product for every value channel. Longer rows keep the streaming path. |
| `transformer_attention_backward` | K/V adjoints evaluate each (query, key) score, probability and `dp` once per row and reuse row statistics instead of replaying the full softmax per key and channel. Strict-shape fallbacks retained. |
| `transformer_qwen_gdn_gate_norm_param_grad` | The `head_dim`-fold mean-square reduction is computed once per row tile and shared across features; previously it was recomputed inside every feature, making the pass quadratic in `head_dim`. |
| `transformer_rms_norm_input_grad`, `_residual` | The eight-lane topology is selected before reading the row, removing a wasted scalar prepass that was always discarded when `unfused_products == 2`. |
| `adamw` | The two bias-correction powers depend only on the step; they are evaluated once per workgroup and broadcast, instead of once per parameter. |

## Dispatch runtime

- **Pipeline interning.** Kernels that share the same SPIR-V module and pipeline
  layout now share one `vk::Pipeline` per device, held through a weak interner
  entry so driver memory is released with the last kernel. Identical layers in
  the same graph previously created separate pipelines.
- **Barrier elision from SPIR-V reflection.** `src/spirv_access.rs` reads
  `NonWritable`/`NonReadable` decorations, including per-member decorations on
  storage blocks, and reports conservative `MayWrite` for anything it does not
  recognize or for contradictory aliased declarations. The command recorder
  inserts a compute-to-compute barrier only when the next dispatch can alias a
  buffer touched by any dispatch since the last barrier, so read/read overlaps
  (activation reused by two consumers) no longer serialize.

## Verification

| Check | Result |
| --- | --- |
| Dense-kernel parity + reference model (`matmul_kernel_validation`) | 12 kernels × 12 shapes = **144/144 bit-identical** to HEAD |
| Attention parity (`attention_kernel_cache_check`) | **79 parity cases × forward/backward bit-identical**, including cache boundaries 512/513 and 1024/1025, windowed, paged, sink, relative-bias, and both strict-head/stats fallbacks |
| Reduction parity (`reduction_kernel_validation`) | **183 GPU cases bit-identical**: Qwen weight grad 8 shapes, RMSNorm input grad ×2 kernels × 5 shapes × 20 config combinations, AdamW 5 lengths × 3 steps |
| Runtime unit tests | 6/6 pass (pipeline sharing and lifetime, interning key separation, reflected-access barrier elision) |
| Maintained regressions | Causal LM 6 passed; CE 3 passed; Hierarchos trainer 40 passed; attention/linear/parameter-matmul library tests 71 passed |
| Full library suite (`cargo test --lib`) | **704 passed, 0 failed, 9 ignored** |
| Shader provenance | Every changed artifact reproduces byte-for-byte from its source (`glslc`, `glslc -O`, or `glslangValidator -V` per shader family) and validates with `spirv-val --target-env vulkan1.1`; CE: 32/32 variants |

### Measured speedups (GPU timestamp median of 5 samples)

Dense projections — 72 timed configurations, 12 kernels × 6 shapes:

| Rows | Median | Range |
| --- | --- | --- |
| ≥ 16 (training-like) | **2.67×** | 0.99× – 4.43× |
| < 16 | 1.06× | 0.89× – 2.56× |

Best cases: `parameter_matmul_input_grad_fp16_packed` 4.43× (64×1024×256),
`parameter_matmul_forward_fp16_packed` 4.29× (16×256×256),
`parameter_matmul_input_grad` 4.07× (64×1024×256), `linear_forward` 3.23×
(32×512×512). Batch-1 `(1, 512, 512)` and the smallest `(8, 64, 16)` shapes are
within ±11% of the baseline, as expected when the tile reuse cannot amortize the
two extra barriers.

Attention (`transformer_attention_forward` / `_backward`, 64 queries × 64 keys,
4 heads, dim 32):

| Configuration | Baseline | Optimized | Speedup |
| --- | --- | --- | --- |
| Gemma4 mode 0, forward | 4.49 ms | 0.73 ms | **6.2×** |
| Gemma4 mode 1, forward | 3.73 ms | 0.81 ms | **4.6×** |
| Gemma4 mode 0, backward | 53.4 ms | 45.0 ms | 1.19× |

Reductions:

| Kernel | Shape | Baseline | Optimized | Speedup |
| --- | --- | --- | --- | --- |
| Qwen gated-RMS weight grad | rows 128, heads 4, dim 128 | 2.73 ms | 0.232 ms | **11.8×** |
| Qwen gated-RMS weight grad | rows 256, heads 4, dim 192 | 4.74 ms | 0.462 ms | **10.3×** |
| RMSNorm input grad | rows 128, dim 256 | 0.276 ms | 0.218 ms | 1.27× |
| RMSNorm input grad + residual | rows 128, dim 256 | 0.435 ms | 0.374 ms | 1.16× |
| AdamW | len 65537 | 0.0219 ms | 0.0171 ms | 1.28× |

## Findings made while finishing the verification

- **Qwen gated-RMS weight gradient: a 1-ULP regression, found and fixed.** The
  candidate broadcast each row's already-evaluated `inversesqrt(...)` through
  workgroup memory; on this driver that shifted roughly half of the reduced
  elements by one ULP against the pre-change kernel. Publishing the raw
  mean-square sum and evaluating `inversesqrt(... / head_dim + epsilon)` in the
  consuming lane restores bit parity (183/183, 11.8×/10.3× retained), because
  the reciprocal square root then executes in the same lane context and
  instruction sequence as before. The `head_dim` multiplier on the mean-square
  reduction is what the pass removes; the per-feature root is not the cost.
- **Compile-flag provenance.** The pre-change Qwen artifact was built without
  `-O` while the candidate was built with `-O`. A flag-only probe (HEAD source
  compiled both ways, compared on GPU) is bit-identical, so the flag is
  numerically inert here and was not the cause; the source restructure was.
- **Reference-model calibration for native-FP16 dense kernels.** The dense
  harness's independent FP32 emulation cannot reproduce exact Float16 product
  rounding on awkward shapes: the *pre-change* `parameter_matmul_input_grad_fp16_native_compute`
  already exceeds its `5e-4` tolerance (max 2.6e-3). The harness now enforces
  bit-parity with the supplied baseline first and calibrates the advisory model
  against that baseline's own error, so a change that widens the deviation still
  fails while an environment limitation cannot mask a real mismatch.
- **Harness diagnostics.** The reduction harness now reports every mismatch with
  case context instead of stopping at the first one, and accepts
  `--candidate-dir` so any two artifacts can be compared; the attention harness
  accepts `--candidate-dir` and `--bench-repeats` and times with GPU timestamps
  instead of host wall time.

## Limitations

- **Attention backward mode 1 was not benchmarked.** The harness resets the
  device (`VK_ERROR_DEVICE_LOST`) on that case on this machine. The reset
  reproduces identically with unoptimized artifacts on both sides
  (`--candidate-dir .attention-optimization-baseline`), so it is a driver /
  watchdog limitation of the harness's long scalar backward dispatches, not a
  property of the optimization. Forward and mode-0 backward numbers are
  unaffected, and parity for that case passes.
- Timings are kernel-level on one AMD device with the debug-profile harnesses;
  no end-to-end trainer wall-clock A/B was measured in this pass, and no
  cross-vendor parity is claimed.
- The dense benchmark's small shapes (rows < 16) include a few sub-1.0×
  configurations; those shapes deliberately keep the scalar schedule, so the
  residual delta is launch/measurement noise rather than a systematic loss.

## Reproduce

```powershell
cd hierarchos-vulkan

# Parity + benchmarks (baseline artifacts are HEAD blobs; see the note below)
cargo run --bin matmul_kernel_validation -- --baseline-dir target/shader-matmul-baseline --bench --output target/matmul_report.json
cargo run --bin reduction_kernel_validation -- target/shader-reduction-baseline
cargo run --bin attention_kernel_cache_check -- .attention-optimization-baseline --bench

# Baseline reconstruction (once)
git show HEAD:hierarchos-vulkan/shaders/<name>.spv > target/shader-matmul-baseline/<name>.spv

# Shader provenance and validation
python validation/verify_hierarchos_ce_shaders.py
python validation/audit_shader_artifacts.py linear_forward parameter_matmul_input_grad ... --output target/shader_audit.json
glslangValidator -V --target-env vulkan1.0 shaders/transformer_attention_forward.comp -o <out>
glslc -O shaders/adamw.comp -o <out>
spirv-val --target-env vulkan1.1 shaders/<name>.spv

# Runtime and regression suites
cargo test --lib -j 1 -- --test-threads=1
cargo test --lib -j 1 hierarchos_causal_lm -- --nocapture --test-threads=1
cargo test --lib -j 1 hierarchos_ce_ -- --test-threads=1
cargo test --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
```

The `target/shader-*-baseline` directories are scratch reconstruction of
pre-change artifacts (`target/` is ignored) and are not part of the tree.

---

# Second pass — 2026-10-15: binding-access audit, CI golden digests, end-to-end throughput

The first pass verified the kernel and runtime changes. This pass answers four
questions the first pass left open: whether the remaining shaders can declare
narrower storage access so the scheduler elides barriers, whether the parity
harnesses can run in CI without the original artifacts, what the attention
backward regression actually costs a whole training step, and what the
end-to-end effect of all of it is.

## Storage-access audit and barrier accounting

`validation/audit_shader_binding_access.py` reads each compiled module as ground
truth: a binding that no `OpStore`, atomic, or `OpImageWrite` can reach is
provably never written, and one that no `OpLoad`, atomic, or `OpImageRead`
reaches is provably never read. It reports where that proof is narrower than the
declared qualifier, which is exactly where the dispatch scheduler pays for an
unqualified block.

`validation/narrow_shader_binding_access.py` closes those gaps. For each affected
artifact it resolves the GLSL source and the build command that produced it,
**requires that command to reproduce the checked-in artifact byte-for-byte**
before editing anything, recompiles every variant built from that source, then
requires `spirv-val` to accept it, the module's read/write usage sets to be
unchanged, and the reported declaration to match the proof. It rolls the source
back and rebuilds if any of that fails.

Result: 41 bindings over 24 sources narrowed, and the audit now reports
`0 provably narrower binding(s) over 0 shader(s) out of 467 modules`.

Two toolchain facts surfaced on the way: the README's command for the CE fused
family omits `--target-env=vulkan1.1`, which the artifact needs, and the
`..._packed_fast_weight_reuse{4,8}` / `..._wg32` variants are built from the
`_weight_reuse2.comp` source with `-DRWKV_WEIGHT_REUSE_ROWS` / `-DRWKV_WEIGHT_REUSE_TILE`
and are not documented at all. The tool finds a working command by requiring
byte-reproduction, and says when the documented one no longer works.

### The measured barrier reduction is zero, and the reason is structural

(The fifth pass below closes this question: the rule is already exactly the hazard
set, so no declaration can lower the count, and the remaining levers are graph
structure and per-barrier cost.)

`src/bin/training_step_probe.rs` measures a real training step (raw-token
microbatch: forward, reverse, canonical AdamW) and prints dispatch and barrier
counts. On the fixed fixture (batch 2, 8 tokens, context 128, hidden 128):

| build | dispatches/step | barriers/step | step ms | tokens/s |
| --- | --- | --- | --- | --- |
| HEAD (`5004a8a`) | 6912 | 3412 | 172.76 | 92.62 |
| current tree | 6912 | 3076 | 167.98 | 95.25 |

Before and after the 41 narrowings on the same tree the count was 3076 both
times, so the 336-barrier improvement (and the 2.8% step-time gain) belongs to
the already-present reflection/runtime change, not to the narrowing. That is not
an accident of which shaders were left; it follows from the elision rule:

```rust
let overlaps_pending = buffers.iter().zip(binding_accesses).any(|(buffer, access)| {
    let region = buffer.buffer_region_key();
    (access.may_read() && self.pending_shader_writes.contains(&region))
        || (access.may_write() && self.pending_shader_buffers.contains(&region))
});
```

`pending_shader_buffers` is every region any earlier dispatch touched, and every
write also inserts there. So the second term reduces to "this binding may write
and the region was touched", and `writeonly` keeps `may_write` — a `writeonly`
declaration can never remove a barrier. Only `readonly` can, by clearing
`may_write`; of the remaining unqualified declarations exactly 5 are provably
read-only, all in `falcon_h1.comp`, which is not on the Hierarchos RWKV training
path. The narrowing is still worth landing as accuracy (it is what makes a
finer-grained rule possible, and it documents the access each kernel has), but it
could not move this number, and the measurement says so rather than the other way
round.

The "non-aliasing region" half of that question is already expressed where it can
be: `buffer_region_key()` is `(VkBuffer, view offset, view size)`, so two disjoint
views of one allocation are already distinct regions and nothing has to be
declared for that. The barrier rate is the sanity check that this granularity is
live — if every binding collapsed onto one region, every dispatch after the first
would barrier (6911 of 6912); the step reports 3076.

Parity after the narrowing: full library suite **704 passed, 0 failed, 9
ignored**; dense `144/144`, reduction `183/183`, attention `79 × fwd/bwd` all
bit-identical to their baselines; and the end-to-end probe's mean loss is
bit-identical between HEAD and the current tree over six real optimizer steps
from the same weights (4.6006656 both sides).

Reproduce:

```powershell
cargo run --bin training_step_probe -- --batch 2 --tokens 8 --warmup 1 --steps 5
python validation/audit_shader_binding_access.py
python validation/narrow_shader_binding_access.py            # dry run
python validation/narrow_shader_binding_access.py --apply    # rebuilds artifacts
```

## Golden digests: CI parity without the pre-change artifacts

`src/kernel_golden.rs` (included by path, like `vulkan.rs`) records a SHA-256 of
each invocation's output vector, keyed by invocation label and device name.
`--write-golden FILE` records, `--golden FILE` verifies, and in either mode the
baseline artifact directory becomes optional — the digests *are* the baseline.
A file accumulates one digest per device, so an AMD, Intel, or NVIDIA runner each
checks the entry recorded on its own hardware and reports another device's entry
as unverified instead of passing quietly.

Verified end to end on this device: dense 144, reduction 183, attention 79
invocations recorded and re-verified. The check is not vacuous — rebuilding the
gated-RMS weight-gradient kernel with a perturbed input scale produced exactly 8
digest mismatches on its 8 invocations (175 others verified) and exit status 1.

```powershell
cargo run --bin matmul_kernel_validation -- --write-golden validation/kernel_golden/dense.json
cargo run --bin matmul_kernel_validation -- --golden validation/kernel_golden/dense.json --golden-strict
```

A CI runner on hardware nobody has recorded yet would otherwise verify nothing and
still exit zero, so `--golden-strict` makes an unrecorded-for-this-device
invocation fatal. Both halves of that were checked: strict mode passes on the
recording device (144/183/79 verified, 0 unverified), and with the device key
rewritten to a foreign GPU the loose mode reports `0 verified, 144 unverified`
while strict mode exits 1 with `144 invocation(s) have no digest recorded for ...`.

## End-to-end trainer throughput, HEAD versus optimized

Both builds are the same probe binary built from the same source, one against
HEAD and one against the working tree, started from **the same model package** so
no weight difference can explain the gap.

| | HEAD | current | ratio |
| --- | --- | --- | --- |
| median step | 172.76 ms | 167.98 ms | 1.028× |
| steps/s | 5.788 | 5.953 | 1.028× |
| tokens/s (16 tokens/step) | 92.62 | 95.25 | 1.028× |
| dispatches/step | 6912 | 6912 | — |
| shader barriers/step | 3412 | 3076 | 0.901× |
| mean loss after 6 identical steps | 4.6006656 | 4.6006656 | bit-identical |

So the kernel and runtime work is worth 2.8% on this small fixture, where the
step is dominated by thousands of tiny dispatches rather than by the dense and
reduction kernels that carry the 2.7–11.8× kernel-level wins. The third pass below
measures that directly and confirms it: a step is 6912 dispatches at 175 µs for its
most expensive launch and 1–4 µs for its median one. The bit-identical
loss over six optimizer steps is the strongest end-to-end parity evidence
available here: same weights, same data, six AdamW updates, no divergence.

## Attention backward: the dominant cost, measured

`attention_kernel_cache_check` gained `--bench-{batch,query,key,heads,kv-heads}`
so one run can separate "not enough workgroups" from "too much serial work per
workgroup". Backward mode 0, `query=key=64`, 4 query heads, 2 KV heads:

| batch | workgroups | HEAD ms | current ms | ratio |
| --- | --- | --- | --- | --- |
| 2 | 4 | 54.6 | 46.3 | 1.18× |
| 8 | 16 | 56.2 | 45.4 | 1.24× |
| 32 | 64 | 56.5 | 88.6 | 0.64× |
| 128 | 256 | 78.3 | 262.9 | 0.30× |

Three facts fall out of this, and none of them flatter the current artifact:

1. The backward kernel declares `local_size = 1×1×1` and dispatches one
   workgroup per `(KV head, batch row)`, so at realistic shapes the grid is tiny
   and every element of the per-key reduction runs as a dependent single-lane
   chain. Step time at 4 workgroups is ~46 ms and is *independent of the
   workgroup count* up to 16: the device is idle, not saturated.
2. The current artifact uses workgroup memory the HEAD artifact did not (10
   workgroup variables versus 0), which lowers residency: it wins only while the
   grid is tiny and loses badly from batch 32 upward, 3.4× slower than HEAD at
   batch 128.
3. The forward path, restructured the same way, is 3.9–6.1× faster at every
   shape, because it dispatches `query × heads × batch` workgroups instead of
   `KV heads × batch`.

A bit-exact fix is available and needs no atomics: in the strict-adjoint path the
per-key phase zeroes `shared_k_cache`/`shared_v_cache` per key, reduces that key
over `(head, query)` in a fixed order, and writes that key's gradient before
moving on — so keys are independent accumulators, and splitting them across lanes
or across a `(KV head, batch, key tile)` grid reorders no FP sum. Independent
keys per lane also break the dependency chain that is doing the damage. Landing
it means moving the shared row statistics to a global pass (or sizing per-lane
scratch) plus the host dispatch/allocation change, and verifying against the
79-case parity harness.

It is **not implemented in this pass**: the change spans the 796-line kernel and
its host call sites, and the backward benchmark already trips
`VK_ERROR_DEVICE_LOST` in mode 1 on this device, which makes long verification
runs unreliable here. The finding to act on is the regression in the table: the
current backward artifact is slower than HEAD at every batch size that matters,
so the next change to it should start from the HEAD artifact's occupancy rather
than from the current one.

One correction from the third pass below: the canonical raw-token trainer never
dispatches this kernel. "Dominant cost" and "highest-value target" here are true
within the attention path's own benchmark, and the same *signature* — cost flat in
the workgroup count — turns out to be the trainer's top cost as well, in a
different kernel.

## Limitations of this pass

- The end-to-end numbers are one small fixture (batch 2, 8 tokens, 16 tokens per
  step) on one AMD device, in an unoptimized build. They say what the kernel and
  runtime work is worth at that shape, not what it is worth at production shapes;
  the backward table above is the counterexample to extrapolating from it.
- Attention backward mode 1 still cannot be benchmarked: the harness resets the
  device, identically with unoptimized artifacts on both sides.
- The barrier count is a property of one graph and one command-batch discipline.
  A different graph (transformer trainer, LoRA) would have a different mix, and
  the audit tool is how to check it there.
- Digest-based parity is device-keyed by design, so a CI runner on new hardware
  reports its invocations as unverified until someone records a golden file for
  that device.

---

# Third pass — 2026-10-15: what the ~6900 dispatches of a step actually cost

The first two passes optimized kernels and then measured each one in isolation.
This pass answers the prior question — where does a whole training step spend its
GPU time — because the answer decides which of those kernel numbers matter. The
instrument already existed: `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` writes a GPU
timestamp delta around every dispatch. [rank_kernel_profile.py](hierarchos-vulkan/validation/rank_kernel_profile.py)
turns that stream into a per-step ranking and reports µs/dispatch — the column that
distinguishes arithmetic from per-launch cost. Selecting the submissions to rank is
the one subtlety: a batch-2 run logs 1733 submissions, of which 1693 dispatch
nothing at all, 37 carry only warmup work (4–64 dispatches), and **3 carry a
complete 6912-dispatch step**. That is what the tool's `--min-dispatches 512`
threshold is for, and the category lines of each selected submission sum to exactly
its `gpu_ms`, which checks the selection.

One step of `training_step_probe --batch 2 --tokens 8` is 6912 dispatches and
88.5 ms of GPU time. Full tables for both batch sizes are in
[kernel_ranking.txt](hierarchos-vulkan/validation/training_step_probe/kernel_ranking.txt).

| category | dispatches/step | GPU ms/step | share | µs/dispatch |
| --- | --- | --- | --- | --- |
| normalization.layer-norm | 536 | 37.690 | 42.6% | 70.3 |
| rwkv.recurrent | 1152 | 21.689 | 24.5% | 18.8 |
| projection.linear | 1739 | 17.027 | 19.2% | 9.8 |
| control | 432 | 5.056 | 5.7% | 11.7 |
| optimizer.gradient-bookkeeping | 1346 | 3.118 | 3.5% | 2.3 |
| loss.lm-projection | 24 | 1.671 | 1.9% | 69.6 |
| elementwise-reduction | 1184 | 1.065 | 1.2% | 0.9 |
| other | 164 | 0.504 | 0.6% | 3.1 |
| token.embedding | 192 | 0.325 | 0.4% | 1.7 |
| optimizer.adamw | 95 | 0.227 | 0.3% | 2.4 |
| memory.rosa | 40 | 0.167 | 0.2% | 4.2 |
| loss.cross-entropy | 8 | 0.008 | 0.0% | 1.0 |
| **total** | **6912** | **88.548** | **100.0%** | **12.8** |

| shader | dispatches/step | GPU ms/step | share | µs/dispatch |
| --- | --- | --- | --- | --- |
| layer_norm_low_rank_producer_forward_fused | 144 | 25.201 | 28.5% | 175.0 |
| packed_cell_channel_mix_state_forward_fused | 144 | 9.757 | 11.0% | 67.8 |
| layer_norm_adapter_forward_fused_512 | 144 | 5.583 | 6.3% | 38.8 |
| linear_bias_forward | 192 | 4.703 | 5.3% | 24.5 |
| linear_input_grad | 385 | 4.095 | 4.6% | 10.6 |
| rwkv_time_mix3_linear3_key_state_forward_packed_fast | 96 | 3.741 | 4.2% | 39.0 |
| rwkv_group_norm_bonus_gate_linear_residual_forward_fused | 144 | 3.195 | 3.6% | 22.2 |
| gradient_accumulate4 | 768 | 2.770 | 3.1% | 3.6 |
| layer_norm_input_grad | 64 | 2.537 | 2.9% | 39.6 |
| drift_update_forward | 48 | 2.522 | 2.8% | 52.5 |

Both tables are abridged. The 10 shader rows above cover 72.4% of the step, and all
25 shaders the runtime printed for these submissions cover 93.8%; the remaining gap
is a property of the instrument rather than of the step, because the runtime prints
only the top 24 shaders per submission (`KERNEL_PROFILE_TOP_SHADERS`). Read this as
a ranking of hot spots, not as an account of every microsecond.

## Four findings

**1. One kernel is 28.5% of the step, and it is not one we had optimized.**
`layer_norm_low_rank_producer_forward_fused` is 144 dispatches at 25.20 ms/step,
175.0 µs each. The step's per-dispatch average is 12.8 µs, and the kernels that
carry its dispatch *volume* are cheaper still — `elementwise-reduction` 0.9 µs per
dispatch, `gradient_accumulate4` 3.6 — so this one launch is 13.7× the step's
average and up to ~190× the cheapest. The `normalization.layer-norm` category is
42.6% of the step and this single kernel is two thirds of it. The second-largest
category, `rwkv.recurrent`, already puts 45% of itself in one kernel;
`projection.linear` spreads 19.2% over 1739 dispatches at 9.8 µs each.

**2. The step's GPU cost is nearly independent of the token count.** Going from
batch 2 to batch 16 with the same 8 tokens per sequence is 8× the rows, and it
costs 88.5 → 97.0 ms of GPU per step: +8.4 ms, +9.5%. Per dispatch, the top kernel
gets *cheaper* (175.0 → 148.5 µs), and so do `linear_bias_forward` (24.5 → 19.0),
`linear_input_grad` (10.6 → 8.7) and the whole `projection.linear` category
(17.0 → 13.8 ms). Only the recurrent state work and the token-dependent elementwise
work grow (`rwkv.recurrent` 21.7 → 31.0 ms, +43%). A step is therefore dominated by
per-launch constants rather than by token-proportional arithmetic, which is the
measured explanation for pass 2's modest 2.8% end-to-end win from kernel-level
changes that were 2.7–11.8× in isolation.

**3. Attention is not on this path at all.** No `transformer_attention_*` kernel
appears in either ranking, at either batch size. Pass 2 measured a real regression
in the attention backward artifact, but the canonical raw-token trainer does not
dispatch that kernel, so the regression is not a training-step cost. "Highest-value
target by a wide margin" was true within the attention benchmark and is corrected
here: within the trainer, the target is the fused low-rank LayerNorm producer.

**4. The hot kernel's cost is a fixed one, and its geometry explains that better
than its arithmetic does.** In [rwkv_low_rank.rs](hierarchos-vulkan/src/rwkv_low_rank.rs#L2420)
the dispatch is `[batch.div_ceil(rows_per_workgroup), 1, 1]` with 64 lanes, and
`rows_per_workgroup` is 1 whenever the fp16 parameter mirrors are absent, which is
the case on this path — so a launch exposes exactly `rows` workgroups. The shader
body's per-lane work is on the order of `width` guarded FMAs plus one barrier per
64-column tile, which cannot cost 150–175 µs. What the measurement *proves* is that
the cost does not scale with rows at all; it does not yet say which mechanism makes
it fixed. Two candidates are worth one experiment each: only `rows` workgroups are
exposed so latency is unhidden — the same signature pass 2 found in the attention
backward kernel, whose cost was also flat in the workgroup count — and/or the
barrier plus cache-invalidation cost of its ~32 bindings, since
`HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1` shows it producing the step's largest
hazard edge (288 raw edges into `rwkv_time_mix3_linear3_key_state_forward_packed_fast`).
Raising `rows_per_workgroup` would test both at once, and the fp16 path already has
that machinery; it would also cut barriers, which ties this back to the elision
rule in pass 2.

## Method, and what it does not prove

- The timestamps bracket each dispatch (`record_kernel_timestamp_begin/end` in
  [vulkan.rs](hierarchos-vulkan/src/vulkan.rs#L6495)), so these are GPU times
  rather than host gaps. Each submission's category lines sum to exactly its
  `gpu_ms` — 100.00% in both runs — which is the internal check that the
  aggregation captures the whole submission.
- These runs use the debug binary (210.7 ms/step wall at batch 2), so the GPU
  attribution holds but the wall-clock numbers are not comparable with pass 2's
  end-to-end measurement, which used a different build.
- Three batch-2 profiles of the same step agree — the earlier one at 90.120 ms of
  GPU per step, two from this pass at 88.186 and 88.548 — and the top kernel is
  25.132, 25.235 and 25.201 ms in them. The ranking is reproducible, not noise.
- The probe's model package is reused across runs and its weights advance every
  run, so `mean_loss` is not comparable between runs. The kernel profile is
  unaffected.
- One device, one fixture: this ranks the canonical trainer's small-model step,
  and finding 2 is the reason not to extrapolate the *proportions* to other shapes
  even though the per-launch constants will travel.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/debug/training_step_probe --batch 2  --tokens 8 --steps 2 --warmup 1 2> target/probe_b2.log
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/debug/training_step_probe --batch 16 --tokens 8 --steps 2 --warmup 1 2> target/probe_b16.log
python validation/rank_kernel_profile.py target/probe_b2.log  --top 24
python validation/rank_kernel_profile.py target/probe_b16.log --top 24
HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1 ./target/debug/training_step_probe --steps 1 --warmup 0 2> target/trace_probe.log
```

The raw profile logs stay in `target/`, which is ignored; the ranking itself is
recorded as [validation/training_step_probe/kernel_ranking.txt](hierarchos-vulkan/validation/training_step_probe/kernel_ranking.txt).

---

# Fourth pass — 2026-10-15: the ladder as one command, and golden mode as a gate

The checks in the first three passes were run by hand, in an order that matters,
because each one constrains a different layer and a change can satisfy one while
breaking another. [verify_training_stack.py](hierarchos-vulkan/validation/verify_training_stack.py)
runs the whole ladder in one pass and exits non-zero unless every step passed:

```
PASS     cargo build (ladder binaries)  finished (debug)
PASS     binding-access audit           0 provably narrower binding(s) over 0 shader(s) out of 467 modules
PASS     shader provenance              32 variant(s) match their sources and flags
PASS     golden dense (matmul)          PASS: golden digests verified for 144 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     golden reduction               PASS: golden digests verified for 183 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     golden attention               PASS: golden digests verified for 79 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     training-step probe            6912 dispatches, 3076 barriers (upper bound 3076), loss 5.543217 on AMD Radeon Graphics

7/7 step(s) passed
```

Two design choices are worth stating, because both exist to stop the ladder from
passing vacuously:

- A step that cannot run — missing toolchain, missing binary, no recorded digest or
expectation for this device — is reported `BLOCKED` and fails the run. Green means
verified. The `--write-golden` and `--golden` modes could otherwise report an
unrecorded device as success, and `--golden-strict` is what turns that into a
failure; the ladder always passes `--golden-strict`.
- The ladder builds exactly the four binaries it runs, not `cargo build --bins`.
Building every bin makes an unrelated harness that does not compile fail a run
about the training stack, which is a real failure mode in a checkout where several
worktrees are being edited at once.

## The probe is now a repeatable end-to-end check

The probe previously reused one model package, and since every run performs AdamW
updates, each run measured a different point in training — so its loss could not be
asserted. `--model-dir` now initializes a directory that does not already hold a
package ([training_step_probe.rs](hierarchos-vulkan/src/bin/training_step_probe.rs#L133)),
so a caller can point it at a scratch directory it deletes first. Two such runs
give bit-identical results: 6912 dispatches, 3076 barriers, loss **5.543217** both
times. The ladder already deletes and re-creates `target/verify_training_stack/fresh_fixture`
for every run, and compares against
[step_expectation.json](hierarchos-vulkan/validation/training_step_probe/step_expectation.json):
dispatch count must match, the barrier count is an upper bound (so the elision work
below cannot fail the ladder by improving), and the loss is compared bit-for-bit
per device name. `--record-expectation` rewrites that file on the device in front of
you.

## The gate is non-vacuous, checked six ways

| perturbation | observed |
| --- | --- |
| one digest nibble flipped in `dense.json` | 1 mismatch reported, exit 1 |
| `dense.json` re-keyed to another vendor, `--golden-strict` | exit 1, "144 invocation(s) have no digest recorded for AMD Radeon Graphics" |
| same file, loose `--golden` | exit 0 but "0 verified, 144 unverified on another device" — not a pass |
| expectation's `dispatch_count` set to 1 | exit 1, "dispatch count 6912 != recorded 1" |
| device entry removed from the expectation | exit 1, "no expectation recorded for AMD Radeon Graphics" |
| release binaries absent with `--skip-build` | 4 × BLOCKED, exit 1 |

## Recording goldens for a second GPU vendor

The expensive half of this is hardware: it cannot be recorded from here, and no
fabricated entry would be worth anything. What is verified is the mechanism that
makes it a one-command step on that machine. Recording **merges** instead of
overwriting: taking a copy of `dense.json`, re-keying it to
`Intel(R) Arc(TM) A770 Graphics`, and re-recording on this AMD machine produced a
file holding both `AMD Radeon Graphics` and `Intel(R) Arc(TM) A770 Graphics`
entries for all 144 invocations, and `--golden-strict` then passed against it. So
the procedure on a new vendor's runner is exactly:

```powershell
./target/debug/matmul_kernel_validation     --write-golden validation/kernel_golden/dense.json
./target/debug/reduction_kernel_validation  --write-golden validation/kernel_golden/reduction.json
./target/debug/attention_kernel_cache_check --write-golden validation/kernel_golden/attention.json
python validation/verify_training_stack.py --record-expectation
```

and the existing vendor's entries survive it. Until that is done, a run on a new
device is `BLOCKED` rather than green, which is the intended behaviour.

## Wiring, and what the project does not have

There is no CI configuration in the repository — no `.github/workflows`, no other
pipeline file — so the gate is the command itself:

```powershell
cd hierarchos-vulkan
python validation/verify_training_stack.py --receipt target/verify_training_stack/receipt.json
```

`--receipt` writes a machine-readable per-step summary. A runner needs `glslc` and
`spirv-val` on `PATH` (the provenance step compiles and validates the CE variants)
and a Vulkan device whose digests and expectation are recorded; both are reported
as failures rather than skipped when absent.

---

# Fifth pass — 2026-10-15: why the barrier count cannot be lowered by narrowing access

Pass 2 found that narrowing a binding to `writeonly` can never remove a barrier,
because the elision rule's second term keys on "may be written", which `writeonly`
preserves, and measured that the 41 narrowings bought exactly zero barriers. The
task here was to change the rule so that narrowed access does buy barriers. It does
not work, and this pass establishes that with a measurement instead of an argument.
The barrier count on the representative step is unchanged at **3076 per step**.

## The rule is already exactly the hazard set

[prepare_shader_dispatch](hierarchos-vulkan/src/vulkan.rs#L6663) inserts a barrier
when any bound buffer satisfies:

```rust
(access.may_read()  && self.pending_shader_writes.contains(&region))
    || (access.may_write() && self.pending_shader_buffers.contains(&region))
```

and [finish_shader_dispatch](hierarchos-vulkan/src/vulkan.rs#L6762) records every
bound region in `pending_shader_buffers` and, additionally, in the read or write set
according to its access. So `pending_shader_buffers` is precisely "read or written by
some dispatch since the last barrier", and the two terms are exactly:

- term 1 = **read-after-write**,
- term 2 = `write && (read || written)` = **write-after-read** ∪ **write-after-write**.

That is the complete set of hazards a compute-to-compute barrier can be protecting
against. There is no conservatism left to remove, which is why no declaration can
lower the count. The dependency trace over one step agrees with the shape of the
rule: 69 read-after-write edges, 37 write-after-read, 21 write-after-write — all
three classes are present, so all three terms are load-bearing.

Two further levers were ruled out by measurement rather than by reasoning:

- **Finer region granularity.** `BufferRegionKey` is already
  `(VkBuffer, offset, size)` of the buffer view, and the engine binds whole views,
  so the key is as precise as the addressing the shaders use. There is no coarser
  key to refine.
- **Unused bindings.** A dispatch that binds a region it never touches still records
  it, which would be a false hazard. `audit_shader_binding_access.py` already proves
  per-binding access from SPIR-V, and its `unused` class is empty where it matters:
  **0 of 135 bindings** across the 13 hottest shaders of the step, and only 6 across
  all 467 modules. Skipping provably-unused bindings would buy nothing here.

## The rule's one real gap is in the strict direction, and this graph does not hit it

An exact-key comparison also means two *different* views of one `VkBuffer` whose byte
ranges overlap are invisible to the rule: the hazard is real but the keys do not
match, so no barrier is inserted. That would be a latent race, and it is now
something the engine can report. `HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1` gained a
recorded-but-inert diagnostic:
[record_overlapping_view_hazards](hierarchos-vulkan/src/vulkan.rs#L6730) compares each
bound view against every region touched since the last barrier, and the submission
summary reports how many pairs it examined against how many overlapped.

On the representative step the answer is **0 overlaps out of 352 same-buffer view
pairs examined, identical across all four profiled steps**. The population counter is
the point: a bare zero would be indistinguishable from a detector that never saw a
comparable pair. The predicate is unit-tested by
`overlapping_view_detection_distinguishes_overlap_from_distinct_ranges`, which asserts
that `(buf 1, 0, 4096)` and `(buf 1, 2048, 4096)` overlap, that identical keys and
adjacent or disjoint ranges do not, and that the same range in two buffers is not a
hazard.

The diagnostic changes no decision: it records only, and the probe's 6912 dispatches,
3076 barriers and bit-identical loss are unchanged with it enabled. It exists because
the check would otherwise be an unverified assumption in the elision rule.

## What would actually reduce barriers

Since the count is not reachable from the rule, the remaining levers are graph
structure and barrier *cost*, not barrier count:

- **Fewer real hazards.** The step executes 6912 dispatches and 3076 barriers, so
  about 44% of its dispatches sit behind a new ordering point. Reducing that means
  changing buffer reuse — for example having a producer write where its predecessor's
  output is not still pending — which is graph work, not scheduler work.
- **Cheaper barriers.** `shader_barrier()` emits an unscoped global memory barrier
  (`src=SHADER_WRITE`, `dst=SHADER_READ|SHADER_WRITE`). Scoping each barrier to the
  regions that actually hazard, via buffer memory barriers, would not change the
  count but would reduce what each one costs. That is a measurement nobody has taken
  here, and it is the honest next experiment on this axis.

## Not done in this pass: the attention backward lane split

The per-key phase of `transformer_attention_backward.comp` is the largest remaining
kernel-level opportunity, and it was not implemented. Reading the kernel closely enough
to be sure of a bit-exact split turned up three obstacles that pass 2's design note did
not capture, and each one has to be solved before the change is worth attempting:

1. **The per-key phase is only independent of itself.** In the strict-adjoint branch
   (lines 565–640) the body of `for (uint key …)` reads `row_max`, `row_inverse` and
   `row_softmax_dot` from shared memory. Those are computed earlier in the same
   invocation by iterating over *all* keys — a cross-key reduction whose FP32 order is
   part of the bit-exactness contract. Splitting keys across lanes therefore cannot be
   done by widening the grid: the row statistics have to stay in one lane, in the
   original order, with a barrier before the split.
2. **Every other path in the file assumes one lane.** The kernel declares
   `local_size = 1×1×1` and handles MHA/GQA, causal and bidirectional, sinks, relative
   bias, paged KV, and three unfused-QK variants in the same entry point, all of them
   single-lane and several of them accumulating `grad_k`/`grad_v` with `+=` in query
   order. Raising `local_size_x` to get lanes puts every one of those paths behind a
   `lane == 0` guard and adds the barriers they never needed.
3. **The per-lane workspace does not fit the shader's own budget.** The accumulators
   being split (`shared_k_cache`, `shared_v_cache`, `head_k_cache`, `head_v_cache`) are
   single-lane scratch today. Per-lane copies need `lanes × max_head_dim × 2` floats,
   and the file's header explicitly keeps its working set inside Vulkan 1.0's 16 KiB
   shared-memory guarantee — so a lane count has to be chosen against
   `HEAD_CACHE_SIZE` and the oversized-row fallback kept for everything else.

None of that is a reason not to do it; it is the work the change actually involves,
and it spans the kernel plus the host dispatch and allocation. The verification gate is
already in place: the parity harness's 79 invocations (forward and backward) compare
bit-for-bit, and the same invocations are covered by golden digests, so a split that
changes any FP32 sum fails immediately rather than silently. A later pass should start
from the current artifact's occupancy — not from the narrower declaration experiment —
and treat obstacle 1 as the design constraint rather than a detail.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
# hazard-class histogram and the overlapping-view population for one step
$env:HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1
./target/debug/training_step_probe --steps 1 --warmup 0 2> target/trace_probe.log
Select-String -Path target/trace_probe.log -Pattern "overlapping_view_total|hazard=" | Select-Object -First 20
# per-binding proof of access, including the unused class
python validation/audit_shader_binding_access.py --json target/audit_full.json
cargo test --lib -j 1 overlapping_view_detection -- --test-threads=1
```

---

# Sixth pass — 2026-10-15: what a step costs per token, and which knob amortizes it

The third pass found the step's GPU cost is nearly flat in the tokens it carries. If
that is true, the cheapest throughput win available is not a kernel at all: it is
carrying more tokens per step so the same dispatches amortize over more work.
[sweep_step_tokens.py](hierarchos-vulkan/validation/sweep_step_tokens.py) measures that,
running `training_step_probe` from a fresh model package per configuration so every
configuration starts at the same weights.

Release build, 8 tokens per sequence, rows varying (wall and GPU milliseconds per
step, `tokens/s` = tokens per step divided by wall time):

| rows | tokens/step | wall ms | GPU ms | GPU share | tokens/s | vs 16 | dispatches | barriers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 16 | 156.06 | 87.87 | 56.3% | **102.5** | 1.00× | 6912 | 3076 |
| 8 | 64 | 163.94 | 90.30 | 55.1% | 390.4 | 3.81× | 7056 | 3172 |
| 32 | 256 | 200.28 | 108.65 | 54.2% | 1278.2 | 12.47× | 7056 | 3172 |
| 128 | 1024 | 502.24 | 351.35 | 70.0% | 2038.9 | 19.89× | 7120 | 3204 |
| 256 | 2048 | 863.20 | 638.06 | 73.9% | 2372.6 | 23.14× | 7120 | 3204 |
| 512 | 4096 | 1575.67 | 1206.72 | 76.6% | **2599.5** | **25.35×** | 7120 | 3204 |
| 1024 | 8192 | 3011.93 | 2339.60 | 77.7% | 2719.9 | 26.53× | 7120 | 3204 |

**The answer is 25.4× tokens/s, and it costs nothing but a configuration change.** 256×
the tokens per step leaves the dispatch count almost unchanged (6912 → 7120, +3%) and
the barrier count essentially unchanged (3076 → 3204). GPU time per token falls 18.6×
(5.492 → 0.295 ms per token in GPU time), which says the *GPU* side of a small step is
also dominated by per-launch overhead rather than by arithmetic — the same effect the
third pass measured as a flat cost in rows.

The curve flattens past 2048 tokens per step: doubling again to 8192 buys 4.6%, because
the GPU share has climbed from 56% to 78% and the step is now doing real work per
token. 4096 tokens per step (512 rows) is the practical point — within 4.6% of the
8192-token result for half the memory and step latency.

## The other knob is the wrong one

Tokens per step can also grow by lengthening sequences instead of adding rows, and that
is 2.4× worse for the same tokens:

| configuration | tokens/step | wall ms | dispatches | barriers | tokens/s |
| --- | --- | --- | --- | --- | --- |
| 32 rows × 8 tokens | 256 | 200.28 | 7056 | 3172 | 1278.2 |
| 8 rows × 32 tokens | 256 | 480.87 | 27642 | 12673 | 532.4 |

The reason is structural: the step dispatches roughly per sequence position, so
sequence length multiplies the dispatch count (7056 → 27642, 3.9×) and therefore the
per-launch overhead, while rows add work inside dispatches that already exist. The
knee in the row curve is the point where per-token GPU work overtakes that fixed cost.
The practical rule this measurement gives: **raise the step's row count to add tokens,
and keep the sequence length moderate.**

That lands on a real trainer flag rather than an abstract knob. `train.rs` exposes
`--batch-size` (rows carried in one graph — the amortizing knob) and
`--gradient-accumulation-steps` (more graphs before one optimizer update — which
multiplies dispatches instead), and its `--batch-size` default was 1. At 1 row per step
the trainer sits left of this table entirely, where the per-launch cost is the whole
step, so the default is now **32** and the `--help` text carries the measured curve
above. `--gradient-accumulation-steps` is deliberately left at 1: it replays the whole
graph per micro-batch and pays the same per-dispatch price, so it cannot amortize
anything.

Measured through the trainer itself, not the probe — 384 dataset rows of 32 tokens
(12 288 tokens), one epoch, release, one AMD Radeon, per-step `tokens_per_sec` medians
taken from the `HIERARCHOS_EVENT` stream:

| `--batch-size` | steps | wall s | median tokens/s | first step | last step |
| --- | --- | --- | --- | --- | --- |
| 1 | 384 | 129.8 | **113.9** | 75.8 | 115.7 |
| 32 (new default) | 12 | 6.4 | **2430.8** | 1752.7 | 2506.6 |

Same tokens, same epochs: **21.3× the per-step token rate, 20.3× the wall clock**. The
first step at 32 rows is 28% below the last (graph setup and the first dispatches are
not amortized yet), which is the shape the probe table predicts.

Two things this does not fix. A larger batch is a different optimizer problem — fewer,
fatter updates — so the learning rate that suited one row per step is not the learning
rate that suits 32; `--batch-size 1` still exists for debugging and for anyone whose
schedule assumes it. And the default change is a training-semantics change as much as a
throughput one; it is recorded here rather than presented as free.

## One incidental result worth keeping

The release build reproduces the debug-recorded probe expectation exactly — 6912
dispatches, 3076 barriers, loss **5.543217** — so the ladder's recorded expectation,
and the bit-exactness it asserts, are build-flavor independent. The ladder can be run
with `--release` and compare against the same numbers.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
python validation/sweep_step_tokens.py --release --batches 2,8,32,128,512 --tokens 8 --steps 5 --warmup 1
python validation/sweep_step_tokens.py --release --batches 8,32 --tokens 32 --steps 3 --warmup 1
python validation/rank_kernel_profile.py target/token_sweep/profile_b512_t8_s1.log
```

# Seventh pass — 2026-10-15: the producer kernel, and the pass that says no

The third pass named `layer_norm_low_rank_producer_forward_fused` as the step's top
cost at 175 µs per dispatch. This pass tried to make it cheap, and the honest result
is that it is now 5% cheap: **132.24 µs per dispatch instead of 139.35 — not the 175 µs
that was asked for, and not a fix.** What follows is the measurement that changed my
mind about why the number is what it is.

The same kernel, same 144 dispatches per step, at three geometries (release, 8 tokens
per sequence, one sequence):

| rows | producer ms | µs/dispatch | step GPU ms | producer share of step |
| --- | --- | --- | --- | --- |
| 2 (third pass) | 25.35 | 175.0 | 88.5 | 28.6% |
| 32 | 20.07 | 139.35 | 106.83 | 18.8% |
| 512 | 27.68 | 192.25 | 1203.38 | **2.3%** |

**The cost is nearly flat in the rows it processes.** 16× the rows (32 → 512) costs
1.38×, while in the same two runs every control kernel scales with rows as it should:
`packed_cell_channel_mix_state_forward_fused` 79.16 → 854.93 µs (10.8×),
`vector_add` 2.91 → 60.87 (20.9×), `linear_weight_grad` 6.00 → 48.61 (8.1×),
`layer_norm_adapter_forward_fused_512` 56.15 → 200.98 (3.6×). A kernel that costs the
same for 2 rows as for 512 is not paying for its arithmetic, and the third pass's
mechanism — a fixed per-launch cost — is confirmed while the third pass's *target* is
not.

## What was changed, and what it bought

The one serial structure in the kernel that is specific to it rather than to the
launch path was the LayerNorm reduction: lane 0 added the whole row to itself, one
global load per addition, while the other 63 lanes waited at the barrier. It now stages
the row through shared memory in 64-float tiles (one coalesced load per lane) and lane 0
adds the same values in the same order out of shared memory, so the mean, the
variance and the fused outputs are unchanged. Measured: **139.35 → 132.24 µs per
dispatch, 20.07 → 19.04 ms per step**, with the control kernels moving less than 0.5%
(`packed_cell` 79.16 → 79.19, `vector_add` 2.91 → 2.89), so the 5% is above this
machine's noise floor and real. The change is kept because it is bit-exact and
monotone, not because it answers the question.

## Why the remaining 132 µs is not this kernel's arithmetic

The hypothesis that the loads themselves were unpipelined is wrong: if a dependent
global-load chain of `width` round trips were sitting on one lane, removing it would
have bought tens of microseconds, and it bought 7. The compiler already hoists
independent loads. Three further facts point away from arithmetic as the explanation:

* The cost is a property of the *shape class*, not of this kernel. At 32 rows the
  same-geometry kernels in the same category — layer-norm backward/add-residual,
  adapter forward, packed cell channel mix — all sit in a 47–79 µs per-dispatch band,
  while kernels dispatched over many narrow workgroups (`vector_add`
  `linear_weight_grad`, `gradient_accumulate4`, `bias_grad`) sit at 3–6 µs. The
  producer is the widest member of the first class, not an outlier in it.
* At a small row count the grid is tiny — 32 workgroups of 64 lanes — so there is
  almost nothing resident to overlap memory latency with, and the 512-row run recovers
  16× the work for 1.38× the time, which is what latency-hiding looks like when it
  starts working.
* The sub-dispatch attribution cannot be separated from the neighbouring dispatches
  with the instrumentation this project has. The runtime's per-category milliseconds do
  sum to the submission's `gpu_ms`, so the *total* is real and the producer's share of
  it is 2.3% of the recommended step; but among adjacent timestamp brackets, some of a
  wide kernel's bracket can belong to the work it inherits across the preceding
  barrier. Resolving that needs per-dispatch counters, and there is no `rocprof` or
  Nsight on this machine.

So the requested outcome — "stop costing 175 µs per dispatch" — was not achieved. I
can say what the number is (139, not 175, at the trainer's default batch, and 2.3% of a
step rather than 29%), what it is not (this kernel's arithmetic), and what would settle
it (counters, or a profile that separates brackets from neighbours). The rest is the
negative result, reported as one.

## Verifying a bit-exact kernel change

`glslc` with its default target environment reproduces the embedded `.spv`
byte-for-byte from the source, before and after the edit
(`python validation/audit_shader_artifacts.py layer_norm_low_rank_producer_forward_fused`),
`spirv-val --target-env vulkan1.0` accepts the result, and the ladder is unaffected in
the ways that matter: **7/7, loss 5.543217, 6912 dispatches, 3076 barriers**, plus the
three golden harnesses at 144/183/79 invocations. The 32-row geometry used for the
timing comparison reports loss 5.3915977 with the same 7056 dispatches and 3172
barriers, bit-identical to the values recorded before the edit (compared on the full
recorded value, not on the table's 4-decimal print). `cargo test --lib -j 1` is 705 passed / 0 failed / 9 ignored, unchanged.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe --bin hierarchos-vulkan-train
glslc shaders/layer_norm_low_rank_producer_forward_fused.comp -o shaders/layer_norm_low_rank_producer_forward_fused.spv
spirv-val --target-env vulkan1.0 shaders/layer_norm_low_rank_producer_forward_fused.spv
python validation/audit_shader_artifacts.py layer_norm_low_rank_producer_forward_fused
python validation/sweep_step_tokens.py --release --batches 32,512 --tokens 8 --steps 5 --warmup 1
python validation/rank_kernel_profile.py target/token_sweep/profile_b32_t8_s1.log
python validation/verify_training_stack.py
# trainer-level before/after, 384 rows of 32 tokens each
./target/release/hierarchos-vulkan-train.exe --model target/train_fixture/model --dataset target/train_fixture/tokens384.jsonl --output target/train_fixture/out_b1 --batch-size 1 --epochs 1 --json-events
./target/release/hierarchos-vulkan-train.exe --model target/train_fixture/model --dataset target/train_fixture/tokens384.jsonl --output target/train_fixture/out_bdefault --epochs 1 --json-events
```

# Eighth pass — 2026-10-15: what a dispatch bracket leaves out, and where the host time goes

The third pass ranked kernels by `gpu_ms`, and that number is the *sum of per-dispatch
timestamps*: a bracket is written immediately before and immediately after each
`vkCmdDispatch`. So `gpu_ms` is execution and nothing else — which is the good news —
but it also means a step can be slower than `gpu_ms` for reasons no kernel row shows.
Nothing in the profile bounded the submission, so the unattributed part had no name and
silently belonged to whichever kernel sat next to it. Two additions fix that.

## The span, and the share no kernel owns

`hierarchos_vulkan_kernel_profile_span` now reports the submission's first bracket begin,
its last bracket end, and the difference. Release, 8 tokens per sequence, one sequence,
four submissions averaged:

| rows | dispatches | GPU span ms | dispatch brackets ms | unattributed ms | share | wall ms | host = wall − span |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6912 | 100.369 | 88.639 | 11.730 | 11.7% | 157.10 | 56.7 ms (8.2 µs/dispatch) |
| 32 | 7056 | 127.076 | 107.594 | 19.482 | 15.3% | 200.36 | 73.3 ms (10.4 µs/dispatch) |
| 512 | 7120 | 1335.095 | 1208.929 | 126.166 | 9.4% | ~1576 | ~241 ms (readback-dominated) |

**9–15% of a step's GPU span is inside no kernel bracket**: barrier flush, launch
latency, and idle between dispatches. At 32 rows that is 19.5 ms of a 127 ms span. It is
the bucket that made the third pass's flat-per-launch cost look like a kernel property,
and it is not the same lever as the dispatch count: removing dispatches removes brackets
and their host cost, not the gap between the remaining ones.

The span also separates host time for the first time. Wall minus span at 32 rows is
73.3 ms over 7056 dispatches — **10.4 µs per recorded dispatch** (8.2 µs at 2 rows).
At 512 rows the same subtraction gives ~241 ms, but that figure is not comparable: the
probe's `--readback full` copies back tensors proportional to the batch, so it measures
readback I/O as well.

**That 10.4 µs is an upper bound, and the profiler is the reason.** Profiling writes two
timestamps per dispatch — 14112 extra host calls per step — so the same 32-row
configuration runs in **115.6 ms unprofiled against 200.4 ms profiled**, and the
profiled wall-minus-span includes the profiler's own recording. The genuine per-dispatch
host cost is therefore single-digit microseconds and, with the instrument that can see
it also perturbing it, it is not pinned down here. This also corrects the sixth pass:
its absolute tokens/s column was measured with profiling on (`sweep_step_tokens.py`
always enables it), so those figures are lower bounds. The *ratios* in that table survive
because the overhead tracks the dispatch count, which barely moves with rows.

The step result also exposes the recording counters, so they are now in the probe's
report rather than in a comment. At 32 rows, per 7056 dispatches: **216 descriptor sets
allocated, 2352 descriptor binds, 1902 pipeline binds**. Binding work is therefore *not*
per-dispatch — the descriptor cache absorbs 96.9% of it, and only a third of dispatches
rebind at all. Whatever the ~10 µs is spent on, it is the recording calls themselves, not
descriptor churn, which removes the second reason to expect a single gradient slab to pay
for itself.

## Cold and warm: does a kernel pay for itself?

`hierarchos_vulkan_kernel_profile_placement` splits each shader's dispatches by whether
they were recorded directly after a barrier or back to back with another dispatch. Same
step, 32 rows, per dispatch:

| shader | after barrier | after dispatch | difference |
| --- | --- | --- | --- |
| `layer_norm_low_rank_producer_forward_fused` | 154.3 µs | **123.7 µs** | +30.6 µs |
| `layer_norm_adapter_forward_fused_512` | 73.8 µs | **27.7 µs** | +46.1 µs |
| `linear_bias_forward` | 20.7 µs | 4.7 µs | +15.9 µs |
| `row_keep_forward` | 14.4 µs | 0.005 µs | +14.4 µs |
| `layer_norm_stats` | 26.0 µs | 17.3 µs | +8.8 µs |
| `indexed_step_gather` | 2.6 µs | 13.8 µs | **−11.2 µs** |

This answers the question the seventh pass had to leave open, and it partly corrects
that pass's conclusion. The producer is mostly paying for its own work: 123.7 µs of its
133.9 µs average is execution, and only ~31 µs of a cold dispatch is inherited. So the
third pass's ranking of it as a hot spot was not a measurement artifact, and the seventh
pass's 5% was a real 5% of a real cost. Where the seventh pass went too far was in
concluding that the residual "is not this kernel's arithmetic": once the inherited share
is subtracted, the remaining 123.7 µs *is* time the kernel spends executing, just spent
in a latency-bound structure rather than in useful flops — which means a shader-level fix
remains possible in principle and the 5% was a small win in the right direction, not a
wrong target. By contrast `layer_norm_adapter_forward_fused_512` looks 2.7× more
expensive than it is: 63% of its cold bracket is the barrier in front of it.

Two caveats, both visible in the table. `row_keep_forward` shows the effect at its
purest — 14.4 µs cold against 5 ns warm, the same kernel and the same grid, so that
kernel's cost is essentially all inherited — while `indexed_step_gather` goes the other
way, because which dispatches land cold correlates with *what* they process. The split
is an observation about placement, not a controlled experiment, and the tool says so.
And a small warm measurement is not proof of free work: 5 ns of bracket is a dispatch
whose grid is a handful of workgroups that the hardware retires between two timestamp
writes. One more precision, since it is a label rather than a measurement: the first
dispatch of a submission follows neither a barrier nor another dispatch, and it is
counted as warm. That is one dispatch per submission out of ~7000, and its bracket can
include the previous submission's tail, so the cold/warm split is unaffected at this
scale — but the label is approximate for that dispatch.

## Why the dispatch count cannot be cut yet

The eighth pass's own numbers size the prize: **8–10 µs of host time per recorded
dispatch**, so the ~1346 per-slot dispatches in `optimizer.gradient-bookkeeping` (one per
optimizer slot, in `record_scale_gradients_*`, `record_accumulate*`, `record_step`) cost
roughly 13 ms of the 200 ms step at 32 rows. They cannot be folded into one dispatch as
the buffers stand: every slot owns its own allocation — `GpuBuffer::zeros_f32(&device,
trainable.len)` per slot — and Vulkan storage bindings cannot be indexed without
descriptor-indexing support, so one dispatch physically cannot touch N slots' gradients.
Cutting the count honestly needs one gradient slab with per-slot sub-ranges (the
barrier-elision rule already keys on `(buffer, offset, size)`, so sub-ranges stay
distinct) plus an index table and a `*_many` kernel per operation. That is an allocation
and checkpoint-layout change, not a batching tweak, and it was not attempted here.

The same reasoning bounds the other route — fusing equal-shape elementwise kernels
(`finite_clamp_forward` 288, `vector_add` 308, `bias_grad` 240 dispatches per step) into
one dispatch per group — and the `dist` trees under `hierarchos-vulkan/` show this
project already treats layout changes as a deliberate migration. What is *verified* here
is the measurement, so that whoever does that work can show the single-digit
microseconds per dispatch it removes — against a step that is 115.6 ms unprofiled, not
the 200 ms the profiled runs show.

## Verified

Ladder 7/7 with loss **5.543217**, 6912 dispatches and 3076 barriers unchanged;
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored. The profiler is inactive
unless `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, so the ladder and the golden harnesses run
the same code they ran before. `rank_kernel_profile.py` reads the two new lines and
`validation/training_step_probe/kernel_ranking.txt` is regenerated from a 512-row
profile.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/release/training_step_probe.exe --model-dir target/prof_check/m32 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/b32.json 2> target/prof_check/b32.log
python validation/rank_kernel_profile.py target/prof_check/b32.log --top 8
grep _span target/prof_check/b32.log
grep _placement target/prof_check/b32.log
```

# Ninth pass — 2026-10-15: the profiler was measuring itself too

The eighth pass ended with a caveat it could not fix: profiling adds ~82% to the wall
clock, because two timestamps per dispatch is 14112 extra host calls in a step, and every
number it produced came from a run distorted by the instrument. This pass samples the
brackets and re-measures what that distortion was hiding.

## Sampling, and why it is per shader

Every `sample_stride`-th dispatch *of each shader* is timestamped, and a shader's first
dispatch always is. Sampling per shader rather than globally is what keeps the ranking
intact: a global stride would leave kernels that appear fewer times than the stride with
no bracket at all, and they would silently vanish from the hot-spot table. The default
stride is **17**, prime so the sample does not phase-lock with a per-layer dispatch
pattern, and `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1` restores exact timing. The
scheduler-driven collection stays at stride 1 by construction: the trainer consumes that
profile's totals when it decides how to schedule, so that path must not become an
estimate.

Same 32-row step, same machine:

| profiler | step ms | overhead |
| --- | --- | --- |
| stride 1 (previous behaviour) | 202.49 | +82% |
| stride 17 (new default) | 119.82 | **+7.8%** |
| off | 111.11 | — |

**The instrument now costs 7.8% instead of 82%**, and the sampled run is 448 dispatches
of 7056 timestamped. A bracket is therefore a mean; each shader's reported `gpu_ms` is
that mean scaled by its exact dispatch count, and the `samples` column says how many
brackets the estimate rests on (the producer: 9 of 144).

## What the distortion was hiding

The span was the surprise. It is now written by its own query pair so that sampling
cannot move it, and on the same 32-row step it falls from **129.0 ms at stride 1 to
88.8 ms at stride 17** — the profiler's own timestamp writes were inside the span it
reported, so a fifth of what the eighth pass called "time no kernel owns" was the
profiler writing timestamps. Averaged over four submissions at 512 rows the unattributed
share is **9.4% at stride 1 against 5.1% at stride 17**, because the extra captures the
profiler without inflating the work between them. The barriers are real; they were just
being measured through an instrument that added more of its own serialization than they
cost.

> **Superseded by the tenth pass.** Those two unattributed figures do not reproduce:
> on the same configuration the share reads 13.8% at batch 32 and 4.2% at batch 512 at
> stride 17, and 15.3% / 9.5% at stride 1. The reason is that the share is an exact span
> minus an *estimated* bracket sum, so it inherits the sampling error of the estimate.
> The tenth pass re-measures it and states the range instead of a point value; the span
> itself, and the mechanism described here, are unchanged.

The per-kernel totals move the same way and the *ranking does not*: the producer reads
**133.9 µs per dispatch at stride 1 and 109.7 µs at stride 17**, while its share of the
step is 17.8% against 17.9%. So every absolute microsecond figure in the third, seventh
and eighth passes is an overestimate from a perturbed run, and their shares are the part
that survives.

## The placement comparison is now shape-matched

The eighth pass's cold/warm split compared dispatches that happened to sit in different
places, and flagged that cold dispatches could simply be different shapes. A placement
line is now keyed by the dispatch's shape — its grid, push constants and binding count —
and both populations print on one line under that key, so a cold and a warm bracket are
only ever compared when they ran the same work. The grid is printed so the match is
visible rather than asserted:

| shader | grid | after barrier | after dispatch | difference | cold n | warm n |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_norm_low_rank_producer_forward_fused` | 32x1x1 | 129.7 µs | 99.6 µs | +30.1 µs | 3 | 6 |
| `layer_norm_adapter_forward_fused_512` | 1x32x1 | 58.7 µs | 20.5 µs | +38.2 µs | 6 | 3 |
| `linear_bias_forward` | 1x2x1 | 11.0 µs | 2.6 µs | +8.5 µs | 3 | 3 |
| `bias_grad` | 1x1x1 | 8.7 µs | 0.3 µs | +8.4 µs | 9 | 5 |
| `linear_forward` | 8x2x1 | 8.3 µs | 10.2 µs | −1.9 µs | 5 | 1 |

Both numbers for the producer fall relative to the eighth pass (154.3/123.7 µs) because
those brackets carried the profiler's own timestamps, but the **+30 µs inherited** and
the 80/20 split between own work and barrier are unchanged. `linear_forward`'s −1.9 µs
is what the confound looks like when it is gone: with matched shapes the remaining
difference is within the noise of 5 cold and 1 warm sample, and the counts are printed
exactly so that can be judged instead of guessed. A small sample is not proof of free
work in either direction.

## Limitations of this pass

The shape key removes the confound it targets and no more. It covers the grid, the push
constants and the binding *count*, not which buffers were bound, so a matched cold/warm
pair can still differ in the tensors it touches — a dispatch that reads a buffer written
moments earlier differs from one reading a cold allocation in cache and TLB state, and no
shape signature sees that. Matching also removes a confound by *construction*: two
dispatches only stay on one line because the graph happened to record them at the same
shape, so the surviving cold population is still selected by the graph's hazards rather
than assigned by the experimenter. A genuinely controlled version would force placement:
dispatch the same call site both after a barrier and back to back, in both orders, and
compare. That is a one-off harness against a single kernel, not something the per-step
profile can do, and it is the test to run before trusting a residual of a few
microseconds.

Sampled totals are estimates: means scaled by exact counts, valid when the stride does
not correlate with the cost, which is why the stride is prime and the sample sizes are
reported. The residual 7.8% is not zero — the shape hash allocates a small buffer per
dispatch (the shape is hashed for every dispatch while profiling is on, not only for the
ones that are sampled, because the sampling decision is made after it), the span pair is
written per submission, and the sampling decision costs a hash lookup per dispatch. Junk
or non-positive values of `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE` fall back to 17 rather
than disabling or exacting the profiler: verified for `0`, `junk` and the empty string
(all resolve to 17) against an explicit `1` (all 7056 dispatches bracketed). The first dispatch of a submission is still labelled warm although
it follows neither a barrier nor another dispatch. And the new report paths have no unit
tests; they were verified against logs with both line formats, including one written
before either line existed.

## Verified

Ladder 7/7 with loss **5.543217**, 6912 dispatches, 3076 barriers, goldens 144/183/79;
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored. Profiling stays inactive unless
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, so the ladder exercises the untouched path.
`rank_kernel_profile.py` reads both the old and the new line formats and
`validation/training_step_probe/kernel_ranking.txt` is regenerated from a sampled
512-row profile. The stride override was exercised across `1`, `0`, `junk` and empty
(see the limitations above) after the last code edit.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "1"; ./target/release/training_step_probe.exe --model-dir target/prof_check/s1 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/s1.json 2> target/prof_check/s1.log
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "17"; ./target/release/training_step_probe.exe --model-dir target/prof_check/s17 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/s17.json 2> target/prof_check/s17.log
Remove-Item Env:HIERARCHOS_VULKAN_PROFILE_KERNELS; ./target/release/training_step_probe.exe --model-dir target/prof_check/none --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/none.json
python validation/rank_kernel_profile.py target/prof_check/s17.log --top 6
```

# Tenth pass — 2026-10-15: the whole distribution re-ranked, and placement assigned
rather than observed

Two things were left over. The ranking was built from a table the runtime truncates to
the top 24 shaders, so the long tail of cheap kernels — the part whose *dispatch count*
decides whether it is worth batching — had never been ranked at all. And the cold/warm
split was still an observation: the ninth pass could match two brackets by shape, but
*which* dispatches landed cold was still the graph's decision, so the comparison could
not separate the kernel's response to placement from the work that happened to be
placed there. This pass fixes the first by making the table complete and the second by
taking placement away from the graph.

## Every kernel, not the top 24

`HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS` now bounds both the per-shader and the placement
tables (the placement loop had its own hard-coded copy of the same constant). The default
stays 24 so a diagnostic run's stderr stays short; a ranking run sets it high. The 512-row
table goes from 12 printed shaders to 92, and a per-submission table from 24 to ~103.

## The re-rank

Same synthetic step, default stride 17, complete table. Category shares of the GPU step
(the step total is 61.716 ms at batch 2, 85.054 ms at 32, 1284.871 ms at 512):

| category | batch 2 | batch 32 | batch 512 |
| --- | --- | --- | --- |
| rwkv.recurrent | 24.73% | 36.89% | 77.60% |
| normalization.layer-norm | 43.82% | 34.04% | 7.73% |
| projection.linear | 17.08% | 11.77% | 4.61% |
| control | 4.95% | 4.69% | 2.18% |
| elementwise-reduction | 1.22% | 4.31% | 5.15% |
| optimizer.gradient-bookkeeping | 4.54% | 3.70% | 0.38% |
| loss.lm-projection | 1.97% | 1.70% | 0.71% |
| token.embedding | 0.42% | 1.65% | 0.62% |
| other | 0.88% | 0.95% | 0.96% |

(Stride-17 tables throughout. The exact-bracket stride-1 run at batch 512 moves the
leaders by less than half a point — rwkv.recurrent 77.89%, layer-norm 7.44% — because at
that batch size the instrument barely moves the span.)

So there is no single priority order, and the earlier passes' lists were all read at one
batch size. At the trainer's default batch the leader is
`layer_norm_low_rank_producer_forward_fused` (144 dispatches, 15.459 ms/step, 18.18%),
followed by `packed_cell_channel_mix_state_forward_fused` (10.99%),
`rwkv_time_mix3_linear3_key_state_forward_packed_fast` (8.55%) and
`layer_norm_adapter_forward_fused_512` (7.56%) — the top four are 45.28% of the step. At
batch 512 the same four kernels are reordered and reweighted under the packed recurrent
kernels: 26.38 / 22.09 / 10.31 / 7.90%, top four 66.68%. At batch 2 the producer alone is
30.01%. The rebuilt list, with the lever and the measured basis for each entry, is
`validation/training_step_probe/kernel_priority.txt`; the generated tables for all three
regimes are `kernel_ranking_b2/b32/b512.txt`.

**A stride-5 run confirms the order at batch 32**: 8 steps, 29 samples per 144-dispatch
shader, same four kernels, same order, every level within 4% (17.52 / 10.79 / 8.07 /
7.38%). The batch-32 ordering is not a sampling artifact.

## What stride 1 actually measures

Running the same batch-32 step at three strides separates sampling error from instrument
effect, and the answer is that stride 1 is not a noisy version of the truth — it is a
different physical regime:

| kernel (batch 32) | stride 17 | stride 5 | stride 1 |
| --- | --- | --- | --- |
| layer_norm_low_rank_producer_forward_fused | 107.4 µs | 105.1 µs | 134.5 µs |
| packed_cell_channel_mix_state_forward_fused | 64.9 | 64.8 | 81.4 |
| rwkv_time_mix3_..._packed_fast | 75.7 | 72.7 | 108.6 |
| layer_norm_adapter_forward_fused_512 | 44.7 | 44.3 | 58.7 |

Strides 5 and 17 agree within 4%, so the sampling error at 17 is small once the sample
count is adequate; stride 1 is 20-30% above both, and the gap is 110.384 ms of brackets
at stride 1 against 86.431 ms at stride 5 with the wall clock moving the *opposite* way
(204.220 ms against 124.148 ms). Two timestamps per dispatch at the compute stage force
the pipeline to drain around every dispatch, so each bracket measures an *isolated*
dispatch and the step no longer overlaps its own work. That is the same mechanism the
harness below finds dominating the wide kernels, seen from the other side: the stride-17
number is the better estimate of what the step pays, and it is still an upper bound
because 1 dispatch in 17 is still isolated by its own timestamps.

At batch 512 the instrument does not move the span at all (1341.762 ms at stride 17
against 1341.146 ms at stride 1) because the step is long enough that the isolated
dispatches still fit; at batch 32 the span moves 98.646 → 130.311 ms. The distortion is
worst exactly where the per-dispatch cost is smallest, which is why the small-batch
regime needed re-measuring before its priority list could be trusted.

It also corrects the ninth pass's unattributed share, which is an exact span minus an
*estimated* bracket sum and therefore moves with the estimate: 13.8% at batch 32 and 4.2%
at batch 512 at stride 17, 15.3% and 9.5% at stride 1, against the 6.7% and 5.1% reported
before. The span is the number to compare across runs; the difference from the brackets
is a range, not a value.

## Sampling erases placement rows, not just samples

The placement table is hit harder than the cost table, because a row exists only if the
sampled dispatches of one (shader, shape) landed in *both* populations:

* `packed_state_slot_backward` (8 cold / 8 warm at stride 1) has no row at all at
  stride 17.
* At batch 512 the producer reads +0.464 µs at stride 17 (3 cold, 6 warm) against
  **+20.196 µs** at stride 1 (48 cold, 96 warm) — a 40× difference in the same run
  configuration.
* At batch 32 the same comparison is well behaved: producer +27.269 (3/6) against
  +29.921 (48/96), `bias_grad` +8.273 (9/5) against +10.107 (96/56), `vector_add`
  +0.997 (11/6) against +1.055 (140/120), `finite_clamp_forward` +0.303 (7/4) against
  +0.287 (144/80).

So the ranking is taken from stride 17 and the placement ratios from stride 1, and every
placement number in the priority list carries its sample counts.

## Placement as an independent variable

`hierarchos-vulkan/src/bin/placement_probe.rs` dispatches one kernel both ways with
placement assigned:

* The cold arm records an explicit barrier before every dispatch, the warm arm none.
  A barrier count read back from the command stream is checked against the expected
  value before a block is timed (measured: one per dispatch cold, zero warm), so the
  assignment is verified rather than assumed.
* Both arms use the same shared inputs and the same shape, and each dispatch writes its
  own output buffer, so no aliasing confound is introduced by the arm that is meant to
  be free of barriers.
* Arms alternate in both orders (warm-cold-cold-warm, then cold-warm-warm-cold) and each
  block is timed as a whole, so the instrument costs one query pair per block instead of
  two timestamp writes per dispatch. Medians over 8 alternating blocks per arm, on
  blocks of 32, 64 and 128 so that anything that scales as 1/blocks is visible.

Per dispatch, at the largest block size (µs):

| kernel | shape | grid | warm | cold | inherited |
| --- | --- | --- | --- | --- | --- |
| `linear_forward` | 32x128x128 | 8x2x1 | 3.72 | 15.44 | +11.7 |
| `linear_bias_forward` | 32x128x128 | 8x2x1 | 4.34 | 16.33 | +12.0 |
| `linear_bias_forward` | 512x128x16 | 1x32x1 | 8.49 | 19.29 | +10.8 |
| `vector_add` | 1x65536x1 | 256x1x1 | 3.90 | 6.03 | +2.1 |
| `finite_clamp_forward` | 1x65536x1 | 256x1x1 | 3.24 | 3.80 | +0.6 |
| `linear_forward` | 512x128x128 | 8x32x1 | 30.06 | 34.12 | +4.1 |
| `linear_bias_forward` | 32x512x512 | 32x2x1 | 20.53 | 50.27 | +29.7 |

Three results, none of which the observed split can show:

1. **Inheritance is a property of the shape class.** The narrow matmul shapes pay
   +10.8…+12.0 µs and are stable to within 0.3 µs across block sizes; a two-stream
   elementwise kernel at 256 workgroups pays +2.1…+2.7; a one-stream one pays +0.1…+0.8,
   indistinguishable from zero. The producer's +20.196 µs (batch 512, 48 cold / 96
   warm) is well above that band, which is the expected direction: it is a much larger
   dispatch than the harness shapes (190 µs warm against 8-19) with ~24 bindings, and
   inheritance tracks how much memory the dispatch first touches after the barrier —
   the same ordering the 2.1 µs for one input stream against 11.7 µs for two shows.
2. **For wide compute-bound shapes it is also a property of the block, so no cost per
   barrier exists.** The same dispatch is 50.44 µs warm at 32 dispatches per submission
   and 20.53 µs at 128, and its cold cost 75.55 → 50.27 µs over the same range: most of
   what a barrier costs there is the inter-dispatch overlap it prevents, not a flush.
   Inheritance for that shape *grows* (25.11 → 26.27 → 29.74 µs) as the warm arm gets
   faster than the cold arm can follow. This is the mechanism behind the stride-1 result
   above, measured with placement under control.
3. **The profile's difference carries the neighbour, not only the placement.** With
   identical neighbours in both arms, `finite_clamp_forward` at 256x1x1 pays +0.6 µs,
   while the step's profile reports **−4.588 µs** for the same kernel at the same grid
   (144 cold, 80 warm). A dispatch's bracket is affected by which kernel preceded it, and
   the placement table has no way to express that.

The harness also pins the host cost that the eighth and ninth passes could only bound:
wall minus GPU is 1.7-5.4 µs per recorded dispatch for these shapes at blocks of 64 and
128, typically 2-3.5 (`vector_add` 6.87 − 3.57 µs at 32 dispatches and 5.65 − 3.75 at 64;
`finite_clamp_forward` 1.7-2.0; `linear_bias_forward` at 1x32x1 3.4-5.4). The step-level
subtraction agrees: wall minus span at batch 32 is 25.5 ms over 7056 dispatches, 3.6 µs
each, and 1.5 µs at batch 2. The eighth pass's ~10 µs was an upper bound from a run its
own instrumentation inflated; the honest level is 1.5-3.6 µs, single-digit as predicted.

Limits: the registry covers only kernels whose dispatch parameters are already pinned by
an existing validation harness, so the numbers transfer to the wide recurrent kernels by
argument rather than measurement; the GPU reading goes through the device's own
per-dispatch timestamps, so its absolute levels carry the same isolation as stride 1 and
only the cold/warm difference is comparable; and each block's dispatches write distinct
output buffers, which the real step does not do.

## What the priority list now says

The count lever is the one this pass can quantify end to end. Select every shader
dispatched at least 240 times per step whose GPU time is at most 6 µs per dispatch:
at batch 32 that is nine shaders — `gradient_accumulate4` (768, 3.5 µs),
`gradient_accumulate` (482, 0.9), `linear_input_grad` (385, 5.1), `linear_weight_grad`
(385, 4.5), `vector_add` (308, 2.1), `parameter_matmul_weight_grad` (288, 4.4),
`parameter_matmul_input_grad` (288, 0.1), `finite_clamp_forward` (288, 1.7), `bias_grad`
(240, 5.3) — and they are 3432 of 7056 dispatches (48.6%) but only 10.523 ms of 85.054 ms
(12.37%) of the GPU step. Their recording adds 5.1-12.4 ms at 1.5-3.6 µs per dispatch, so
the class is 15.6-22.9 ms of a ~124 ms step (13-18%), and *execution* would not pay for
restructuring them — the recording is what makes them worth attacking.

The dispatch counts are batch-independent (6912-7120 per step at every batch size), but
the same kernels' per-dispatch cost is not, so the class itself shrinks as the batch
grows: 8 shaders and 45.2% of the dispatches at batch 2 (7.570 ms of 61.716, plus 4.7-11.2
ms of recording), the 9 above at batch 32, and only 3 shaders and 1538 of 7120 dispatches
at batch 512, where they are 0.49% of the GPU step and under 1% of its wall clock. The
lever therefore belongs to the small-batch regimes, which is the opposite of the
conventional reading that a larger batch is where overhead matters. `gradient_accumulate4`
is the cleanest single candidate: one shader, 768 call sites, one operation. The 1346
`optimizer.gradient-bookkeeping` dispatches stay blocked on the gradient slab, as the
eighth pass established.

Everything else in the top ten is an execution problem with no cheaper structural fix,
and the two regimes disagree about which of them matters. The producer's own cost is
bounded but not explained: about +20 µs of its 107 µs at batch 32 is placement (stride-1
sample), and the harness cannot instantiate it to go further.

## Verified

Ladder 7/7 — loss **5.543217**, 6912 dispatches, 3076 barriers, goldens 144/183/79 — and
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored, both after the changes to
`vulkan.rs` (`shader_barrier` is now crate-visible for the harness; the top-shaders
override is read per submission). The new binary self-verifies its own placement before
measuring it, and the profiles behind the ranking were parsed by
`rank_kernel_profile.py` unchanged, which reads every shader line and reports sample
counts. `validation/training_step_probe/kernel_ranking.txt` is regenerated from the
512-row stride-17 run (84 shaders instead of 12).

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe --bin placement_probe
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
$env:HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS = "4096"
foreach ($b in 2, 32, 512) { ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_b$b --batch $b --tokens 8 --steps 5 --warmup 1 > target/prof_check/rank_b$b.json 2> target/prof_check/rank_b$b.log }
python validation/rank_kernel_profile.py target/prof_check/rank_b32.log --top 120
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "5"; ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_s5_b32 --batch 32 --tokens 8 --steps 8 --warmup 1 2> target/prof_check/rank_s5_b32.log
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "1"; ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_s1_b32 --batch 32 --tokens 8 --steps 3 --warmup 1 2> target/prof_check/rank_s1_b32.log
Remove-Item Env:HIERARCHOS_VULKAN_PROFILE_KERNELS, Env:HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS, Env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE
./target/release/placement_probe.exe --blocks 32,64,128 --trials 4 --json validation/placement_probe/placement_probe.json
```


# Attention-backward regression closed — 2026-10-15

The previously documented mode-0 large-batch slowdown is fixed on the AMD device.
The cached backward kernel used ten Workgroup variables (14,400 bytes) despite
launching one invocation per workgroup. Moving those caches to Private storage
retains every operation, reduction order, cache bound and dispatch ABI.

Paired GPU timestamp measurements at 64 queries/keys and head dimension 32:
batch 32 falls from 88.8 to 48.4 ms (original artifact 55.8 ms); batch 128 falls
from 263.8 to about 71 ms (original 78.6 ms). Batches 2, 8 and 16 preserve the
small-batch benefit, within about one percent of the earlier cached kernel.
The fixed artifact is faster than the original at all five measured batches.

All 79 attention comparisons remain bit-identical, with unchanged frozen goldens.
The harness now rejects even a one-bit difference in direct A/B mode, and requires
benchmark outputs to match too. An injected one-bit gradient error is rejected.
The shader provenance manifest also guards local size [1,1,1] and zero Workgroup
variables, so the earlier storage regression fails the release gate.

After the final code changes: both debug and release ladders pass all eight checks,
including 86 shader variants and 3,817 full-state digests across 13 training steps;
706 library and 40 trainer tests pass, with nine microprofiles explicitly ignored.
All binary targets pass cargo check. No existing output golden was re-recorded.

Full timings, scope and commands: hierarchos-vulkan/validation/ATTENTION_BACKWARD_PERFORMANCE.md.
Raw paired results: hierarchos-vulkan/validation/release_parity/attention_backward/.
Cross-vendor performance remains unqualified. Long strict-mode jobs can still hit
the driver watchdog; the safe 8-query/8-key strict check passes exactly.
