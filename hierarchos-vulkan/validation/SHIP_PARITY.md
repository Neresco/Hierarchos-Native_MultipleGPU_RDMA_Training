# Training optimization numerical sign-off — 2026-10-15

All numerical gates passed on the local **AMD Radeon Graphics** device in debug
and release builds. The optimizations were retained. No tolerances were widened
and the existing kernel goldens were not re-recorded in this sign-off.

| Check | Result |
| --- | --- |
| Direct comparison against original embedded shaders | 144 dense + 183 reduction + 79 attention invocations; all bit-identical |
| Full training trajectory against the pre-optimization implementation | 3,817 exact tensor digests over 13 optimizer steps; zero differences |
| Shader source/artifact provenance and SPIR-V validity | 86/86 affected variants reproduce byte-for-byte and validate |
| Debug and release verification ladder | All eight checks pass in both profiles |
| Library tests | 706 passed, zero failed, nine ignored |
| Trainer tests | 40 passed, zero failed |
| Eight-way gradient accumulation | Exact match to both four-way and scalar fallbacks across 12 tensor counts, tails and 256-element group boundaries |
| Verification negative controls | Parameter/moment drift, missing tensors/steps and wrong optimizer clocks are rejected |
| All binary targets | `cargo check --bins` passes |
| Standalone source package | All 62 Cargo target source paths exist after staging |

## Baseline and trajectory coverage

The baseline restores `vulkan.rs`, `rwkv_optimizer.rs` and 50 optimization-touched
shader artifacts from commit `5004a8a2923aa0ac89deeffa241f5d7d7f65edc1` in an
isolated source copy. It retains the CE/objective changes that existed before
this optimization task, along with the other current host code. Baseline and
candidate consume the same initialized model package for each direct comparison.
The original artifact hashes are in `release_parity/baseline_manifest.json`.

The trajectory fixtures cover batch 2 for six updates, batch 32 for four updates,
and batch 9 with two independent sequences and width 256 for three updates.
After every update, checks include every trainable parameter, both Adam moments,
global/per-slot optimizer clocks, token losses, final H/L recurrent states,
initial-state adjoints, and context carriers. Hashing uses the full raw FP32 bit
patterns and vector lengths; non-finite snapshots fail. State goldens were frozen
from the verified **baseline**, not from candidate outputs.

## Repairs to verification and packaging

The reduction harness's optional A/B mode compared baseline outputs with candidate
buffers before dispatching the candidate. All three affected paths now dispatch
first. The verification ladder exercises both A/B and frozen-golden comparison
on every reduction run, so this ordering regression cannot hide behind golden-only
CI runs. The corrected harness passes all 183 direct comparisons with HEAD.

The ladder formerly checked only the 32 CE variants for provenance. It now uses
`training_shader_manifest.json` to check all 86 affected variants with their actual
compiler/flags, including the latest low-rank producer and `gradient_accumulate8`.
The manifest records attention's glslang build, reduction optimization flags, and
the native FP16 specialization separately.

`training_step_probe --state-output FILE` adds untimed full-state snapshots.
`verify_training_state.py` compares those snapshots against the immutable baseline
goldens, and is now a mandatory ladder check. The standalone packager also includes
`validation/`, which Cargo's new attention harness target requires.

## Reproduce

From `hierarchos-vulkan`:

```powershell
python validation/verify_training_stack.py --release --receipt target/release_parity.json
python validation/verify_training_stack.py --receipt target/debug_parity.json
cargo test --lib --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
python -m unittest discover -s validation -p test_training_state.py -v
cargo check --bins
```

Run the two ladder profiles sequentially. New hardware requires independently
qualified device goldens; an unrecorded device fails the gate. Do not re-record
expectations to make an unintended numerical change pass.

Machine-readable receipts, direct comparison logs, full test output and a hash
inventory of the tested sources/artifacts are under `release_parity/`.

The subsequent [attention-backward performance fix](ATTENTION_BACKWARD_PERFORMANCE.md)
retains the same output goldens and numerical trajectory: invocation-private
caches remove the unnecessary 14,400-byte shared-memory reservation and close the
measured slowdown at larger batches. The shader provenance gate now also checks
that this kernel keeps its single-invocation ABI and zero workgroup variables.

This sign-off establishes numerical parity for the tested fixtures on this AMD
device. Cross-vendor parity remains unverified. Long strict-backward benchmark
jobs can still hit this driver's watchdog with either artifact. The nine
explicitly ignored library tests were not run.
