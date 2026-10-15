# Hierarchos training objectives — 2026-10-15

Scope: use the existing Transformer trainers as a read-only reference for
Hierarchos training discipline. Transformer implementations, shaders, and
training behavior are unchanged. Pre-existing deletions in the packaged `dist/`
tree are outside this work.

The earlier causal-LM audit already established shifted labels, `-100` prompt
masks, weighted gradient accumulation, tied embedding/head ownership, padding
state identity, and dense/sparse replay equivalence. This follow-up repairs
remaining objective and data-boundary issues rather than replacing coherent-v9
with a Transformer graph.

## Changes

- Hierarchos CE kernels now treat an exactly zero source coefficient as an
  absent language-loss edge before evaluating probabilities or reducing LM-head
  parameter gradients. This carries the Transformer CE kernel's zero-weight
  discipline into streaming FP32, packed/native FP16, tape, and fused variants.
  The guard does not suppress recurrent, embedding, or auxiliary gradients.
- CE statistic kernels evaluate `log(exp_sum) - (target_logit - max_logit)`.
  This is the same cross-entropy in real arithmetic, but preserves small losses
  when every logit has a large common offset. Explicit `precise` staging is
  required: the Vulkan regression reproduced CE = 0 at a common offset of
  `1e8` when the driver could reassociate the subtraction. Z-loss uses the unshifted
  log partition required by its existing definition.
- Hierarchos text preparation appends EOS once, retains BOS, and removes a
  tokenizer-inserted closing EOS from an unfinished prompt before attaching the
  response. Literal prompt tokens remain intact. A private Hierarchos-only
  option keeps shared helpers' Transformer behavior unchanged.
- Native Hierarchos chat resolves EOS with the same tokenizer metadata and
  standard token names as training, so termination and early-EOS guards use
  the token that was supervised during fine-tuning. Chat prefill and training
  share the unfinished-prompt token handling, including literal EOS and shared
  BOS/EOS conventions.
- `HierarchosTokenTapeTrainResult.language_loss` exposes the original weighted
  language numerator and supervision mass. Trainer batch/window/epoch metrics
  merge these quantities instead of averaging unweighted row diagnostics or
  independently normalized means. This reports CE + z-loss before backward
  caps, excluding ponder, commitment, and LTM. Historical `losses` diagnostics
  retain their existing meaning. No extra GPU readback or optimizer operation
  is added.

For shifted targets with weights `w`, the language objective remains

```text
sum(w * (-log softmax(logits)[target] + z_weight * logsumexp(logits)^2))
    / sum(w)
```

H/L recurrence, hard ACT, worker refinement, context interpolation, state-derived
drift, ROSA/LTM/persistent memory, ponder/commitment denominators, TBPTT cuts,
backward-cap policy, and AdamW equations are preserved. Nonfinite active model
paths still require the existing numerical checks and loss-scaling policy; a
zero CE source is not a guarantee that every other graph edge is finite.

## Validation

Maintained regressions cover independent Rust CPU/Vulkan causal loss and
continuation from nonzero H/L/context states across manager updates and bounded
ROSA rollover, dense/sparse replay and TBPTT, weighted metrics with unequal
fractional supervision masses, and isolated CE adjoints with poisoned ignored
rows. The metric fixture also checks ignored prompt labels, right padding,
lookahead targets, z-loss, and a backward cap below the reported loss.

`hierarchos-vulkan/validation/verify_hierarchos_ce_shaders.py` reproduces and
validates the 32 affected embedded SPIR-V variants. `--write` explicitly
regenerates artifacts; the default only verifies source/artifact hashes.

These are deterministic native correctness and learning checks, not a
production-scale language-quality or mixed-precision convergence certificate.

## Final results

All checks below exited successfully against the final source and artifacts.
The Vulkan device was AMD Radeon Graphics.

| Check | Result |
| --- | --- |
| Native causal-LM suite | 6 passed; Intel-specific dispatch comparison unavailable on this AMD device |
| Isolated CE adjoints and common-offset stability | 3 passed |
| Hierarchos trainer | 40 passed |
| Objective/TBPTT weighting and backward caps | 10 passed |
| Token frontend | 4 passed; 1 existing occupancy benchmark ignored |
| Pure-Rust inference | 12 passed |
| Native CLI, including shared Transformer behavior | 38 passed |
| CE shader reproduction and Vulkan 1.1 validation | 32/32 matched |
| Whitespace and Transformer/inference source diff | Passed; protected source diff empty |

The existing 12-update native learning/export check still reduced shifted CE
from **2.5682123** to **0.9756531**, measured finally by Rust inference on the
exported weights. The independent weighted-metric fixture reported a numerator
of **21.9226581** with supervision mass **7.5**, giving **2.9230211**; the old
all-row diagnostic mean was **2.9628715**.

Reproduce with:

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib hierarchos_causal_lm -j 1 -- --nocapture --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib hierarchos_ce_ -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib pytorch_tbptt_weighting_tests -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib token_frontend::tests -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-inference/Cargo.toml --lib -j 1
cargo test --manifest-path hierarchos-native-cli/Cargo.toml --lib -j 1 -- --test-threads=1
python hierarchos-vulkan/validation/verify_hierarchos_ce_shaders.py --output hierarchos-vulkan/validation/hierarchos_ce_shader_report.json
```

Local execution evidence is preserved in `.hierarchos-objective-port-*.log`
and `.hierarchos-cli-objective-tests.log`; shader provenance is checked in at
`hierarchos-vulkan/validation/hierarchos_ce_shader_report.json`.
