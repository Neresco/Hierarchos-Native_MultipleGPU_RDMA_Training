# Attention backward regression closure — 2026-10-15

The measured mode-0 attention-backward slowdown at larger batches is closed on
the local AMD Radeon Graphics device. Numerical outputs remain bit-identical.

## Cause and change

`transformer_attention_backward` launches one invocation per workgroup. Its cache
optimization introduced ten workgroup variables, reserving 14,400 bytes even for
attention modes that do not use the strict-adjoint caches. Those arrays have only
one reader/writer in each workgroup, so they can use invocation-private storage.

Changing the ten storage declarations removes all Workgroup variables from the
compiled module. No arithmetic, loop order, descriptor binding, push constant,
dispatch geometry, cache bound or fallback changes. Driver register/scratch
allocation replaces the fixed shared-memory reservation.

## Measured results

GPU timestamp medians of five samples, one dispatch per sample. All rows use
mode 0, 64 queries, 64 keys, four query heads, two KV heads and head dimension 32.
The harness compares the complete output vectors bit for bit before reporting a
benchmark result.

| Batch | Original artifact | Regressed shared caches | Fixed private caches |
| ---: | ---: | ---: | ---: |
| 2 | 52.9 ms | 44.1 ms | 44.4 ms |
| 8 | 54.3 ms | 45.2 ms | 45.5 ms |
| 16 | 54.8 ms | 45.4 ms | 45.9 ms |
| 32 | 55.8 ms | 88.8 ms | 48.4 ms |
| 128 | 78.6 ms | 263.8 ms | 71.1 ms |

Original-versus-fixed and regressed-versus-fixed are separate paired runs; the
table shows the fixed result from the original pair. Both paired measurements
are retained in the JSON receipt. The batch-128 regressed-versus-fixed pair is
263.776 / 70.874 ms, a 3.72× improvement. The fixed kernel is 1.10–1.20× faster
than the original across the tested batches, while small-batch differences from
the earlier cached kernel are about one percent.

Strict mode 1 was also compared at a safe eight-query/eight-key shape: batch 32
is 46.1 ms original versus 1.4 ms fixed, with exact gradients. This smaller shape
does not establish that long strict-mode jobs avoid the driver's watchdog.

## Regression protection and verification

- All 79 existing attention invocations pass direct original-artifact comparison
  and unchanged frozen output digests, including strict modes and cache fallbacks.
- Both numerical verification profiles retain their original full-state goldens.
  See [SHIP_PARITY.md](SHIP_PARITY.md) and the refreshed release receipts.
- The shader manifest requires local size `[1,1,1]` and zero workgroup variables.
  The former artifact has ten and fails that storage policy.
- Baseline comparison in the attention harness now fails on any differing FP32
  bit, including one-ULP differences. Benchmark mode also checks complete output
  vectors instead of discarding them.
- `--bench-only` cannot bypass golden verification, and `--bench-mode 0` avoids
  automatically running the long strict-mode benchmark that resets this driver.

From `hierarchos-vulkan`, with the original shader artifacts in `BASELINE_DIR`:

```powershell
cargo build --bin attention_kernel_cache_check
./target/debug/attention_kernel_cache_check.exe BASELINE_DIR --golden validation/kernel_golden/attention.json --golden-strict
./target/debug/attention_kernel_cache_check.exe BASELINE_DIR --bench --bench-only --bench-mode 0 --bench-repeats 1 --bench-batch 128
python validation/verify_training_stack.py --release
```

Detailed timings and comparison logs are under `release_parity/attention_backward/`.
These are kernel timings on one AMD device, not whole-training throughput or
cross-vendor performance claims.
