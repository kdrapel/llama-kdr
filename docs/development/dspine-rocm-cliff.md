# DSpine verification workspace investigation

Target: `Qwen3.8-27B-UD-IQ2_XXS.gguf`, as explicitly corrected by the user. The IQ3_S model is not used in these measurements. RX 9060 XT 16 GB / gfx1200, Windows HIP, Ryzen 7800X3D. Source base: `47e76a3da64c3d7287fa7f9a09cf5cd8145b4324`. No commits or pushes were made.

## Finding

The confirmed source trigger is a retained, full-context **prefill workspace reservation**, combined with width-dependent recurrent rollback storage. A short verification graph inherits the large device allocation even though its populated KV is only 256-512 padded cells. The static reservation adds hundreds of MiB of unused attention mask and FP16 K/V staging space. Windows' process memory budget varies independently of configured context capacity, so there is no universal 28K or 32K threshold.

The source change limits the synthetic attention reservation to 256 cells for single-sequence speculative contexts. It preserves KV capacity, prompt microbatch size, attention operations, quantization, and acceptance logic. The existing graph allocator grows the workspace for actual inputs. Explicit full graph reservation and no-allocation memory-fit estimation keep their original full-capacity behavior.

In the paired, same-binary 131072 / n_max=15 / b=2048 / ub=512 control, target workspace fell from **780.02 to 190.51 MiB**, steady target graph time from **194.231 to 93.924 ms**, and decode throughput rose from **23.02 to 39.66 tok/s**. Both runs proposed 727 tokens, accepted 205, made 50 verification calls, and returned the identical message (SHA-256 `85e0768e2c2035d33bec16c322715272f85ede026970a1cb05eee90c7738a034`).

This establishes the allocation trigger and a corrective source change. It does **not** establish individual eviction, page-in, or migration events. WPR GPU recording failed with `0xc5585011` both normally and under elevated execution. DXGI budget/usage is not a residency or fault trace.

## Allocation and execution path

1. `llama_context::sched_reserve()` calls `memory->init_full()` and builds prompt-processing and token-generation graphs. The prompt shape uses configured `n_ubatch`.
2. The full `llama_kv_cache_context` initializes `n_kv` from `kv->get_size()`, the configured capacity. This is appropriate for a worst-case estimate but expensive as a permanent initial allocation.
3. Attention masks in `llama-graph.cpp` use `n_kv * n_tokens`. Their reservation therefore couples capacity with prompt microbatch size.
4. `ggml_cuda_flash_attn_ext_get_alloc_size()` calls `ggml_cuda_flash_attn_ext_get_f16_extra_data()`. Tile/MMA paths append FP16 conversions sized from the K/V tensor dimensions. At 131072 cells, the target's per-layer K+V conversion space alone can reach 512 MiB. These are allocator-managed bytes appended to the attention output allocation, not merely a separate attention pool visible in normal logs.
5. The graph allocator reuses its allocated buffers; executing a smaller graph does not automatically shrink them. Zero graph reallocations in the old logs therefore did not establish an efficient allocation size.
6. Actual decoding calls `llama_kv_cache_context::apply()`, which obtains populated/padded KV through `get_n_kv()`. The short request stays at 256-512, regardless of capacity. The selected attention shapes remain small, but the large allocation stays alive.
7. `ggml_backend_sched_alloc_splits()` / `ggml_gallocr_reserve_n()` already synchronize and grow allocations when the actual graph no longer fits. The fix uses this existing path rather than introducing a second allocator or changing kernel semantics.

Width has a second, large allocation effect. `llama_memory_recurrent` allocates `mem_size * (1 + n_rs_seq)` rows. For this target's 48 recurrent layers, each matrix-state plane totals 144 MiB. At n_max=15, matrix snapshots use 2304 MiB, plus 90 MiB of convolution history: **2394 MiB**. At n_max=6, the total is **1047.375 MiB**. This explains why reducing width is a powerful memory control even with identical populated KV.

Other measured persistent allocations:

| Category | MiB |
|---|---:|
| Target device weights | 6521.13 |
| Draft device weights | 1189.18 |
| Target Q4_0 KV, 24576 / 32768 / 65536 / 131072 capacity | 432 / 576 / 1152 / 2304 |
| Draft SWA KV | 165.00 |
| Target DFlash feature buffer, b=2048 | 200.00 |
| Draft compute buffer, draft ubatch=32 | 30.94 |
| Attention rotations | 4.25 |

The feature buffer is batch-sized because features from multiple target microbatches are indexed using original batch rows. Shrinking it to verification width without changing that lifetime/indexing contract would be unsafe. It was left unchanged.

## Memory telemetry and interpretation

`GGML_CUDA_CLIFF_PROFILE=1` enables backend allocation/free logs, requested-byte live/peak accounting, in-process DXGI queries, and synchronized whole-graph timings. The accounting distinguishes backend buffers from legacy pool allocations. Normal initialization logs identify weights, KV, rollback state, features, and compute buffers by category.

The counter covers allocations routed through `ggml_cuda_device_malloc`; it does not cover opaque HIP/hipBLAS/driver allocations or host-pinned buffers. Therefore tracked requested-byte peaks are not total GPU usage. DXGI queries enumerate both AMD adapters; adapter 0 is the discrete GPU on this machine. Adapter 1 is the integrated adapter and is not added to the discrete GPU figures.

For the paired 131K / n15 control:

| Measurement | Full reservation | Reduced reservation |
|---|---:|---:|
| Tracked peak device allocations, MiB | 13877.24 | 13287.73 |
| Decode DXGI local usage range, MiB | 15029.69-15131.56 | 13737.44-13839.31 |
| Decode DXGI local budget range, MiB | 13991.87-14049.15 | 14100.52-14197.32 |
| Decode DXGI nonlocal usage range, MiB | 949.20-964.36 | 693.70-708.86 |
| Target device workspace, MiB | 780.02 | 190.51 |

The DXGI difference exceeds the requested-byte difference. The untracked difference must not be assigned to a particular runtime allocation or migration mechanism without another trace. Nonlocal usage includes legitimate host-visible resources and cannot, by itself, prove paging.

[Microsoft documents DXGI budget and current process usage separately](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info), and describes possible performance penalties when usage exceeds budget. Exceeding budget is a pressure indicator, not a sufficient test for a performance cliff: the current old-policy n6 control is slightly over budget yet remains fast.

Early runs had approximately 13200 MiB of budget; later controls had approximately 14100 MiB. Under the earlier budget, the old 64K / n15 case ran at 25.38 tok/s, while reduced reservation ran at 39.75 tok/s even with a slightly smaller budget. Under the later budget, the old 64K case already ran at 41.48 tok/s. This challenges the original handoff's interpretation of a fixed capacity threshold. Under the earlier budget, reduced reservation alone still left 131K / n15 near 25 tok/s; sufficient headroom is still required.

## Benchmark method

Use `tools/dspine-cliff-bench.py` and `tools/dspine-cliff-validate.py`. They port the existing `build-rocm/bench/conv_ab.sh` request to native Python, use the same model pair and JSON request, start one server at a time, and terminate only that server. No dependency installation is needed. Each run stores exact argv/environment/request, raw log, and response under `build-rocm/bench/`. New runs also store measured request wall time.

Common settings: np=1, all layers on GPU, load-mode none, 8 threads, Flash Attention on, Q4_0 K/V, b=2048, draft-dflash, draft ubatch=32, temperature=0.6, top-p=0.95, top-k=20, request seed=42, max_tokens=256. Configured capacity is varied; the short Fibonacci request contains 78 prompt tokens. Investigation comparisons use `GGML_CUDA_GDN_FUSION=0` to match the handoff. Production-default fusion has not been checked in this continuation.

`predicted_per_second` is the server's complete decode throughput, including draft generation and verification. It excludes prefill, and its effective numerator is 255 for this 256-token response. Prompt throughput and total/request timing are separate. Steady target graph time is measured directly with synchronization, excluding graph capture/update calls; it is not inferred from tokens/s. Instrumented comparisons have identical instrumentation on both sides.

`LLAMA_GRAPH_RESERVE_FULL=1` restores the old reservation policy in the same rebuilt binary. It is a diagnostic A/B switch, not a required production setting. No-allocation fitting is always full-capacity, independently of that switch.

### Paired policy controls

All rows use n_max=15 and b=2048; draft ubatch=32. These later controls had a larger WDDM budget than the early reproduction.

| Capacity | ub | Old tok/s | Fixed tok/s | Old verification ms | Fixed verification ms |
|---:|---:|---:|---:|---:|---:|
| 24576 | 1024 | 39.06 | 39.80 | 98.73 | 97.86 |
| 32768 | 1024 | 40.76 | 38.94 | 95.02 | 96.64 |
| 65536 | 1024 | 41.48 | 39.68 | 93.80 | 93.98 |
| 131072 | 512 | 23.02 | 39.66 | 194.23 | 93.92 |

The 32K/64K later throughput differences are not speedups and should not be presented as such. Their steady graph times are nearly unchanged. The reduction is an allocation/headroom improvement when the old reservation is already within the available budget.

For comparison, the initial eight-case reproduction used a smaller budget:

| Capacity | n_max | ub | Initial old-policy tok/s |
|---:|---:|---:|---:|
| 24576 | 15 | 1024 | 40.96 |
| 32768 | 15 | 1024 | 36.52 |
| 32768 | 15 | 512 | 42.01 |
| 65536 | 15 | 1024 | 25.38 |
| 65536 | 15 | 512 | 25.22 |
| 65536 | 6 | 1024 | 45.76 |
| 131072 | 15 | 512 | 22.72 |
| 131072 | 6 | 1024 | 32.92 |

These are a distinct budget epoch, not paired controls for later 32K/64K runs. The raw logs and portable result JSON retain their measured budgets where profiling was enabled.

### Normalized width sweep with the fix

All cells use b=2048, ub=512, draft ubatch=32, the same 78-token prompt and 256-token generation limit. These are directly comparable configuration settings; Windows budgets still vary by run.

| Configured capacity | n_max=6 tok/s | n_max=9 tok/s | n_max=12 tok/s | n_max=15 tok/s |
|---:|---:|---:|---:|---:|
| 24576 | 47.95 | 38.94 | 39.91 | 40.41 |
| 32768 | 47.76 | 38.24 | 39.15 | 39.69 |
| 65536 | 47.98 | 38.28 | 39.65 | 39.74 |
| 131072 | 47.91 | 38.97 | 39.82 | 41.57 |

At 131072, steady target verification times were 65.86 / 90.78 / 93.65 / 93.46 ms for widths 6 / 9 / 12 / 15. The earlier wide-window cliff is absent in this sweep. The remaining width penalty is normal execution/acceptance efficiency for this workload.

| n_max | Proposed tokens | Accepted tokens | Verification calls | Acceptance | Accepted/call | Proposed/call + anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 330 | 199 | 55 | 60.30% | 3.618 | 7.00 |
| 9 | 490 | 200 | 55 | 40.82% | 3.636 | 9.91 |
| 12 | 611 | 203 | 52 | 33.22% | 3.904 | 12.75 |
| 15 | 727 | 205 | 50 | 28.20% | 4.100 | 15.54 |

The final column is a proposed-token-derived window estimate, not a separate count of every target graph invocation. The profiler also records actual query columns and padded populated KV for each graph. Width comparisons produce different sampling trajectories; the old/new comparisons **within a width** preserve output and acceptance. Wider speculation gains relatively few accepted tokens per verification here while costing substantially more per target step. n_max=6 is the throughput recommendation for this prompt/model/backend, not a universal optimum for every workload.

### Operation-level diagnostic comparison

The intrusive 48-token paired diagnostic requests at 24K and 64K each proposed 69 and accepted 41 draft tokens in six calls. Summed synchronized wall times after sampler initialization were:

| First operation in a fused group | 24K ms | 64K ms |
|---|---:|---:|
| MUL_MAT | 1682.48 | 2323.88 |
| RMS_NORM | 181.52 | 260.84 |
| ADD | 112.99 | 190.83 |
| GATED_DELTA_NET | 55.53 | 55.70 |
| FLASH_ATTN_EXT | 37.00 | 44.94 |

Most of the additional diagnostic time is in matrix operations, not Flash Attention execution. FA's **reservation path** is implicated separately by its allocation sizing. The timer serializes operations and changes caching/launch behavior, so these totals must not be treated as production kernel timings or extrapolated to the 256-token request. Tiny HIP event intervals were sometimes invalid; whole-graph timings are the stronger execution evidence. GPU transfers, CPU graph construction, and synchronization are not separately attributed by a kernel trace.

## Rejected experiments

- Disable HIP graphs: approximately 40.31 / 24.88 / 22.45 tok/s for 24K / 64K / 131K n15, versus 40.43 / 25.10 / 22.55 with graphs in the early instrumented epoch. It does not remove the cliff.
- Split KV into per-layer allocations: no improvement at 131K; 64K worsened in that run. Not retained.
- Reduce the initial token-batch reservation as well as KV reservation: no useful additional 131K/n15 improvement. Not retained; prompt microbatch reservation remains unchanged.
- Split IQ1_M fallback matmuls into existing MMVQ calls: helped 64K in one run, did not fix 131K, and changed rounding/acceptance trajectories. Not retained.
- Chunk BLAS weight dequantization into 32 MiB row groups: reduced pool allocations but worsened total throughput and changed rounding trajectories. Not retained.

The latter experiments exposed retained 105 MiB and 178.5 MiB legacy-pool allocations from unsupported-for-MMQ IQ1_M matrices. They are additional pressure, but replacing their arithmetic path was not a justified fix here. Source for the discarded experiments is archived locally in `build-rocm/bench/experimental-source.patch`.

## Validation and remaining limits

- Build: incremental Release HIP/gfx1200 build through `build-rocm-inc.ps1`, existing ROCm toolchain, no new dependencies.
- `test-alloc`: passes.
- New `reserve_grow` context test: passes with Q4_0 and FP16 KV. It asserts lower initial workspace, unchanged cache capacity, and matching logits against an explicit full-reservation reference across 256- and 512-cell boundaries.
- The same test fails with the old policy, as intended: `initial workspace was not reduced`.
- Existing `noalloc` context test: passes; memory-fit simulation remains full-capacity.
- Existing FP16 `rollback` case passed. The broader `rollback_bound` case hit an HIP launch failure with the new policy and also with `LLAMA_GRAPH_RESERVE_FULL=1`. This pre-existing failure is unresolved and is not counted as a passing regression check. See `cliff-context-f16.log` and `cliff-rollback-bound-full.log` in the benchmark directory.
- Matched-width short requests return identical message hashes, proposal counts, and acceptance counts across old/new policies. No attention or acceptance algorithm was changed.

The final rebuilt test executable passed `reserve_grow|noalloc` with Q4_0 KV (2/2) and `reserve_grow|rollback` with FP16 KV (2/2). The context test setup can disable FA when its unified-KV row path is unsupported; those unit checks establish allocator growth and logit correctness, while the server benchmarks exercise the requested FA-on configuration. Growth/reference reservation also produces existing destructor warnings that actual buffer size differs from the initial expected size; these were observed and are not silent test failures.

The optimization reduces an initial retained reservation; it does not reduce the physical KV cache or required rollback snapshots. Allocator growth after a long prefill can retain larger workspace again. A fully populated 131K conversation was not benchmarked. Lower available WDDM budgets or competing GPU workloads can still make wide verification unattractive. No universal paging-free guarantee is established.

## Recommended test/production configuration

Use [dspine-rocm-launcher.json](dspine-rocm-launcher.json) with the rebuilt `build-rocm/bin` libraries. It preserves context=131072, b=2048, ub=1024, Q4_0 KV, and draft ubatch=32, and recommends n_max=6 for throughput and memory margin. The fix is active by default; leave `LLAMA_GRAPH_RESERVE_FULL` unset. A wide-window comparison profile can use n_max=15 and ub=512; it now reaches approximately 40 tok/s with sufficient budget, but remains slower than width 6 on this request.

The measured investigation environment sets `GGML_CUDA_GDN_FUSION=0`. Set that environment variable before starting the launcher to reproduce these results; the JSON follows the existing launcher schema and does not invent an environment-setting field. Fusion enabled/default still needs the user's production test.

### Handoff at the user's stop request

The source and final test executable are built. The eight-case reproduction, instrumented matrix, paired reservation-policy controls, and all 16 normalized width cases are complete. Further automated testing was stopped at the user's request.

The following checks are prepared in `dspine-cliff-validate.py` but **were not run**: longer-prompt old/new comparison (`prompt`), ordinary decoding comparison (`ordinary`), final uninstrumented paired comparison (`unprofiled`), an explicit draft-ubatch 0 versus 32 comparison (`draftub`), and default-fusion launcher validation (`production`). All reported speculative results preserve draft ubatch=32. Short-request prompt speeds are recorded, but a long-prompt processing regression assessment remains pending.

## Reproduction commands

```powershell
.\build-rocm-inc.ps1 -SkipConfigure -ROCmPath D:/AMD/ROCM/build -Parallel 4
python tools/dspine-cliff-validate.py controls
python tools/dspine-cliff-validate.py widths
python tools/dspine-cliff-validate.py prompt
python tools/dspine-cliff-validate.py ordinary
python tools/dspine-cliff-validate.py unprofiled
python tools/dspine-cliff-validate.py draftub
python tools/dspine-cliff-validate.py production
python tools/dspine-cliff-export.py
```

Use fresh tags for ad-hoc runs. `GGML_CUDA_CLIFF_OPS=1` adds intrusive per-operation synchronization and HIP event intervals; this mode disables graph capture. Small/negative HIP event intervals were observed for tiny operations, so these are diagnostic comparisons rather than a trustworthy kernel trace. The final policy controls use whole-graph synchronization, not per-operation profiling.

Portable metrics are in [dspine-rocm-cliff-results.json](dspine-rocm-cliff-results.json). Full raw logs, responses, commands, and experiment build logs remain under `build-rocm/bench/`.
