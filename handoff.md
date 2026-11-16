# Intermediate handoff: 2026-11-16

## Measurement retention requirement

User direction (2026-11-16): always keep every measured statistic. Save raw timing/acceptance, exact requests, commands/environment, responses and binary hashes under `docs/development/measurements/rdna4-q4/`, not only inside the disposable build tree. Keep prior samples; use distinct run tags instead of overwriting. Record failures and validity limits alongside successful runs. The 8K/64K A/B artifacts and isolated kernel timing data have been copied there. The 98304-token prompt, 256-output-token candidate/stock follow-up is complete. New harness `tools/dspine-fattn-ab.py` defaults to this archive and refuses existing request/summary tags.

## Latest measurement: 96K, 256 output tokens

### Later short-context audit and historical CSV check

- User reports prior fresh GSQ-RCO singleton-question rates of 35-40 tok/s, sometimes 55. Exact current prompt `What is a singleton in C#?` renders to 48 tokens. Same 131072 capacity, temperature 0.6, seed 42 and DFlash width 6, fresh server per sample.
- Fixed 256-token candidate/stock/stock/candidate: candidate 28.450/28.587 tok/s, mean 28.519; stock 29.441/29.369, mean 29.405. Candidate is 3.014% slower. Acceptance repeats at 185/416 candidate versus 188/388 stock; outputs differ across variants. Full-answer pair: stock 643 tokens at 28.095 tok/s; candidate 708 at 29.605; both EOS, but different output lengths prevent a direct latency comparison.
- User specifically requested historical CSV checks. `dspine-rocm-speed-curve-gsq-rco-8k-100k.csv` records GSQ 8K generation 45.733 tok/s from two 96-output-token runs with 78/102 draft acceptance. The `bc7c78ef95ac8b65ae50eeff7d02b544d6cc62ee` curve already records GSQ 8K 28.420 tok/s for 256 outputs, with 195/360 acceptance, before the new attention kernel. Both use temperature 0.6, but prompts, lengths and binaries differ. The 1K-96K curve reporting rates up to 55 tok/s is labeled UD-IQ2, not GSQ. None of those named GSQ curves contains the empty singleton prompt.
- Policy discussion: populated KV length and query width are useful dispatch inputs; allocated capacity alone is not. Measure 16K/32K crossover and distinguish nb=1 from verification batches before choosing thresholds. Low short-prompt acceptance also motivates a separate draft-width sweep. No dispatch or draft-width policy was changed.
- Full report: `docs/development/dspine-rocm-short-context-audit.md`. Every sample and historical comparison is retained in the measurement archive, including `fa-qr-ab-singleton-20261116*`, `fa-qr-ab-singleton-full-20261116*` and `historical-gsq-and-singleton-audit-20261116.json`. Harness now supports exact prompt text, sampling temperature and EOS-complete requests.
- Existing user-created commit `f7ccf262f` contains three MMVQ files. It does not include the measurement archive/report. The earlier request to commit measurements remained pending after the staging turn was interrupted.

- One candidate/stock pair, same binaries and settings as the prior checks, exact same 98304-token prompt, greedy sampling, profiling off and zero prompt-cache reuse. Both emitted exactly 256 tokens and completed without inference errors; processes were cleaned up.
- Candidate: prefill 622.872609 s / 157.823604 tok/s; generation 7.579862 s / 33.641773 tok/s; acceptance 210/266 (78.947%); wall time 630.521724 s.
- Stock: prefill 726.335260 s / 135.342459 tok/s; generation 12.960412 s / 19.675300 tok/s; acceptance 205/294 (69.728%); wall time 739.353807 s.
- Measured candidate differences: generation throughput +70.9848%; prefill throughput +16.6106%; prefill latency -14.2445%, saving 103.462651 s. Outputs differ from token index 3, and acceptance changes. Bulk prefill width 512 is outside the new gate. These are single-pair observed gains with confounding, not confirmed kernel-only speedups or a prefill optimization.
- Preserve every sample. Exact requests, full responses, hashes, commands, logs and derived statistics are in `docs/development/measurements/rdna4-q4/fa-qr-ab-96k-20261116*`; latest aggregate is `fa-qr-ab-comparison-through-96k-20261116-v2.json`. Earlier aggregates were kept. Full report: `docs/development/dspine-rocm-rdna4-q4-ab.md`.
- No kernel changes, rebuilds or additional commits were made for this run. Previous requested commit is `7dfebaa12`. Source is still broadly enabled for eligible 1-8 token calls. Confirm gains with reversed repeats and matched generated work before deciding on dispatch; do not infer a context-scaling curve from different outputs/prompts.

## Latest continuation: dense RDNA4 Q4 decode

### Real-model A/B completed (2026-11-16)

This A/B supersedes the pending-inference notes below. Full method and evidence: `docs/development/dspine-rocm-rdna4-q4-ab.md`. No kernel or dispatch changes were made during this measurement task.

- Same-binary GSQ-RCO/DSpine, 131072 capacity, target Q4_0 KV, ubatch 512, draft F16 KV/ubatch 32, DFlash n_max=6, profiling off, greedy sampling, seed 42, 256 generated tokens. Exact synthetic-history prompt arrays were identical within each depth; every measured request had zero cache reuse.
- 8K, candidate/stock/stock/candidate: candidate 27.708/31.134 tok/s (mean 29.421), stock 31.909/31.883 (mean 31.896). Candidate mean is 7.76% lower. Acceptance repeats at 204/301 candidate versus 206/294 stock. Each variant repeats its own output, but variants differ from output token index 122. No measured 8K gain.
- 64K, single stock/candidate pair: 21.829 versus 26.073 tok/s, candidate +19.44%, identical 256 output token IDs. Acceptance 206/292 stock versus 206/290 candidate. Prefill also varied materially (415.159 versus 355.745 s), so the decode gain is provisional; a reversed pair is needed before claiming a kernel-only improvement.
- Separate completed-answer checks at 64K: both Fibonacci functions matched an iterative reference for n=0..1000 and rejected six invalid inputs. Both answers contain explanatory errors (stock's proof drops a square; candidate's bit-complexity claim is false). This is bounded code validation, not general quality validation.
- Evidence is now in `build-rocm/bench/`, including `run_fa_qr_ab.py`, `fa-qr-ab-{8k,64k}-20261116*`, `fa-qr-ab-comparison.json` and `fa-qr-ab-quality-validation.json`. Processes were cleaned up. The previous note that `bench/` is absent applies only to historical artifacts missing before this new harness was created.
- Next decision: do not assume the broad current gate improves real inference. Reverse/repeat the 64K pair to confirm the possible long-context gain; any context-dependent dispatch change needs its own measured decision. Current source still has the preceding agent's enabled eligibility gate.

This section supersedes older statements below about the current small-batch attention dispatch and latest build. The objective remains GSQ-RCO inference on gfx1200; sparse-prefill work remains stopped.

- Resumed the preceding agent's `fattn-rdna4-q4.{cu,cuh}` / `fattn-rdna4-q4-kernel.cuh` prototype. It uses RDNA4 WMMA and direct Q4_0 K/V reads for D256, GQA6, 1-8 query tokens. `fattn.cu` routes matching calls and reserves split-output scratch. Existing eligibility checks reject sinks, bias, softcap and auxiliary row/block inputs. `GGML_CUDA_DISABLE_FA_QR=1` selects the stock path. The prototype remains enabled for eligible shapes; this is local experimental work, not end-to-end validated production performance.
- The previous agent's mask fix was already present: the launcher and kernel now pass `const char * mask`, apply byte-valued `mask_nb1`, then cast the selected row to `const half *`. The former half-pointer arithmetic doubled mask row offsets, causing multi-token NaNs. No kernel change was needed during this continuation.
- Expanded existing backend tests to widths 1-8, KV lengths 65/512/8192, masked and unmasked, plus masked KV65536 widths 1/7/8. Added four performance cases for KV8192/65536 and widths 1/7.
- Full ROCm build succeeded in 386 s (`build-fa-qr-resume.log`); final incremental test build succeeded in 15 s (`build-fa-qr-resume-final.log`). Focused correctness: 64/64 (`fa-qr-resume-correctness.log`). Broader Q4_0/Q4_0 attention: 429/429 (`fa-qr-resume-q4-suite.log`), followed by the three newly added exact-64K cases: 3/3 (`fa-qr-resume-64k-correctness.log`). These compare GPU output against CPU. Gate logging confirms the specialized kernel runs on gfx1200. `git diff --check` passed.
- Same-binary isolated timings, order candidate/stock/stock/candidate, microseconds per operation: KV8192 nb1 candidate 46.05/46.35 versus stock 188.79/189.78; KV8192 nb7 162.30/162.03 versus 894.62/966.66; KV65536 nb1 294.49/294.23 versus 1309.85/1294.76; KV65536 nb7 876.34/876.59 versus 5932.41/5907.30. The key seven-token 64K call is about 6.75x faster in this backend benchmark. These random-input microbenchmarks do not establish generation throughput or task quality.
- Artifacts are at the repository root: `fa-qr-resume-perf-{candidate,stock}-{1,2}.log` and `fa-qr-resume-perf-results.json`. Source and tests remain uncommitted; unrelated pre-existing modifications were preserved.
- Next: paired GSQ-RCO inference with identical prompts/settings, populated contexts and output length, profiling disabled, candidate versus `GGML_CUDA_DISABLE_FA_QR=1`. Check output quality, draft acceptance and decode throughput. Existing handoff command-JSON paths below refer to historical artifacts; this current build tree has no `bench/` directory, so verify those files exist before relying on them. No real-model request was run in this continuation.

## Active objective and user direction

Optimize this private DSpine llama.cpp fork on RX 9060 XT 16 GB, gfx1200, ROCm/HIP for real inference at 8K-128K populated context. Goal remains active. Prioritize demonstrated Q2_K improvements and the existing dense Flash Attention path. Sparse-prefill implementation is explicitly stopped. Preserve README, functionality and unrelated work. Prefer focused 8K measurements before expensive large-context runs. Compare identical prompts/settings and output lengths (256 tokens when comparing against a 256-token record). GSQ-RCO remains the target model.

Target: `D:/40-49 Assets/44 AI Models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`. Draft: `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` in the same directory. Typical settings: capacity 131072, target Q4_0 K/V, DFlash n_max=6, target ubatch 512, draft ubatch 32. Reuse saved command JSON rather than reconstructing settings. Qwen has 16 full-attention and 48 GDN layers; preserve every recurrent token update.

## Current authoritative state

- Concise optimization record: `docs/development/dspine-rocm-progress.md`. Q2_K investigation: `dspine-rocm-8k-audit.md`. Sparse evidence/design: `dspine-rocm-block-sparse-prefill.md`. Older generation work: `dspine-rocm-longctx-generation.md`; many of those runs use UD-IQ2 rather than GSQ and cannot be merged into a GSQ A/B.
- The validated gfx1200 Q2_K dispatch uses dequantization plus hipBLAS for exactly 512 columns and five projection shapes: (K,M)=(5120,6144/10240/12288/17408), (17408,5120). It remains in `ggml-cuda.cu`. CPU reference tests passed. Earlier isolated MMQ 26-76 ms became approximately 1.14-2.62 ms. One historical 8K pair improved prefill 142.21 -> 200.37 tok/s; subsequent same-build drift prevents treating those absolute values as a current stable baseline.
- Tail dispatch expansion (442-512 columns) and a lower-spill J64 MMQ experiment were rejected and reverted after no repeatable request-level improvement. Do not repeat them without new evidence.
- Quantized-KV vector attention for small decode/verification batches is present and default enabled. Dense prefill D256/Q512/GQA6 selects F16 WMMA with full K/V conversion. Keep that production path.
- Latest full build succeeded after retiring the failed sparse WMMA edits: `build-rocm/bench/wmma-restored-build.log`, duration 86 seconds. `git diff --check` passed. The four block-reference tests passed together on the restored build after that rebuild.
- No inference process was active at the last process check. No new decode attribution request has started. Do not assume old profiling hooks mentioned below survived upstream synchronization; inspect current source.

- Decode attribution with GPU events exists at 8K and 64K. At 8K the target graph is MUL_MAT 63.6%, FLASH_ATTN_EXT 24.1%, Q2_K only 6.4%. At 64K (65,536 real prompt tokens, 256 output tokens) it is FLASH_ATTN_EXT 72.4% at 16.92 ms/call, MUL_MAT 20.5%, GATED_DELTA_NET 2.0%, so attention dominates. A forced MMQ comparison for IQ3_S and IQ3_XXS at 7 columns was worth about 2% with changed numerics, so no dispatch change was kept. Details in `docs/development/dspine-rocm-progress.md`. Latest full build: `build-rocm/bench/decode-gdn-build.log`; `git diff --check` passed.

## Sparse experiments: stopped

- Row-gather sparse at 64K: 117.56 ms versus dense 46.48 ms. Compact-block vector sparse: 128.90 ms versus matched dense 46.99 ms, selecting 11776/65536 KV rows (17.97%).
- One authorized direct Q4_0-to-LDS WMMA prototype reused matrix primitives and online softmax. It built and passed 32K, 64K, partial-tail and two-slice-with-sinks CPU-reference cases together. Generated D256/32-query/2-head specialization used 256 VGPRs, 105 VGPR spills, one SGPR spill and 312 bytes private memory/thread. It failed the no-significant-spilling gate; no WMMA latency benchmark was run.
- Failed WMMA loader/dispatch edits were removed from active source and saved in `docs/development/dspine-sparse-wmma-rejected.patch`. Local generated-code evidence: `build-rocm/bench/sparse-wmma.hsaco` and `sparse-wmma-metadata.txt`. Earlier fixed-block vector experiment and API remain default-off in the worktree; do not activate for ordinary inference.
- Partial-tail intermittent NaNs were uninitialized test harness sentinels, not established attention algorithm failures. Retained fix: `test_flash_attn_ext_blocks::initialize_tensors` calls `test_case::initialize_tensors(ctx)` before overwriting deterministic inputs. Four cases passed together both on experimental WMMA and restored vector build. This supersedes older unresolved-NaN notes.
- Future ideas only: smaller query tiles or splitting softmax across workgroups could reduce register pressure but require another design and add overhead. User has stopped further sparse implementation.

## Result of the 2026-11-15 decode attribution

- Added a default-off decode-only GPU-event profiler (`GGML_CUDA_PROFILE_DECODE=1`) in `ggml/src/ggml-cuda/ggml-cuda.cu`. It disables graph capture, gates on graphs whose attention columns are 1-8 over at least 4096 populated KV rows, and logs `decode_graph` plus per-op `decode_op` event times. Harness: `build-rocm/bench/run_gsq_decode_events.py`; summary: `summarize_gsq_decode_events.py`.
- One GSQ 8K request, 256 output tokens, zero cache reuse, saved settings: 288 proposed / 207 accepted drafts, 263.87 tok/s prefill and 13.67 tok/s generation under profiling. Unprofiled on a binary without the gate active it was 37.49 tok/s with the same acceptance.
- Target decode graph over 46 steady graphs, 188.96 ms summed event time each: MUL_MAT 63.6%, FLASH_ATTN_EXT 24.1% (736 calls, all 7 columns), GET_ROWS 3.7%, GLU 2.3%, ADD 2.2%, CONCAT 2.1%. Q2_K is 6.4% of the seven-column matmul subset, so the prefill dispatch is not the decode lever. Dominant types: IQ2_S 1143 ms, IQ3_XXS 1105, IQ3_S 1053. The draft graph is 88.9% MUL_MAT at 15.94 ms mean.
- Forced-path comparison at (K,M)=(5120,17408), 7 columns: IQ2_S MMVQ 0.2588 vs MMQ 0.3510 ms; IQ3_S MMQ 0.2127 vs MMVQ 0.2263; IQ3_XXS MMQ 0.2013 vs MMVQ 0.2298. Routing the two IQ3 types to MMQ saves about 2% of target decode time, below same-build variation, and changes acceptance (197/344 vs 207/288). No dispatch change retained.
- `GGML_CUDA_DIAG_IQ7` now covers IQ2_S, IQ3_XXS and IQ3_S and disables graph capture when enabled. Without that, `hipEventSynchronize` fails inside capture and aborts the server.

Next:

1. Attention now dominates: 72.4% of target decode at 64K against 20.5% for matmul, so investigate the existing dense attention kernel for the 7-column D=256, 24 Q heads over 4 KV heads path. That call is `flash_attn_ext_vec<256, 8, 2, Q4_0, Q4_0, false>` and spills 288 VGPRs with 1,116 B/lane scratch, so the online-softmax reduction loop is the place to look, not KV traffic. Do not spend more effort on seven-column IQ dispatch; its alternatives were worth about 2% at 8K.
2. The profiler now also times the fused GDN branch. Keep `GGML_CUDA_PROFILE_DECODE` default off and reuse it for any further depth, but do not use it for throughput claims.
3. A 100K confirmation point is optional and costs about 10 minutes of prefill; run it only if an attention change needs a deeper data point.

Useful existing harness: `build-rocm/bench/run_gguf_8k_ab_256.py` contains readiness, clean environment, saved settings, hash recording and cleanup. It currently runs UD/GSQ/GSQ/UD: do not execute unchanged for GSQ-only diagnosis. Its baseline is `unchanged-memory-20261115-122744-1.command.json`; request max_tokens becomes 256. `run-profile.py` also runs both GGUFs with full serialized profiling and should not be executed unchanged. `tools/dspine-profile-summary.py` summarizes those logs but does not separate phases.

## Worktree and caveats

Dirty worktree is intentional. Modified paths include `ggml/include/ggml.h`, `ggml/src/ggml.c`, CPU ops, CUDA fattn common/vector/dispatch, `ggml-cuda.cu`, `mmq.cu`, and `tests/test-backend-ops.cpp`; many docs/plots and tools are untracked. Preserve existing changes. `fattn-mma-f16.cuh` has no retained experiment diff. No commits or pushes were made.

Build via `.\\build-rocm.ps1 -SkipConfigure -Parallel 4` to load MSVC/ROCm environment; plain cmake build previously failed linking without MSVC paths. Redirect verbose compiler output to a bench log. CodeGraph tools are unavailable and `.codegraph` is absent; user explicitly declined initialization. Local `skills/code-review/SKILL.md` has been read. Do not spawn agents unless user requests delegation.

WDDM counters showed near-budget dedicated memory and several GiB shared accounting, but these are not physical residency proof. Separate-context hipMemGetInfo samples were invalid for server memory. Output quality and full 128K populated performance remain unproven. Historical 100K GSQ run: 169.62 tok/s prefill, 19.88 tok/s generation, 96 generated tokens, 76/106 drafts accepted; it is a capacity/performance test, not quality validation.

---

# Historical handoff: 100K decode, quantized-KV vec attention (state at end of Nov 9)

The following section is retained as historical context. Current state and caveats above take precedence.

Previous handoff (spec-decode cliff) is archived as `handoff-2026-11-08-cliff.md`.
Full writeup of this work: `docs/development/dspine-rocm-longctx-generation.md`.

## Headline

Decode after a conversation has really populated 100K KV went from **5.77 to 13.75 tok/s**
(2.38x) by serving the verification batch on the vec attention kernel, which reads Q4_0
K/V directly instead of expanding them to F16 on every call.

| 100K populated, 96 tokens, no instrumentation | Prefill tok/s | Decode tok/s | Draft accepted |
| --- | ---: | ---: | ---: |
| before (TILE, F16 expansion per call) | 159.7 | 5.77 | 72 / 129 |
| after (vec, Q4_0 read directly) | 158.8 | **13.75** | 72 / 129 |

Same prompt (99,991 tokens), same sampling, same acceptance count, so the gain is pure
kernel work, not a different output path. Prefill is unaffected as expected (it never
used the vec path). Single sample per row.

Caveats to keep honest:
- The 5.77 row was measured on the pre-change binary in the same harness with the same
  flags. The tile control on the current binary (`v100k_tile`) is the confirmation run;
  check its number before quoting the ratio outside this note.
- 96 tokens is about 23 verification steps, so a small part of the delta is step-count
  noise. Acceptance is identical (72), so the step count is the same in both runs.
- 128K populated is still not measurable on 16 GB (100K already peaks about 13.3 GB).

## What changed (3 files, about 40 lines)

- `ggml/src/ggml-cuda/fattn.cu`
  - `ggml_cuda_get_best_fattn_kernel`: quantized K/V now selects the vec kernel for
    `Q->ne[1] <= 8` instead of `<= 2`.
  - New `ggml_cuda_fattn_vec_quant_enabled()` reading `GGML_CUDA_FATTN_VEC_QUANT`
    (default on, `=0` restores the old TILE behavior). Keep this kill switch for now.
- `ggml/src/ggml-cuda/fattn-vec.cuh`
  - `nthreads_KQ_q_wide = 8` when `D/4 % 8 == 0`, used only when `ncols > 2`. Wide tiles
    keep the q8_1 copy of Q in registers; 8 threads caps `Q_reg` at 16 `half2` per column.
  - New `cols_per_block = 4` dispatch branch for `Q->ne[1] > 2`, so one K/V pass serves
    4 columns instead of 4 passes of 2.
- `ggml/src/ggml-cuda/{cliff-profile.cuh,ggml-cuda.cu}`: decode-only gate for per-op
  profiling (1-8 attention columns, kv >= 4096) so prefill is not instrumented.

Why 4 columns, not 8: `Q_ds[j][it] = tmp_q_ds[(i0+lane)/QI8_1]` spans two q8_1 blocks if
`nthreads_KQ > QI8_1` (8), so 8 columns cannot be done with 8 threads. 8 columns would
need a different register layout or a wider index, which is the next experiment if 4
columns still spills.

## Verified

- Build: `cmd //c "build-rocm\build-hip.bat"` -> 14 steps, 0 errors. `llama-server.exe`
  rebuilt and used for all runs above.
- Correctness: `cmd //c "build-rocm\build-test.bat"` then
  `build-rocm/bin/test-backend-ops.exe -b ROCm0 -o FLASH_ATTN_EXT -p 'hsk=64.*type_K=q4_0'`
  -> all cases OK, including nb=3, kv 512/1024, permuted and GQA. Log:
  `build-rocm/fa-vec-q4_0.log`.
- Not covered by that test: D=256 (this model's head size). The quantized KV cases only
  exist at hsk 64/72. Runtime evidence for D=256 is the 100K run: identical draft
  acceptance and coherent output at 2.4x the speed, but no reference comparison yet.
  Tomorrow: compare the vec reply against the tile reply text at the same prompt and seed.

## State of the benchmark chain

`build-rocm/bench/longctx-decode.py` runs four paired cases, serial, port 9092. Each
prints one JSON line into `<tag>.run.log` and `<tag>.summary.json`.

| tag | env | status |
| --- | --- | --- |
| `v100k_vec` | default | done: 13.745 tok/s |
| `v100k_tile` | `GGML_CUDA_FATTN_VEC_QUANT=0` | started 23:50, about 11 min total |
| `v12k_vec` | default | queued |
| `v12k_tile` | `GGML_CUDA_FATTN_VEC_QUANT=0` | queued |

Collect with:

```sh
for t in v100k_vec v100k_tile v12k_vec v12k_tile; do
  echo "== $t"
  grep -E 'target|prompt_tokens|prompt_per_second|predicted_per_second|draft_n' \
    build-rocm/bench/$t.summary.json 2>/dev/null
done
```

If the chain died with the session, rerun the missing tags (each is one command):

```sh
python build-rocm/bench/longctx-decode.py v100k_tile --sizes 100000 --tokens 96 --no-profile --no-ops --env GGML_CUDA_FATTN_VEC_QUANT=0
python build-rocm/bench/longctx-decode.py v12k_vec   --sizes 12800  --tokens 96 --no-profile --no-ops
python build-rocm/bench/longctx-decode.py v12k_tile  --sizes 12800  --tokens 96 --no-profile --no-ops --env GGML_CUDA_FATTN_VEC_QUANT=0
```

Tags are fixed filenames, so archive old files before reusing a tag.

## Expected next results, and what to do with them

- `v100k_tile` near 5.7: confirms 2.38x. Then the doc gets the table above as final.
- `v100k_tile` clearly higher (say 7 or more): part of the delta is night-to-night
  variance, so quote the vec/tile ratio from the same evening instead.
- `v12k_vec` vs `v12k_tile`: at 12.8K populated the old TILE path cost 1.09 ms/call, so
  vec can win or lose here. If vec loses at 12.8K, the selection rule may want a
  populated-length floor (vec reads K/V once per query head, 6x redundancy with
  gqa_ratio 6, so it pays off only once traffic dominates). Measure before adding it.

## Open work, in priority order

1. Finish the four runs, put the final table into
   `docs/development/dspine-rocm-longctx-generation.md`, replacing the "Next
   implementation step" section with the outcome. That doc still says "do not quote a
   speedup before it is measured"; the numbers above are the measurement.
2. Per-op confirmation at 100K: `--sizes 100000 --tokens 24` with `GGML_CUDA_CLIFF_OPS=1`
   should show `FLASH_ATTN_EXT` far below 4,821 ms and no F16 scratch allocation. Cheap,
   and it is the only direct evidence that the expansion is gone rather than amortized.
3. Remaining headroom, biggest first:
   - GQA packing in the vec kernel. Blocks are sized `gqa_ratio * K->ne[2]`, so each of
     the 24 query heads re-reads its KV slice; that is 6x redundant Q4_0 traffic. Packing
     heads per block would cut the 115.6 MB per layer toward 19 MB. This is the single
     largest remaining lever and it is independent of columns.
   - 8 columns per block, if a register-clean layout can be found.
   - WMMA path for wider verify batches (needs `Q->ne[1] * gqa_ratio_eff > 16`; 7 columns
     gives 14, so it is 2 short by construction for this model).
4. Do not enable `GGML_GRAPH_ALLOC_GROWTH` for long context: 34.31 vs 34.04 ms/call at
   100K, 5.66 vs 5.77 tok/s end to end. It stays a short-chat-only experiment.
5. LSA stays rejected for now: distilled maps exist for this model, but it is an SGLang +
   CUDA FA3 plugin with host-side selection, no HIP path, and it needs new persistent
   per-position structures against a 2.3 GB Q4_0 cache on a 16 GB card.

## Environment traps already paid for

- Any running `llama-server.exe` breaks the link step with "unable to remove file:
  permission denied". `taskkill //F //IM llama-server.exe` first.
- `cmd //c build-rocm\\build-test.bat` fails; the backslashes must stay inside the quoted
  string: `cmd //c "build-rocm\build-test.bat"`.
- Backend name for this card in `test-backend-ops` is `ROCm0`, not `CUDA0`. A wrong `-p`
  filter prints "Skipping" and looks like a pass.
- Compare throughput only within this harness. The archived 256-token cold run reported
  7.18 tok/s for the same settings; that is not a baseline for these 96-token runs.
- Use `event_ms` (GPU event time) for attribution, not `CLIFF op us` (adds sync).

## Uncommitted tree

`README.md` (user's edits, unrelated), `ggml/src/ggml-alloc.c` (growth experiment),
`cliff-profile.cuh`, `ggml-cuda.cu` (profiler gate), `fattn-vec.cuh`, `fattn.cu` (the
fix). Docs and handoffs are untracked. Nothing was committed or pushed.
