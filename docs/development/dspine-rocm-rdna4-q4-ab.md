# RDNA4 Q4 attention: real-model A/B (2026-11-16)

The prototype did not improve the measured 8K workload. The single 64K pair measured a 19.4% decode improvement with identical fixed-length output, but substantial prefill variation prevents treating that pair as a confirmed kernel-only gain. The isolated 6.75x attention result does not translate into a general generation speedup.

## 96K follow-up: 98304 prompt tokens, 256 output tokens

One fresh-process candidate/stock pair used exactly the same request, binaries and settings as the earlier protocol, with the history extended to 98304 tokens. This reverses the order of the earlier 64K pair. Both measured requests had `cache_n=0`, 256 generated tokens, `truncated=false`, and normal token-limit stops. No kernel changes or rebuilds occurred.

| Metric | Stock | Candidate |
| --- | ---: | ---: |
| Prefill throughput, tok/s | 135.342 | 157.824 |
| Prefill time, seconds | 726.335 | 622.873 |
| Generation throughput, tok/s | 19.675 | 33.642 |
| Generation time, seconds | 12.960 | 7.580 |
| Request wall time, seconds | 739.354 | 630.522 |
| Draft accepted/proposed | 205/294 | 210/266 |
| Draft acceptance | 69.728% | 78.947% |

The candidate measured 70.98% higher generation throughput and 16.61% higher prefill throughput. Prefill latency was 14.24% lower, saving 103.463 seconds. These are measured differences in one pair, not established causal kernel speedups. Bulk prefill uses width 512, outside the prototype's 1-8 token gate, so its large timing difference is a run-variation/confounding signal. No prefill optimization was introduced by this measurement task.

The generated outputs differ from token index 3 (zero-based), and draft acceptance also differs materially. Both 256-token responses are incomplete reasoning text about the Fibonacci task. Therefore generation work is not identical, and this sample does not establish completed-answer quality or a kernel-only 70.98% improvement. Do not compare the 96K absolute throughput with 64K as a controlled context-scaling curve: the prompt contents, output and acceptance differ.

All 96K raw data and previous measurements are now retained outside the build tree in `docs/development/measurements/rdna4-q4/`. The exact 96K request is `fa-qr-ab-96k-20261116.request.json`; individual samples are in `.summary.json`, and calculated comparisons including prefill gains are in `fa-qr-ab-96k-20261116.comparison-v2.json`. The earlier comparison files were preserved. `fa-qr-ab-comparison-through-96k-20261116-v2.json` combines the saved 8K, 64K and 96K statistics while retaining individual samples. The reusable harness is `tools/dspine-fattn-ab.py`, which defaults to this persistent archive and rejects existing run tags. User direction is to always retain measured statistics, including failures and validity limits.

## Method

- RX 9060 XT gfx1200; GSQ-RCO target `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`, DSpine draft `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`.
- Same binaries for every run, with hashes saved for `llama-server.exe`, `llama-server-impl.dll`, `ggml-hip.dll`, `llama.dll`, and `ggml-base.dll`.
- Reused the saved baseline command and engineering conversation in `dspine-rocm-prefill-results.json`, substituting the GSQ target. Capacity 131072, one slot, all GPU layers, Q4_0 target KV, F16 draft KV, target batch/ubatch 2048/512, draft ubatch 32, DFlash maximum draft width 6, eight threads, low reasoning.
- Greedy sampling (temperature 0), seed 42, 256 output tokens. The saved conversation was rendered through `/apply-template`; its synthetic history was repeated/trimmed before the final fast-doubling Fibonacci question to produce exactly 8192 or 65536 prompt tokens. The exact token arrays and request hashes are saved. This is a model inference test using synthetic history, not a natural-document quality benchmark.
- Fresh server for each measured request. Profiling and inherited GGML/LLAMA overrides were removed. Candidate uses normal dispatch; stock sets `GGML_CUDA_DISABLE_FA_QR=1`. Both set the one-time gate logger. Identical requests within each depth; all measured requests report `cache_n=0`, `predicted_n=256`, and `truncated=false`.
- 8K order: candidate/stock/stock/candidate. 64K order: stock/candidate, one run each. No concurrent benchmark or inference process was launched by this harness. Server processes were stopped after their requests.

## Measured results

| Prompt tokens | Variant | Generation tok/s, individual runs | Mean tok/s | Draft accepted/proposed | Prefill tok/s |
| --- | --- | --- | --- | --- | --- |
| 8192 | Stock | 31.909, 31.883 | 31.896 | 206/294 both runs (70.1%) | 213.13, 202.94 |
| 8192 | Candidate | 27.708, 31.134 | 29.421 | 204/301 both runs (67.8%) | 218.66, 214.82 |
| 65536 | Stock | 21.829 | 21.829 | 206/292 (70.5%) | 157.86 |
| 65536 | Candidate | 26.073 | 26.073 | 206/290 (71.0%) | 184.22 |

At 8K the candidate mean is 7.76% lower; both candidate samples are below both stock samples. Candidate variation is material, so this does not quantify an invariant regression. It does establish no measured speedup for this prompt. Each variant repeats its own exact output, but outputs differ across variants beginning at generated token index 122 (zero-based). Consequently acceptance and generated work are not perfectly matched across variants.

At 64K the candidate is 19.44% higher in the single pair. The 256 output token IDs and decoded text are identical across variants; both accept 206 draft tokens, with slightly different proposal totals. Prefill also improves from 415.159 to 355.745 seconds (157.86 to 184.22 tok/s), despite the prototype being limited to 1-8 query tokens and bulk prefill using width 512. This is evidence of run variation/confounding, not a demonstrated large-prefill optimization. Repeat in reverse order before claiming a reliable 64K kernel-only or end-to-end gain.

## Bounded answer-quality check

The timed 256-token responses remain in the reasoning section. They are not completed task answers. After each timed 64K request, a separate request reused the same long prompt, closed the thinking section, and allowed up to 1536 output tokens. These quality requests used prompt-cache reuse and are excluded from all timing comparisons above.

Both quality answers ended normally at EOS: stock 911 tokens, candidate 567 tokens. Both generated fast-doubling Python functions passed all 1001 values F(0) through F(1000), checked against an independent iterative reference, and rejected negative integers, float, string, None and bool inputs with the expected exception classes.

Neither full answer is free of explanatory errors. Stock drops a square from the stated odd doubling identity and uses that incorrect expression in its proof. Candidate gives the correct identities/proof for positive indices, but claims the computed integers have O(log n) bits and that the bit complexity is O(log^2 n); F(n) has O(n) bits, so that complexity claim is false. These two samples cannot establish general quality preservation or a quality advantage for either kernel.

## Decision and artifacts

Do not treat the current broad 1-8 token gate as an established inference-speed improvement. A possible long-context benefit merits a reversed 64K repeat; the 8K results argue against enabling this path indiscriminately. This A/B task did not change kernel dispatch, rebuild binaries, commit, or push.

Earlier artifacts originated under `build-rocm/bench/` and have also been copied to the persistent `docs/development/measurements/rdna4-q4/` archive:

- `run_fa_qr_ab.py`: harness, using the saved baseline settings and fresh-process cleanup.
- `fa-qr-ab-{8k,64k}-20261116.request.json`: exact request token arrays and sampling settings.
- `fa-qr-ab-{8k,64k}-20261116.summary.json`: raw timing, output hashes, acceptance and binary hashes.
- `fa-qr-ab-{8k,64k}-20261116-<trial>-<variant>.{command.json,log,response.json}`: per-run evidence.
- `fa-qr-ab-64k-20261116-<trial>-<variant>.quality-{request,response}.json`: completed-answer evidence.
- `fa-qr-ab-comparison.json`: calculated means, relative changes and request/output equality checks.
- `validate_fa_qr_quality.py` and `fa-qr-ab-quality-validation.json`: generated-function validation.

The build directory is ignored by Git. The persistent archive now keeps these measurements outside that tree; new measurements must use that archive or another durable source location.
