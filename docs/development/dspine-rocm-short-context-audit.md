# Short-context and historical throughput audit (2026-11-16)

User reported that a fresh GSQ-RCO conversation answering `What is a singleton in C#?` previously reached roughly 35-40 tok/s, sometimes up to 55 tok/s. This audit preserves the current measurements and checks the saved curves without treating different workloads as paired regressions.

## Saved CSV evidence

| Source | Populated context | Output length | Generation tok/s | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| `dspine-rocm-speed-curve-gsq-rco-8k-100k.csv` | 8192 | 96 | 45.733 mean, 44.185-47.280 | 317.648 mean |
| Same GSQ curve | 32768 | 96 | 23.814 | 207.267 |
| Same GSQ curve | 99991 | 96 | 19.879 | 169.623 |
| `dspine-rocm-speed-curve-bc7c78ef95ac8b65ae50eeff7d02b544d6cc62ee.csv` | 8192 | 256 | 28.420 mean, 27.634-29.206 | 273.948 mean |
| Same `bc7c78e` curve | 16384 | 256 | 28.745 mean | 227.045 mean |
| Same `bc7c78e` curve | 32768 | 256 | 26.688 mean | 231.495 mean |

The companion JSON files identify both curves as GSQ-RCO with the DSpine draft, capacity 131072, Q4_0 target KV, DFlash maximum draft width 6, seed 42, temperature 0.6, top-p 0.95 and top-k 20. The earlier 8K curve uses 96 generated tokens, while the commit-specific curve uses 256. The prompt construction and binary hashes also differ. The commit-specific curve predates the newly added RDNA4 Q4 attention kernel, so approximately 28-29 tok/s at 8K was already recorded before that addition.

The earlier GSQ 8K runs accepted 78/102 drafts (76.47%); the `bc7c78e` 8K runs accepted 195/360 (54.17%). This is a substantial workload/acceptance difference. It prevents attributing the throughput difference solely to attention. It does not, by itself, identify which prior change or workload difference lowered acceptance.

The `dspine-rocm-speed-curve-1k-8k` and `1k-96k` companion JSON files identify the target as UD-IQ2, not GSQ-RCO. Their 1K/2K/8K means of roughly 39.75/55.47/54.16 tok/s must not be substituted for a GSQ baseline. The named GSQ CSV starts at 8K; it contains no empty-context singleton sample.

## Exact singleton prompt: current same-binary A/B

Fresh server per sample, user message exactly `What is a singleton in C#?`, rendered prompt 48 tokens, capacity 131072, GSQ-RCO/DSpine, Q4_0 target KV, F16 draft KV, DFlash width 6, temperature 0.6, seed 42, profiling off, no prompt-cache reuse. Stock sets `GGML_CUDA_DISABLE_FA_QR=1`; candidate uses the current default gate. No kernel changes or rebuilds were made.

| Variant | Generation tok/s, first 256 tokens | Mean | Draft acceptance per run |
| --- | --- | ---: | --- |
| Candidate | 28.450, 28.587 | 28.519 | 185/416 (44.47%) |
| Stock | 29.441, 29.369 | 29.405 | 188/388 (48.45%) |

Order was candidate/stock/stock/candidate. Each variant reproduced its own output; outputs differ across variants. Candidate mean is 3.014% lower on this fixed-length test. Both are below the user's historical range, so disabling this new attention kernel alone does not reproduce the older speed. These measurements do not establish what caused the remaining difference.

A separate complete-answer pair used the same 48-token prompt, temperature 0.6 and seed 42, with a 1536-token limit and EOS allowed. Stock completed 643 tokens at 28.095 tok/s (22.851 s; 465/1062 draft acceptance, 43.79%); candidate completed 708 tokens at 29.605 tok/s (23.881 s; 520/1122, 46.35%). Both ended normally at EOS. Candidate has a higher reported rate but a longer total generation time because it produced more tokens. These are different outputs/amounts of work, not proof of a full-answer speedup.

## Strategy implied by the evidence

Choose attention dispatch using actual populated KV length and query batch width, together with existing dtype, head geometry and device checks. Allocated context capacity alone is insufficient: the singleton test has capacity 131072 but only 48 prompt tokens. Keep allocation sizing and execution under the same eligibility predicate if adding a context threshold.

The short/8K measurements support considering stock attention there; long-context samples support investigating the WMMA path there. They do not establish a crossover threshold: measure the intermediate 16K/32K region and separate one-token decode from seven-token verification before hardcoding one. Do not add continuous autotuning to the request path on this evidence.

Short-context decode is not explained solely by attention choice. Low draft acceptance suggests a separate, bounded maximum-draft-width comparison (for example 2/4/6) and a matched replay of historical prompts/output lengths. These are next experiments, not implemented policies or established fixes. No attention gate or draft-width default was changed in this audit.

## Retained statistics

All raw request token arrays, commands/environment, binary hashes, full responses, timings, acceptance and logs are under `docs/development/measurements/rdna4-q4/`:

- `fa-qr-ab-singleton-20261116*`: four fixed-256-token samples and calculated comparison.
- `fa-qr-ab-singleton-full-20261116*`: the two EOS-complete answer samples.
- `historical-gsq-and-singleton-audit-20261116.json`: historical source metadata/curves and current summaries together.

Historical CSV/JSON files remain separate from new measurements; no old samples were overwritten. The pending measurement archive commit is separate from existing commit `f7ccf262f`, whose actual diff contains three MMVQ files rather than this benchmark archive.
