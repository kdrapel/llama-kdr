# RDNA4 Q4 attention measurement archive

User requirement: always retain measured statistics, including unsuccessful or inconclusive comparisons. This directory is outside the disposable build tree. Keep distinct run tags; never overwrite earlier samples.

The archive contains the 8K/64K A/B requests, responses, command/environment snapshots, binary hashes, logs, timing summaries, bounded quality checks and isolated-kernel timing results measured on 2026-11-16. The 96K follow-up uses 98304 prompt tokens and 256 generated tokens, candidate then stock. Its summary is saved after each completed request; errors are retained separately when a request fails.

Use `tools/dspine-fattn-ab.py` for subsequent runs. Its default output directory is this archive, and it rejects an existing request or summary tag. The `.py.snapshot` files preserve the exact harness and validator used for the earlier measurements. Exact saved request token arrays can be replayed directly through `/completion` without reconstructing the synthetic history.

Each run retains:

- `<tag>.request.json`: exact prompt token IDs and sampling settings.
- `<tag>-<trial>-<variant>.command.json`: server command, environment overrides and binary SHA256 values.
- `<tag>-<trial>-<variant>.response.json`: full output, output token IDs and all server-reported timing/acceptance fields.
- `<tag>.summary.json`: individual measured samples, hashes and wall times. Do not replace individual samples with an average alone.
- `<tag>-<trial>-<variant>.log`: server output, including prompt progress and diagnostics.
- `<tag>-<trial>-<variant>.error.json`: failure details if a launched run fails.

`fa-qr-ab-comparison.json` is the earlier 8K/64K aggregate; preserve it when adding a newer aggregate. The corresponding report is `../../dspine-rocm-rdna4-q4-ab.md`. Quality requests are explicitly separated from timed fixed-length requests and must not be mixed into generation-rate averages.
