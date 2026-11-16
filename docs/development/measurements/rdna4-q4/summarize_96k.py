"""Preserve calculated 96K A/B comparisons without replacing prior measurements."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / 'fa-qr-ab-96k-20261116.summary.json').read_text())
rows = {row['variant']: row for row in data['rows']}
responses = {
    variant: json.loads(Path(row['response_file']).read_text(encoding='utf-8'))
    for variant, row in rows.items()
}
first_difference = next((
    i for i, (a, b) in enumerate(zip(responses['candidate']['tokens'], responses['stock']['tokens']))
    if a != b
), None)
result = {
    'context_tokens': 98304, 'output_tokens': 256, 'samples_per_variant': 1,
    'order': data['order'], 'candidate': rows['candidate'], 'stock': rows['stock'],
    'generation_change_percent': 100 * (
        rows['candidate']['timings']['predicted_per_second'] /
        rows['stock']['timings']['predicted_per_second'] - 1
    ),
    'prefill_throughput_change_percent': 100 * (
        rows['candidate']['timings']['prompt_per_second'] /
        rows['stock']['timings']['prompt_per_second'] - 1
    ),
    'prefill_latency_change_percent': 100 * (
        rows['candidate']['timings']['prompt_ms'] / rows['stock']['timings']['prompt_ms'] - 1
    ),
    'prefill_seconds_saved': (rows['stock']['timings']['prompt_ms'] - rows['candidate']['timings']['prompt_ms']) / 1000,
    'same_request': len({row['request_sha256'] for row in data['rows']}) == 1,
    'all_uncached': all(row['timings']['cache_n'] == 0 for row in data['rows']),
    'all_contexts_exact': all(row['timings']['prompt_n'] == 98304 for row in data['rows']),
    'all_outputs_exact': all(row['timings']['predicted_n'] == 256 for row in data['rows']),
    'identical_output_tokens': responses['candidate']['tokens'] == responses['stock']['tokens'],
    'first_differing_output_token_index': first_difference,
    'binary_sha256': data['binary_sha256'],
    'same_binaries_as_8k_64k': all(
        json.loads((root / f'fa-qr-ab-{size}-20261116.summary.json').read_text())['binary_sha256']
        == data['binary_sha256'] for size in ['8k', '64k']
    ),
    'limitations': [
        'Single candidate/stock pair; run-to-run variation remains.',
        'Outputs and draft acceptance differ, so generated work is not identical.',
        'Prefill varies despite bulk prefill width being outside the prototype gate.',
        'The fixed-length responses do not establish completed-answer quality.',
    ],
}
for variant, row in rows.items():
    row['acceptance_percent'] = 100 * row['timings']['draft_n_accepted'] / row['timings']['draft_n']
    log = (root / f'fa-qr-ab-96k-20261116-{row["trial"]}-{variant}.log').read_text(encoding='utf-8', errors='replace')
    row['additional_server_stats'] = [
        line for line in log.splitlines()
        if any(key in line for key in ['graphs reused =', 'graph reallocs =', 'acc per pos =', 'spec common_specu: statistics'])
    ]
target = root / 'fa-qr-ab-96k-20261116.comparison-v2.json'
if target.exists():
    raise RuntimeError(f'refusing to overwrite {target}')
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
previous = json.loads((root / 'fa-qr-ab-comparison.json').read_text())
previous['96k'] = result
(root / 'fa-qr-ab-comparison-through-96k-20261116-v2.json').write_text(json.dumps(previous, indent=2), encoding='utf-8')
print(json.dumps({
    'generation_change_percent': result['generation_change_percent'],
    'prefill_throughput_change_percent': result['prefill_throughput_change_percent'],
    'prefill_latency_change_percent': result['prefill_latency_change_percent'],
    'prefill_seconds_saved': result['prefill_seconds_saved'],
    'first_differing_token_index': first_difference,
    'same_binaries_as_8k_64k': result['same_binaries_as_8k_64k'],
    'validity': {key: result[key] for key in ['same_request', 'all_uncached', 'all_contexts_exact', 'all_outputs_exact']},
    'runs': {variant: {'timings': row['timings'], 'acceptance_percent': row['acceptance_percent']}
             for variant, row in rows.items()},
}, indent=2))
