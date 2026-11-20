"""Summarize a single cold request's opt-in HIP prefill trace."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def records(text, label, fields):
    pattern = ' '.join(f'{key}=({regex})' for key, regex, _ in fields)
    return [dict(zip((key for key, _, _ in fields),
                     (convert(value) for (_, _, convert), value in zip(fields, match))))
            for match in re.findall(rf'\b{label} {pattern}', text)]


def save(folder, name, rows):
    (folder / f'{name}.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    if rows:
        with (folder / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(log_path, response_path, output_dir):
    text = log_path.read_text(encoding='utf-8', errors='replace').split('new prompt,', 1)[1]
    text = text.split('prompt eval time', 1)[0]
    timings = json.loads(response_path.read_text(encoding='utf-8'))['timings']
    integer = (r'\d+', int)
    decimal = (r'[\d.]+', float)
    word = (r'\S+', str)
    graph = records(text, 'prefill_gpu', [('graph', *integer), ('queries', *integer), ('kv', *integer), ('span_ms', *decimal)])
    host = records(text, 'prefill_graph', [(key, *integer) for key in
                   ('count', 'prepare_us', 'build_us', 'alloc_us', 'compute_call_us')])
    ops = records(text, 'prefill_gpu_op', [('graph', *integer), ('op', *word), ('name', r'.*?', str),
                  ('type', *word), ('fused', *integer), ('capture', *integer), ('gpu_ms', *decimal)])
    batches = records(text, 'prefill_profile', [(key, *integer) for key in ('pos', 'count', 'target_us', 'inject_us')])
    checkpoints = records(text, 'prefill_checkpoint', [(key, *integer) for key in ('pos', 'bytes', 'target_us', 'draft_state_us')])
    routes = records(text, 'prefill_q2_route', [('n', *integer), ('k', *integer), ('m', *integer),
                     ('route', *word), ('name', r'[^\r\n]*', str)])
    if not graph or not ops or sum(row['count'] for row in batches) != timings['prompt_n']:
        raise RuntimeError('missing GPU events or incomplete server prefill boundary coverage')
    if [row['queries'] for row in graph] != [row['count'] for row in host]:
        raise RuntimeError('GPU graphs and host microbatches do not align by token count')

    by_type = defaultdict(lambda: [0, 0.0])
    by_graph = defaultdict(lambda: defaultdict(float))
    q2_ms = defaultdict(float)
    for op in ops:
        family = 'capture_SET_ROWS' if op['capture'] else op['op']
        key = (family, op['type'])
        by_type[key][0] += 1
        by_type[key][1] += op['gpu_ms']
        by_graph[op['graph']][family] += op['gpu_ms']
        if op['op'] == 'MUL_MAT' and op['type'] == 'q2_K':
            q2_ms[op['graph']] += op['gpu_ms']
    totals = [{'op': key[0], 'src_type': key[1], 'calls': count, 'gpu_ms': ms}
              for key, (count, ms) in sorted(by_type.items(), key=lambda pair: -pair[1][1])]
    combined = []
    pos = 0
    for gpu, cpu in zip(graph, host):
        families = by_graph[gpu['graph']]
        combined.append(dict(gpu, **cpu, pos=pos, measured_op_ms=sum(families.values()),
                             matmul_ms=families['MUL_MAT'], q2_matmul_ms=q2_ms[gpu['graph']], attention_ms=families['FLASH_ATTN_EXT'],
                             gdn_ms=families['GATED_DELTA_NET'], capture_ms=families['capture_SET_ROWS']))
        pos += gpu['queries']
    if pos > timings['prompt_n']:
        raise RuntimeError('logged GPU graphs exceed the prompt length')

    metrics = {
        'prompt_tokens': timings['prompt_n'], 'gpu_profiled_tokens': pos,
        'prefill_ms': timings['prompt_ms'],
        'target_ms': sum(row['target_us'] for row in batches) / 1000,
        'draft_injection_ms': sum(row['inject_us'] for row in batches) / 1000,
        'checkpoint_ms': sum(row['target_us'] + row['draft_state_us'] for row in checkpoints) / 1000,
        'graph_prepare_ms': sum(row['prepare_us'] for row in host) / 1000,
        'graph_build_ms': sum(row['build_us'] for row in host) / 1000,
        'graph_alloc_ms': sum(row['alloc_us'] for row in host) / 1000,
        'graph_compute_call_ms': sum(row['compute_call_us'] for row in host) / 1000,
        'gpu_stream_span_ms': sum(row['span_ms'] for row in graph),
        'gpu_event_op_ms': sum(row['gpu_ms'] for row in ops),
        'capture_writes_ms': sum(row['gpu_ms'] for row in ops if row['capture']),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [('profile-summary', [metrics]), ('microbatches', combined), ('operator-totals', totals),
                       ('operators', ops), ('server-batches', batches), ('checkpoints', checkpoints), ('q2-routes', routes)]:
        save(output_dir, name, rows)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('response', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    main(args.log, args.response, args.output_dir or args.log.parent)
