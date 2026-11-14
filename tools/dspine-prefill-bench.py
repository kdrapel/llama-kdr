"""Replay fixed prefill requests serially with allocation and whole-graph telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('tag')
parser.add_argument('--cases', default='c16384,c32768,c65536,c131072,draftq4,b512,u256')
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--sizes', default='4096,8192')
parser.add_argument('--no-profile', action='store_true')
parser.add_argument('--data', type=Path, default=Path('docs/development/dspine-rocm-prefill-results.json'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / 'build-rocm/bench'
data = json.loads(args.data.read_text())
base = 'http://127.0.0.1:9092'
cases = {f'c{n}': {'--ctx-size': str(n)} for n in [16384, 32768, 65536, 131072]}
cases.update({'draftq4': {'--spec-draft-type-k': 'q4_0', '--spec-draft-type-v': 'q4_0'},
              'b512': {'-b': '512'}, 'u256': {'-ub': '256'},
              'growth': {}, 'growth16': {'--ctx-size': '16384'}})
if args.repeats < 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
    parser.error('positive repeats and a simple filename tag are required')


def post(endpoint, payload):
    request = urllib.request.Request(base + endpoint, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def telemetry(text):
    # Sampler initialization precedes the final small prompt decode. Keep this boundary explicit.
    bulk = text.split('new prompt,', 1)[-1].split('init_sampler', 1)[0]
    graphs = []
    for nodes, captured, update, us, columns, kv in re.findall(
            r'CLIFF graph nodes=(\d+) captured=(\d+) update=(\d+) elapsed_us=(\d+) columns=(\d+) kv=(\d+)', bulk):
        graphs.append(dict(zip(['nodes', 'captured', 'update', 'us', 'columns', 'kv'],
                               map(int, [nodes, captured, update, us, columns, kv]))))
    memory = []
    for budget, usage, shared in re.findall(
            r'CLIFF memory phase=graph adapter=0 .*?local_budget=(\d+) local_usage=(\d+).*?shared_usage=(\d+)', bulk):
        memory.append({'budget': int(budget), 'usage': int(usage), 'nonlocal': int(shared)})
    return {'boundary': 'new prompt through first init_sampler, excluding final small prompt tail',
            'graphs': graphs, 'memory': memory,
            'target_graph_ms': sum(g['us'] for g in graphs if g['nodes'] > 1000) / 1000,
            'small_graph_ms': sum(g['us'] for g in graphs if g['nodes'] <= 1000) / 1000}


for case in args.cases.split(','):
    overrides = cases[case]
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 9092)) == 0:
            raise RuntimeError('benchmark port 9092 is occupied')
    tag = f'{args.tag}_{case}'
    command = list(data['cases']['baseline']['command']['argv'])
    for key, value in overrides.items():
        command[command.index(key) + 1] = value
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('GGML_CUDA_CLIFF_', 'GGML_CUDA_GDN_')) or key in ('LLAMA_GRAPH_RESERVE_FULL', 'GGML_GRAPH_ALLOC_GROWTH'):
            env.pop(key)
    if not args.no_profile:
        env['GGML_CUDA_CLIFF_PROFILE'] = '1'
    if case.startswith('growth'):
        env['GGML_GRAPH_ALLOC_GROWTH'] = '1'
    command_path = out / (tag + '.command.json')
    if command_path.exists():
        raise RuntimeError(f'refusing to overwrite {command_path}')
    command_path.write_text(json.dumps({'argv': command, 'env': {k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_'))}}, indent=2))
    log_path = out / (tag + '.log')
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=Path(command[command.index('-m') + 1]).parent,
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError('server exited during initialization')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('server readiness timeout')
            for repeat in range(1, args.repeats + 1):
                for size in args.sizes.split(','):
                    payload = data['requests'][size]
                    offset = log_path.stat().st_size
                    started = time.perf_counter()
                    result = post('/v1/chat/completions', payload)
                    elapsed = time.perf_counter() - started
                    with log_path.open('rb') as log_read:
                        log_read.seek(offset)
                        section = log_read.read().decode(errors='replace')
                    row = {'case': case, 'repeat': repeat, 'size': int(size), 'wall_s': elapsed,
                           'timings': result['timings'], 'usage': result.get('usage'),
                           'message_sha256': hashlib.sha256(json.dumps(result['choices'][0]['message'], sort_keys=True).encode()).hexdigest(),
                           'telemetry': telemetry(section)}
                    prefix = out / f'{tag}.{size}.r{repeat}'
                    prefix.with_suffix(prefix.suffix + '.json').write_text(json.dumps({'metrics': row, 'response': result}, indent=2))
                    print(json.dumps({k: v for k, v in row.items() if k != 'telemetry'}), flush=True)
                    if row['timings'].get('cache_n', 0) != 0 or row['timings']['prompt_n'] != int(size):
                        raise RuntimeError('cached or incorrectly sized prompt')
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=30)
