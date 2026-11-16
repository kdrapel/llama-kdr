"""Run fresh-process, unprofiled GSQ attention A/B requests."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/development/measurements/rdna4-q4'
BASE = 'http://127.0.0.1:9092'
parser = argparse.ArgumentParser(description=__doc__)
workload = parser.add_mutually_exclusive_group(required=True)
workload.add_argument('--size', type=int, choices=[8192, 65536, 98304])
workload.add_argument('--prompt-text')
parser.add_argument('--temperature', type=float, default=0)
parser.add_argument('--n-predict', type=int, default=256)
parser.add_argument('--allow-eos', action='store_true')
parser.add_argument('--order', default='candidate,stock,stock,candidate')
parser.add_argument('--tag', required=True)
parser.add_argument('--quality', action='store_true')
parser.add_argument('--output-dir', type=Path, default=OUT)
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
variants = args.order.split(',')
assert all(v in ('candidate', 'stock') for v in variants)
assert args.tag.replace('-', '').replace('_', '').isalnum()
assert args.n_predict > 0
source = json.loads((ROOT / 'docs/development/dspine-rocm-prefill-results.json').read_text())
command = [v.replace('\\\\', '\\') for v in source['cases']['baseline']['command']['argv']]
command[0] = str(ROOT / 'build-rocm/bin/llama-server.exe')
command[command.index('-m') + 1] = 'D:/40-49 Assets/44 AI Models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf'
command[command.index('--alias') + 1] = 'GSQ-RCO-FA-AB'
command[command.index('--temp') + 1] = str(args.temperature)
binary_hashes = {}
for name in ['llama-server.exe', 'llama-server-impl.dll', 'ggml-hip.dll', 'llama.dll', 'ggml-base.dll']:
    binary_hashes[name] = hashlib.sha256((ROOT / 'build-rocm/bin' / name).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def post(endpoint, payload):
    request = urllib.request.Request(BASE + endpoint, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors='replace')) from exc


def tokenize(content, add_special):
    return post('/tokenize', {'content': content, 'add_special': add_special, 'parse_special': True})['tokens']


def make_request():
    if args.prompt_text is not None:
        rendered = post('/apply-template', {'messages': [{'role': 'user', 'content': args.prompt_text}]})['prompt']
        tokens = tokenize(rendered, True)
        args.size = len(tokens)
        return {'prompt': tokens, 'n_predict': args.n_predict, 'temperature': args.temperature, 'seed': 42,
                'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0,
                'cache_prompt': False, 'return_tokens': True, 'stream': False}
    messages = copy.deepcopy(source['requests']['8192']['messages'])
    marker = '\n\nNow switch to the numerical utility'
    history, separator, question = messages[-1]['content'].rpartition(marker)
    assert separator
    for repeats in range(1, 30):
        messages[-1]['content'] = (history + '\n') * repeats + separator + question
        rendered = post('/apply-template', {'messages': messages})['prompt']
        before, separator, after = rendered.rpartition(marker)
        assert separator
        prefix = tokenize(before, True)
        suffix = tokenize(separator + after, False)
        if len(prefix) + len(suffix) >= args.size:
            tokens = prefix[:args.size - len(suffix)] + suffix
            assert len(tokens) == args.size
            return {'prompt': tokens, 'n_predict': args.n_predict, 'temperature': args.temperature, 'seed': 42,
                    'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0,
                    'cache_prompt': False, 'return_tokens': True, 'stream': False}
    raise RuntimeError('could not construct the requested populated context')


request_path = OUT / (args.tag + '.request.json')
summary_path = OUT / (args.tag + '.summary.json')
if summary_path.exists() or request_path.exists():
    raise RuntimeError('refusing to overwrite an earlier run')
rows = []
payload = None
for trial, variant in enumerate(variants, 1):
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 9092)) == 0:
            raise RuntimeError('port 9092 is already occupied')
    prefix = OUT / f'{args.tag}-{trial}-{variant}'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_'))}
    env['GGML_CUDA_FA_QR_LOG'] = '1'
    if variant == 'stock':
        env['GGML_CUDA_DISABLE_FA_QR'] = '1'
    write(prefix.with_suffix('.command.json'), {
        'argv': command, 'environment_overrides': {k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_'))},
        'binary_sha256': binary_hashes, 'request_file': str(request_path),
        'prompt_source': args.prompt_text if args.prompt_text is not None else 'saved 8192-token engineering history, repeated/trimmed before final Fibonacci request',
    })
    print(json.dumps({'event': 'starting', 'trial': trial, 'variant': variant, 'size': args.size}), flush=True)
    log_path = prefix.with_suffix('.log')
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited with {proc.returncode}; see {log_path}')
                try:
                    with urllib.request.urlopen(BASE + '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('server startup timed out')
            if payload is None:
                payload = make_request()
                write(request_path, payload)
            request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            write(prefix.with_suffix('.status.json'), {'event': 'generating', 'pid': proc.pid, 'request_sha256': request_hash})
            print(json.dumps({'event': 'generating', 'trial': trial, 'variant': variant,
                              'pid': proc.pid, 'request_sha256': request_hash}), flush=True)
            started = time.perf_counter()
            result = post('/completion', payload)
            wall_s = time.perf_counter() - started
            write(prefix.with_suffix('.response.json'), result)
            timings = result['timings']
            text = result['content']
            row = {'trial': trial, 'variant': variant, 'context_tokens': args.size, 'wall_s': wall_s,
                   'request_sha256': request_hash, 'timings': timings,
                   'tokens_evaluated': result.get('tokens_evaluated'), 'truncated': result.get('truncated'),
                   'stop_type': result.get('stop_type'), 'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'output_token_sha256': hashlib.sha256(json.dumps(result.get('tokens', [])).encode()).hexdigest(),
                   'output_characters': len(text), 'response_file': str(prefix.with_suffix('.response.json'))}
            rows.append(row)
            write(summary_path, {'size': args.size, 'order': variants, 'rows': rows, 'binary_sha256': binary_hashes})
            print(json.dumps(row), flush=True)
            if timings.get('cache_n', 0) != 0 or timings['prompt_n'] != args.size:
                raise RuntimeError('prompt cache or incorrect context length invalidates the comparison')
            if timings['predicted_n'] != args.n_predict and not (args.allow_eos and result.get('stop_type') == 'eos'):
                raise RuntimeError('response did not reach the requested comparison length or an allowed EOS')
            if not text or '\ufffd' in text:
                raise RuntimeError('empty or invalid decoded text')
            if args.quality:
                quality = dict(payload)
                quality['prompt'] = payload['prompt'] + tokenize('\n</think>\n\n', False)
                quality['n_predict'] = 1536
                quality['cache_prompt'] = True
                write(prefix.with_suffix('.quality-request.json'), quality)
                print(json.dumps({'event': 'quality', 'trial': trial, 'variant': variant}), flush=True)
                quality_result = post('/completion', quality)
                write(prefix.with_suffix('.quality-response.json'), quality_result)
                print(json.dumps({'event': 'quality_done', 'trial': trial, 'variant': variant,
                                  'stop_type': quality_result.get('stop_type'),
                                  'tokens': quality_result['timings']['predicted_n']}), flush=True)
        except BaseException as exc:
            write(prefix.with_suffix('.error.json'), {
                'error_type': type(exc).__name__, 'message': str(exc),
                'server_exit_code': proc.poll(), 'log_file': str(log_path),
            })
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    time.sleep(2)
