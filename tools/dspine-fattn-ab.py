"""Run fresh-process, unprofiled GSQ attention A/B requests."""
import argparse
import copy
from datetime import datetime, timezone
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
workload.add_argument('--request-file', type=Path)
parser.add_argument('--temperature', type=float, default=0)
parser.add_argument('--n-predict', type=int, default=256)
parser.add_argument('--allow-eos', action='store_true')
parser.add_argument('--exact-saved-history', action='store_true')
parser.add_argument('--baseline-bin', type=Path)
parser.add_argument('--order', default='candidate,stock,stock,candidate')
parser.add_argument('--draft-order', type=int, nargs='+', choices=[2, 4, 6])
parser.add_argument('--context-order', type=int, nargs='+', help='set reserved context capacity for each trial')
parser.add_argument('--initial-kv-order', type=int, nargs='+',
                    help='set initial target KV cells for each trial (0 uses static allocation)')
parser.add_argument('--target-model', type=Path, help='override the target GGUF')
parser.add_argument('--compare-speculation', action='store_true',
                    help='compare DSpine with speculation and draft-model loading disabled')
parser.add_argument('--compare-mask-scan', action='store_true',
                    help='compare the experimental HIP prefill mask scan on the same build')
parser.add_argument('--compare-q2-508', action='store_true', help='compare HIP Q2_K projection dispatch at 508 columns')
parser.add_argument('--tag', required=True)
parser.add_argument('--quality', action='store_true')
parser.add_argument('--profile-prefill', action='store_true', help='collect synchronized server and HIP event prefill timings')
parser.add_argument('--profile-dflash-draft', action='store_true', help='time DFlash2 draft stages without extra device waits')
parser.add_argument('--profile-dflash-gpu', action='store_true', help='record HIP events around the draft graph and lattice readback')
parser.add_argument('--profile-dflash-ops', action='store_true', help='record GPU event timings for DFlash2 draft operators')
parser.add_argument('--disable-hip-graphs', action='store_true', help='run with HIP graph capture and replay disabled')
parser.add_argument('--output-dir', type=Path, default=OUT)
args = parser.parse_args()
if args.compare_speculation and (args.baseline_bin or args.draft_order):
    parser.error('--compare-speculation requires the same binary and a fixed draft width')
if args.compare_mask_scan and (args.compare_speculation or args.baseline_bin or args.draft_order):
    parser.error('--compare-mask-scan requires the same binary and speculation settings')
if args.compare_q2_508 and (args.compare_mask_scan or args.compare_speculation or args.baseline_bin or args.draft_order or not args.profile_prefill):
    parser.error('--compare-q2-508 requires --profile-prefill and matching binary and speculation settings')
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
variants = args.order.split(',')
if args.draft_order:
    if args.baseline_bin:
        parser.error('--draft-order uses the current binary for every trial')
    variants = ['candidate'] * len(args.draft_order)
if args.context_order and (len(args.context_order) != len(variants) or min(args.context_order) <= args.n_predict):
    parser.error('--context-order needs one capacity per trial with room for generation')
if args.initial_kv_order and (len(args.initial_kv_order) != len(variants) or min(args.initial_kv_order) < 0 or
                              any(variant != 'candidate' for variant in variants)):
    parser.error('--initial-kv-order needs one nonnegative size per trial and the same candidate build and attention path')
comparison_mode = ('speculation' if args.compare_speculation else
                   'hip_mask_scan' if args.compare_mask_scan else
                   'hip_q2_508' if args.compare_q2_508 else
                   'initial_target_kv' if args.initial_kv_order else
                   'draft_width' if args.draft_order else
                   'different_builds' if args.baseline_bin else 'same_binary_dispatch')
assert all(v in ('candidate', 'stock') for v in variants)
assert args.tag.replace('-', '').replace('_', '').isalnum()
assert args.n_predict > 0
assert not args.exact_saved_history or args.size == 8192
source = json.loads((ROOT / 'docs/development/dspine-rocm-prefill-results.json').read_text())
command = [v.replace('\\\\', '\\') for v in source['cases']['baseline']['command']['argv']]
command[0] = str(ROOT / 'build-rocm/bin/llama-server.exe')
command[command.index('-m') + 1] = str(args.target_model.resolve()) if args.target_model else (
    'D:/40-49 Assets/44 AI Models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf')
command[command.index('--alias') + 1] = 'GSQ-RCO-FA-AB'
command[command.index('--temp') + 1] = str(args.temperature)
current_bin = ROOT / 'build-rocm/bin'
variant_bins = {'candidate': current_bin, 'stock': args.baseline_bin.resolve() if args.baseline_bin else current_bin}
variant_hashes = {}
for variant, bin_path in variant_bins.items():
    variant_hashes[variant] = {}
    for name in ['llama-server.exe', 'llama-server-impl.dll', 'ggml-hip.dll', 'llama.dll', 'ggml-base.dll']:
        variant_hashes[variant][name] = hashlib.sha256((bin_path / name).read_bytes()).hexdigest()
binary_hashes = variant_hashes['candidate']


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
    if args.request_file is not None:
        request = json.loads(args.request_file.read_text(encoding='utf-8'))
        tokens = request['prompt']
        if not isinstance(tokens, list) or not tokens or not all(type(token) is int for token in tokens):
            raise ValueError('saved request must contain a nonempty array of prompt token IDs')
        args.size = len(tokens)
        return dict(request, n_predict=args.n_predict, temperature=args.temperature,
                    cache_prompt=False, return_tokens=True, stream=False)
    if args.exact_saved_history:
        rendered = post('/apply-template', {'messages': source['requests']['8192']['messages']})['prompt']
        tokens = tokenize(rendered, True)
        if len(tokens) != 8192:
            raise RuntimeError(f'exact saved history rendered to {len(tokens)} tokens rather than 8192')
        return {'prompt': tokens, 'n_predict': args.n_predict, 'temperature': args.temperature, 'seed': 42,
                'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0,
                'cache_prompt': False, 'return_tokens': True, 'stream': False}
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
if summary_path.exists() or request_path.exists() or any(OUT.glob(args.tag + '-[0-9]*')):
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
    initial_kv = args.initial_kv_order[trial - 1] if args.initial_kv_order else 0
    if initial_kv:
        env['LLAMA_KV_CACHE_INITIAL_SIZE'] = str(initial_kv)
    if args.profile_prefill:
        env['LLAMA_PROFILE_PREFILL'] = '1'
        env['GGML_HIP_PROFILE_PREFILL'] = '1'
    if args.profile_dflash_draft:
        env['LLAMA_PROFILE_DFLASH_DRAFT'] = '1'
    if args.profile_dflash_gpu:
        env['GGML_HIP_PROFILE_DFLASH_GPU'] = '1'
    if args.profile_dflash_ops:
        env['GGML_HIP_PROFILE_DFLASH_OPS'] = '1'
    if args.disable_hip_graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    if args.compare_q2_508:
        env['GGML_HIP_Q2_K_BLAS_508'] = str(int(variant == 'candidate'))
    elif args.compare_mask_scan:
        env['GGML_HIP_FA_MASK_SCAN_AB'] = str(int(variant == 'candidate'))
    elif variant == 'stock' and not args.compare_speculation:
        env['GGML_CUDA_DISABLE_FA_QR'] = '1'
    run_command = list(command)
    run_command[0] = str(variant_bins[variant] / 'llama-server.exe')
    if args.compare_speculation and variant == 'stock':
        run_command[run_command.index('--spec-type') + 1] = 'none'
        model_draft = run_command.index('--model-draft')
        del run_command[model_draft:model_draft + 2]
    if args.draft_order:
        run_command[run_command.index('--spec-draft-n-max') + 1] = str(args.draft_order[trial - 1])
    if args.context_order:
        run_command[run_command.index('--ctx-size') + 1] = str(args.context_order[trial - 1])
    spec_type = run_command[run_command.index('--spec-type') + 1]
    draft_width = 0 if spec_type == 'none' else int(run_command[run_command.index('--spec-draft-n-max') + 1])
    write(prefix.with_suffix('.command.json'), {
        'argv': run_command, 'environment_overrides': {k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_'))},
        'binary_sha256': variant_hashes[variant], 'request_file': str(request_path),
        'comparison_mode': comparison_mode,
        'spec_type': spec_type,
        'draft_width': draft_width,
        'exact_saved_history': args.exact_saved_history,
        'prompt_source': (str(args.request_file) if args.request_file is not None else
                          args.prompt_text if args.prompt_text is not None else
                          'saved 8192-token engineering history'),
    })
    print(json.dumps({'event': 'starting', 'trial': trial, 'variant': variant, 'size': args.size,
                      'draft_width': draft_width}), flush=True)
    log_path = prefix.with_suffix('.log')
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(run_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
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
            write(prefix.with_suffix('.status.json'), {'event': 'generating', 'pid': proc.pid, 'request_sha256': request_hash,
                                                       'request_start_utc': datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'event': 'generating', 'trial': trial, 'variant': variant,
                              'pid': proc.pid, 'request_sha256': request_hash}), flush=True)
            started = time.perf_counter()
            result = post('/completion', payload)
            wall_s = time.perf_counter() - started
            write(prefix.with_suffix('.response.json'), result)
            timings = result['timings']
            text = result['content']
            row = {'trial': trial, 'variant': variant, 'context_tokens': args.size, 'wall_s': wall_s,
                   'spec_type': spec_type,
                   'draft_width': draft_width, 'initial_target_kv_cells': initial_kv,
                   'binary_sha256': variant_hashes[variant], 'server_bin': str(variant_bins[variant]),
                   'request_sha256': request_hash, 'timings': timings,
                   'tokens_evaluated': result.get('tokens_evaluated'), 'truncated': result.get('truncated'),
                   'stop_type': result.get('stop_type'), 'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'output_token_sha256': hashlib.sha256(json.dumps(result.get('tokens', [])).encode()).hexdigest(),
                   'output_characters': len(text), 'response_file': str(prefix.with_suffix('.response.json'))}
            rows.append(row)
            write(summary_path, {'size': args.size, 'order': variants, 'rows': rows,
                                'binary_sha256': None if args.baseline_bin else binary_hashes,
                                'binary_sha256_by_variant': variant_hashes,
                                'comparison_mode': comparison_mode, 'draft_order': args.draft_order})
            print(json.dumps(row), flush=True)
            if timings.get('cache_n', 0) != 0 or timings['prompt_n'] != args.size:
                raise RuntimeError('prompt cache or incorrect context length invalidates the comparison')
            if timings['predicted_n'] != args.n_predict and not (args.allow_eos and result.get('stop_type') == 'eos'):
                raise RuntimeError('response did not reach the requested comparison length or an allowed EOS')
            if not text or '\ufffd' in text:
                raise RuntimeError('empty or invalid decoded text')
            if initial_kv and 'growing target KV enabled' not in log_path.read_text(encoding='utf-8', errors='replace'):
                raise RuntimeError('initial target KV setting did not enable cache growth')
            if args.compare_mask_scan:
                route = log_path.read_text(encoding='utf-8', errors='replace')
                expected = f'scan={int(variant == "candidate")}'
                if not any(expected in line for line in route.splitlines() if 'fa_mask_scan queries=512 ' in line):
                    raise RuntimeError(f'512-row mask scan did not reach the expected setting ({expected})')
            if args.compare_q2_508:
                route = log_path.read_text(encoding='utf-8', errors='replace').split('new prompt,', 1)[1]
                expected = 'route=blas' if variant == 'candidate' else 'route=mmq'
                if not any(expected in line for line in route.splitlines() if 'prefill_q2_route n=508 ' in line):
                    raise RuntimeError(f'508-column Q2_K projections did not reach the expected setting ({expected})')
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
