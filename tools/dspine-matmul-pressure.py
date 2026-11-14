"""Windows diagnostic: replay real GGUF FFN matrices with resident background buffers."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--background-mib', type=int, default=0)
parser.add_argument('--columns', type=int, default=512)
parser.add_argument('--samples', type=int, default=20)
parser.add_argument('--touch', action='store_true', help='touch background before each timed compute, outside its interval')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--runtime-bin', type=Path, default=Path('D:/AMD/ROCM/build/bin'))
args = parser.parse_args()
if args.background_mib < 0 or args.background_mib > 14336 or args.samples < 1 or args.columns < 1:
    parser.error('invalid allocation or sample dimensions')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'gguf-py'))
import numpy as np
from gguf import GGUFReader

dll_directory = os.add_dll_directory(str(root / 'build-rocm/bin'))
runtime_directory = os.add_dll_directory(str(args.runtime_bin.resolve()))
libraries = [ct.CDLL(str(root / 'build-rocm/bin' / name)) for name in ['ggml-base.dll', 'ggml.dll']]
ptr = ct.c_void_p
size = ct.c_size_t


def api(name, result, *arguments):
    fn = next((getattr(lib, name) for lib in libraries if hasattr(lib, name)), None)
    if fn is None:
        raise RuntimeError(f'missing ggml API: {name}')
    fn.restype = result
    fn.argtypes = list(arguments)
    return fn


class InitParams(ct.Structure):
    _fields_ = [('mem_size', size), ('mem_buffer', ptr), ('no_alloc', ct.c_bool)]


load = api('ggml_backend_load_all_from_path', None, ct.c_char_p)
initialize = api('ggml_backend_init_by_name', ptr, ct.c_char_p, ct.c_char_p)
alloc = api('ggml_backend_alloc_buffer', ptr, ptr, size)
clear = api('ggml_backend_buffer_clear', None, ptr, ct.c_uint8)
free_buffer = api('ggml_backend_buffer_free', None, ptr)
free_backend = api('ggml_backend_free', None, ptr)
sync = api('ggml_backend_synchronize', None, ptr)
init = api('ggml_init', ptr, InitParams)
free_context = api('ggml_free', None, ptr)
tensor = api('ggml_new_tensor_2d', ptr, ptr, ct.c_int, ct.c_int64, ct.c_int64)
mul_mat = api('ggml_mul_mat', ptr, ptr, ptr, ptr)
new_graph = api('ggml_new_graph_custom', ptr, ptr, size, ct.c_bool)
expand = api('ggml_build_forward_expand', None, ptr, ptr)
alloc_context = api('ggml_backend_alloc_ctx_tensors', ptr, ptr, ptr)
tensor_set = api('ggml_backend_tensor_set', None, ptr, ptr, size, size)
tensor_get = api('ggml_backend_tensor_get', None, ptr, ptr, size, size)
compute = api('ggml_backend_graph_compute', ct.c_int, ptr, ptr)
load(str(root / 'build-rocm/bin').encode())
backend = initialize(b'ROCm0', None)
if not backend:
    raise RuntimeError('ROCm0 unavailable')
background = []
results = []
try:
    for mib in range(0, args.background_mib, 256):
        buf = alloc(backend, min(256, args.background_mib - mib) * 1048576)
        if not buf:
            raise RuntimeError('background allocation failed')
        background.append(buf)
        clear(buf, 0)
    sync(backend)
    reader = GGUFReader(str(args.model))
    for name in ['blk.1.ffn_gate.weight', 'blk.1.ffn_down.weight', 'blk.20.ffn_gate.weight']:
        weight = next(t for t in reader.tensors if t.name == name)
        k, m = map(int, weight.shape)
        ctx = init(InitParams(2 * 1048576, None, True))
        if not ctx:
            raise RuntimeError('ggml_init failed')
        tensor_buffer = None
        try:
            a = tensor(ctx, int(weight.tensor_type), k, m)
            b = tensor(ctx, 0, k, args.columns)
            c = mul_mat(ctx, a, b)
            graph = new_graph(ctx, 32, False)
            expand(graph, c)
            tensor_buffer = alloc_context(ctx, backend)
            if not tensor_buffer:
                raise RuntimeError('tensor allocation failed')
            values = np.sin(np.arange(k * args.columns, dtype=np.float32) * np.float32(0.01))
            tensor_set(a, ptr(weight.data.ctypes.data), 0, weight.data.nbytes)
            tensor_set(b, ptr(values.ctypes.data), 0, values.nbytes)
            for _ in range(10):
                if compute(backend, graph) != 0:
                    raise RuntimeError('warmup compute failed')
            sync(backend)
            samples = []
            for _ in range(args.samples):
                if args.touch:
                    for buf in background:
                        clear(buf, 0)
                    sync(backend)
                started = time.perf_counter()
                if compute(backend, graph) != 0:
                    raise RuntimeError('compute failed')
                sync(backend)
                samples.append((time.perf_counter() - started) * 1000)
            output = np.empty(m * args.columns, dtype=np.float32)
            tensor_get(c, ptr(output.ctypes.data), 0, output.nbytes)
            if not np.isfinite(output).all():
                raise RuntimeError('nonfinite output')
            row = {'name': name, 'type': weight.tensor_type.name, 'k': k, 'm': m,
                   'columns': args.columns, 'median_ms': statistics.median(samples), 'samples_ms': samples,
                   'output_sum': float(output.sum(dtype=np.float64)),
                   'output_sha256': hashlib.sha256(output.tobytes()).hexdigest()}
            results.append(row)
            print(json.dumps(row), flush=True)
        finally:
            if tensor_buffer:
                free_buffer(tensor_buffer)
            free_context(ctx)
    args.out.write_text(json.dumps({'argv': sys.argv, 'background_mib': args.background_mib,
                                   'touch': args.touch, 'results': results}, indent=2))
finally:
    sync(backend)
    for buf in background:
        free_buffer(buf)
    free_backend(backend)
