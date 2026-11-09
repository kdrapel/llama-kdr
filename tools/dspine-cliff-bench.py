"""Run the handoff request serially, preserving exact commands and responses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import socket
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--cases", default="24576:15:1024,32768:15:1024,32768:15:512,65536:15:1024,65536:15:512,65536:6:1024,131072:15:512,131072:6:1024")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--tokens", type=int, default=256)
parser.add_argument("--request", type=Path)
parser.add_argument("--fusion", choices=["on", "off"], default="off")
parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / "build-rocm/bench"
models = Path("D:/40-49 Assets/44 AI Models")
request_data = json.loads((args.request or out / "req.json").read_bytes())
request_data["max_tokens"] = args.tokens
request = json.dumps(request_data).encode()
for case in args.cases.split(","):
    try:
        existing = socket.create_connection(("127.0.0.1", 9092), timeout=1)
    except OSError:
        pass
    else:
        existing.close()
        raise RuntimeError("port 9092 is already in use")
    ctx, width, ub = case.split(":")
    tag = f"{args.tag}_c{ctx}_n{width}_u{ub}"
    cmd = [str(root / "build-rocm/bin/llama-server.exe"), "--host", "127.0.0.1", "--port", "9092",
           "-m", str(models / "Qwen3.8-27B-UD-IQ2_XXS.gguf"), "--model-draft", str(models / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"),
           "-np", "1", "-ngl", "all", "--load-mode", "none", "-t", "8", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
           "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "2048", "-ub", ub, "-c", ctx,
           "--spec-type", "draft-dflash", "--spec-draft-n-max", width, "--spec-draft-n-ubatch", "32", "-n", "-1", "-lv", "4"] + args.extra
    env = dict(os.environ)
    if args.fusion == "off":
        env["GGML_CUDA_GDN_FUSION"] = "0"
    else:
        env.pop("GGML_CUDA_GDN_FUSION", None)
    if args.profile:
        env["GGML_CUDA_CLIFF_PROFILE"] = "1"
    (out / f"{tag}.command.json").write_text(json.dumps({"argv": cmd, "request": json.loads(request), "env": {k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_"))}}, indent=2))
    with (out / f"{tag}.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=models, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {tag}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:9092/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("server readiness timeout")
            started = time.perf_counter()
            req = urllib.request.Request("http://127.0.0.1:9092/v1/chat/completions", request, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as response:
                data = response.read()
            elapsed = time.perf_counter() - started
            (out / f"{tag}.json").write_bytes(data)
            (out / f"{tag}.request-time.json").write_text(json.dumps({"request_wall_seconds": elapsed}))
            print(tag, round(elapsed, 3), json.loads(data).get("timings"), flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    time.sleep(2)
