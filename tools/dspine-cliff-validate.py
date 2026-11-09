"""Paired policy controls, normalized width sweep, and prompt/decode checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("group", choices=["controls", "widths", "prompt", "ordinary", "unprofiled", "draftub", "production"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
harness = root / "tools/dspine-cliff-bench.py"

def run(tag, cases, eager=False, extra=()):
    env = dict(os.environ)
    env.pop("LLAMA_GRAPH_RESERVE_FULL", None)
    if eager:
        env["LLAMA_GRAPH_RESERVE_FULL"] = "1"
    subprocess.run([sys.executable, str(harness), tag, "--cases", cases, *extra], cwd=root, env=env, check=True)

if args.group == "controls":
    cases = "24576:15:1024,32768:15:1024,65536:15:1024,131072:15:512,131072:6:1024"
    run("control_full", cases, eager=True, extra=["--profile"])
    run("control_fixed", cases, extra=["--profile"])
elif args.group == "widths":
    cases = ",".join(f"{ctx}:{width}:512" for ctx in [24576, 32768, 65536, 131072] for width in [6, 9, 12, 15])
    run("widths", cases, extra=["--profile"])
elif args.group == "prompt":
    request = json.loads((root / "build-rocm/bench/req.json").read_text())
    prefix = "Reference notes:\n" + "\n".join(f"Record {i}: the stored value equals {i % 17}." for i in range(150))
    request["messages"][0]["content"] = prefix + "\nIgnore the reference notes. " + request["messages"][0]["content"]
    path = root / "build-rocm/bench/cliff-long-request.json"
    path.write_text(json.dumps(request))
    for eager in [True, False]:
        run("prompt_full" if eager else "prompt_fixed", "65536:15:1024,131072:6:1024", eager=eager, extra=["--request", str(path), "--tokens", "64", "--profile"])
elif args.group == "ordinary":
    for eager in [True, False]:
        run("ordinary_full" if eager else "ordinary_fixed", "32768:6:1024,131072:6:1024", eager=eager, extra=["--extra", "--spec-type", "none"])
elif args.group == "unprofiled":
    for eager in [True, False]:
        run("unprofiled_full" if eager else "unprofiled_fixed", "65536:15:1024,131072:15:512", eager=eager)
elif args.group == "draftub":
    for draft_ub in [0, 32]:
        run(f"draftub_{draft_ub}", "32768:15:1024,131072:6:1024", extra=["--profile", "--extra", "--spec-draft-n-ubatch", str(draft_ub)])
else:
    run("production", "131072:6:1024,131072:15:512", extra=["--fusion", "on"])
