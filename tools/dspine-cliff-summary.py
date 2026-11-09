"""Summarize responses and opt-in HIP/DXGI logs; byte counts are not residency traces."""
import json
import hashlib
from pathlib import Path
import re
import statistics
import sys

bench = Path(__file__).resolve().parents[1] / "build-rocm/bench"
for pattern in sys.argv[1:] or ["prof2_*.log"]:
    for path in sorted(bench.glob(pattern)):
        text = path.read_text(errors="replace")
        response = path.with_suffix(".json")
        result = {"tag": path.stem}
        if response.exists():
            data = json.loads(response.read_text())
            result.update(data.get("timings", {}))
            result["message_sha256"] = hashlib.sha256(json.dumps(data["choices"][0]["message"], sort_keys=True).encode()).hexdigest()
        stats = re.search(r"#calls\(b,g,a\) =\s+(\d+)\s+(\d+)\s+(\d+).*dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms", text)
        if stats:
            _, draft_calls, verify_calls, _, draft_ms, _ = stats.groups()
            result["verification_calls"] = int(verify_calls)
            result["draft_mean_wall_ms"] = round(float(draft_ms) / int(draft_calls), 3)
            if result.get("draft_n"):
                result["acceptance_rate"] = round(result["draft_n_accepted"] / result["draft_n"], 6)
                result["accepted_per_verification"] = round(result["draft_n_accepted"] / int(verify_calls), 4)
                result["proposed_per_verification"] = round(result["draft_n"] / int(verify_calls), 4)
                result["estimated_window_with_anchor"] = round(1 + result["draft_n"] / int(verify_calls), 4)
        allocs = [int(x) for x in re.findall(r" peak=(\d+)", text)]
        result["tracked_peak_MiB"] = round(max(allocs) / 2**20, 2) if allocs else None
        samples = re.findall(r"adapter=0 dedicated=(\d+) local_budget=(\d+) local_usage=(\d+) shared_budget=(\d+) shared_usage=(\d+)", text)
        if samples:
            for col, key in [(1, "budget"), (2, "local"), (4, "shared")]:
                vals = [int(x[col]) / 2**20 for x in samples]
                result[key + "_MiB_min_max"] = [round(min(vals), 2), round(max(vals), 2)]
        groups = {}
        for nodes, captured, update, elapsed in re.findall(r"CLIFF graph nodes=(\d+) captured=(\d+) update=(\d+) elapsed_us=(\d+)", text):
            groups.setdefault(f"{nodes}:{captured}:{update}", []).append(int(elapsed) / 1000)
        result["graphs_nodes_captured_update"] = {k: {"count": len(v), "mean_ms": round(statistics.mean(v), 3), "min_ms": round(min(v), 3), "max_ms": round(max(v), 3)} for k, v in groups.items()}
        steady = text.split("init_sampler", 1)[-1]
        samples = re.findall(r"adapter=0 dedicated=(\d+) local_budget=(\d+) local_usage=(\d+) shared_budget=(\d+) shared_usage=(\d+)", steady)
        if samples:
            result["decode_dxgi_MiB"] = {key: [round(min(int(x[col]) for x in samples)/2**20, 2), round(max(int(x[col]) for x in samples)/2**20, 2)] for col, key in [(1, "budget"), (2, "local"), (4, "nonlocal")]}
        windows = re.findall(r"CLIFF graph nodes=3910 .* columns=(\d+) kv=(\d+)", steady)
        if windows:
            result["target_graph_columns"] = sorted(set(int(x[0]) for x in windows))
            result["target_populated_kv_padded"] = sorted(set(int(x[1]) for x in windows))
        compute = re.findall(r"ROCm0 compute buffer size =\s+([\d.]+) MiB", text)
        if compute:
            result["last_target_compute_MiB"] = float(compute[-1])
        print(json.dumps(result))
