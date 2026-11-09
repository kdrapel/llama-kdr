"""Keep a portable result summary beside the investigation report."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
patterns = sys.argv[1:] or ["baseline_*.log", "prof2_*.log", "lazy_c*.log", "fixed_*.log", "control_*.log", "widths_*.log", "prompt_*.log", "ordinary_*.log", "unprofiled_*.log", "draftub_*.log", "production_*.log"]
output = subprocess.check_output([sys.executable, str(root / "tools/dspine-cliff-summary.py"), *patterns], cwd=root, text=True)
rows = [row for line in output.splitlines() if "predicted_n" in (row := json.loads(line))]
path = root / "docs/development/dspine-rocm-cliff-results.json"
path.write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    if not row["tag"].startswith(("control_", "widths_", "prompt_", "ordinary_", "draftub_", "unprofiled_", "production_")):
        continue
    perf = row.get("graphs_nodes_captured_update", {}).get("3910:1:0", {})
    print(row["tag"], "tg", round(row.get("predicted_per_second", 0), 2), "pp", round(row.get("prompt_per_second", 0), 2),
          "verify", perf.get("mean_ms"), "peak", row.get("tracked_peak_MiB"), "decode", row.get("decode_dxgi_MiB"), "compute", row.get("last_target_compute_MiB"))
print("saved", path)
