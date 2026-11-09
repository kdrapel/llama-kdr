import collections
from pathlib import Path
import re
import statistics
import sys

bench = Path(__file__).resolve().parents[1] / "build-rocm/bench"
for pattern in sys.argv[1:] or ["ops_*.log"]:
    for path in sorted(bench.glob(pattern)):
        groups = collections.defaultdict(list)
        active = False
        for line in path.read_text(errors="replace").splitlines():
            if "init sampler" in line:
                active = True
            if not active:
                continue
            match = re.search(r"CLIFF op name=(.*?) type=(\S+) us=(\d+)(?: dispatch_us=(\d+) event_ms=(-?[\d.]+))? ne=(\S+)", line)
            if match:
                name, op, us, dispatch, gpu, shape = match.groups()
                # Preserve tensor shape to distinguish verification from one-token graphs.
                groups[(re.sub(r"-\d+$", "", name), op, shape)].append((int(us), float(gpu or 0)))
        print(path.stem)
        totals = collections.defaultdict(float)
        for (_, op, _), values in groups.items():
            totals[op] += sum(v[0] for v in values) / 1000
        print("synchronized wall ms by first op in each fused group:", dict(sorted(totals.items(), key=lambda x: x[1], reverse=True)))
        for key, values in sorted(groups.items(), key=lambda x: sum(v[0] for v in x[1]), reverse=True)[:25]:
            print(key, "count", len(values), "wall_ms", round(sum(v[0] for v in values)/1000, 2), "mean_us", round(statistics.mean(v[0] for v in values), 2), "gpu_ms", round(sum(v[1] for v in values), 2))
