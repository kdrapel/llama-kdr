"""Summarize opt-in GGML_CUDA_PROFILE_OPS logs; timings include synchronization overhead."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def summarize(path):
    groups = defaultdict(lambda: [0, 0])
    tensors = defaultdict(lambda: [0, 0])
    pattern = re.compile(r"cuda_profile: device=(\d+) op=(\S+) name=(.*?) src0=(.*?) type=(\S+) ne=(\S+) fused=(\d+) us=(\d+)")
    with path.open(errors="replace") as source:
        for line in source:
            match = pattern.search(line)
            if not match:
                continue
            device, op, name, weight, quant, shape, fused, elapsed = match.groups()
            for table, key in ((groups, (device, op, quant)), (tensors, (device, name, weight, quant, shape))):
                table[key][0] += int(elapsed)
                table[key][1] += 1

    def rows(table):
        return sorted(
            [{"key": list(key), "seconds": value[0] / 1e6, "calls": value[1]} for key, value in table.items()],
            key=lambda row: -row["seconds"],
        )

    return {
        "source": str(path),
        "measurement": "Serialized device synchronization plus host dispatch; not normal throughput. Fused groups are charged to their first node. Includes warmup, prefill and decode.",
        "total_serialized_seconds": sum(value[0] for value in groups.values()) / 1e6,
        "by_op_type": rows(groups),
        "by_tensor_shape": rows(tensors),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.logs:
        result = summarize(path)
        destination = path.with_suffix(".summary.json")
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(destination)
        for row in result["by_op_type"][:8]:
            print(row)
