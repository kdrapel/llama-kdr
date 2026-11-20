#!/usr/bin/env python3
"""Combine the recorded DSpine small- and long-context speed curves."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/development"
SMALL = DOCS / "dspine-rocm-speed-curve-1k-8k.json"
LONG = DOCS / "dspine-rocm-speed-curve-16k-96k.json"
OUT = DOCS / "dspine-rocm-speed-curve-1k-96k"


def load_data():
    small = json.loads(SMALL.read_text(encoding="utf-8"))
    long = json.loads(LONG.read_text(encoding="utf-8"))
    points = sorted(small["points"] + long["points"], key=lambda p: p["context_tokens"])
    if [p["context_tokens"] for p in points] != [1024, 2048, 4096, 8192, 16384, 32768, 65536, 96000]:
        raise ValueError("Unexpected populated-context points")
    return small, long, points


def main():
    small, long, points = load_data()
    metadata = {
        "title": "DSpine generation and prefill speed across populated context",
        "model": long["model"],
        "gpu": long["gpu"],
        "context_capacity": long["context_capacity"],
        "target_kv": long["target_kv"],
        "prompt_cache": False,
        "flash_attention": True,
        "speculative_decoding": long["speculative_decoding"],
        "sampling": long["sampling"],
        "averaging": {
            "1k_to_8k": "Equal-weight mean of one 96-token and one 256-token run; whiskers show their range.",
            "16k_to_96k": "Equal-weight mean of two 256-token runs; whiskers show their range.",
        },
        "x_axis": "log2 populated context; positions correspond to token count",
        "points": points,
    }
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "context_tokens", "generation_tok_s_mean", "generation_tok_s_min", "generation_tok_s_max",
            "prefill_tok_s_mean", "prefill_tok_s_min", "prefill_tok_s_max", "sample_count", "sample_basis",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            is_small = point["context_tokens"] <= 8192
            writer.writerow({
                "context_tokens": point["context_tokens"],
                "generation_tok_s_mean": point["generation_tok_s_mean"],
                "generation_tok_s_min": point["generation_tok_s_min"],
                "generation_tok_s_max": point["generation_tok_s_max"],
                "prefill_tok_s_mean": point["prefill_tok_s_mean"],
                "prefill_tok_s_min": point["prefill_tok_s_min"],
                "prefill_tok_s_max": point["prefill_tok_s_max"],
                "sample_count": len(point["samples"]),
                "sample_basis": "96+256 tokens" if is_small else "two 256-token runs",
            })

    # Log2 spacing keeps the 1K-8K points readable while preserving context ratios.
    x = [point["context_tokens"] / 1024 for point in points]
    labels = ["1K", "2K", "4K", "8K", "16K", "32K", "64K", "96K"]
    gen = [point["generation_tok_s_mean"] for point in points]
    gen_min = [point["generation_tok_s_min"] for point in points]
    gen_max = [point["generation_tok_s_max"] for point in points]
    prefill = [point["prefill_tok_s_mean"] for point in points]
    prefill_min = [point["prefill_tok_s_min"] for point in points]
    prefill_max = [point["prefill_tok_s_max"] for point in points]

    fig, gen_axis = plt.subplots(figsize=(10.2, 5.8), layout="constrained")
    prefill_axis = gen_axis.twinx()
    gen_color, prefill_color = "#1769aa", "#d97706"
    gen_axis.errorbar(
        x, gen,
        yerr=[[m - lo for m, lo in zip(gen, gen_min)], [hi - m for m, hi in zip(gen, gen_max)]],
        color=gen_color, marker="o", linewidth=2.3, markersize=6.5, capsize=3.5,
    )
    prefill_axis.errorbar(
        x, prefill,
        yerr=[[m - lo for m, lo in zip(prefill, prefill_min)], [hi - m for m, hi in zip(prefill, prefill_max)]],
        color=prefill_color, marker="s", linewidth=2.3, markersize=5.5, capsize=3.5,
    )
    for xv, yv in zip(x, gen):
        gen_axis.annotate(f"{yv:.1f}", (xv, yv), xytext=(0, 8), textcoords="offset points",
                          ha="center", color=gen_color, fontsize=8)
    for xv, yv in zip(x, prefill):
        prefill_axis.annotate(f"{yv:.0f}", (xv, yv), xytext=(0, -15), textcoords="offset points",
                              ha="center", va="top", color=prefill_color, fontsize=8)

    gen_axis.set_xscale("log", base=2)
    gen_axis.set_xticks(x, labels)
    gen_axis.set_xlim(0.85, 112)
    gen_axis.set_ylim(0, max(gen) * 1.28)
    prefill_axis.set_ylim(0, max(prefill) * 1.28)
    gen_axis.set_title(
        "Generation and prefill speed vs populated context\n"
        f"{long['model']}\nRX 9060 XT | Q4_0 KV cache",
        fontsize=12,
    )
    gen_axis.set_xlabel("Populated context (log scale)")
    gen_axis.set_ylabel("Generation speed (tok/s)", color=gen_color)
    prefill_axis.set_ylabel("Prefill speed (tok/s)", color=prefill_color)
    gen_axis.tick_params(axis="y", colors=gen_color)
    prefill_axis.tick_params(axis="y", colors=prefill_color)
    gen_axis.grid(axis="y", alpha=0.25)
    gen_axis.legend(handles=[
        Line2D([0], [0], color=gen_color, marker="o", linewidth=2.3, label="Generation (left axis)"),
        Line2D([0], [0], color=prefill_color, marker="s", linewidth=2.3, label="Prefill (right axis)"),
    ], loc="upper right")
    fig.text(0.5, -0.01,
             "Whiskers show sample ranges. 1K-8K averages 96/256-token runs; 16K-96K averages two 256-token runs. Prompt cache disabled.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
