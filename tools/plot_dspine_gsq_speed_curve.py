#!/usr/bin/env python3
"""Plot saved GSQ-RCO DSpine throughput measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "build-rocm" / "bench"
OUT = ROOT / "docs" / "development" / "dspine-rocm-speed-curve-gsq-rco-8k-100k"
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
DRAFT = "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
SAMPLES = {
    8192: [
        ("unchanged-memory-20261115-122744-1", 1),
        ("unchanged-memory-20261115-122744-2", 1),
    ],
    32768: [("q2-32k-populated", 1)],
    99991: [("gsq-rco-100k-populated-20261115-125143", 1)],
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_name(command):
    args = command["argv"]
    return Path(args[args.index("-m") + 1]).name


def load_points():
    points = []
    summary = read_json(BENCH / "unchanged-memory-summary.json")
    common_hashes = None
    summary_runs = {Path(row["run"]).stem: row for row in summary}
    for context_tokens, runs in SAMPLES.items():
        samples = []
        for tag, run in runs:
            result = read_json(BENCH / f"{tag}.json")
            command = read_json(BENCH / f"{tag}.command.json")
            if model_name(command) != MODEL:
                raise ValueError(f"Unexpected target GGUF in {tag}")
            if Path(command["argv"][command["argv"].index("--model-draft") + 1]).name != DRAFT:
                raise ValueError(f"Unexpected DSpine draft in {tag}")
            timings = result["timings"]
            if timings["cache_n"] != 0 or timings["predicted_n"] != 96:
                raise ValueError(f"Unexpected cache or generation length in {tag}")
            if timings["prompt_n"] != context_tokens:
                raise ValueError(f"Unexpected prompt length in {tag}")
            if context_tokens == 8192:
                summary_row = summary_runs.get(tag)
                hashes = summary_row.get("binary_sha256") if summary_row else None
                if not hashes:
                    raise ValueError(f"Missing binary hashes in {tag}")
                if common_hashes is None:
                    common_hashes = hashes
                elif hashes != common_hashes:
                    raise ValueError("The 8K samples do not use the same binaries")
            samples.append({
                "source": tag,
                "run": run,
                "prefill_tok_s": timings["prompt_per_second"],
                "generation_tok_s": timings["predicted_per_second"],
                "prompt_ms": timings["prompt_ms"],
                "generation_ms": timings["predicted_ms"],
                "prompt_tokens": timings["prompt_n"],
                "output_tokens": timings["predicted_n"],
                "drafts_proposed": timings.get("draft_n"),
                "drafts_accepted": timings.get("draft_n_accepted"),
            })
        points.append({
            "context_tokens": context_tokens,
            "prefill_tok_s_mean": sum(sample["prefill_tok_s"] for sample in samples) / len(samples),
            "prefill_tok_s_min": min(sample["prefill_tok_s"] for sample in samples),
            "prefill_tok_s_max": max(sample["prefill_tok_s"] for sample in samples),
            "generation_tok_s_mean": sum(sample["generation_tok_s"] for sample in samples) / len(samples),
            "generation_tok_s_min": min(sample["generation_tok_s"] for sample in samples),
            "generation_tok_s_max": max(sample["generation_tok_s"] for sample in samples),
            "samples": samples,
        })
    return points, common_hashes


def main():
    points, hashes = load_points()
    metadata = {
        "title": "DSpine generation and prefill speed vs populated context",
        "model": MODEL,
        "draft": DRAFT,
        "engine": "DSpine llama.cpp ROCm build with gfx1200 Q2_K hipBLAS dispatch",
        "gpu": "AMD Radeon RX 9060 XT (gfx1200)",
        "context_capacity": 131072,
        "target_kv": "Q4_0",
        "prompt_cache": False,
        "flash_attention": True,
        "speculative_decoding": "DFlash, max draft width 6",
        "generated_tokens_per_run": 96,
        "sampling": {"seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "averaging": "8K is the arithmetic mean of two same-binary runs; 32K and 100K are single samples.",
        "prompt_construction": "Synthetic history. The 32K and 100K prompts were tokenized from the saved 8K history, then trimmed to the stated token counts.",
        "eight_k_binary_sha256": hashes,
        "points": points,
    }
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fields = [
        "context_tokens", "generation_tok_s_mean", "generation_tok_s_min", "generation_tok_s_max",
        "prefill_tok_s_mean", "prefill_tok_s_min", "prefill_tok_s_max", "sample_count",
    ]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({
                "context_tokens": point["context_tokens"],
                "generation_tok_s_mean": point["generation_tok_s_mean"],
                "generation_tok_s_min": point["generation_tok_s_min"],
                "generation_tok_s_max": point["generation_tok_s_max"],
                "prefill_tok_s_mean": point["prefill_tok_s_mean"],
                "prefill_tok_s_min": point["prefill_tok_s_min"],
                "prefill_tok_s_max": point["prefill_tok_s_max"],
                "sample_count": len(point["samples"]),
            })

    x = [point["context_tokens"] / 1024 for point in points]
    labels = ["8K", "32K", "100K"]
    generation = [point["generation_tok_s_mean"] for point in points]
    generation_min = [point["generation_tok_s_min"] for point in points]
    generation_max = [point["generation_tok_s_max"] for point in points]
    prefill = [point["prefill_tok_s_mean"] for point in points]
    prefill_min = [point["prefill_tok_s_min"] for point in points]
    prefill_max = [point["prefill_tok_s_max"] for point in points]

    fig, generation_axis = plt.subplots(figsize=(9.2, 5.6), layout="constrained")
    prefill_axis = generation_axis.twinx()
    generation_color = "#1769aa"
    prefill_color = "#d97706"
    generation_axis.errorbar(
        x, generation,
        yerr=[[mean - low for mean, low in zip(generation, generation_min)],
              [high - mean for mean, high in zip(generation, generation_max)]],
        color=generation_color, marker="o", linewidth=2.4, markersize=7, capsize=4,
    )
    prefill_axis.errorbar(
        x, prefill,
        yerr=[[mean - low for mean, low in zip(prefill, prefill_min)],
              [high - mean for mean, high in zip(prefill, prefill_max)]],
        color=prefill_color, marker="s", linewidth=2.4, markersize=6, capsize=4,
    )

    for x_value, y_value in zip(x, generation):
        generation_axis.annotate(f"{y_value:.1f}", (x_value, y_value), xytext=(-12, 8),
                                 textcoords="offset points", ha="right", color=generation_color, fontsize=8)
    for x_value, y_value in zip(x, prefill):
        prefill_axis.annotate(f"{y_value:.0f}", (x_value, y_value), xytext=(12, 8),
                              textcoords="offset points", ha="left", color=prefill_color, fontsize=8)

    generation_axis.set_xscale("log", base=2)
    generation_axis.set_xticks(x, labels)
    generation_axis.set_xlim(6.5, 112)
    generation_axis.set_ylim(0, max(generation) * 1.4)
    prefill_axis.set_ylim(0, max(prefill) * 1.25)
    generation_axis.set_title(
        "Generation and prefill speed vs populated context\n"
        "Qwen3.8-27B-GSQ-RCO-IQ3_XXS | DSpine ROCm\nRX 9060 XT | Q4_0 target KV",
        fontsize=12,
    )
    generation_axis.set_xlabel("Populated context (log scale)")
    generation_axis.set_ylabel("Generation speed (tok/s)", color=generation_color)
    prefill_axis.set_ylabel("Prefill speed (tok/s)", color=prefill_color)
    generation_axis.tick_params(axis="y", colors=generation_color)
    prefill_axis.tick_params(axis="y", colors=prefill_color)
    generation_axis.grid(axis="y", alpha=0.25)
    generation_axis.legend(handles=[
        Line2D([0], [0], color=generation_color, marker="o", linewidth=2.4,
               label="Generation (left axis)"),
        Line2D([0], [0], color=prefill_color, marker="s", linewidth=2.4,
               label="Prefill (right axis)"),
    ], loc="upper right")
    fig.text(
        0.5, -0.02,
        "96 generated tokens per run; only 8K has two samples (whiskers show range); prompt cache disabled.",
        ha="center", fontsize=8, color="#555555",
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
