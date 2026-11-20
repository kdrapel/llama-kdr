#!/usr/bin/env python3
"""Plot paired DSpine generation and prefill measurements at long contexts."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    16384: [
        ("speed_curve_16k_96k_256x2_allmsg_20261114", 1),
        ("speed_curve_16k_64k_r2_20261114", 1),
    ],
    32768: [
        ("speed_curve_16k_96k_256x2_allmsg_20261114", 1),
        ("speed_curve_16k_64k_r2_20261114", 1),
    ],
    65536: [
        ("speed_curve_16k_96k_256x2_allmsg_20261114", 1),
        ("speed_curve_16k_64k_r2_20261114", 1),
    ],
    96000: [
        ("speed_curve_96k_256x2_20261114", 1),
        ("speed_curve_96k_256x2_20261114", 2),
    ],
}
OUT = ROOT / "docs/development/dspine-rocm-speed-curve-16k-96k"


def load_points():
    points = []
    for size, sources in SOURCES.items():
        samples = []
        for tag, repeat in sources:
            path = ROOT / "build-rocm/bench" / f"{tag}.{size}.r{repeat}.metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            timings = metrics["timings"]
            if metrics["size_tokenizer"] != size or timings["cache_n"] != 0:
                raise ValueError(f"Invalid cold-context sample in {path}")
            if timings["predicted_n"] != 256:
                raise ValueError(f"Unexpected generation length in {path}")
            samples.append({
                "source": tag,
                "repeat": repeat,
                "prefill_tok_s": timings["prompt_per_second"],
                "generation_tok_s": timings["predicted_per_second"],
                "prompt_ms": timings["prompt_ms"],
                "generation_ms": timings["predicted_ms"],
                "drafts_proposed": timings["draft_n"],
                "drafts_accepted": timings["draft_n_accepted"],
                "cache_tokens": timings["cache_n"],
                "message_sha256": metrics["message_sha256"],
            })
        points.append({
            "context_tokens": size,
            "prefill_tok_s_mean": sum(sample["prefill_tok_s"] for sample in samples) / len(samples),
            "prefill_tok_s_min": min(sample["prefill_tok_s"] for sample in samples),
            "prefill_tok_s_max": max(sample["prefill_tok_s"] for sample in samples),
            "generation_tok_s_mean": sum(sample["generation_tok_s"] for sample in samples) / len(samples),
            "generation_tok_s_min": min(sample["generation_tok_s"] for sample in samples),
            "generation_tok_s_max": max(sample["generation_tok_s"] for sample in samples),
            "samples": samples,
        })
    return points


def main():
    points = load_points()
    metadata = {
        "title": "DSpine long-context generation and prefill speed",
        "model": "Qwen3.8-27B-UD-IQ2_XXS.gguf",
        "gpu": "AMD Radeon RX 9060 XT (gfx1200)",
        "context_capacity": 131072,
        "target_kv": "Q4_0",
        "prompt_cache": False,
        "flash_attention": True,
        "speculative_decoding": "DFlash, max draft width 6",
        "sampling": {"seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "output_tokens_per_sample": 256,
        "samples_per_context": 2,
        "averaging": "Arithmetic mean; whiskers show the two-sample min/max.",
        "points": points,
    }
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fields = [
        "context_tokens", "generation_tok_s_run1", "generation_tok_s_run2", "generation_tok_s_mean",
        "prefill_tok_s_run1", "prefill_tok_s_run2", "prefill_tok_s_mean",
        "drafts_accepted_run1", "drafts_proposed_run1", "drafts_accepted_run2", "drafts_proposed_run2",
    ]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            run1, run2 = point["samples"]
            writer.writerow({
                "context_tokens": point["context_tokens"],
                "generation_tok_s_run1": run1["generation_tok_s"],
                "generation_tok_s_run2": run2["generation_tok_s"],
                "generation_tok_s_mean": point["generation_tok_s_mean"],
                "prefill_tok_s_run1": run1["prefill_tok_s"],
                "prefill_tok_s_run2": run2["prefill_tok_s"],
                "prefill_tok_s_mean": point["prefill_tok_s_mean"],
                "drafts_accepted_run1": run1["drafts_accepted"],
                "drafts_proposed_run1": run1["drafts_proposed"],
                "drafts_accepted_run2": run2["drafts_accepted"],
                "drafts_proposed_run2": run2["drafts_proposed"],
            })

    x = [point["context_tokens"] / 1000 for point in points]
    labels = ["16K", "32K", "64K", "96K"]
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
        generation_axis.annotate(f"{y_value:.1f}", (x_value, y_value),
                                 xytext=(-12, 8), textcoords="offset points", ha="right",
                                 color=generation_color, fontsize=8)
    for x_value, y_value in zip(x, prefill):
        prefill_axis.annotate(f"{y_value:.0f}", (x_value, y_value), xytext=(12, 8),
                              textcoords="offset points", ha="left", color=prefill_color, fontsize=8)

    generation_axis.set_title("Generation and prefill speed vs populated context\nQwen3.8-27B | RX 9060 XT | Q4_0 target KV")
    generation_axis.set_xlabel("Populated context")
    generation_axis.set_ylabel("Generation speed (tok/s)", color=generation_color)
    prefill_axis.set_ylabel("Prefill speed (tok/s)", color=prefill_color)
    generation_axis.set_xticks(x, labels)
    generation_axis.set_xlim(10, 102)
    generation_axis.set_ylim(0, max(generation) * 1.32)
    prefill_axis.set_ylim(0, max(prefill) * 1.3)
    generation_axis.tick_params(axis="y", colors=generation_color)
    prefill_axis.tick_params(axis="y", colors=prefill_color)
    generation_axis.grid(axis="y", alpha=0.25)
    generation_axis.legend(
        handles=[
            Line2D([0], [0], color=generation_color, marker="o", linewidth=2.4, label="Generation (left axis)"),
            Line2D([0], [0], color=prefill_color, marker="s", linewidth=2.4, label="Prefill (right axis)"),
        ],
        loc="upper right",
    )
    fig.text(0.5, -0.02, "Mean of two 256-token samples per size; whiskers show their range; prompt cache disabled.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
