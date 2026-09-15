#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


OUTPUT = None
ROOT = None

METHODS = (
    ("t00", "allctx-t00"),
    ("df", "df"),
    ("sampled_df_notail", "sampled-df-notail"),
    ("t50", "allctx-t50"),
    ("p01", "allctx-p01"),
    ("sampled_df", "sampled-df"),
    ("p00", "allctx-p00"),
)

# perf/cumulative_PFLOP added per optimizer step for the two compute families.
PFLOPS_PER_STEP = {
    "t00": 3.68965809340416,
    "t50": 3.68965809340416,
    "p01": 3.68965809340416,
    "p00": 3.68965809340416,
    "df": 1.94158962671616,
    "sampled_df_notail": 1.94158962671616,
    "sampled_df": 1.94158962671616,
}

QUANTITIES = (
    (
        "lr-exp1",
        "0-1000",
        "long_full",
        "Full long-rollout FVD vs. training compute",
        "256 videos, 100 × 10-frame clips per video (frames 1–1000)",
    ),
    (
        "lr-exp2",
        "0-249",
        "long_0-249",
        "Long-rollout FVD vs. training compute — frames 0–249",
        "256 videos, 25 × 10-frame clips per video",
    ),
    (
        "lr-exp2",
        "250-499",
        "long_250-499",
        "Long-rollout FVD vs. training compute — frames 250–499",
        "256 videos, 25 × 10-frame clips per video",
    ),
    (
        "lr-exp2",
        "500-749",
        "long_500-749",
        "Long-rollout FVD vs. training compute — frames 500–749",
        "256 videos, 25 × 10-frame clips per video",
    ),
    (
        "lr-exp2",
        "750-999",
        "long_750-999",
        "Long-rollout FVD vs. training compute — frames 750–999",
        "256 videos, 25 × 10-frame clips per video",
    ),

)


def training_step(root: str) -> int:
    match = re.search(r"step[_-]?(\d+)", root)
    if not match:
        raise ValueError(f"Cannot find checkpoint step in {root}")
    return int(match.group(1))


def load_rows():
    rows = []
    seen = set()
    with (ROOT / "fvd_all_results.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["experiment"] not in ("lr-exp1", "lr-exp2"):
                continue
            family, step = re.fullmatch(r"(.+)_s(\d+)", raw["run_key"]).groups()
            if int(raw["videos"]) != 256:
                raise RuntimeError("PFLOPs campaign requires 256 videos per run")
            key = (family, int(step), raw["experiment"], raw["window"])
            if key in seen:
                raise RuntimeError(f"Duplicate method/checkpoint/FVD quantity: {key}")
            seen.add(key)
            pflops = int(step) * PFLOPS_PER_STEP[family]
            rows.append({**raw, "family": family, "checkpoint_step": int(step),
                         "cumulative_pflops": pflops, "cumulative_pflops_k": pflops / 1000})
    return rows


def write_data(rows: list[dict[str, str | int | float]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["experiment"]),
            str(row["window"]),
            float(row["cumulative_pflops"]),
            str(row["run_key"]),
        ),
    )
    with (OUTPUT / "fvd_vs_pflops_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    (OUTPUT / "plot_config.json").write_text(
        json.dumps(
            {
                "source_results": str(ROOT / "fvd_all_results.csv"),
                "compute_reference": "August 2026 pflop_threshold_fvd_10f campaign: perf/cumulative_PFLOP per optimizer step",
                "methods": [
                    {"run_key": key, "label": label, "tab10_index": index}
                    for index, (key, label) in enumerate(METHODS)
                ],
                "pflops_per_step": PFLOPS_PER_STEP,
                "quantities": [
                    {
                        "experiment": experiment,
                        "window": window,
                        "slug": slug,
                        "title": title,
                        "subtitle": subtitle,
                    }
                    for experiment, window, slug, title, subtitle in QUANTITIES
                ],
            },
            indent=2,
        )
        + "\n"
    )


def plot_quantity(
    rows: list[dict[str, str | int | float]],
    experiment: str,
    window: str,
    slug: str,
    title: str,
    subtitle: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["experiment"] == experiment and row["window"] == window
    ]
    expected = len({row["run_key"] for row in rows})
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} unique method/checkpoint points for {slug}; "
            f"found {len(selected)}"
        )

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    for index, (run_key, label) in enumerate(METHODS):
        method_rows = sorted(
            (row for row in selected if row["family"] == run_key),
            key=lambda row: float(row["cumulative_pflops"]),
        )
        if not method_rows:
            continue
        ax.plot(
            [float(row["cumulative_pflops"]) for row in method_rows],
            [float(row["fvd"]) for row in method_rows],
            color=cmap(index % 10),
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            label=label,
        )

    tick_values = (37_000, 74_000, 111_000, 148_000, 184_000, 295_000, 406_000)
    ax.set_xticks(tick_values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    ax.set_xlabel("Cumulative PFLOPs")
    ax.set_ylabel("FVD ↓")
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = OUTPUT / f"fvd_vs_pflops_{slug}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    global ROOT, OUTPUT
    parser = argparse.ArgumentParser(description="Plot 256-video long-rollout FVD against cumulative training PFLOPs")
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.analysis.resolve()
    OUTPUT = ROOT / "fvd_vs_pflops"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    expected = len({row["run_key"] for row in rows}) * len(QUANTITIES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} FVD rows; found {len(rows)}")
    write_data(rows)
    for quantity in QUANTITIES:
        plot_quantity(rows, *quantity)


if __name__ == "__main__":
    main()
