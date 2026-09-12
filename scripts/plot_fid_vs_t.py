#!/usr/bin/env python
"""Plot FID (or any per-frame metric) vs. frame index t for several methods.

Designed for a single manuscript column: compact, thick color-blind-safe
lines, and direct on-line labels (no legend box) by default.

Example:
    python plot_fid_vs_t.py \
        --labels PERSIST,Oasis,WorldMem \
        --csvs persist_base_fid_per_frame.csv,ab_pixonly_fid_per_frame.csv,worldmem_fid_per_frame.csv \
        --smooth 9 --output fid_vs_t

Each CSV is expected to have columns `frame_index,fid` (header row). Curves may
have different lengths (e.g. one method ending earlier than the others).
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# Color-blind-safe qualitative palette (seaborn "colorblind"), strong in print.
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
           "#CA9161", "#949494", "#ECE133", "#56B4E9"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Use LaTeX when installed; keep headless analysis hosts supported.
        "text.usetex": shutil.which("latex") is not None,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Embed editable TrueType text in vector output (camera-ready friendly).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def smooth_series(y, window):
    """Centered moving average; endpoints averaged over the available window."""
    if window <= 1:
        return y
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def spread_labels(ys, min_gap):
    """Nudge near-coincident label y-positions apart, preserving order."""
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    s = ys[order].copy()
    for i in range(1, len(s)):
        if s[i] - s[i - 1] < min_gap:
            s[i] = s[i - 1] + min_gap
    out = np.empty_like(s)
    out[order] = s
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot FID vs frame index t for several methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--labels", required=True,
                   help="Comma-separated curve labels, in the same order as --csvs.")
    p.add_argument("--csvs", required=True,
                   help="Comma-separated CSV paths (columns: frame_index,fid).")
    p.add_argument("--smooth", type=int, default=1,
                   help="Moving-average window (frames). 1 = no smoothing.")
    p.add_argument("--shift", default=None,
                   help="Comma-separated per-curve frame offsets (same order as "
                        "--csvs), e.g. '0,0,400' to align a curve to later frames.")
    p.add_argument("--show-raw", action="store_true",
                   help="When smoothing, also draw the raw curve faintly behind it.")
    p.add_argument("--label-mode", choices=["direct", "legend"], default="direct",
                   help="'direct' = labels on each line; 'legend' = legend box.")
    p.add_argument("--highlight", type=int, default=0,
                   help="Index of the curve to emphasize (thicker, on top). -1 = none.")
    p.add_argument("--linewidth", type=float, default=2.2,
                   help="Base line width for the curves.")
    p.add_argument("--logy", action="store_true", help="Log-scale the y axis.")
    p.add_argument("--xlabel", default=r"Frame $t$")
    p.add_argument("--fps", type=float,
                   help="Convert frame indices to rollout time in seconds.")
    p.add_argument("--ylabel", default=r"FID$\downarrow$")
    p.add_argument("--title", default=None)
    p.add_argument("--xmax", type=float, default=None, help="Clip x axis at this frame, or seconds when --fps is set.")
    p.add_argument("--width", type=float, default=3.35, help="Figure width (inches).")
    p.add_argument("--height", type=float, default=2.5, help="Figure height (inches).")
    p.add_argument("--output", default="fid_vs_t",
                   help="Output basename; writes <name>.pdf and <name>.png.")
    p.add_argument("--png-dpi", type=int, default=1600,
                   help="Raster resolution for the PNG output.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        sys.exit("--fps must be positive")
    labels = [s.strip() for s in args.labels.split(",")]
    csvs = [s.strip() for s in args.csvs.split(",")]

    if len(labels) != len(csvs):
        sys.exit(f"--labels has {len(labels)} entries but --csvs has {len(csvs)}.")

    missing = [c for c in csvs if not os.path.isfile(c)]
    if missing:
        sys.exit("CSV file(s) not found: " + ", ".join(missing))

    if args.shift is None:
        shifts = [0] * len(csvs)
    else:
        shifts = [int(s) for s in args.shift.split(",")]
        if len(shifts) != len(csvs):
            sys.exit(f"--shift has {len(shifts)} entries but --csvs has {len(csvs)}.")

    set_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    series = []  # (label, color, x, y)
    for i, (label, csv) in enumerate(zip(labels, csvs)):
        df = pd.read_csv(csv)
        x = df["frame_index"].to_numpy() + shifts[i]
        if args.fps is not None:
            x = x / args.fps
        y = df["fid"].to_numpy()
        if args.xmax is not None:
            keep = x <= args.xmax
            x, y = x[keep], y[keep]
        color = PALETTE[i % len(PALETTE)]
        emphasized = (i == args.highlight)
        lw = args.linewidth + (0.8 if emphasized else 0.0)
        z = 5 if emphasized else 3

        y_s = smooth_series(y, args.smooth)
        if args.show_raw and args.smooth > 1:
            ax.plot(x, y, color=color, lw=0.8, alpha=0.25, zorder=z - 1)
        ax.plot(x, y_s, color=color, lw=lw, zorder=z,
                label=label, solid_capstyle="round")
        series.append((label, color, x, y_s))

    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("Rollout time (s)" if args.fps is not None else args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    ax.margins(x=0.01)
    ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)

    if args.label_mode == "legend":
        ax.legend(loc="best")
    else:
        # Direct labels at each curve's end; spread the right-edge cluster and
        # add a white halo so text stays readable where it crosses other lines.
        x_hi = max(x[-1] for _, _, x, _ in series)
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.17 * (x1 - x0))  # room for end labels
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        gap = 0.075 * (y1 - y0)
        pad = 0.012 * (x1 - x0)

        # With the LaTeX engine, fontweight="bold" is ignored; emit bold markup
        # instead (\boldmath also bolds any math such as PERSIST$+\bm{w}_0$).
        def bold(lab):
            if plt.rcParams["text.usetex"]:
                return r"\textbf{\boldmath " + lab + "}"
            return lab

        # Cluster: curves ending at (near) the rightmost frame.
        edge = [(bold(lab), c, xx[-1], yy[-1]) for lab, c, xx, yy in series
                if xx[-1] >= x_hi - 1e-9]
        other = [(bold(lab), c, xx[-1], yy[-1]) for lab, c, xx, yy in series
                 if xx[-1] < x_hi - 1e-9]

        if edge:
            new_ys = spread_labels([e[3] for e in edge], gap)
            for (lab, c, xe, _), ye in zip(edge, new_ys):
                t = ax.text(xe + pad, ye, lab, color=c, va="center",
                            ha="left", fontsize=9, fontweight="bold",
                            clip_on=False)
                t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
        for lab, c, xe, ye in other:
            t = ax.text(xe + pad, ye, lab, color=c, va="center", ha="left",
                        fontsize=9, fontweight="bold", clip_on=False)
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    fig.tight_layout(pad=0.4)
    base = os.path.splitext(args.output)[0]
    for ext in ("pdf", "png"):
        out = f"{base}.{ext}"
        fig.savefig(out, dpi=args.png_dpi if ext == "png" else None)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
