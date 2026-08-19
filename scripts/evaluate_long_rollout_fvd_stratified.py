#!/usr/bin/env python3
"""Run repeatable stratified FVD analyses for configurable clip lengths."""

from __future__ import annotations

import argparse
import csv
import glob
import inspect
import json
import math
import os
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


T95 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.77644510520,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413520,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    25: 2.05953855275,
    30: 2.04227245630,
    40: 2.02107539031,
    60: 2.00029782106,
    120: 1.97993040505,
}

DEFAULT_GT_GLOB = (
    "outputs/long_rollout_fvd_full_0_1000_20260819/"
    "split/sampled_df/gt/*.mp4"
)
DEFAULT_GENERATED_GLOB = (
    "outputs/long_rollout_fvd_full_0_1000_20260819/"
    "split/sampled_df/generated/*.mp4"
)
DEFAULT_OUTPUT_DIR = "outputs/long_rollout_fvd_stratified_sampled_df_20260819"


@dataclass(frozen=True)
class Condition:
    clip_len: int
    clips_per_episode: int
    start_frame: int
    end_frame_exclusive: int
    total_clips: int


def parse_positive_csv(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "fid_metrics" / "main.py").is_file():
            return parent
    raise SystemExit("Could not find repository root containing fid_metrics/main.py")


def absolute_glob(repo: Path, pattern: str) -> str:
    path = Path(pattern)
    return str(path if path.is_absolute() else repo / path)


def t_critical_95(df: int) -> float:
    if df in T95:
        return T95[df]
    smaller = [key for key in T95 if key <= df]
    return T95[max(smaller)] if smaller else 1.95996398454


def adjusted_window(frame_start: int, frame_end_inclusive: int, clip_len: int) -> tuple[int, int, int]:
    span = frame_end_inclusive - frame_start + 1
    if span < clip_len:
        raise ValueError(f"frame interval has {span} frames, fewer than clip length {clip_len}")
    dropped = span % clip_len
    start = frame_start + dropped
    total_clips = (frame_end_inclusive + 1 - start) // clip_len
    return start, frame_end_inclusive + 1, total_clips


def build_conditions(args: argparse.Namespace) -> tuple[list[Condition], list[dict]]:
    conditions = []
    skipped = []
    for clip_len in args.clip_lengths:
        start, end, total = adjusted_window(args.frame_start, args.frame_end_inclusive, clip_len)
        for count in args.clip_counts:
            if count > total:
                item = {"clip_len": clip_len, "clips_per_episode": count,
                        "available_clips_per_episode": total, "reason": "infeasible"}
                if args.strict_counts:
                    raise SystemExit(
                        f"clip_len={clip_len}: requested {count} clips but only {total} exist"
                    )
                skipped.append(item)
                continue
            conditions.append(Condition(clip_len, count, start, end, total))
    return conditions, skipped


def discover_gpus(repo: Path, python: Path, explicit: str | None) -> list[str]:
    if explicit:
        return [item.strip() for item in explicit.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("", "NoDevFiles"):
        slots = [item.strip() for item in visible.split(",") if item.strip()]
        if slots:
            return slots
    count = int(subprocess.check_output(
        [str(python), "-c", "import torch; print(torch.cuda.device_count())"],
        cwd=repo, text=True,
    ).strip())
    return [str(index) for index in range(count)]


def validate_dataset_api(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from fid_metrics.dataset import VideoDataset

    params = inspect.signature(VideoDataset.__init__).parameters
    missing = {"stratified_num_clips", "stratified_seed"} - set(params)
    if missing:
        raise SystemExit(f"VideoDataset lacks required parameters: {sorted(missing)}")


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("clip_len", "clips_per_episode", "seed", "episodes", "clips_per_distribution"):
            row[key] = int(row[key])
        row["fvd"] = float(row["fvd"])
    return rows


def command_for(
    args: argparse.Namespace,
    repo: Path,
    python: Path,
    output: Path,
    condition: Condition,
    seed: int,
) -> tuple[list[str], Path]:
    tag = f"len_{condition.clip_len}_clips_{condition.clips_per_episode}_seed_{seed}"
    hydra_dir = output / "hydra" / tag
    log_path = output / "logs" / f"fvd_{tag}.log"
    command = [
        str(python), str(repo / "fid_metrics" / "main.py"),
        f"paths=[{args.gt_glob_resolved},{args.generated_glob_resolved}]",
        f"metrics.0.data.dataset.sequence_length={condition.clip_len}",
        f"metrics.0.data.dataset.start_frame={condition.start_frame}",
        f"metrics.0.data.dataset.end_frame={condition.end_frame_exclusive}",
        f"metrics.0.data.dataset.max_videos={args.episodes}",
        f"+metrics.0.data.dataset.stratified_num_clips={condition.clips_per_episode}",
        f"+metrics.0.data.dataset.stratified_seed={seed}",
        f"metrics.0.data.batch_size={args.batch_size}",
        f"metrics.0.data.num_workers={args.num_workers}",
        f"hydra.run.dir={hydra_dir}", "hydra.job.chdir=False",
        *args.hydra_override,
    ]
    return command, log_path


def evaluate(
    args: argparse.Namespace,
    repo: Path,
    python: Path,
    output: Path,
    condition: Condition,
    seed: int,
    gpu: str,
) -> dict:
    command, log_path = command_for(args, repo, python, output, condition, seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (output / "hydra").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command, cwd=repo, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"FVD failed for {condition}, seed={seed}; see {log_path}")
    match = re.search(r"^FVD:\s*([-+0-9.eE]+)\s*$", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"No FVD value found in {log_path}")
    fvd = float(match.group(1))
    print(
        f"gpu={gpu} clip_len={condition.clip_len} clips={condition.clips_per_episode} "
        f"seed={seed} FVD={fvd:.6f}", flush=True,
    )
    return {
        "clip_len": condition.clip_len,
        "clips_per_episode": condition.clips_per_episode,
        "seed": seed,
        "label": args.label,
        "episodes": args.episodes,
        "clips_per_distribution": condition.clips_per_episode * args.episodes,
        "start_frame": condition.start_frame,
        "end_frame_inclusive": condition.end_frame_exclusive - 1,
        "sampling": "one_random_clip_per_temporal_stratum_per_episode",
        "fvd": fvd,
    }


def stratified_indices(total: int, count: int, seed: int, episode_idx: int) -> tuple[int, ...]:
    rng = random.Random(seed * 1_000_003 + episode_idx)
    return tuple(
        rng.randint(stratum * total // count, (stratum + 1) * total // count - 1)
        for stratum in range(count)
    )


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(row["clip_len"], row["clips_per_episode"]) for row in rows})
    for clip_len, count in keys:
        group = [row for row in rows if row["clip_len"] == clip_len and row["clips_per_episode"] == count]
        values = [row["fvd"] for row in group]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_ci = t_critical_95(len(values) - 1) * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary.append({
            "clip_len": clip_len,
            "clips_per_episode": count,
            "clips_per_distribution": count * group[0]["episodes"],
            "repeats": len(values),
            "mean_fvd": mean,
            "sample_std_fvd": std,
            "sample_variance_fvd": std * std,
            "min_fvd": min(values), "max_fvd": max(values),
            "range_fvd": max(values) - min(values),
            "coefficient_of_variation_pct": 100.0 * std / mean,
            "mean_95ci_low": mean - half_ci, "mean_95ci_high": mean + half_ci,
        })
    return summary


def write_table(output: Path, stem: str, rows: list[dict]) -> None:
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")


def plot(output: Path, summary: list[dict], label: str, episodes: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = ("#0173B2", "#D55E00", "#029E73", "#CC78BC", "#CA9161")
    markers = ("o", "s", "^", "D", "v")
    for index, clip_len in enumerate(sorted({row["clip_len"] for row in summary})):
        group = sorted(
            (row for row in summary if row["clip_len"] == clip_len),
            key=lambda row: row["clips_per_episode"],
        )
        xs = [row["clips_per_episode"] for row in group]
        means = [row["mean_fvd"] for row in group]
        stds = [row["sample_std_fvd"] for row in group]
        color = colors[index % len(colors)]
        ax.errorbar(xs, means, yerr=stds, color=color, marker=markers[index % len(markers)],
                    markersize=6, linewidth=2.2, capsize=5, label=f"{clip_len}-frame clips")
        for x, mean in zip(xs, means):
            ax.annotate(f"{mean:.1f}", (x, mean), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=color)
    ax.set_xticks(sorted({row["clips_per_episode"] for row in summary}))
    ax.set_xlabel("Stratified clips per episode")
    ax.set_ylabel("FVD ↓")
    repeats = sorted({row["repeats"] for row in summary})
    repeat_text = str(repeats[0]) if len(repeats) == 1 else "/".join(map(str, repeats))
    ax.set_title(
        f"{label} FVD sampling variance by clip length\n"
        f"{repeat_text} repeats, {episodes} episodes"
    )
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.margins(y=0.17)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fvd_stratified_variance.{suffix}",
                    dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    for clip_len in sorted({row["clip_len"] for row in summary}):
        group = sorted(
            (row for row in summary if row["clip_len"] == clip_len),
            key=lambda row: row["clips_per_episode"],
        )
        xs = [row["clips_per_episode"] for row in group]
        means = [row["mean_fvd"] for row in group]
        stds = [row["sample_std_fvd"] for row in group]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.errorbar(
            xs, means, yerr=stds, color="#0173B2", marker="o",
            markersize=6, linewidth=2.2, capsize=5, label="mean ± sample SD",
        )
        for x, mean in zip(xs, means):
            ax.annotate(
                f"{mean:.1f}", (x, mean), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8,
            )
        ax.set_xticks(xs)
        ax.set_xlabel(f"Stratified {clip_len}-frame clips per episode")
        ax.set_ylabel("FVD ↓")
        ax.set_title(
            f"{label} {clip_len}-frame FVD sampling variance\n"
            f"{repeat_text} repeats, {episodes} episodes"
        )
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        ax.margins(y=0.17)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"fvd_stratified_variance_len_{clip_len}.{suffix}",
                dpi=240 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-glob", default=DEFAULT_GT_GLOB)
    parser.add_argument("--generated-glob", default=DEFAULT_GENERATED_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-lengths", type=parse_positive_csv, default=(8, 16))
    parser.add_argument("--clip-counts", type=parse_positive_csv, default=(64, 32, 16, 8, 4))
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end-inclusive", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--label", default="sampled-df")
    parser.add_argument("--gpus", help="comma-separated CUDA device IDs; auto-detect by default")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--python", help="Python executable; defaults to repository .venv")
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end_inclusive < args.frame_start:
        parser.error("invalid frame interval")
    if min(args.episodes, args.repeats, args.batch_size, args.num_workers + 1) <= 0:
        parser.error("episodes, repeats, batch-size, and num-workers must be valid positive values")
    return args


def main() -> None:
    args = parse_args()
    repo = find_repo_root()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = repo / output
    python = Path(args.python) if args.python else repo / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    args.gt_glob_resolved = absolute_glob(repo, args.gt_glob)
    args.generated_glob_resolved = absolute_glob(repo, args.generated_glob)
    gt_paths = sorted(glob.glob(args.gt_glob_resolved))[:args.episodes]
    generated_paths = sorted(glob.glob(args.generated_glob_resolved))[:args.episodes]
    if len(gt_paths) < args.episodes or len(generated_paths) < args.episodes:
        raise SystemExit(
            f"Need {args.episodes} videos per distribution; found GT={len(gt_paths)}, "
            f"generated={len(generated_paths)}"
        )
    validate_dataset_api(repo)
    conditions, skipped = build_conditions(args)
    seeds = tuple(range(args.seed_offset, args.seed_offset + args.repeats))
    existing_path = output / "fvd_stratified_repeats.csv"
    existing = load_rows(existing_path)
    completed_keys = {
        (row["clip_len"], row["clips_per_episode"], row["seed"])
        for row in existing
    }
    tasks = [
        (condition, seed)
        for condition in conditions for seed in seeds
        if args.force or (condition.clip_len, condition.clips_per_episode, seed) not in completed_keys
    ]
    plan = {
        "repository": str(repo), "gt_glob": args.gt_glob_resolved,
        "generated_glob": args.generated_glob_resolved, "output_dir": str(output),
        "episodes": args.episodes, "repeats": args.repeats, "seeds": seeds,
        "frame_start_requested": args.frame_start,
        "frame_end_inclusive_requested": args.frame_end_inclusive,
        "conditions": [asdict(item) for item in conditions], "skipped_conditions": skipped,
        "pending_evaluations": len(tasks),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        if tasks:
            command, _ = command_for(args, repo, python, output, *tasks[0])
            print("example command:\n" + " ".join(command), flush=True)
        return

    if args.plot_only:
        if not existing:
            raise SystemExit(f"No repeats found at {existing_path}")
        summary = summarize(existing)
        output.mkdir(parents=True, exist_ok=True)
        write_table(output, "fvd_stratified_summary", summary)
        plot(output, summary, args.label, args.episodes)
        return

    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    new_rows = []
    worker_errors = []
    rows_lock = threading.Lock()
    task_queue: queue.Queue[tuple[Condition, int]] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    def worker(gpu: str) -> None:
        while True:
            try:
                condition, seed = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                row = evaluate(args, repo, python, output, condition, seed, gpu)
                with rows_lock:
                    new_rows.append(row)
            except Exception as error:
                with rows_lock:
                    worker_errors.append(error)
            finally:
                task_queue.task_done()

    if tasks:
        gpus = discover_gpus(repo, python, args.gpus)
        if not gpus:
            raise SystemExit("No GPUs found")
        threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus[:len(tasks)]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if worker_errors:
            details = "\n".join(f"- {error}" for error in worker_errors)
            raise RuntimeError(f"{len(worker_errors)} FVD evaluation(s) failed:\n{details}")
    rows_by_key = {
        (row["clip_len"], row["clips_per_episode"], row["seed"]): row
        for row in existing
    }
    for row in new_rows:
        rows_by_key[(row["clip_len"], row["clips_per_episode"], row["seed"])] = row
    requested_keys = {
        (condition.clip_len, condition.clips_per_episode, seed)
        for condition in conditions for seed in seeds
    }
    rows = [row for key, row in rows_by_key.items() if key in requested_keys]
    rows.sort(key=lambda row: (row["clip_len"], -row["clips_per_episode"], row["seed"]))
    expected = len(conditions) * len(seeds)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} completed rows, found {len(rows)}; inspect logs")
    summary = summarize(rows)
    write_table(output, "fvd_stratified_repeats", rows)
    write_table(output, "fvd_stratified_summary", summary)
    selections = {
        str(clip_len): {
            str(condition.clips_per_episode): {
                str(seed): {
                    Path(gt_paths[episode_idx]).name: stratified_indices(
                        condition.total_clips, condition.clips_per_episode, seed, episode_idx
                    )
                    for episode_idx in range(args.episodes)
                }
                for seed in seeds
            }
            for condition in conditions if condition.clip_len == clip_len
        }
        for clip_len in sorted({condition.clip_len for condition in conditions})
    }
    (output / "clip_selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    plot(output, summary, args.label, args.episodes)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
