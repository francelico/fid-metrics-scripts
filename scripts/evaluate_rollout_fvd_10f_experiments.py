#!/usr/bin/env python3
"""Prepare and run the complete 10-frame rollout FVD experiment set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "outputs" / "rollout_fvd_10f_experiments_20260819"
DEFAULT_LONG_SPLIT = REPO / "outputs" / "long_rollout_fvd_full_0_1000_20260819" / "split"
CLIP_LEN = 10
LONG_IDS = tuple(range(20))
FREE_IDS = tuple(range(32))


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    run_id: str
    root: Path
    family: str | None = None
    step: int | None = None


RUNS = (
    Run(
        "t50", "allctx-t50", "87c1c46368a77032",
        Path("/projects/u6ni/output/evals/"
             "eval_ctx1-allctx-t50-trim-load-screen_latest_32free-64long-nocaptions_stream_20260818-2146/"
             "wandb/run-20260818_214757-87c1c46368a77032/files/media/videos/video"),
    ),
    Run(
        "p00", "allctx-p00", "6dfeee85efec55e0",
        Path("/projects/u6ni/output/evals/"
             "eval_ctx1-allctx-p00-trim-load-screen_latest_32free-64long-nocaptions_stream_20260818-2146/"
             "wandb/run-20260818_214757-6dfeee85efec55e0/files/media/videos/video"),
    ),
    Run(
        "sampled_df", "sampled-df", "bc0cd6fb2b1dcc78",
        Path("/projects/u6ni/output/evals/"
             "eval_ctx1-sampled-df-trim-load-screen_latest_32free-64long-nocaptions_stream_20260818-2146/"
             "wandb/run-20260818_214756-bc0cd6fb2b1dcc78/files/media/videos/video"),
    ),
    Run(
        "p01", "allctx-p01", "660de8990982f6c1",
        Path("/projects/u6ni/output/evals/"
             "eval_ctx1-allctx-p01-trim-load-screen_latest_32free-64long-nocaptions_stream_20260818-2146/"
             "wandb/run-20260818_214756-660de8990982f6c1/files/media/videos/video"),
    ),
)


@dataclass(frozen=True)
class Evaluation:
    experiment: str
    video_kind: str
    window: str
    requested_start: int
    requested_end: int
    videos: int

    @property
    def clip_start(self) -> int:
        return self.requested_start + (
            (self.requested_end - self.requested_start + 1) % CLIP_LEN
        )

    @property
    def clip_end(self) -> int:
        return self.requested_end + 1

    @property
    def clips_per_video(self) -> int:
        return (self.clip_end - self.clip_start) // CLIP_LEN


EVALUATIONS = (
    Evaluation("lr-exp1", "long_rollout", "0-1000", 0, 1000, 20),
    Evaluation("lr-exp2", "long_rollout", "0-249", 0, 249, 20),
    Evaluation("lr-exp2", "long_rollout", "250-499", 250, 499, 20),
    Evaluation("lr-exp2", "long_rollout", "500-749", 500, 749, 20),
    Evaluation("lr-exp2", "long_rollout", "750-999", 750, 999, 20),
    Evaluation("fr-exp1", "free_running", "0-76", 0, 76, 32),
)


def configure(campaign_root: Path | None, videos: int, long_only: bool) -> None:
    global RUNS, LONG_IDS, EVALUATIONS
    if videos <= 0:
        raise ValueError("--videos must be positive")
    if campaign_root is not None:
        RUNS = discover_campaign_runs(campaign_root)
    LONG_IDS = tuple(range(videos))
    evaluations = [
        Evaluation("lr-exp1", "long_rollout", "0-1000", 0, 1000, videos),
        Evaluation("lr-exp2", "long_rollout", "0-249", 0, 249, videos),
        Evaluation("lr-exp2", "long_rollout", "250-499", 250, 499, videos),
        Evaluation("lr-exp2", "long_rollout", "500-749", 500, 749, videos),
        Evaluation("lr-exp2", "long_rollout", "750-999", 750, 999, videos),
    ]
    if not long_only:
        evaluations.append(
            Evaluation("fr-exp1", "free_running", "0-76", 0, 76, 32)
        )
    EVALUATIONS = tuple(evaluations)


def video_id(path: Path) -> int:
    match = re.match(r"s(\d+)(?:_|\.mp4$)", path.name)
    if not match:
        raise ValueError(f"Cannot parse sample ID from {path.name}")
    return int(match.group(1))


def discover_campaign_runs(campaign_root: Path) -> tuple[Run, ...]:
    """Discover <family>_s<step> runs from a completed long-rollout campaign."""
    family_order = {"t50": 0, "p00": 1, "sampled_df": 2, "p01": 3}
    family_label = {
        "t50": "allctx-t50",
        "p00": "allctx-p00",
        "sampled_df": "sampled-df",
        "p01": "allctx-p01",
    }
    runs = []
    for run_dir in (campaign_root / "runs").iterdir():
        if not run_dir.is_dir():
            continue
        match = re.fullmatch(r"(.+)_s(\d+)", run_dir.name)
        if not match:
            continue
        family, step_text = match.groups()
        step = int(step_text)
        video_root = run_dir / "eval_outputs" / f"step_{step}"
        if not (video_root / "long_rollout").is_dir():
            raise RuntimeError(f"Missing long_rollout directory at {video_root}")
        wandb_dirs = sorted((run_dir / "wandb").glob("run-*-*"))
        run_id = wandb_dirs[-1].name.rsplit("-", 1)[-1] if wandb_dirs else run_dir.name
        runs.append(Run(
            run_dir.name,
            f"{family_label.get(family, family)}-step{step}",
            run_id,
            video_root,
            family,
            step,
        ))
    if not runs:
        raise RuntimeError(f"No campaign runs found below {campaign_root / 'runs'}")
    return tuple(sorted(
        runs,
        key=lambda run: (family_order.get(run.family or "", 99), run.step or -1),
    ))


def sources(run: Run, kind: str, selected_ids: tuple[int, ...]) -> list[Path]:
    paths = sorted((run.root / kind).glob("*.mp4"), key=video_id)
    by_id = {video_id(path): path for path in paths}
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"{run.key}/{kind}: missing IDs {missing}")
    return [by_id[index] for index in selected_ids]


def write_source_manifest(output: Path) -> None:
    rows = []
    for run in RUNS:
        kinds = tuple(dict.fromkeys(item.video_kind for item in EVALUATIONS))
        for kind in kinds:
            selected_ids = LONG_IDS if kind == "long_rollout" else FREE_IDS
            all_paths = sorted((run.root / kind).glob("*.mp4"), key=video_id)
            selected = set(selected_ids)
            for path in all_paths:
                sample = video_id(path)
                rows.append({
                    "run_key": run.key,
                    "label": run.label,
                    "run_id": run.run_id,
                    "video_kind": kind,
                    "video_id": f"s{sample:06d}",
                    "selected": sample in selected,
                    "source": str(path),
                })
    with (output / "source_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "source_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")


def stage_inputs(output: Path, run: Run, kind: str, selected_ids: tuple[int, ...]) -> Path:
    stage = output / "selected_inputs" / kind / run.key
    stage.mkdir(parents=True, exist_ok=True)
    selected = sources(run, kind, selected_ids)
    expected_names = {path.name for path in selected}
    for old in stage.glob("*.mp4"):
        if old.name not in expected_names:
            old.unlink()
    for source in selected:
        target = stage / source.name
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    return stage


def link_long_splits(output: Path, long_split_root: Path) -> None:
    for run in RUNS:
        for side in ("gt", "generated"):
            source = (long_split_root / run.key / side).resolve()
            paths = sorted(source.glob("*.mp4"), key=video_id)
            ids = [video_id(path) for path in paths]
            if ids != list(LONG_IDS):
                raise RuntimeError(
                    f"Expected long split IDs 0-19 at {source}; found {ids}"
                )
            target = output / "split" / "long_rollout" / run.key / side
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == source:
                continue
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Refusing to replace existing split path {target}")
            target.symlink_to(source, target_is_directory=True)


def stage_free_inputs(output: Path, run: Run) -> Path:
    return stage_inputs(output, run, "free_running", FREE_IDS)


def validate_split_ids(run: Run, kind: str, output: Path, selected_ids: tuple[int, ...]) -> None:
    for side in ("gt", "generated"):
        folder = output / "split" / kind / run.key / side
        paths = sorted(folder.glob("*.mp4"), key=video_id)
        ids = [video_id(path) for path in paths]
        if ids != list(selected_ids):
            raise RuntimeError(
                f"{run.key}/{kind}/{side}: expected IDs "
                f"{selected_ids[0]}-{selected_ids[-1]}; found {len(ids)} files"
            )


def prepare_long_run(output: Path, run: Run, overwrite: bool) -> None:
    stage = stage_inputs(output, run, "long_rollout", LONG_IDS)
    gt = output / "split" / "long_rollout" / run.key / "gt"
    generated = output / "split" / "long_rollout" / run.key / "generated"
    log = output / "logs" / f"prepare_long_rollout_{run.key}.log"
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "split_reencode_trim.py"),
        str(stage), str(gt), str(generated),
        "--target-fps", "20", "--num-frames", "1001", "--vertical-split",
    ]
    if overwrite:
        command.append("--overwrite")
    with log.open("w") as handle:
        subprocess.run(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True
        )
    validate_split_ids(run, "long_rollout", output, LONG_IDS)


def prepare_free_run(output: Path, run: Run, overwrite: bool) -> None:
    stage = stage_free_inputs(output, run)
    gt = output / "split" / "free_running" / run.key / "gt"
    generated = output / "split" / "free_running" / run.key / "generated"
    log = output / "logs" / f"prepare_free_running_{run.key}.log"
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "split_reencode_trim.py"),
        str(stage), str(gt), str(generated),
        "--target-fps", "20", "--num-frames", "77", "--vertical-split",
    ]
    if overwrite:
        command.append("--overwrite")
    with log.open("w") as handle:
        subprocess.run(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True
        )
    for side, folder in (("gt", gt), ("generated", generated)):
        paths = sorted(folder.glob("*.mp4"), key=video_id)
        ids = [video_id(path) for path in paths]
        if ids != list(FREE_IDS):
            raise RuntimeError(f"{run.key}/{side}: expected free-running IDs 0-31; found {ids}")


def prepare(
    output: Path,
    long_split_root: Path,
    overwrite: bool,
    campaign_root: Path | None,
    long_only: bool,
    prepare_workers: int,
) -> None:
    (output / "logs").mkdir(parents=True, exist_ok=True)
    write_source_manifest(output)
    if campaign_root is None:
        link_long_splits(output, long_split_root)
    else:
        with ThreadPoolExecutor(max_workers=min(prepare_workers, len(RUNS))) as pool:
            futures = {
                pool.submit(prepare_long_run, output, run, overwrite): run for run in RUNS
            }
            for future in as_completed(futures):
                run = futures[future]
                future.result()
                print(f"prepared long_rollout {run.key}", flush=True)
    if long_only:
        return
    with ThreadPoolExecutor(max_workers=min(prepare_workers, len(RUNS))) as pool:
        futures = {
            pool.submit(prepare_free_run, output, run, overwrite): run for run in RUNS
        }
        for future in as_completed(futures):
            run = futures[future]
            future.result()
            print(f"prepared free_running {run.key}", flush=True)


def gpu_slots(explicit: str | None) -> list[str]:
    if explicit:
        return [item.strip() for item in explicit.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("", "NoDevFiles"):
        slots = [item.strip() for item in visible.split(",") if item.strip()]
        if slots:
            return slots
    count = int(subprocess.check_output(
        [str(REPO / ".venv" / "bin" / "python"), "-c",
         "import torch; print(torch.cuda.device_count())"], text=True,
    ).strip())
    return [str(index) for index in range(count)]


def inputs_for(output: Path, run: Run, evaluation: Evaluation) -> tuple[Path, Path]:
    root = output / "split" / evaluation.video_kind / run.key
    return root / "gt" / "*.mp4", root / "generated" / "*.mp4"


def result_from_log(log: Path) -> float | None:
    if not log.is_file():
        return None
    match = re.search(r"^FVD:\s*([-+0-9.eE]+)\s*$", log.read_text(), re.MULTILINE)
    return float(match.group(1)) if match else None


def make_row(run: Run, evaluation: Evaluation, fvd: float) -> dict:
    return {
        "experiment": evaluation.experiment,
        "run_key": run.key,
        "label": run.label,
        "run_id": run.run_id,
        "video_kind": evaluation.video_kind,
        "window": evaluation.window,
        "requested_start_inclusive": evaluation.requested_start,
        "requested_end_inclusive": evaluation.requested_end,
        "clip_start_inclusive": evaluation.clip_start,
        "clip_end_exclusive": evaluation.clip_end,
        "dropped_leading_frames": evaluation.clip_start - evaluation.requested_start,
        "clip_len": CLIP_LEN,
        "frames_per_video": evaluation.clip_end - evaluation.clip_start,
        "clips_per_video": evaluation.clips_per_video,
        "videos": evaluation.videos,
        "clips_per_distribution": evaluation.videos * evaluation.clips_per_video,
        "fvd": fvd,
    }


def evaluate_one(
    output: Path,
    run: Run,
    evaluation: Evaluation,
    gpu: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> dict:
    tag = f"{evaluation.experiment}_{run.key}_{evaluation.window}"
    log = output / "logs" / f"fvd_{tag}.log"
    if not force:
        prior = result_from_log(log)
        if prior is not None:
            print(f"gpu={gpu} reuse {tag} FVD={prior:.6f}", flush=True)
            return make_row(run, evaluation, prior)
    gt, generated = inputs_for(output, run, evaluation)
    hydra_dir = output / "hydra" / evaluation.experiment / run.key / evaluation.window
    hydra_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "fid_metrics" / "main.py"),
        f"paths=[{gt},{generated}]",
        f"metrics.0.data.dataset.sequence_length={CLIP_LEN}",
        f"metrics.0.data.dataset.start_frame={evaluation.clip_start}",
        f"metrics.0.data.dataset.end_frame={evaluation.clip_end}",
        f"metrics.0.data.dataset.max_videos={evaluation.videos}",
        f"metrics.0.data.batch_size={batch_size}",
        f"metrics.0.data.num_workers={num_workers}",
        f"hydra.run.dir={hydra_dir}", "hydra.job.chdir=False",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        command, cwd=REPO, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{tag} failed; see {log}")
    fvd = result_from_log(log)
    if fvd is None:
        raise RuntimeError(f"No FVD result in {log}")
    print(f"gpu={gpu} {tag} FVD={fvd:.6f}", flush=True)
    return make_row(run, evaluation, fvd)


def evaluate_run(
    output: Path,
    run: Run,
    gpu: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> list[dict]:
    return [
        evaluate_one(output, run, evaluation, gpu, batch_size, num_workers, force)
        for evaluation in EVALUATIONS
    ]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


def write_results(output: Path, rows: list[dict]) -> None:
    run_order = {run.key: index for index, run in enumerate(RUNS)}
    eval_order = {
        (evaluation.experiment, evaluation.window): index
        for index, evaluation in enumerate(EVALUATIONS)
    }
    rows.sort(key=lambda row: (
        eval_order[(row["experiment"], row["window"])], run_order[row["run_key"]]
    ))
    write_rows(output / "fvd_all_results", rows)
    for experiment in sorted({row["experiment"] for row in rows}):
        subset = [row for row in rows if row["experiment"] == experiment]
        write_rows(output / experiment / "fvd_results", subset)


def compute(
    output: Path,
    requested_gpus: str | None,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> None:
    slots = gpu_slots(requested_gpus)
    if not slots:
        raise RuntimeError("No GPUs visible; run compute inside an interactive allocation")
    print(f"using GPU slots: {slots[:len(RUNS)]}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=min(len(slots), len(RUNS))) as pool:
        futures = {
            pool.submit(
                evaluate_run, output, run, slots[index], batch_size, num_workers, force
            ): (run, slots[index])
            for index, run in enumerate(RUNS[:len(slots)])
        }
        # If fewer than four GPUs are supplied, enqueue remaining runs as workers free up.
        remaining = iter(RUNS[len(slots):])
        while futures:
            for future in as_completed(tuple(futures)):
                run, slot = futures.pop(future)
                rows.extend(future.result())
                try:
                    next_run = next(remaining)
                except StopIteration:
                    pass
                else:
                    futures[pool.submit(
                        evaluate_run, output, next_run, slot,
                        batch_size, num_workers, force,
                    )] = (next_run, slot)
                break
    write_results(output, rows)


def plot_quarters(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with (output / "lr-exp2" / "fvd_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {(row["run_key"], row["window"]): float(row["fvd"]) for row in rows}
    windows = [evaluation.window for evaluation in EVALUATIONS
               if evaluation.experiment == "lr-exp2"]
    families = {}
    for run in RUNS:
        families.setdefault(run.family, []).append(run)
    use_panels = len(RUNS) > 8 and None not in families
    colors = ("#0173B2", "#DE8F05", "#029E73", "#D55E00")
    if use_panels:
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharey=True)
        for ax, (family, family_runs) in zip(axes.flat, families.items()):
            x = np.arange(len(family_runs))
            width = 0.19
            for index, window in enumerate(windows):
                ys = [values[(run.key, window)] for run in family_runs]
                offset = (index - (len(windows) - 1) / 2) * width
                ax.bar(x + offset, ys, width, label=window, color=colors[index])
            ax.set_title(family_runs[0].label.rsplit("-step", 1)[0])
            ax.set_xticks(x, [f"step {run.step}" for run in family_runs])
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        axes[0, 0].set_ylabel("FVD ↓")
        axes[1, 0].set_ylabel("FVD ↓")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Frame window", frameon=False,
                   loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.95))
        fig.suptitle(
            f"Long-rollout FVD by frame subset\n"
            f"{len(LONG_IDS)} videos, 25 × 10-frame clips per video",
            y=1.01,
        )
    else:
        x = np.arange(len(windows))
        width = 0.8 / len(RUNS)
        fig, ax = plt.subplots(figsize=(max(9.4, len(RUNS) * 1.1), 5.3))
        cmap = plt.get_cmap("tab20")
        for index, run in enumerate(RUNS):
            ys = [values[(run.key, window)] for window in windows]
            offset = (index - (len(RUNS) - 1) / 2) * width
            bars = ax.bar(x + offset, ys, width, label=run.label, color=cmap(index))
            if len(RUNS) <= 6:
                ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5, rotation=90)
        ax.set_xlabel("Original rollout frame window (inclusive)")
        ax.set_ylabel("FVD ↓")
        ax.set_title(
            f"Long-rollout FVD by frame subset\n"
            f"{len(LONG_IDS)} videos, 25 × 10-frame clips per video"
        )
        ax.set_xticks(x, windows)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncols=2)
        ax.margins(y=0.15)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / "lr-exp2" / f"fvd_by_window.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def plot_single_window(
    output: Path,
    experiment: str,
    title: str,
    subtitle: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / experiment / "fvd_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_run = {row["run_key"]: float(row["fvd"]) for row in rows}
    values = [by_run[run.key] for run in RUNS]
    labels = [run.label for run in RUNS]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(index) for index in range(len(RUNS))]

    fig, ax = plt.subplots(figsize=(max(8.2, len(RUNS) * 0.85), 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    if len(RUNS) > 6:
        ax.tick_params(axis="x", rotation=55)
    ax.set_ylabel("FVD ↓")
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.14)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / experiment / f"fvd_by_run.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def plot_all(output: Path) -> None:
    experiments = {item.experiment for item in EVALUATIONS}
    if "lr-exp1" in experiments:
        plot_single_window(
            output,
            "lr-exp1",
            "Full long-rollout FVD",
            f"{len(LONG_IDS)} videos, 100 × 10-frame clips per video",
        )
    if "lr-exp2" in experiments:
        plot_quarters(output)
    if "fr-exp1" in experiments:
        plot_single_window(
            output,
            "fr-exp1",
            "Full free-running FVD",
            "32 videos, 7 × 10-frame clips per video",
        )


def validate_plan(output: Path) -> None:
    import cv2

    for run in RUNS:
        kinds = tuple(dict.fromkeys(item.video_kind for item in EVALUATIONS))
        for kind in kinds:
            expected_ids = list(LONG_IDS if kind == "long_rollout" else FREE_IDS)
            expected_frames = 1001 if kind == "long_rollout" else 77
            evaluation = next(item for item in EVALUATIONS if item.video_kind == kind)
            gt, generated = inputs_for(output, run, evaluation)
            gt_paths = sorted(Path(gt.parent).glob(gt.name), key=video_id)
            generated_paths = sorted(Path(generated.parent).glob(generated.name), key=video_id)
            if [video_id(path) for path in gt_paths] != expected_ids:
                raise RuntimeError(f"GT inputs invalid for {run.key}/{kind}")
            if [video_id(path) for path in generated_paths] != expected_ids:
                raise RuntimeError(f"generated inputs invalid for {run.key}/{kind}")
            for path in (*gt_paths, *generated_paths):
                cap = cv2.VideoCapture(str(path))
                properties = (
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    float(cap.get(cv2.CAP_PROP_FPS)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                cap.release()
                expected = (expected_frames, 20.0, 224, 128)
                if properties != expected:
                    raise RuntimeError(f"Unexpected video properties for {path}: {properties}, expected {expected}")
    plan = {
        "clip_len": CLIP_LEN,
        "runs": [{**asdict(run), "root": str(run.root)} for run in RUNS],
        "evaluations": [
            {
                **asdict(evaluation),
                "clip_start": evaluation.clip_start,
                "clip_end_exclusive": evaluation.clip_end,
                "dropped_leading_frames": evaluation.clip_start - evaluation.requested_start,
                "clips_per_video": evaluation.clips_per_video,
                "clips_per_distribution": evaluation.videos * evaluation.clips_per_video,
            }
            for evaluation in EVALUATIONS
        ],
    }
    (output / "analysis_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--long-split-root", type=Path, default=DEFAULT_LONG_SPLIT)
    parser.add_argument(
        "--campaign-root", type=Path,
        help="discover <family>_s<step> runs and split their local long_rollout videos",
    )
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument(
        "--long-only", action="store_true",
        help="run only full and quarter-window long-rollout experiments",
    )
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--phase", choices=("all", "prepare", "compute", "plot"), default="all")
    parser.add_argument("--gpus", help="comma-separated CUDA device IDs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-fvd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve() if args.campaign_root else None
    if campaign_root is not None and not args.long_only:
        raise SystemExit("--campaign-root currently requires --long-only")
    configure(campaign_root, args.videos, args.long_only)
    output = args.output.resolve()
    if args.phase in ("all", "prepare"):
        output.mkdir(parents=True, exist_ok=True)
        prepare(
            output,
            args.long_split_root.resolve(),
            args.force_prepare,
            campaign_root,
            args.long_only,
            args.prepare_workers,
        )
        validate_plan(output)
    if args.phase in ("all", "compute"):
        validate_plan(output)
        compute(output, args.gpus, args.batch_size, args.num_workers, args.force_fvd)
    if args.phase in ("all", "plot"):
        plot_all(output)


if __name__ == "__main__":
    main()
