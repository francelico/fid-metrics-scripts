#!/usr/bin/env python3
"""Run lr-exp3: per-frame FID for the validated long-rollout campaign splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import evaluate_rollout_fvd_10f_experiments as fvd


def valid_csv(path: Path, frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == frames
            and [int(row["frame_index"]) for row in rows] == list(range(frames))
            and all(math.isfinite(float(row["fid"])) for row in rows)
        )
    except (KeyError, ValueError):
        return False


def evaluate(output: Path, run: fvd.Run, gpu: str, args) -> None:
    folder = output / "lr-exp3" / run.key
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "fid_per_frame.csv"
    if not args.force and valid_csv(csv_path, 1001):
        print(f"gpu={gpu} reuse lr-exp3 {run.key}", flush=True)
        return
    split = output / "split" / "long_rollout" / run.key
    command = [
        str(fvd.REPO / ".venv/bin/python"), "-u", "fid_metrics/main.py",
        "--config-name", "fid_per_frame",
        f"paths=[{split / 'gt/*.mp4'},{split / 'generated/*.mp4'}]",
        "metrics.0.data.dataset.start_frame=0",
        "metrics.0.data.dataset.end_frame=1001",
        f"metrics.0.data.dataset.max_videos={args.videos}",
        f"metrics.0.data.batch_size={args.batch_size}",
        f"metrics.0.data.num_workers={args.num_workers}",
        f"metrics.0.output_csv={csv_path}",
        f"hydra.run.dir={folder / 'hydra'}", "hydra.job.chdir=False",
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(fvd.REPO)
    log = output / "logs" / f"fid_lr-exp3_{run.key}.log"
    print(f"gpu={gpu} start lr-exp3 {run.key}", flush=True)
    with log.open("w") as handle:
        subprocess.run(command, cwd=fvd.REPO, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    if not valid_csv(csv_path, 1001):
        raise RuntimeError(f"Invalid or incomplete per-frame FID: {csv_path}")
    print(f"gpu={gpu} completed lr-exp3 {run.key}", flush=True)


def plot(output: Path) -> None:
    groups = {}
    for run in fvd.RUNS:
        groups.setdefault(run.family or "all", []).append(run)
    combined = []
    for family, runs in groups.items():
        csvs = [output / "lr-exp3" / run.key / "fid_per_frame.csv" for run in runs]
        if not all(valid_csv(path, 1001) for path in csvs):
            raise RuntimeError(f"Missing or incomplete lr-exp3 CSV for {family}")
        for run, path in zip(runs, csvs):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    combined.append({"run_key": run.key, "run_id": run.run_id,
                                     "label": run.label, **row,
                                     "time_seconds": int(row["frame_index"]) / 20})
        for seconds in (False, True):
            command = [
                str(fvd.REPO / ".venv/bin/python"), "scripts/plot_fid_vs_t.py",
                "--labels", ",".join(run.label for run in runs),
                "--csvs", ",".join(map(str, csvs)),
                "--label-mode", "legend", "--highlight", "-1",
                "--width", "7", "--height", "4", "--png-dpi", "240",
                "--title", f"lr-exp3: {family}, {len(fvd.LONG_IDS)} videos per frame",
                "--output", str(output / "lr-exp3" /
                                f"fid_vs_{'time' if seconds else 'frame'}_{family}"),
            ]
            if seconds:
                command += ["--fps", "20"]
            subprocess.run(command, cwd=fvd.REPO, check=True)
    fvd.write_rows(output / "lr-exp3/fid_all_results", combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=256)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--phase", choices=("all", "compute", "plot"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    fvd.configure(args.campaign_root.resolve(), args.videos, True)
    fvd.validate_plan(output)
    config = {"experiment": "lr-exp3", "frames_inclusive": [0, 1000],
              "fps": 20, "videos_per_frame": args.videos, "dims": 2048,
              "batch_size": args.batch_size, "smoothing": "none"}
    (output / "lr-exp3").mkdir(exist_ok=True)
    (output / "lr-exp3/analysis_config.json").write_text(json.dumps(config, indent=2))
    if args.phase in ("all", "compute"):
        pending = queue.Queue()
        for run in fvd.RUNS:
            pending.put(run)

        def worker(gpu):
            while True:
                try:
                    run = pending.get_nowait()
                except queue.Empty:
                    return
                evaluate(output, run, gpu, args)

        slots = fvd.gpu_slots(args.gpus)
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in slots]
            for future in futures:
                future.result()
    if args.phase in ("all", "plot"):
        plot(output)


if __name__ == "__main__":
    main()
