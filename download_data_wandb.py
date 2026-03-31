#!/usr/bin/env python3
"""
Download specific mp4s from one or more Weights & Biases runs and rename them
to save_dir/level_XXX.mp4.

Supported inputs:
  1. A single run path via --run
  2. A CSV file with a "Name" column via --file-csv

Expected file layout inside each run:
  media/videos/pixel_videos/val_level_001_0_def7f5e1e9380dd8dfb3.mp4

Examples:

Single run:
  python wandb_download_levels.py \
      --run entity/project/abcd1234 \
      --save-dir /tmp/videos \
      --prefix media/videos/pixel_videos \
      --keep latest

CSV of runs:
  python wandb_download_levels.py \
      --file-csv runs.csv \
      --save-dir /tmp/videos \
      --prefix media/videos/pixel_videos \
      --keep latest
"""

import argparse
import csv
import re
from pathlib import Path

import wandb


LEVEL_RE = re.compile(r"(?:^|_)level_(\d{3})(?:_|\.mp4$)", re.IGNORECASE)


def choose_one(files, keep: str):
    """Choose one file from a list of wandb File objects."""
    if keep == "first":
        return files[0]
    if keep == "latest":
        # Heuristic: W&B file objects sometimes expose updatedAt / updated_at.
        def key(f):
            return getattr(f, "updatedAt", None) or getattr(f, "updated_at", None) or f.name

        return sorted(files, key=key)[-1]
    raise ValueError(f"Unknown keep policy: {keep}")


def remove_empty_media_dir(save_dir: Path):
    media_dir = save_dir / "media"
    if not media_dir.exists():
        return

    # remove empty directories bottom-up
    for p in sorted(media_dir.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass

    try:
        media_dir.rmdir()
    except OSError:
        pass


def normalize_run_path(run_path: str) -> str:
    """
    Normalize possible W&B run paths.

    Accepts either:
      - entity/project/run_id
      - entity/project/runs/run_id

    Returns:
      - entity/project/run_id
    """
    run_path = run_path.strip().strip("/")

    parts = run_path.split("/")
    if len(parts) == 4 and parts[2] == "runs":
        return "/".join([parts[0], parts[1], parts[3]])

    if len(parts) == 3:
        return run_path

    raise ValueError(
        f"Unrecognized run path format: {run_path!r}. "
        "Expected 'entity/project/run_id' or 'entity/project/runs/run_id'."
    )


def load_runs_from_csv(file_csv: Path) -> list[str]:
    """
    Load run paths from a CSV file with a required 'Name' column.
    """
    runs = []
    seen = set()

    with file_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "Name" not in (reader.fieldnames or []):
            raise ValueError(f"{file_csv} does not contain a 'Name' column.")

        for row in reader:
            raw_name = (row.get("Name") or "").strip()
            if not raw_name:
                continue

            run_path = normalize_run_path(raw_name)
            if run_path not in seen:
                seen.add(run_path)
                runs.append(run_path)

    if not runs:
        raise ValueError(f"No valid run paths found in {file_csv}.")

    return runs


def collect_matching_files(run, prefix: str) -> dict[str, list]:
    """
    Collect matching mp4 files from a run, grouped by level string.
    """
    matches_by_level = {}

    prefix_with_slash = prefix.rstrip("/") + "/"
    for f in run.files():
        name = f.name  # path inside run

        if not name.startswith(prefix_with_slash):
            continue
        if not name.lower().endswith(".mp4"):
            continue

        m = LEVEL_RE.search(Path(name).name)
        if not m:
            continue

        level = m.group(1)  # e.g. "001"
        matches_by_level.setdefault(level, []).append(f)

    return matches_by_level


def main():
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run",
        help="Single run path: entity/project/run_id (or entity/project/runs/run_id)",
    )
    src.add_argument(
        "--file-csv",
        type=Path,
        help="CSV file containing a 'Name' column with run paths",
    )

    ap.add_argument("--save-dir", required=True, type=Path)
    ap.add_argument(
        "--prefix",
        default="media/videos/pixel_videos",
        help="Path prefix inside each run to search under",
    )
    ap.add_argument(
        "--split",
        default="val_",
        help="Unused except for readability; regex still finds level_###.",
    )
    ap.add_argument(
        "--keep",
        choices=["first", "latest"],
        default="latest",
        help="If multiple files match the same level_### within a run, which one to keep",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        run_paths = [normalize_run_path(args.run)]
    else:
        run_paths = load_runs_from_csv(args.file_csv)

    api = wandb.Api()

    total_downloaded = 0
    total_skipped_existing = 0
    total_runs_with_matches = 0

    for run_path in run_paths:
        print(f"[run] {run_path}")
        run = api.run(run_path)

        matches_by_level = collect_matching_files(run, args.prefix)

        if not matches_by_level:
            print(f"[warn] No matching mp4s found under prefix '{args.prefix}' in run {run_path}")
            continue

        total_runs_with_matches += 1

        for level in sorted(matches_by_level.keys()):
            chosen = choose_one(matches_by_level[level], args.keep)
            dst = save_dir / f"level_{level}.mp4"

            # Duplicate handling across all runs:
            # if another run already produced this destination, skip.
            if dst.exists():
                print(f"[skip] exists: {dst} (from run {run_path})")
                total_skipped_existing += 1
                continue

            print(f"[download] {run_path}: {chosen.name} -> {dst}")
            if args.dry_run:
                total_downloaded += 1
                continue

            # Download into save_dir, preserving run-internal subfolders
            downloaded_path = Path(
                chosen.download(root=str(save_dir), replace=False).name
            )

            # Move/rename to desired destination
            downloaded_path.replace(dst)
            total_downloaded += 1

    remove_empty_media_dir(save_dir)

    print(
        f"Done. Downloaded {total_downloaded} file(s) into {save_dir}. "
        f"Skipped {total_skipped_existing} duplicate/existing file(s). "
        f"Runs with matches: {total_runs_with_matches}/{len(run_paths)}"
    )


if __name__ == "__main__":
    main()