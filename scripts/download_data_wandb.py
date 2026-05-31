#!/usr/bin/env python3
"""
Download specific mp4s from one or more Weights & Biases runs and rename them
to save_dir/level_XXX.mp4.

Supported inputs:
  1. A single run path via --run
  2. A CSV file with a "Name" column via --file-csv
  3. A partial run match via --run-pattern

By default, files are grouped by level_### and renamed to save_dir/level_XXX.mp4.
Alternatively, when a single --run is provided, --video_filenames selects files by
filename substring (OR match) and downloads ALL of them under their original names.

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

Pattern match:
  python wandb_download_levels.py \
      --run-pattern entity/project/persist260k-xl-vox \
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
        def key(f):
            return getattr(f, "updatedAt", None) or getattr(f, "updated_at", None) or f.name

        return sorted(files, key=key)[-1]
    raise ValueError(f"Unknown keep policy: {keep}")


def remove_empty_media_dir(save_dir: Path):
    media_dir = save_dir / "media"
    if not media_dir.exists():
        return

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


def parse_run_pattern(run_pattern: str):
    """
    Parse a run pattern of the form:
      - entity/project/pattern
      - entity/project/runs/pattern

    Returns:
      project_path: entity/project
      pattern: partial string to match
    """
    run_pattern = run_pattern.strip().strip("/")
    parts = run_pattern.split("/")

    if len(parts) == 4 and parts[2] == "runs":
        project_path = "/".join(parts[:2])
        pattern = parts[3]
        return project_path, pattern

    if len(parts) == 3:
        project_path = "/".join(parts[:2])
        pattern = parts[2]
        return project_path, pattern

    raise ValueError(
        f"Unrecognized --run-pattern format: {run_pattern!r}. "
        "Expected 'entity/project/pattern' or 'entity/project/runs/pattern'."
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


def load_runs_from_pattern(api: wandb.Api, run_pattern: str) -> list[str]:
    """
    Find all runs in a project whose id, name, display_name, or path contains
    the provided partial pattern.

    --run-pattern must be:
      - entity/project/pattern
      - entity/project/runs/pattern
    """
    project_path, pattern = parse_run_pattern(run_pattern)
    pattern_lc = pattern.lower()

    matched = []
    seen = set()

    for run in api.runs(project_path):
        run_id = str(getattr(run, "id", "") or "")
        run_name = str(getattr(run, "name", "") or "")
        run_display_name = str(getattr(run, "display_name", "") or "")
        run_path = "/".join(getattr(run, "path", []) or [])
        normalized_path = normalize_run_path(run_path) if run_path else ""

        haystacks = [
            run_id,
            run_name,
            run_display_name,
            run_path,
            normalized_path,
        ]

        if any(pattern_lc in h.lower() for h in haystacks if h):
            if normalized_path and normalized_path not in seen:
                seen.add(normalized_path)
                matched.append(normalized_path)

    if not matched:
        raise ValueError(
            f"No runs matched pattern {pattern!r} in project {project_path!r}."
        )

    return matched


def collect_files_by_filename(run, prefix: str, substrings: list[str]) -> list:
    """
    Collect ALL mp4 files from a run (under prefix) whose basename contains ANY of
    the provided substrings (case-insensitive OR match).
    """
    prefix_with_slash = prefix.rstrip("/") + "/"
    subs_lc = [s.lower() for s in substrings]

    matches = []
    for f in run.files():
        name = f.name

        if not name.startswith(prefix_with_slash):
            continue
        if not name.lower().endswith(".mp4"):
            continue

        basename_lc = Path(name).name.lower()
        if any(s in basename_lc for s in subs_lc):
            matches.append(f)

    return matches


def collect_matching_files(run, prefix: str) -> dict[str, list]:
    """
    Collect matching mp4 files from a run, grouped by level string.
    """
    matches_by_level = {}

    prefix_with_slash = prefix.rstrip("/") + "/"
    for f in run.files():
        name = f.name

        if not name.startswith(prefix_with_slash):
            continue
        if not name.lower().endswith(".mp4"):
            continue

        m = LEVEL_RE.search(Path(name).name)
        if not m:
            continue

        level = m.group(1)
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
    src.add_argument(
        "--run-pattern",
        help=(
            "Partial run match of the form entity/project/pattern "
            "(or entity/project/runs/pattern). "
            "All matching runs in that project will be used."
        ),
    )

    ap.add_argument(
        "--video_filenames",
        help=(
            "Comma-separated list of filename substrings. Only valid with a single --run. "
            "Downloads ALL mp4s whose filename contains ANY of the substrings (OR match), "
            "e.g. '--video_filenames vid_1,vid_2' downloads every video with 'vid_1' or "
            "'vid_2' in its name, keeping their original filenames."
        ),
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

    video_filenames = None
    if args.video_filenames:
        if not args.run:
            ap.error("--video_filenames can only be used together with a single --run")
        video_filenames = [s.strip() for s in args.video_filenames.split(",") if s.strip()]
        if not video_filenames:
            ap.error("--video_filenames did not contain any non-empty entries")

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    if args.run:
        run_paths = [normalize_run_path(args.run)]
    elif args.file_csv:
        run_paths = load_runs_from_csv(args.file_csv)
    else:
        run_paths = load_runs_from_pattern(api, args.run_pattern)

    print(f"[info] Using {len(run_paths)} run(s)")

    total_downloaded = 0
    total_skipped_existing = 0
    total_runs_with_matches = 0

    for run_path in run_paths:
        print(f"[run] {run_path}")
        run = api.run(run_path)

        if video_filenames is not None:
            files = collect_files_by_filename(run, args.prefix, video_filenames)

            if not files:
                print(
                    f"[warn] No mp4s matching {video_filenames} under prefix "
                    f"'{args.prefix}' in run {run_path}"
                )
                continue

            total_runs_with_matches += 1

            for f in sorted(files, key=lambda x: x.name):
                dst = save_dir / Path(f.name).name

                if dst.exists():
                    print(f"[skip] exists: {dst} (from run {run_path})")
                    total_skipped_existing += 1
                    continue

                print(f"[download] {run_path}: {f.name} -> {dst}")
                if args.dry_run:
                    total_downloaded += 1
                    continue

                downloaded_path = Path(
                    f.download(root=str(save_dir), replace=False).name
                )
                downloaded_path.replace(dst)
                total_downloaded += 1

            continue

        matches_by_level = collect_matching_files(run, args.prefix)

        if not matches_by_level:
            print(f"[warn] No matching mp4s found under prefix '{args.prefix}' in run {run_path}")
            continue

        total_runs_with_matches += 1

        for level in sorted(matches_by_level.keys()):
            chosen = choose_one(matches_by_level[level], args.keep)
            dst = save_dir / f"level_{level}.mp4"

            if dst.exists():
                print(f"[skip] exists: {dst} (from run {run_path})")
                total_skipped_existing += 1
                continue

            print(f"[download] {run_path}: {chosen.name} -> {dst}")
            if args.dry_run:
                total_downloaded += 1
                continue

            downloaded_path = Path(
                chosen.download(root=str(save_dir), replace=False).name
            )

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