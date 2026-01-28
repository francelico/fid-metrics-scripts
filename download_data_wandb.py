#!/usr/bin/env python3
"""
Download specific mp4s from a Weights & Biases run and rename to save_dir/level_XXX.mp4.

Expected file layout inside the run:
  media/videos/pixel_videos/val_level_001_0_def7f5e1e9380dd8dfb3.mp4

Example:
  python wandb_download_levels.py \
      --run entity/project/abcd1234 \
      --save-dir /tmp/videos \
      --prefix media/videos/pixel_videos \
      --split val_ \
      --keep latest
"""

import argparse
import re
from pathlib import Path

import wandb


LEVEL_RE = re.compile(r"(?:^|_)level_(\d{3})(?:_|\.mp4$)", re.IGNORECASE)


def choose_one(files, keep: str):
    """Choose one file from a list of wandb File objects."""
    if keep == "first":
        return files[0]
    if keep == "latest":
        # Heuristic: W&B file objects have updatedAt sometimes; if not, fallback to name sort.
        # We'll try attribute access safely.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="Run path: entity/project/run_id (e.g., myteam/myproj/abcd1234)",
    )
    ap.add_argument("--save-dir", required=True, type=Path)
    ap.add_argument(
        "--prefix",
        default="media/videos/pixel_videos",
        help="Path prefix inside the run to search under",
    )
    ap.add_argument(
        "--split",
        default="val_",
        help="Optional string prefix before level_### in the filename (e.g. 'val_'). Used only for readability; regex still finds level_###.",
    )
    ap.add_argument(
        "--keep",
        choices=["first", "latest"],
        default="latest",
        help="If multiple files match the same level_###, which one to keep",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(args.run)

    # Collect matching mp4 files
    matches_by_level = {}  # level_str -> list[wandb.apis.public.File]
    for f in run.files():
        name = f.name  # path inside run
        if not name.startswith(args.prefix.rstrip("/") + "/"):
            continue
        if not name.lower().endswith(".mp4"):
            continue

        m = LEVEL_RE.search(Path(name).name)
        if not m:
            continue
        level = m.group(1)  # e.g. "001"
        matches_by_level.setdefault(level, []).append(f)

    if not matches_by_level:
        raise SystemExit(
            f"No matching mp4s found under prefix '{args.prefix}' in run {args.run}."
        )

    # Download + rename
    n = 0
    for level in sorted(matches_by_level.keys()):
        chosen = choose_one(matches_by_level[level], args.keep)
        dst = save_dir / f"level_{level}.mp4"

        if dst.exists():
            print(f"[skip] exists: {dst}")
            continue

        print(f"[download] {chosen.name} -> {dst}")
        if args.dry_run:
            n += 1
            continue

        # Download into save_dir, preserving run-internal subfolders
        downloaded_path = Path(
            chosen.download(root=str(save_dir), replace=False).name
        )  # local path of downloaded file

        # Move/rename to desired destination
        downloaded_path.replace(dst)
        n += 1

    remove_empty_media_dir(save_dir)

    print(f"Done. Downloaded {n} file(s) into {save_dir}")


if __name__ == "__main__":
    main()
