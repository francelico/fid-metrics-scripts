#!/usr/bin/env python3
"""
Copy dataset RGB videos that correspond to selected level mp4s.

Inputs:
  --selected-dir   directory containing files like level_021.mp4
  --dataset-dir    dataset root containing subdirs like level_021/rgb.mp4
  --output-dir     destination directory

For each level_X.mp4 found in selected-dir, this script copies:
  dataset-dir/level_X/rgb.mp4
to:
  output-dir/level_X.mp4
"""

import argparse
import re
import shutil
from pathlib import Path


LEVEL_RE = re.compile(r"^(level_\d{3})\.mp4$", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_dir", required=True, type=Path)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    selected_dir = args.selected_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not selected_dir.is_dir():
        raise NotADirectoryError(f"--selected_dir is not a directory: {selected_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"--dataset_dir is not a directory: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_files = sorted(p for p in selected_dir.iterdir() if p.is_file())

    n_copied = 0
    n_missing = 0

    for selected_file in selected_files:
        m = LEVEL_RE.match(selected_file.name)
        if not m:
            continue

        level_name = m.group(1)  # e.g. level_021
        src = dataset_dir / level_name / "rgb.mp4"
        dst = output_dir / f"{level_name}.mp4"

        if not src.exists():
            print(f"[missing] {src}")
            n_missing += 1
            continue

        shutil.copy2(src, dst)
        print(f"[copied] {src} -> {dst}")
        n_copied += 1

    print(f"Done. Copied {n_copied} file(s). Missing {n_missing} file(s).")


if __name__ == "__main__":
    main()