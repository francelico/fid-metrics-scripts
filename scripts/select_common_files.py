#!/usr/bin/env python3
"""
Given a comma-separated list of directories, find the files that are common to all
directories by filename, then copy those files into a `selected` subdirectory
inside each directory.

Example:
    python select_common_files.py --dirs ab-gtcam,ab-novoxvae

This will create:
    ab-gtcam/selected/
    ab-novoxvae/selected/

and copy only the shared filenames into each `selected` directory.
"""

import argparse
import shutil
from pathlib import Path


def parse_dirs(dirs_arg: str) -> list[Path]:
    dirs = [Path(d.strip()).expanduser().resolve() for d in dirs_arg.split(",") if d.strip()]
    if not dirs:
        raise ValueError("No directories were provided.")
    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
    return dirs


def list_files(directory: Path) -> set[str]:
    """
    Return the set of filenames directly inside `directory`.
    Only regular files are included. Subdirectories are ignored.
    """
    return {p.name for p in directory.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser(
        description="Find common filenames across directories and copy them into selected/ subdirectories."
    )
    parser.add_argument(
        "--dirs",
        required=True,
        help="Comma-separated list of directory paths, e.g. ab-gtcam,ab-novoxvae",
    )
    args = parser.parse_args()

    dirs = parse_dirs(args.dirs)

    # Find intersection of filenames across all directories
    common_files = None
    for d in dirs:
        files = list_files(d)
        if common_files is None:
            common_files = files
        else:
            common_files &= files

    common_files = sorted(common_files or [])

    if not common_files:
        print("No common files found across all directories.")
        return

    print("Common files:")
    for name in common_files:
        print(f"  {name}")

    # Copy common files into selected/ subdirectory for each directory
    for d in dirs:
        selected_dir = d / "selected"
        selected_dir.mkdir(exist_ok=True)

        for name in common_files:
            src = d / name
            dst = selected_dir / name
            shutil.copy2(src, dst)
            print(f"[copied] {src} -> {dst}")

    print(f"Done. Copied {len(common_files)} common file(s) into each selected/ directory.")


if __name__ == "__main__":
    main()