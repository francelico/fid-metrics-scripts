#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize videos from root_dir/.../dir1/dirA/*.mp4 "
                    "to save_dir/dirA/dir1_dirA/*.mp4"
    )
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    n = 0

    for mp4 in root_dir.rglob("*.mp4"):
        # Expect structure: .../dir1/dirA/file.mp4
        try:
            dirA = mp4.parent.name
            dir1 = mp4.parent.parent.name
        except Exception:
            print(f"[skip] unexpected path depth: {mp4}")
            continue

        out_dir = save_dir / dirA / f"{dir1}_{dirA}"
        out_dir.mkdir(parents=True, exist_ok=True)

        dst = out_dir / mp4.name

        print(f"[copy] {mp4} -> {dst}")
        if not args.dry_run:
            shutil.copy2(mp4, dst)

        n += 1

    print(f"Done. Copied {n} video(s) into {save_dir}")


if __name__ == "__main__":
    main()