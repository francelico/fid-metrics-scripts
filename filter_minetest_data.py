#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path


LEVEL_RE = re.compile(r"^level[_-]?(?P<id>\d+)$", re.IGNORECASE)


def read_selected_instances(path: Path) -> list[str]:
    """
    Returns a list of normalized IDs as strings (keeps leading zeros if present in file).
    Ignores blank lines and comments starting with '#'.
    """
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # keep the token as-is (e.g., "001"), but validate it's numeric
        token = line.split()[0]
        if not token.isdigit():
            raise ValueError(f"Non-numeric instance id '{token}' in {path}")
        ids.append(token)
    # de-dup while preserving order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def find_level_dir(subdir: Path, instance_id: str) -> Path | None:
    """
    Tries to locate a directory in `subdir` corresponding to this instance id,
    accepting variations like level_001, level001, level-001, etc.
    """
    # First try canonical name (common case)
    canonical = subdir / f"level_{instance_id}"
    if canonical.is_dir():
        return canonical

    # Otherwise scan children for a matching pattern
    for child in subdir.iterdir():
        if not child.is_dir():
            continue
        m = LEVEL_RE.match(child.name)
        if not m:
            continue
        found_id = m.group("id")
        # Compare numerically to be resilient to missing leading zeros
        if int(found_id) == int(instance_id):
            return child

    return None


def process_root(root_dir: Path, save_dir: Path, *, dry_run: bool, move: bool) -> int:
    """
    Returns number of videos copied/moved.
    Supports both:
      - root/subdir/selected_instances.txt
      - root/dir/subdir/selected_instances.txt
    """
    count = 0

    for sel_path in root_dir.rglob("selected_instances.txt"):
        subdir = sel_path.parent
        rel = subdir.relative_to(root_dir)
        rel_parts = rel.parts

        if len(rel_parts) >= 2:
            # root/dir/subdir/...
            dir_name, subdir_name = rel_parts[0], rel_parts[1]
            out_subdir = save_dir / f"{dir_name}-{subdir_name}"
        elif len(rel_parts) == 1:
            # root/subdir/...
            (subdir_name,) = rel_parts
            out_subdir = save_dir / subdir_name
        else:
            print(f"[skip] {sel_path} (unexpected location under root)")
            continue

        selected = read_selected_instances(sel_path)

        for instance_id in selected:
            level_dir = find_level_dir(subdir, instance_id)
            if level_dir is None:
                print(f"[miss] {subdir}: no level dir for instance {instance_id}")
                continue

            src = level_dir / "rgb.mp4"
            if not src.is_file():
                print(f"[miss] {src} does not exist")
                continue

            out_subdir.mkdir(parents=True, exist_ok=True)
            dst = out_subdir / f"{instance_id}.mp4"

            if dst.exists():
                print(f"[skip] already exists: {dst}")
                continue

            action = "MOVE" if move else "COPY"
            print(f"[{action}] {src} -> {dst}")
            if not dry_run:
                if move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
            count += 1

    return count


def main():
    p = argparse.ArgumentParser(
        description="Extract selected level_X/rgb.mp4 files into save_dir/dir-subdir/X.mp4"
    )
    p.add_argument("root_dir", type=Path, help="Root directory containing dir/subdir/... structure")
    p.add_argument("save_dir", type=Path, help="Output directory")
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without copying")
    p.add_argument("--move", action="store_true", help="Move instead of copy")
    args = p.parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    n = process_root(root_dir, save_dir, dry_run=args.dry_run, move=args.move)
    print(f"Done. Extracted {n} video(s) into {save_dir}")


if __name__ == "__main__":
    main()
