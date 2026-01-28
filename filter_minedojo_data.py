#!/usr/bin/env python3
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import imageio.v3 as iio


def read_selected_filenames(sel_path: Path) -> list[str]:
    """
    Read selected_instances.txt.
    - Ignores blank lines and lines starting with '#'
    - Accepts entries with or without .mp4
    Returns a de-duplicated list preserving order.
    """
    raw = []
    for line in sel_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if not token.lower().endswith(".mp4"):
            token = token + ".mp4"
        raw.append(token)

    seen = set()
    out = []
    for x in raw:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def tail_frames(video_path: Path, n_frames: int) -> np.ndarray:
    """
    Load ONLY the last n_frames of a video using a streaming iterator.
    Keeps a deque(maxlen=n_frames) of frames, then stacks at the end.

    Returns: (T, H, W, C) uint8 array (or whatever dtype imageio provides).
    Raises if video has 0 frames.
    """
    buf = deque(maxlen=n_frames)
    # iio.imiter yields frames one by one (streaming)
    for frame in iio.imiter(video_path):
        buf.append(frame)

    if not buf:
        raise RuntimeError(f"No frames found in video: {video_path}")

    # Stack into a single array for iio.imwrite
    return np.stack(list(buf), axis=0)


def process(root_dir: Path, save_dir: Path, *, last_n_frames: int, dry_run: bool) -> int:
    """
    Returns number of videos written.
    Expects selected_instances.txt inside each immediate subdir of root_dir.
    """
    root_dir = root_dir.expanduser().resolve()
    save_dir = save_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    written = 0
    root_tag = root_dir.name  # used in output folder name

    # Only iterate immediate subdirectories: root_dir/subdir1
    for subdir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        sel_path = subdir / "selected_instances.txt"
        if not sel_path.is_file():
            continue

        selected = read_selected_filenames(sel_path)
        out_subdir = save_dir / f"{root_tag}_{subdir.name}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        for fname in selected:
            src = subdir / fname
            if not src.is_file():
                print(f"[miss] {src} does not exist")
                continue

            dst = out_subdir / fname
            if dst.exists():
                print(f"[skip] already exists: {dst}")
                continue

            print(f"[trim+write] {src} -> {dst} (last {last_n_frames} frames)")
            if dry_run:
                written += 1
                continue

            frames = tail_frames(src, last_n_frames)

            # Write with your requested args
            iio.imwrite(
                dst,
                frames,
                fps=24,
                codec="h264",  # or "libx264" depending on your ffmpeg build
                quality=10,    # 0 (worst) .. 10 (best)
                macro_block_size=1,
            )
            written += 1

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Extract selected mp4s from root_dir/subdir/ and save last-N frames to save_dir/root_subdir/."
    )
    ap.add_argument("root_dir", type=Path)
    ap.add_argument("save_dir", type=Path)
    ap.add_argument("--last-n-frames", type=int, default=200, help="Number of trailing frames to keep (default: 200)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    n = process(args.root_dir, args.save_dir, last_n_frames=args.last_n_frames, dry_run=args.dry_run)
    print(f"Done. Wrote {n} trimmed video(s) into {Path(args.save_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
