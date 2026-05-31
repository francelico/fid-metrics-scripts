#!/usr/bin/env python3
import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np


def crop_top_half(frames: np.ndarray) -> np.ndarray:
    """
    frames: (T, H, W, C) or (T, H, W)
    Returns top half along height dimension.
    """
    if frames.ndim not in (3, 4):
        raise ValueError(f"Unexpected video array shape {frames.shape} (ndim={frames.ndim})")
    h = frames.shape[1]
    return frames[:, : h // 2, ...]


def process(root_dir: Path, save_dir: Path, *, dry_run: bool) -> int:
    root_dir = root_dir.expanduser().resolve()
    save_dir = save_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    n_written = 0

    # Find mp4 files exactly under root_dir/subdir/*.mp4 (and deeper under subdir if present)
    for src in sorted(root_dir.rglob("*.mp4")):
        if not src.is_file():
            continue
        rel = src.relative_to(root_dir)  # subdir/.../filename.mp4
        dst = save_dir / Path(str(save_dir.name) + '_' + str(rel))

        dst.parent.mkdir(parents=True, exist_ok=True)

        print(f"[crop] {src} -> {dst}")
        if dry_run:
            n_written += 1
            continue

        # Load full video (no temporal cropping requested)
        frames = iio.imread(src)  # typically (T, H, W, C)
        cropped = crop_top_half(frames)

        # Save with your preferred defaults; adjust if you want different encoding args.
        iio.imwrite(
            dst,
            cropped,
            fps=24,
            codec="h264",        # or "libx264" depending on ffmpeg build
            quality=10,          # 0..10
            macro_block_size=1,
        )
        n_written += 1

    return n_written


def main():
    ap = argparse.ArgumentParser(
        description="Crop each video in root_dir/subdir/filename.mp4 to the top half of frames and save to save_dir/subdir/filename.mp4"
    )
    ap.add_argument("root_dir", type=Path)
    ap.add_argument("save_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    n = process(args.root_dir, args.save_dir, dry_run=args.dry_run)
    print(f"Done. Wrote {n} cropped video(s) into {args.save_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
