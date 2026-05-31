#!/usr/bin/env python3
"""
For every video in a folder:
  1. Split each frame into two halves (default: a horizontal cut -> top/bottom halves;
     use --vertical-split for a vertical cut -> left/right halves).
  2. Write the first half to FOLDER_A and the second half to FOLDER_B, preserving
     the relative directory structure under each.
  3. Re-encode the new videos to --target-fps.
  4. Trim the new videos to the first --num-frames frames.

Uses imageio.v3 for reading/writing (same conventions as the other scripts here).
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import json
import subprocess
from fractions import Fraction

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _parse_rate(rate_str: str) -> float | None:
    # rate_str like "30000/1001" or "30/1" or "0/0"
    try:
        f = Fraction(rate_str)
        if f.numerator == 0:
            return None
        return float(f)
    except Exception:
        return None


def get_video_fps(path: Path) -> float:
    """Try imageio metadata first; if fps is 0/None, fall back to ffprobe."""
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 0) or 0)
    if fps > 0:
        return fps

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{p.stderr.strip()}")

    info = json.loads(p.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")

    s0 = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = s0.get(key)
        val = _parse_rate(rate) if rate else None
        if val and val > 0:
            return val

    raise RuntimeError(f"Could not determine FPS for {path} (ffprobe returned {s0})")


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def read_first_n_frames(video_path: Path, n: int | None) -> np.ndarray:
    """Stream and keep the first n frames (all frames if n is None). Returns (T, H, W, C)."""
    frames = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if n is not None and i >= n:
            break
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return np.stack(frames, axis=0)


def split_halves(frames: np.ndarray, *, vertical: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Split (T, H, W, C) (or (T, H, W)) frames into two halves.

    vertical=False (default): cut along the horizontal axis -> (top, bottom) halves.
    vertical=True:            cut along the vertical axis -> (left, right) halves.
    """
    if frames.ndim not in (3, 4):
        raise ValueError(f"Unexpected video array shape {frames.shape} (ndim={frames.ndim})")

    if vertical:
        w = frames.shape[2]
        first = frames[:, :, : w // 2, ...]
        second = frames[:, :, w // 2 : 2 * (w // 2), ...]
    else:
        h = frames.shape[1]
        first = frames[:, : h // 2, ...]
        second = frames[:, h // 2 : 2 * (h // 2), ...]

    return first, second


def write_video(dst: Path, frames: np.ndarray, fps: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        dst,
        frames,
        fps=fps,
        codec="h264",
        quality=10,
        macro_block_size=1,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Split each video in a folder into two halves, re-encode to a target FPS, and trim to N frames."
    )
    ap.add_argument("input_dir", type=Path, help="Folder of videos (searched recursively)")
    ap.add_argument("folder_a", type=Path, help="Output folder for the first half (top/left)")
    ap.add_argument("folder_b", type=Path, help="Output folder for the second half (bottom/right)")
    ap.add_argument(
        "--target-fps",
        type=float,
        required=True,
        help="Re-encode the output videos to this frame rate.",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Trim the output videos to the first N frames. If omitted, keep all frames.",
    )
    ap.add_argument(
        "--vertical-split",
        action="store_true",
        help="Cut each frame vertically (left/right halves) instead of horizontally (top/bottom halves).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if both halves exist)",
    )
    args = ap.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    folder_a = args.folder_a.expanduser().resolve()
    folder_b = args.folder_b.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")
    if args.num_frames is not None and args.num_frames <= 0:
        raise SystemExit("--num-frames must be a positive integer")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be positive")

    n_written = 0
    n_skipped = 0
    n_errors = 0

    for src in sorted(input_dir.rglob("*")):
        if not is_video(src):
            continue

        rel = src.relative_to(input_dir).with_suffix(".mp4")
        dst_a = folder_a / rel
        dst_b = folder_b / rel

        if dst_a.exists() and dst_b.exists() and not args.overwrite:
            print(f"[skip] exists: {dst_a}, {dst_b}")
            n_skipped += 1
            continue

        axis = "vertically (left/right)" if args.vertical_split else "horizontally (top/bottom)"
        trim = f"first {args.num_frames} frames" if args.num_frames is not None else "all frames"
        print(f"[split] {src} -> {dst_a} | {dst_b} (split {axis}, fps={args.target_fps}, {trim})")
        if args.dry_run:
            n_written += 1
            continue

        try:
            frames = read_first_n_frames(src, args.num_frames)
            first, second = split_halves(frames, vertical=args.vertical_split)

            print(
                f"[info] {src.name}: {frames.shape[0]} frames, "
                f"half A {first.shape[1:3]}, half B {second.shape[1:3]}"
            )
            write_video(dst_a, first, args.target_fps)
            write_video(dst_b, second, args.target_fps)
            n_written += 1
        except Exception as e:
            n_errors += 1
            print(f"[error] {src}: {e}")

    print(
        f"Done. Wrote {n_written} video pair(s), skipped {n_skipped}, errors {n_errors}. "
        f"Halves at: {folder_a} and {folder_b}"
    )


if __name__ == "__main__":
    main()
