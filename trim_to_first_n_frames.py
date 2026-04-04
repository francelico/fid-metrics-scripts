#!/usr/bin/env python3
"""
Recursively find all videos under root_dir and write trimmed versions to save_dir,
preserving the relative directory structure. Output videos contain ONLY the first
N frames (default: 200).

Optionally, build a grid video (save_dir/grid.mp4) from the trimmed outputs.

Uses imageio.v3 for reading/writing.
"""

import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np

import json
import subprocess
from fractions import Fraction
from typing import Iterable, Iterator, Optional, Sequence

try:
    from PIL import Image
except ImportError as e:
    Image = None  # handled if grid is requested

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
    """
    Try imageio metadata first; if fps is 0/None, fall back to ffprobe.
    """
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 0) or 0)
    if fps > 0:
        return fps

    ffprobe = "ffprobe"
    cmd = [
        ffprobe,
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


def trim_first_n_frames(video_path: Path, n: int, start_from: int = 0) -> np.ndarray:
    """
    Stream frames and keep n frames starting from start_from.
    Returns (T, H, W, C).
    """
    frames = []
    end = start_from + n

    for i, frame in enumerate(iio.imiter(video_path)):
        if i < start_from:
            continue
        if i >= end:
            break
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No frames found in video: {video_path} "
            f"(start_from={start_from}, n={n})"
        )

    return np.stack(frames, axis=0)


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _first_frame_and_size(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Returns (first_frame, (H, W)).
    """
    it = iio.imiter(path)
    first = next(it)  # may raise StopIteration
    h, w = first.shape[:2]
    return first, (h, w)


def _resize_frame(frame: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize (H,W,C) frame to out_hw using PIL. Keeps dtype uint8 if possible.
    """
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for --make-grid (pip install pillow).")
    out_h, out_w = out_hw
    if frame.shape[0] == out_h and frame.shape[1] == out_w:
        return frame
    img = Image.fromarray(frame)
    img = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(img)


def _iter_frames_with_resize(
    path: Path,
    out_hw: tuple[int, int],
) -> Iterator[np.ndarray]:
    """
    Stream frames from video, resized to out_hw.
    """
    for frame in iio.imiter(path):
        yield _resize_frame(frame, out_hw)


def make_video_grid(
    video_paths: Sequence[Path],
    out_path: Path,
    cols: int,
    rows: int,
    shuffle: bool,
    dry_run: bool,
) -> None:
    """
    Make a grid video from up to cols*rows videos and save using iio.imwrite.

    - Uses first video's FPS
    - Resizes all videos to smallest resolution among selected videos
    - Pads shorter videos with black frames
    """

    if cols <= 0 or rows <= 0:
        raise ValueError(f"cols and rows must be positive, got cols={cols}, rows={rows}")

    video_paths = [p for p in video_paths if p.resolve() != out_path.resolve()]
    if not video_paths:
        print("[grid] no videos found")
        return

    vids = sorted(video_paths)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(vids)

    vids = vids[: cols * rows]
    n_tiles = cols * rows

    # Determine tile size
    sizes = []
    for p in vids:
        it = iio.imiter(p)
        try:
            frame = next(it)
        except StopIteration:
            continue
        sizes.append(frame.shape[:2])

    if not sizes:
        print("[grid] no usable videos")
        return

    tile_h = min(h for h, w in sizes)
    tile_w = min(w for h, w in sizes)

    fps = get_video_fps(vids[0])
    out_h = rows * tile_h
    out_w = cols * tile_w

    print(f"[grid] building grid {cols}x{rows}")
    print(f"[grid] tile={tile_w}x{tile_h}, output={out_w}x{out_h}, fps={fps:.2f}")

    if dry_run:
        return

    # Load and resize all videos
    video_frames = []
    max_len = 0

    for p in vids:
        frames = []
        for frame in iio.imiter(p):
            if frame.shape[0] != tile_h or frame.shape[1] != tile_w:
                frame = _resize_frame(frame, (tile_h, tile_w))
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frames.append(frame.astype(np.uint8, copy=False))
        video_frames.append(frames)
        max_len = max(max_len, len(frames))

    # Pad missing tiles
    black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    while len(video_frames) < n_tiles:
        video_frames.append([])

    # Assemble grid frames
    grid_frames = []

    for t in range(max_len):

        tiles = []
        for vid in video_frames:
            if t < len(vid):
                tiles.append(vid[t])
            else:
                tiles.append(black)

        rows_imgs = []
        for r in range(rows):
            row = np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
            rows_imgs.append(row)

        grid = np.concatenate(rows_imgs, axis=0)
        grid_frames.append(grid)

    grid_frames = np.stack(grid_frames, axis=0)

    print(f"[grid] writing {out_path} ({grid_frames.shape[0]} frames)")

    iio.imwrite(
        out_path,
        grid_frames,
        fps=fps,
        codec="h264",
        quality=10,
        macro_block_size=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=Path, help="Input root directory (searched recursively)")
    ap.add_argument("save_dir", type=Path, help="Output root directory (structure replicated)")
    ap.add_argument("--n-frames", type=int, default=200, help="Number of frames to keep (default: 200)")

    ap.add_argument(
        "--make-grid",
        action="store_true",
        help="Save a grid of all trimmed videos in save_dir/grid.mp4",
    )
    ap.add_argument("--grid-cols", type=int, default=4, help="Number of columns in the grid (default: 4)")
    ap.add_argument("--grid-rows", type=int, default=8, help="Number of rows in the grid (default: 8)")
    ap.add_argument(
        "--shuffle-grid",
        action="store_true",
        help="Shuffle videos before making grid (default: keep sorted)",
    )

    ap.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start trimming from this frame index (default: 0)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if exists)",
    )
    args = ap.parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    n_written = 0
    n_skipped = 0
    n_errors = 0

    for src in sorted(root_dir.rglob("*")):
        if not is_video(src):
            continue

        rel = src.relative_to(root_dir)
        dst = (save_dir / rel).with_suffix(".mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            print(f"[skip] exists: {dst}")
            n_skipped += 1
            continue

        print(f"[trim] {src} -> {dst} (first {args.n_frames} frames)")
        if args.dry_run:
            n_written += 1
            continue

        try:
            fps = get_video_fps(src)
            frames = trim_first_n_frames(src, args.n_frames, args.start_from)

            print(f"[info] encoding {dst} with fps={fps:.2f}, frames={frames.shape[0]}, size={frames.shape[1:3]}")
            iio.imwrite(
                dst,
                frames,
                fps=fps,
                codec="h264",
                quality=10,
                macro_block_size=1,
            )
            n_written += 1
        except Exception as e:
            n_errors += 1
            print(f"[error] {src}: {e}")

    print(
        f"Done. Wrote {n_written} video(s), skipped {n_skipped}, errors {n_errors}. "
        f"Output at: {save_dir}"
    )

    if args.make_grid:
        grid_out = save_dir / "grid.mp4"
        # Build grid from *all* mp4s in save_dir (i.e., the trimmed outputs)
        all_trimmed = [p for p in sorted(save_dir.rglob("*.mp4")) if p.name != "grid.mp4"]
        make_video_grid(
            video_paths=all_trimmed,
            out_path=grid_out,
            cols=args.grid_cols,
            rows=args.grid_rows,
            shuffle=args.shuffle_grid,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()