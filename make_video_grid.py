#!/usr/bin/env python3
"""
Merge many videos into a grid.

New: --crop-h / --crop-w can be either:
- a single float (applies to all rows), e.g. --crop-w 0.6
- a comma-separated list of floats (one per row), e.g. --crop-w 0.6,0.8,1.0,0.7
  (must match number of rows, else error)

Other assumptions:
- root_dir contains subfolders; each subfolder corresponds to a *row* in the output grid.
- Each subfolder contains the same number of video files (columns).
- All videos have the same resolution (before optional cropping).
- Videos are aligned by sorted filename order within each folder.
- Trimming:
  - If --n-frames is provided: write at most N frames.
  - If --n-frames is omitted: automatically trim to the shortest video across the entire grid
    by stopping when *any* stream ends.

Uses imageio.v3 for reading/writing.
"""

import argparse
import re
import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Return uint8 RGB frame with shape (H, W, 3)."""
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    return frame


def _center_crop(
    frame: np.ndarray,
    *,
    frac_h: float | None,
    frac_w: float | None,
    crop_y: float = 0.5,  # 0=bottom, 1=top
) -> np.ndarray:
    """
    Crop by fractional height/width. Horizontal crop is centered.
    Vertical crop centerpoint is controlled by crop_y in [0,1]:
      - 1.0 => crop centered at top
      - 0.5 => centered
      - 0.0 => centered at bottom
    """
    frame = _ensure_rgb(frame)
    H, W, _ = frame.shape

    crop_h = H if frac_h is None else max(1, int(round(H * frac_h)))
    crop_w = W if frac_w is None else max(1, int(round(W * frac_w)))

    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)

    # X: always centered
    x0 = (W - crop_w) // 2

    # Y: parameterized centerpoint. Convert [0,1] bottom->top into a center position in pixels.
    crop_y = float(crop_y)
    crop_y = 0.0 if crop_y < 0.0 else (1.0 if crop_y > 1.0 else crop_y)
    center_y = (1.0 - crop_y) * (H - 1)  # crop_y=0 -> bottom (large y), crop_y=1 -> top (small y)
    center_y = int(round(center_y))

    y0 = center_y - crop_h // 2
    y0 = max(0, min(y0, H - crop_h))

    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w, :]

def _resize_frame_to_width(frame: np.ndarray, out_w: int) -> np.ndarray:
    """Resize a single RGB frame to out_w, preserving aspect ratio."""
    if out_w <= 0:
        raise ValueError(f"--out-w-px must be > 0, got {out_w}")

    frame = _ensure_rgb(frame)
    h, w, _ = frame.shape
    if w == out_w:
        return frame

    new_w = int(out_w)
    new_h = max(1, int(round(h * (new_w / w))))

    pil = Image.fromarray(frame, mode="RGB")
    downscaling = new_w < w
    if downscaling:
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS, reducing_gap=3.0)
    else:
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    return np.asarray(pil, dtype=np.uint8)


def _make_even_hw_white(frames: np.ndarray) -> np.ndarray:
    """
    Ensure even H/W for H.264 yuv420p by padding with white pixels.
    frames: (T, H, W, 3)
    """
    T, H, W, C = frames.shape
    new_H = H if (H % 2 == 0) else (H + 1)
    new_W = W if (W % 2 == 0) else (W + 1)
    if new_H == H and new_W == W:
        return frames

    out = np.full((T, new_H, new_W, C), 255, dtype=frames.dtype)
    out[:, :H, :W, :] = frames
    return out


def _infer_fps_from_ffmpeg_stderr(video_path: Path) -> float | None:
    """
    Uses the ffmpeg binary imageio bundles (imageio_ffmpeg) to infer fps from stderr.
    Works when immeta() reports fps=0 or duration=0 (common with some webm files).
    """
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    cmd = [ffmpeg, "-hide_banner", "-i", str(video_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    s = p.stderr or ""

    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", s)
    if m:
        try:
            val = float(m.group(1))
            if val > 0:
                return val
        except Exception:
            pass
    return None


def get_video_fps(video_path: Path, fallback: float = 24.0) -> float:
    meta = iio.immeta(video_path)
    fps = float(meta.get("fps", 0) or 0)
    if fps > 0:
        return fps

    fps2 = _infer_fps_from_ffmpeg_stderr(video_path)
    if fps2 is not None and fps2 > 0:
        return fps2

    return float(fallback)


def collect_grid_videos(root_dir: Path) -> list[list[Path]]:
    """
    Returns a list of rows; each row is a sorted list of video paths (columns).
    Rows are ordered by subfolder name.
    """
    subdirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under: {root_dir}")

    rows: list[list[Path]] = []
    for d in subdirs:
        vids = [p for p in sorted(d.iterdir()) if is_video(p)]
        if not vids:
            continue
        rows.append(vids)

    if not rows:
        raise RuntimeError(f"No videos found under: {root_dir}")

    ncols = len(rows[0])
    for r in rows:
        if len(r) != ncols:
            raise RuntimeError(
                "All folders must contain the same number of videos.\n"
                f"Expected {ncols}, got {len(r)} in folder {r[0].parent}"
            )

    return rows

def _parse_row_spec(
    spec: str | None,
    nrows: int,
    flag_name: str,
    *,
    lo: float,
    hi: float,
    default: float | None = None,
) -> list[float | None]:
    """
    Parse a per-row spec from:
      - None -> [default]*nrows
      - "x" -> [x]*nrows
      - "x,y,z" -> per-row list length==nrows
    Validates each value in [lo, hi].
    """
    if spec is None:
        return [default] * nrows

    s = spec.strip()
    if not s:
        return [default] * nrows

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 0:
        return [default] * nrows

    def _to_val(x: str) -> float:
        try:
            v = float(x)
        except ValueError:
            raise ValueError(f"{flag_name}: could not parse float from '{x}'") from None
        if not (lo <= v <= hi):
            raise ValueError(f"{flag_name}: values must be in [{lo}, {hi}], got {v}")
        return v

    if len(parts) == 1:
        v = _to_val(parts[0])
        return [v] * nrows

    if len(parts) != nrows:
        raise ValueError(
            f"{flag_name}: provided {len(parts)} value(s) but grid has {nrows} row(s). "
            f"Provide exactly {nrows} comma-separated values, or a single value."
        )

    return [_to_val(p) for p in parts]

def _resize_to_fit_and_pad_white(frame: np.ndarray, *, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize frame uniformly (same scale for x/y) to fit within (target_w, target_h)
    without changing aspect ratio, then pad with white to exactly (target_h, target_w).
    """
    frame = _ensure_rgb(frame)
    h, w, _ = frame.shape
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size: {target_w}x{target_h}")

    # Uniform scale that fits inside target
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil = Image.fromarray(frame, mode="RGB")
    downscaling = scale < 1.0
    if downscaling:
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS, reducing_gap=3.0)
    else:
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    resized = np.asarray(pil, dtype=np.uint8)

    # Pad to target with white, centered
    out = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w, :] = resized
    return out


def _row_strip_size_for_videos(
    video_hw: tuple[int, int],
    ncols: int,
    crop_h: float | None,
    crop_w: float | None,
) -> tuple[int, int]:
    """
    Compute the expected (H, W) of a row-strip after per-video crop, then horizontal concat.
    video_hw is (H, W) of original frames.
    """
    H, W = video_hw
    ch = 1.0 if crop_h is None else crop_h
    cw = 1.0 if crop_w is None else crop_w
    h = max(1, int(round(H * ch)))
    w = max(1, int(round(W * cw)))
    return (h, w * ncols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=Path, help="Input root directory (searched recursively)")
    ap.add_argument("save_path", type=Path, help="Output video path")
    ap.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Trim to the first N frames. If omitted, trim all videos to the shortest video length.",
    )
    # NOTE: now strings so we can accept comma-separated per-row values.
    ap.add_argument(
        "--crop-h",
        type=str,
        default=None,
        help="Center-crop height fraction. Either a single float (applies to all rows) or comma-separated floats (one per row).",
    )
    ap.add_argument(
        "--crop-y",
        type=str,
        default="0.5",
        help="Vertical crop centerpoint in [0,1] where 0=bottom, 1=top. "
             "Either a single value (all rows) or comma-separated values (one per row). Default: 0.5",
    )
    ap.add_argument(
        "--crop-w",
        type=str,
        default=None,
        help="Center-crop width fraction. Either a single float (applies to all rows) or comma-separated floats (one per row).",
    )
    ap.add_argument(
        "--out-w-px",
        type=int,
        default=1920,
        help="Resize output video width to this size (in pixels). Aspect ratio is preserved.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if exists)",
    )
    args = ap.parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    save_path = args.save_path.expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    if args.n_frames is not None and args.n_frames <= 0:
        raise ValueError("--n-frames must be a positive integer when provided")

    if save_path.exists() and not args.overwrite:
        print(f"[skip] exists: {save_path}")
        return

    rows = collect_grid_videos(root_dir)
    nrows = len(rows)
    ncols = len(rows[0])

    crop_h_by_row = _parse_row_spec(args.crop_h, nrows, "--crop-h", lo=1e-12, hi=1.0, default=None)
    crop_w_by_row = _parse_row_spec(args.crop_w, nrows, "--crop-w", lo=1e-12, hi=1.0, default=None)
    crop_y_by_row = _parse_row_spec(args.crop_y, nrows, "--crop-y", lo=0.0, hi=1.0, default=0.5)

    # Use FPS from the first video in the grid.
    fps = get_video_fps(rows[0][0], fallback=24.0)

    print(f"[info] root_dir: {root_dir}")
    print(f"[info] grid: {nrows} rows x {ncols} cols")
    print(f"[info] fps: {fps}")
    print(f"[info] crop_h_by_row: {crop_h_by_row}")
    print(f"[info] crop_w_by_row: {crop_w_by_row}")
    print(f"[info] out_w_px: {args.out_w_px}")
    print(f"[info] n_frames: {args.n_frames if args.n_frames is not None else '(auto shortest)'}")
    print(f"[info] save_path: {save_path}")

    if args.dry_run:
        print("[dry-run] Would read videos in each subfolder as one row (sorted filenames).")
        for r, vids in enumerate(rows):
            print(f"  row {r}: {vids[0].parent.name}  ({len(vids)} videos)  crop_h={crop_h_by_row[r]} crop_w={crop_w_by_row[r]}")
            for v in vids:
                print(f"    - {v.name}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    iters: list[list] = [[iio.imiter(p) for p in row] for row in rows]

    out_frames: list[np.ndarray] = []
    t = 0
    max_t = args.n_frames  # None => until any ends

    while True:
        if max_t is not None and t >= max_t:
            break

        grid_rows: list[np.ndarray] = []
        try:
            for r in range(nrows):
                row_frames: list[np.ndarray] = []
                for c in range(ncols):
                    fr = next(iters[r][c])
                    fr = _ensure_rgb(fr)

                    ch = crop_h_by_row[r]
                    cw = crop_w_by_row[r]
                    if ch is not None or cw is not None:
                        cy = crop_y_by_row[r]
                        fr = _center_crop(fr, frac_h=ch, frac_w=cw, crop_y=float(cy))

                    row_frames.append(fr)

                grid_rows.append(np.concatenate(row_frames, axis=1))

            # Normalize all row strips to the smallest (H,W) among them to allow vertical stacking
            row_hws = [(r.shape[0], r.shape[1]) for r in grid_rows]
            target_w = min(w for h, w in row_hws)

            grid_rows = [_resize_frame_to_width(r, target_w) for r in grid_rows]

            grid = np.concatenate(grid_rows, axis=0)

        except StopIteration:
            break

        grid = _resize_frame_to_width(grid, args.out_w_px)
        out_frames.append(grid)
        t += 1

    if not out_frames:
        raise RuntimeError("No frames were produced (videos may be empty/unreadable).")

    frames = np.stack(out_frames, axis=0).astype(np.uint8)
    frames = _make_even_hw_white(frames)

    H, W = frames.shape[1], frames.shape[2]
    print(f"[info] wrote frames: {frames.shape[0]}  output size: {W}x{H}")

    iio.imwrite(
        save_path,
        frames,
        fps=fps,
        codec="h264",  # or "libx264" depending on your ffmpeg build
        quality=10,    # 0 (worst) .. 10 (best)
        macro_block_size=1,
    )
    print(f"[ok] Wrote: {save_path}")


if __name__ == "__main__":
    main()