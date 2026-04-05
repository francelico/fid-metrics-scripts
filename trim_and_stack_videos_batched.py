#!/usr/bin/env python3
"""
Stacks two videos A and B vertically (one on top of the other) or horizontally (side by side).
Optionally resizes video A to match the width (if vertical) or height (if horizontal) of video B
while preserving aspect ratio.
Output is saved to output_path as an .mp4 file.
Uses imageio.v3 for reading/writing.

Batch mode:
- If --root_dir is provided, positional video_a/video_b are *selectors* (glob by default, regex with prefix "re:").
- For each directory under root_dir that contains matches for both selectors, produce one stacked output.
"""

import argparse
import fnmatch
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageFilter


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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


def _resize_preserve_aspect(
    frame: np.ndarray,
    *,
    target_w: int | None = None,
    target_h: int | None = None,
    sharpen: bool = False,
) -> np.ndarray:
    """Resize frame preserving aspect ratio. Specify exactly one of target_w or target_h."""
    if (target_w is None) == (target_h is None):
        raise ValueError("Specify exactly one of target_w or target_h")

    frame = _ensure_rgb(frame)
    h, w, _ = frame.shape

    if target_w is not None:
        if target_w <= 0:
            raise ValueError(f"target_w must be > 0, got {target_w}")
        if w == target_w:
            return frame
        new_w = int(target_w)
        new_h = max(1, int(round(h * (new_w / w))))
    else:
        if target_h is None or target_h <= 0:
            raise ValueError(f"target_h must be > 0, got {target_h}")
        if h == target_h:
            return frame
        new_h = int(target_h)
        new_w = max(1, int(round(w * (new_h / h))))

    pil = Image.fromarray(frame, mode="RGB")
    resample = Image.Resampling.LANCZOS

    downscaling = (new_w < w) or (new_h < h)
    if downscaling:
        pil = pil.resize((new_w, new_h), resample=resample, reducing_gap=3.0)
        if sharpen:
            pil = pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    else:
        pil = pil.resize((new_w, new_h), resample=resample)

    return np.asarray(pil, dtype=np.uint8)


def _resize_by_factor(frame: np.ndarray, factor: float) -> np.ndarray:
    frame = _ensure_rgb(frame)
    if factor == 1.0:
        return frame
    h, w, _ = frame.shape
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))

    pil = Image.fromarray(frame, mode="RGB")
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _validate_frac(name: str, frac: float | None) -> None:
    if frac is None:
        return
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {frac}")


def _validate_pad_frac(name: str, frac: float | None) -> None:
    if frac is None:
        return
    # padding can be > 1.0; just must be positive
    if not (frac > 0.0):
        raise ValueError(f"{name} must be > 0, got {frac}")


def _center_crop(frame: np.ndarray, *, frac_h: float | None, frac_w: float | None) -> np.ndarray:
    frame = _ensure_rgb(frame)
    H, W, C = frame.shape

    crop_h = H if frac_h is None else max(1, int(round(H * frac_h)))
    crop_w = W if frac_w is None else max(1, int(round(W * frac_w)))

    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)

    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w, :]


def _pad_width_white(frame: np.ndarray, frac_w: float) -> np.ndarray:
    """
    Add white padding on left+right so new width = round(W * (1 + frac_w)).
    Padding is split approximately evenly.
    """
    frame = _ensure_rgb(frame)
    H, W, C = frame.shape

    new_W = max(W, int(round(W * (1.0 + frac_w))))
    pad_total = new_W - W
    if pad_total <= 0:
        return frame

    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    out = np.full((H, new_W, C), 255, dtype=frame.dtype)
    out[:, pad_left : pad_left + W, :] = frame
    return out


def _make_even_hw(frames: np.ndarray) -> np.ndarray:
    T, H, W, C = frames.shape
    new_H = H if (H % 2 == 0) else (H + 1)
    new_W = W if (W % 2 == 0) else (W + 1)

    if new_H == H and new_W == W:
        return frames

    out = np.full((T, new_H, new_W, C), 255, dtype=frames.dtype)
    out[:, :H, :W, :] = frames
    return out


def _paired_frames_until_shorter(video_a: Path, video_b: Path, max_pairs: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ita = iio.imiter(video_a)
    itb = iio.imiter(video_b)

    frames_a: list[np.ndarray] = []
    frames_b: list[np.ndarray] = []

    n = 0
    while True:
        if max_pairs is not None and n >= max_pairs:
            break
        try:
            fa = next(ita)
            fb = next(itb)
        except StopIteration:
            break
        frames_a.append(_ensure_rgb(fa))
        frames_b.append(_ensure_rgb(fb))
        n += 1

    return frames_a, frames_b


def _match_selector(name: str, selector: str, filename: str) -> bool:
    """
    selector:
      - default: glob (fnmatch), e.g. 'val_*', '*.mp4', 'vid.mp4'
      - regex: prefix with 're:', e.g. 're:^val_.*\\.mp4$'
    Matching is against the *basename* (filename), not the full path.
    """
    if selector.startswith("re:"):
        pat = selector[3:]
        try:
            return re.search(pat, filename) is not None
        except re.error as e:
            raise ValueError(f"Invalid regex for {name}: {selector!r}: {e}") from e
    return fnmatch.fnmatch(filename, selector)


def _select_one_video_in_dir(dirpath: Path, selector: str, label: str) -> Path | None:
    """
    Returns:
      - None if no matches
      - Path if >=1 matches (chooses first in sorted order, but raises on ambiguity by default)
    """
    files = [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    matches = [p for p in files if _match_selector(label, selector, p.name)]

    if not matches:
        return None

    matches = sorted(matches)
    if len(matches) > 1:
        # Ambiguity is usually a bug in the selector; fail loudly so you notice.
        # If you prefer "pick first", change this to a warning + return matches[0].
        raise RuntimeError(
            f"Ambiguous selector in {dirpath}: {label} selector {selector!r} matched {len(matches)} files:\n"
            + "\n".join(f"  - {m.name}" for m in matches)
        )
    return matches[0]


def _stack_pair_to_file(src_a: Path, src_b: Path, dst: Path, args: argparse.Namespace) -> None:
    if not src_a.exists():
        raise FileNotFoundError(f"video_a does not exist: {src_a}")
    if not src_b.exists():
        raise FileNotFoundError(f"video_b does not exist: {src_b}")
    if src_a.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported extension for video_a: {src_a.suffix} (expected one of {sorted(VIDEO_EXTS)})")
    if src_b.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported extension for video_b: {src_b.suffix} (expected one of {sorted(VIDEO_EXTS)})")
    if args.n_frames is not None and args.n_frames <= 0:
        raise ValueError("--n-frames must be a positive integer when provided")

    frames_a, frames_b = _paired_frames_until_shorter(src_a, src_b, args.n_frames)
    if len(frames_a) == 0 or len(frames_b) == 0:
        raise RuntimeError(f"No paired frames could be read for: {src_a} vs {src_b}")

    if args.upscale_b != 1.0:
        frames_b = [_resize_by_factor(fb, args.upscale_b) for fb in frames_b]

    if args.crop_a_h is not None or args.crop_a_w is not None:
        frames_a = [_center_crop(fa, frac_h=args.crop_a_h, frac_w=args.crop_a_w) for fa in frames_a]

    if args.pad_a_w is not None:
        frames_a = [_pad_width_white(fa, args.pad_a_w) for fa in frames_a]

    b0 = frames_b[0]
    hb, wb, _ = b0.shape
    stack_h = bool(args.stack_horizontally)

    if stack_h:
        resized_a = [_resize_preserve_aspect(fa, target_h=hb) for fa in frames_a]
        out_frames = []
        for fa, fb in zip(resized_a, frames_b):
            if fa.shape[0] != fb.shape[0]:
                fa = _resize_preserve_aspect(fa, target_h=fb.shape[0])
            out_frames.append(np.concatenate([fa, fb], axis=1))
    else:
        resized_a = [_resize_preserve_aspect(fa, target_w=wb) for fa in frames_a]
        out_frames = []
        for fa, fb in zip(resized_a, frames_b):
            if fa.shape[1] != fb.shape[1]:
                fa = _resize_preserve_aspect(fa, target_w=fb.shape[1])
            out_frames.append(np.concatenate([fa, fb], axis=0))

    frames = np.stack(out_frames, axis=0)
    frames = _make_even_hw(frames)

    a0 = frames_a[0]
    ha, wa, _ = a0.shape
    mode = "horizontal" if stack_h else "vertical"
    print(f"[info] video_a: {src_a} (first frame {wa}x{ha})")
    print(f"[info] video_b: {src_b} (first frame {wb}x{hb})")
    print(f"[info] mode: {mode}")
    print(f"[info] paired frames: {frames.shape[0]}")
    print(f"[info] output frames shape: {frames.shape[1]}x{frames.shape[2]} (HxW), RGB")

    if args.dry_run:
        print(f"[dry-run] Would write to: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        dst,
        frames,
        fps=24,
        codec="h264",
        quality=10,
        macro_block_size=1,
    )
    print(f"[ok] Wrote: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_a", type=str, help="Input video A path (single mode) OR selector (batch mode)")
    ap.add_argument("video_b", type=str, help="Input video B path (single mode) OR selector (batch mode)")
    ap.add_argument("output_path", type=Path, help="Output video path (single mode) OR output directory (batch mode)")
    ap.add_argument("--root-dir", type=Path, default=None, help="Root directory to search for videos recursively")

    ap.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Trim to the first N frames. If omitted, trim to the shorter video length.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")

    ap.add_argument(
        "--stack-horizontally",
        action="store_true",
        help=(
            "Stack videos horizontally (side by side) instead of vertically (one on top of the other). "
            "Video A is always on the left/top, video B on the right/bottom. "
            "Video A is automatically resized to match the height (if horizontal) or width (if vertical) "
            "of video B while preserving aspect ratio."
        ),
    )
    ap.add_argument("--crop-a-h", type=float, default=None, help="Center-crop Video A height fraction (0<frac<=1).")
    ap.add_argument("--crop-a-w", type=float, default=None, help="Center-crop Video A width fraction (0<frac<=1).")
    ap.add_argument(
        "--pad-a-w",
        type=float,
        default=None,
        help=(
            "Add white pixel padding to the left and right of video A corresponding to a fraction of its "
            "original width (frac>0). Applied before resizing. "
            "Example: --pad-a-w 0.25 makes the frame 25%% wider before resizing."
        ),
    )
    ap.add_argument(
        "--upscale-b",
        type=float,
        default=1.0,
        help="Upscale video B by this factor (>1 increases resolution) before matching/resizing/stacking. Default: 1.0",
    )

    args = ap.parse_args()
    _validate_frac("--crop-a-h", args.crop_a_h)
    _validate_frac("--crop-a-w", args.crop_a_w)
    _validate_pad_frac("--pad-a-w", args.pad_a_w)

    # -------------------------
    # Batch mode
    # -------------------------
    if args.root_dir is not None:
        root = args.root_dir
        if not root.exists():
            raise FileNotFoundError(f"--root-dir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"--root-dir is not a directory: {root}")

        out_dir = args.output_path
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        sel_a = args.video_a
        sel_b = args.video_b

        # Search all directories under root (including root itself)
        dirs = [root] + sorted([p for p in root.rglob("*") if p.is_dir()])

        n_done = 0
        n_skipped = 0

        for d in dirs:
            try:
                a_path = _select_one_video_in_dir(d, sel_a, "video_a")
                b_path = _select_one_video_in_dir(d, sel_b, "video_b")
            except RuntimeError as e:
                # Ambiguous patterns: fail loudly (better than silently picking the wrong file)
                raise

            if a_path is None or b_path is None:
                n_skipped += 1
                continue

            dst_name = f"{b_path.stem}_stacked.mp4"
            dst_path = out_dir / dst_name

            print(f"[batch] dir: {d}")
            print(f"[batch]   A: {a_path.name}")
            print(f"[batch]   B: {b_path.name}")
            print(f"[batch]   -> {dst_path}")

            _stack_pair_to_file(a_path, b_path, dst_path, args)
            n_done += 1

        print(f"[batch] completed: {n_done} outputs (skipped dirs without matches: {n_skipped})")
        return

    # -------------------------
    # Single-pair mode
    # -------------------------
    src_a = Path(args.video_a)
    src_b = Path(args.video_b)
    dst = args.output_path
    _stack_pair_to_file(src_a, src_b, dst, args)


if __name__ == "__main__":
    main()