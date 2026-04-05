#!/usr/bin/env python3
"""
Stacks two videos A and B vertically (one on top of the other) or horizontally (side by side).
Optionally resizes video A to match the width (if vertical) or height (if horizontal) of video B
while preserving aspect ratio.
Output is saved to output_path as an .mp4 file.
Uses imageio.v3 for reading/writing.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageFilter


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Return uint8 RGB frame with shape (H, W, 3)."""
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)

    # Convert dtype to uint8 if needed (best-effort)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        # Grayscale HxW -> HxWx3
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        # Drop alpha
        frame = frame[..., :3]
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    return frame


# def _resize_preserve_aspect(frame: np.ndarray, *, target_w: int | None = None, target_h: int | None = None) -> np.ndarray:
#     """Resize frame preserving aspect ratio. Specify exactly one of target_w or target_h."""
#     if (target_w is None) == (target_h is None):
#         raise ValueError("Specify exactly one of target_w or target_h")
#
#     frame = _ensure_rgb(frame)
#     h, w, _ = frame.shape
#
#     if target_w is not None:
#         if target_w <= 0:
#             raise ValueError(f"target_w must be > 0, got {target_w}")
#         if w == target_w:
#             return frame
#         new_w = int(target_w)
#         new_h = max(1, int(round(h * (new_w / w))))
#         pil = Image.fromarray(frame, mode="RGB")
#         pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)
#         return np.asarray(pil, dtype=np.uint8)
#
#     # target_h is not None
#     if target_h <= 0:
#         raise ValueError(f"target_h must be > 0, got {target_h}")
#     if h == target_h:
#         return frame
#     new_h = int(target_h)
#     new_w = max(1, int(round(w * (new_h / h))))
#     pil = Image.fromarray(frame, mode="RGB")
#     pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)
#     return np.asarray(pil, dtype=np.uint8)


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

    # Best quality resampling. LANCZOS is especially good for downscaling.
    resample = Image.Resampling.LANCZOS

    # If we're downscaling, reducing_gap helps reduce aliasing / moiré.
    downscaling = (new_w < w) or (new_h < h)
    if downscaling:
        pil = pil.resize((new_w, new_h), resample=resample, reducing_gap=3.0)
        if sharpen:
            # mild sharpening; adjust radius/percent/threshold if you want stronger/weaker
            pil = pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    else:
        # For upscaling, reducing_gap doesn't apply; LANCZOS still tends to look best.
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


def _center_crop(frame: np.ndarray, *, frac_h: float | None, frac_w: float | None) -> np.ndarray:
    """
    Center-crop a frame by fractional height/width. Fractions are relative to the frame's current size.
    If a fraction is None, that dimension is not cropped.
    """
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
    """
    Ensures video frames have even height and width by padding with white pixels.
    frames: (T, H, W, 3), uint8
    """
    T, H, W, C = frames.shape

    new_H = H if (H % 2 == 0) else (H + 1)
    new_W = W if (W % 2 == 0) else (W + 1)

    if new_H == H and new_W == W:
        return frames

    # Fill with white (255)
    out = np.full((T, new_H, new_W, C), 255, dtype=frames.dtype)

    # Copy original frames into top-left corner
    out[:, :H, :W, :] = frames

    return out

def _paired_frames_until_shorter(video_a: Path, video_b: Path, max_pairs: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Read frames from both videos in lockstep until one ends (or until max_pairs).
    This naturally trims to the shorter video length when max_pairs is None.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_a", type=Path, help="Input video A path")
    ap.add_argument("video_b", type=Path, help="Input video B path")
    ap.add_argument("output_path", type=Path, help="Output video path")
    ap.add_argument("--root_dir", type=Path, default=None, help="Root directory to search for videos recursively")
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
    ap.add_argument(
        "--crop-a-h",
        type=float,
        default=None,
        help="Center-crop Video A height to this fraction of its original height (0<frac<=1). Applied before resizing.",
    )
    ap.add_argument(
        "--crop-a-w",
        type=float,
        default=None,
        help="Center-crop Video A width to this fraction of its original width (0<frac<=1). Applied before resizing.",
    )
    ap.add_argument(
        "--pad-a-w",
        type=float,
        default=None,
        help="Add white pixel padding to the left and right of video A corresponding to a fraction of its original width (frac>0). Applied before resizing."
             "For example, --pad-a-w 0.25 means the video is 25% wider prior to resizing",
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

    src_a: Path = args.video_a
    src_b: Path = args.video_b
    dst: Path = args.output_path

    # Basic validation
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

    # Read in lockstep so "no --n-frames" naturally trims to shorter length
    frames_a, frames_b = _paired_frames_until_shorter(src_a, src_b, args.n_frames)

    if len(frames_a) == 0 or len(frames_b) == 0:
        raise RuntimeError("No paired frames could be read (one or both videos appear empty or unreadable).")

    if args.upscale_b != 1.0:
        frames_b = [_resize_by_factor(fb, args.upscale_b) for fb in frames_b]

    if args.crop_a_h is not None or args.crop_a_w is not None:
        frames_a = [_center_crop(fa, frac_h=args.crop_a_h, frac_w=args.crop_a_w) for fa in frames_a]

    if args.pad_a_w is not None:
        frames_a = [_pad_width_white(fa, args.pad_a_w) for fa in frames_a]

    # Determine target dimension from first B frame (assumed consistent)
    b0 = frames_b[0]
    hb, wb, _ = b0.shape

    stack_h = bool(args.stack_horizontally)

    # Resize A frames to match B width/height (preserve aspect)
    if stack_h:
        # match heights
        resized_a = [_resize_preserve_aspect(fa, target_h=hb) for fa in frames_a]
        # After resizing, ensure heights match exactly (they should), then concat width-wise
        out_frames = []
        for fa, fb in zip(resized_a, frames_b):
            if fa.shape[0] != fb.shape[0]:
                # Safety: force exact height
                fa = _resize_preserve_aspect(fa, target_h=fb.shape[0])
            # Channels already RGB; heights match; concat along width axis=1
            out_frames.append(np.concatenate([fa, fb], axis=1))
    else:
        # vertical: match widths
        resized_a = [_resize_preserve_aspect(fa, target_w=wb) for fa in frames_a]
        out_frames = []
        for fa, fb in zip(resized_a, frames_b):
            if fa.shape[1] != fb.shape[1]:
                # Safety: force exact width
                fa = _resize_preserve_aspect(fa, target_w=fb.shape[1])
            # concat along height axis=0
            out_frames.append(np.concatenate([fa, fb], axis=0))

    frames = np.stack(out_frames, axis=0)  # (T, H, W, 3), uint8
    frames = _make_even_hw(frames)

    # Print summary
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

    # Save with your preferred args (same as earlier)
    iio.imwrite(
        dst,
        frames,
        fps=24,
        codec="h264",  # or "libx264" depending on your ffmpeg build
        quality=10,    # 0 (worst) .. 10 (best)
        macro_block_size=1,
    )
    print(f"[ok] Wrote: {dst}")


if __name__ == "__main__":
    main()