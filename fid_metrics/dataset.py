import bisect
import glob
from typing import List

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    'ImageDataset',
    'ImageSequenceDataset',
    'VideoDataset',
    'is_image_dir_path',
    'is_video_path',
]


class SequeceTransform:
    def __init__(self, transforms, stack_dim=1):
        self.transforms = transforms
        self.stack_dim = stack_dim

    def __call__(self, imgs: List) -> torch.Tensor:
        imgs = [self.transforms(img) for img in imgs]
        return torch.stack(imgs, dim=self.stack_dim) if len(imgs) > 1 else imgs[0]


class ImageDataset(Dataset):
    def __init__(self, image_dirs, resize_shape=(256, 512), ext='jpg'):
        self.image_dirs = glob.glob(image_dirs)
        self.transforms = T.Compose([T.ToTensor(), T.Resize(resize_shape)])

        self.files = []
        for image_dir in self.image_dirs:
            imgs = glob.glob(image_dir + (f'/*.{ext}' if ext else '/*'))
            self.files.extend(imgs)
        print(f'Loaded {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        image_dirs,
        sequence_length=16,
        resize_shape=(224, 224),
        ext='jpg',
        no_overlap=True,
    ):
        self.image_dirs = glob.glob(image_dirs)
        self.sequence_length = sequence_length
        self.no_overlap = no_overlap
        self.transforms = SequeceTransform(T.Compose([T.ToTensor(), T.Resize(resize_shape)]))

        self.frame_paths = []
        for image_dir in self.image_dirs:
            frame_paths = sorted(glob.glob(image_dir + (f'/*.{ext}' if ext else '/*')))
            self.frame_paths.extend(frame_paths)
        print(f'Loaded {len(self.frame_paths)} images')

    def __len__(self):
        return (
            len(self.frame_paths) // self.sequence_length
            if self.no_overlap
            else len(self.frame_paths)
        )

    def __getitem__(self, idx):
        if self.no_overlap:
            frame_paths = self.frame_paths[
                idx * self.sequence_length : (idx + 1) * self.sequence_length
            ]
        else:
            frame_paths = self.frame_paths[idx : idx + self.sequence_length]

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        frames = self.transforms(frames)
        return frames



class VideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        max_videos=173,
        sequence_length=16,
        resize_shape=(224, 224),
        no_overlap=True,
        start_frame=0,
        end_frame=600,
    ):
        self.video_paths = sorted(glob.glob(video_path))
        if max_videos is not None:
            self.video_paths = self.video_paths[:int(max_videos)]
        self.sequence_length = int(sequence_length)
        self.no_overlap = bool(no_overlap)
        self.start_from_frame = start_frame
        self.end_frame = end_frame

        self.transforms = SequeceTransform(T.Compose([T.ToTensor(), T.Resize(resize_shape)]))

        total_frames = 0
        self.num_accum_sequences = []
        # store per-video selection window so __getitem__ doesn't have to recompute
        self._start_frame = []
        self._usable_frames = []

        for vp in self.video_paths:
            cap = cv2.VideoCapture(vp)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Select only the last_n_frames window if provided
            start = self.start_from_frame
            end = min(num_frames, self.end_frame) if self.end_frame is not None else num_frames
            usable = end - start

            total_frames += usable

            self._start_frame.append(start)
            self._usable_frames.append(usable)

            # Split the selected window into clips of sequence_length
            if usable < self.sequence_length:
                num_sequences = 0
            else:
                num_sequences = (
                    (usable // self.sequence_length) if self.no_overlap
                    else (usable - self.sequence_length + 1)
                )

            if self.num_accum_sequences:
                self.num_accum_sequences.append(self.num_accum_sequences[-1] + num_sequences)
            else:
                self.num_accum_sequences.append(num_sequences)

        print(f"Loaded {len(self.video_paths)} video files, {total_frames} frames total")

    def __len__(self):
        return 0 if not self.num_accum_sequences else self.num_accum_sequences[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        video_idx = bisect.bisect_left(self.num_accum_sequences, idx + 1)
        video_path = self.video_paths[video_idx]

        local_idx = idx - (self.num_accum_sequences[video_idx - 1] if video_idx > 0 else 0)

        start = self._start_frame[video_idx]
        usable = self._usable_frames[video_idx]

        # local_idx is an index over sequences inside the selected window
        if self.no_overlap:
            offset = local_idx * self.sequence_length
        else:
            offset = local_idx

        frame_idx = start + offset

        # Safety: ensure we never read beyond the selected window
        # (this should already be guaranteed by num_sequences computation)
        end_allowed = start + usable
        if frame_idx + self.sequence_length > end_allowed:
            raise RuntimeError(
                f"Computed clip [{frame_idx}, {frame_idx + self.sequence_length}) exceeds "
                f"selected window [{start}, {end_allowed}) for {video_path}"
            )

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read frame at {frame_idx} from {video_path}")
            frames.append(frame)

        cap.release()
        frames = self.transforms(frames)
        return frames


def is_image_dir_path(path, ext='jpg'):
    return len(glob.glob(path + (f'/*.{ext}' if ext else '/*'))) > 0


def is_video_path(path, ext='mp4'):
    return glob.glob(path)[0].endswith(f'.{ext}')
