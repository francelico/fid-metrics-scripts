# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The working goal is **evaluating world-model rollouts** (primarily WorldMem, plus MineDojo/Minetest) by comparing generated rollout videos against the corresponding ground-truth dataset videos with **FVD** (Fréchet Video Distance) — and **FID** (Fréchet Inception Distance) for the image case. The end-to-end pipeline (download rollouts → copy matching dataset videos → compute the distance) is the main use; the generic metrics engine in `fid_metrics/` is the means.

The `fid_metrics/` package computes the metrics; the scripts at the repo root prepare and align the two video sets being compared. Metric code is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [fvd-comparison](https://github.com/universome/fvd-comparison).

## Setup & running

The package is **not installed** — set the path and run from the repo root:

```bash
pip install -r requirements.txt
export PYTHONPATH=`pwd`:$PYTHONPATH
python fid_metrics/main.py paths=[PATH_A,PATH_B]
```

`paths` is a Hydra override taking **exactly two** entries — the two distributions to compare. Each entry can be an image, a video, a folder of images, or a `glob` pattern (e.g. `dir/*.mp4`). Config lives in `configs/config.yaml`; override any field on the command line (Hydra syntax), e.g. `num_iters=1000` or `metrics.0.data.batch_size=8`. Outputs are written under `outputs/` (Hydra's run dir). Runs on CUDA when available, else CPU. There is no test suite or linter.

## Architecture

`main.py` is the Hydra entrypoint. For each metric in `cfg.metrics` it builds one dataloader per path, runs every clip/image through a feature extractor, concatenates the features per distribution, then `calculate_fid` returns the Fréchet distance between the two feature sets.

- **`fid_metrics/main.py`** — orchestration. `build_loaders` dispatches on path type (`is_video_path` / `is_image_dir_path` from `dataset.py`) and metric type; `build_model` picks the extractor. Note the input handling differences: FID reshapes 5-D video tensors to a batch of frames, while **FVD rescales inputs to [-1, 1]** (`x * 2 - 1`).
- **`fid_metrics/fid.py`** — the Fréchet-distance math (`calculate_frechet_distance`, `calculate_fid`) plus model builders: `build_inception` (FID, InceptionV3), `build_inception3d` (FVD), `build_resnet3d` (unused alt). `postprocess_i2d_pred` pools FID features to a vector.
- **`fid_metrics/dataset.py`** — three datasets: `ImageDataset` (FID over image dirs), `ImageSequenceDataset` (FVD over image dirs), `VideoDataset` (FVD over video files; slices each video into non-overlapping clips of `sequence_length` frames within a `[start_frame, end_frame)` window, capped at `max_videos`).
- **`fid_metrics/inception.py`** — `InceptionV3` 2-D extractor for FID.
- **`fid_metrics/inception3d.py`** — `InceptionI3d` 3-D extractor for FVD (the `videogpt` model).
- **`fid_metrics/resnet3d.py`** — ResNet-50 3-D, an alternative FVD backbone.

### FVD backbone choice

`configs/config.yaml` selects the FVD model. Two options, **not interchangeable** in weights or call convention:
- `type: videogpt`, `path: weights/i3d_pretrained_400.pt` — loads `InceptionI3d` via `load_state_dict`. This is the committed default; the weights file is present.
- `type: styleganv`, `path: i3d_torchscript.pt` — loads a TorchScript module; `main.py` calls it with `return_features=True`. Weights not in repo.

## FVD frame window / video count

The frame window and video count for FVD live in `configs/config.yaml` under `metrics.0.data.dataset` (`start_frame`, `end_frame`, `max_videos`) and are overridable on the CLI, e.g. `metrics.0.data.dataset.end_frame=400 metrics.0.data.dataset.max_videos=64`. These keys are passed straight through `build_loaders` (`main.py`) as `**dataset_cfgs` to `VideoDataset.__init__`. Aim for ≥64 videos, ideally 173 (no more). Going from 173→64 videos can shift FVD by ~40 points, so keep the count consistent when comparing runs.

## Typical end-to-end workflow (from README)

1. `scripts/download_data_wandb.py` — download WorldMem rollout videos from W&B (`wandb login` first).
2. `scripts/copy_from_dataset.py` — extract the dataset videos that correspond to the downloaded rollouts (uses `azcopy`).
3. Edit `dataset.py` window (see gotcha above).
4. `python fid_metrics/main.py paths=[DATASET_SELECTED/*.mp4,ROLLOUTS/*.mp4]`.

## Data-prep / processing scripts (`scripts/`)

The processing scripts live under `scripts/`. Each is a standalone `argparse` CLI (run `python scripts/<script> -h`), independent of the `fid_metrics` package:

- `download_data_wandb.py` — pull rollout videos from W&B runs.
- `copy_from_dataset.py` — copy dataset videos matching downloaded rollouts.
- `filter_minedojo_data.py` / `filter_minetest_data.py` — extract/filter mp4s from MineDojo / Minetest dataset layouts.
- `crop_top_half.py` — crop videos to the top half of each frame.
- `split_reencode_trim.py` — split every video in a folder into two halves (horizontal cut by default → top/bottom into FOLDER_A/FOLDER_B; `--vertical-split` for left/right), re-encode to `--target-fps`, and trim to `--num-frames`.
- `trim_to_first_n_frames.py`, `trim_and_stack_videos.py`, `trim_and_stack_videos_batched.py` — trim and side-by-side stack videos.
- `make_video_grid.py` — assemble videos into a grid.
- `select_common_files.py` — find filenames common across directories and copy them out.
- `split_run_names.py` — split a CSV's `Name` column by substring patterns.
- `path_magic.py` — reorganize a `root/dir1/dirA/*.mp4` tree into `save/dirA/dir1_dirA/*.mp4`.
- `plot_fid_vs_t.py` — plot FID/FVD vs. time/frame window from run outputs.
