# FID/FVD Metrics

This repository provides a toolkit for computing Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD) metrics, widely utilized for assessing the quality of generative models in the fields of image and video generation.

## Acknowledgements

This repository is derived from [fid-metrics](https://github.com/npurson/fid-metrics). We re-implement certain methods to run metric computation directly on GPU and we add utility scripts to download & pre-process videos and to plot results.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` installs Python 3.11.13 and the dependencies recorded in `uv.lock`.
The default environment matches the core Isambard evaluation stack: PyTorch
2.13.0 and torchvision 0.28.0 with CUDA 13.0. Linux x86-64 (Verda) and aarch64
(Isambard) are supported. Use an NVIDIA driver compatible with CUDA 13.0;
the Python CUDA libraries are installed automatically. On Isambard, continue
to source `~/.hpc_env` for the cluster's driver compatibility libraries.

No separate `pip install` or PyTorch index flags are needed. `pyproject.toml`
declares dependencies and the PyTorch index; commit `uv.lock` whenever these
dependencies change. To verify the installation on a GPU node:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device='cuda'))"
uv run python -m fid_metrics.main --help
```

Some preparation scripts also invoke external tools (`ffmpeg`, `ffprobe`, or
`azcopy`); install those separately when using those scripts. Model weights and
datasets are also separate from the Python environment.

## Usage

0. **Setup**

    ```shell
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```
1. download worldmem rollouts
Note: there are multiple options to do this based on whether the videos are in multiple runs or one, check the script itself and its arguments for detailed instructions
```bash
wandb login
python scripts/download_data_wandb.py --run-pattern samuel-garcin-research-projects/voxel_wm/ab-gtcam --save-dir WORLDMEMPATH
```

2. **download dataset and create a filtered set corresponding to rollout videos you just downloaded **
use azcopy to download free_noop_and_look_around
```bash
python scripts/copy_from_dataset.py --selected_dir WORLDMEMPATH --dataset_dir PATH/free_noop_and_look_around --output_dir PATH/free_noop_and_look_around_selected 
```

1. **Calculating FID/FVD**

Set the frame window and video count as command-line overrides. `end_frame` is the window end in frames; `max_videos` caps how many videos are compared. Defaults live in `configs/config.yaml` under `metrics.0.data.dataset`.

script supports regex:
    ```shell
    python fid_metrics/main.py \
      paths=[PATH/free_noop_and_look_around_selected/*.mp4,WORLDMEMPATH/*.mp4] \
      metrics.0.data.dataset.end_frame=400 \
      metrics.0.data.dataset.max_videos=64
    ```

or any path1, path2:
    `path1`, `path2` can either be images, videos, folders of images or pathnames of the aforementioned that match the pattern of `glob`.

## Rollout experiment names

| ID | Analysis |
| --- | --- |
| `fr-exp1` | Full free-running FVD, frames 7–76 |
| `lr-exp1` | Full long-rollout FVD, frames 1–1000 |
| `lr-exp2` | Long-rollout FVD in four 250-frame windows |
| `lr-exp3` | Per-frame FID over long rollouts, frames 0–1000 / 0–50 seconds |

Run the FVD experiments with `scripts/evaluate_rollout_fvd_10f_experiments.py`.
For a 256-video campaign, pass `--campaign-root`, `--videos 256`, and
`--long-only` to omit `fr-exp1`. Run `scripts/evaluate_rollout_fid_per_frame.py`
against the same validated splits for `lr-exp3`; it uses the existing per-frame
metric engine and `scripts/plot_fid_vs_t.py`. The latter accepts `--fps 20` to
plot seconds and works without a LaTeX installation.

See the [rollout analysis skill](.codex/skills/run-stratified-fvd/SKILL.md) for
W&B downloading, duplicate handling, Verda commands, sample counts, and output
verification. New output directories use the IDs above; historical output
folders retain their original names until explicitly migrated.

## License

Released under the [MIT](LICENSE) License.
