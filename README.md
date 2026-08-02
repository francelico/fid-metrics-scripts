# FID/FVD Metrics

This repository provides a toolkit for computing Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD) metrics, widely utilized for assessing the quality of generative models in the fields of image and video generation.

## Acknowledgements

This repository is derived from [fid-metrics](https://github.com/npurson/fid-metrics). We re-implement certain methods to run metric computation directly on GPU and we add utility scripts to download & pre-process videos and to plot results.

## Installation

    ```bash
    pip install -r requirements.txt
    ```

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

## License

Released under the [MIT](LICENSE) License.
