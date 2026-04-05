# FID/FVD Metrics

This repository provides a toolkit for computing Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD) metrics, widely utilized for assessing the quality of generative models in the fields of image and video generation.

## Installation

    ```bash
    pip install -r requirements.txt
    ```

## Usage

In the absence of setting up the package, users can currently utilize it by the following as a temporal fix.

0. **Setup**

    ```shell
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```
1. download worldmem rollouts
Note: there are multiple options to do this based on whether the videos are in multiple runs or one, check the script itself and its arguments for detailed instructions
```bash
wandb login
python download_data_wandb.py --run-pattern samuel-garcin-research-projects/voxel_wm/ab-gtcam --save-dir WORLDMEMPATH
```

2. **download dataset and create a filtered set corresponding to rollout videos you just downloaded **
use azcopy to download free_noop_and_look_around
```bash
python copy_from_dataset.py --selected_dir WORLDMEMPATH --dataset_dir PATH/free_noop_and_look_around --output_dir PATH/free_noop_and_look_around_selected 
```

1. **Calculating FID/FVD**

Manually modify l100-105 of fid_metrics/dataset.py to set end_frame (200/400/600) and max_videos (set to number you downloaded, aim for at least 64).

Note: Previous results used 173 videos. For reference going from 173->64 videos persist_base increases ~40 pts FVD. Going anywhere 64 would be ill-advised.

script supports regex:
    ```shell
    python fid_metrics/main.py paths=[PATH/free_noop_and_look_around_selected/*.mp4,WORLDMEMPATH/*.mp4]
    ```

or any path1, path2:
    `path1`, `path2` can either be images, videos, folders of images or pathnames of the aforementioned that match the pattern of `glob`.

## Acknowledgements

The code in this repository is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [fvd-comparison](https://github.com/universome/fvd-comparison).

## License

Released under the [MIT](LICENSE) License.
