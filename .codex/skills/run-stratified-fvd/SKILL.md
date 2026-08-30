---
name: run-stratified-fvd
description: Reproduce rollout FVD analyses, including uniformly stratified sampling-variance studies and the four-run 10-frame full, quarter-window, and free-running experiment suite. Use when evaluating FVD across clip counts, clip lengths, rollout windows, or generation modes; repeating temporal samples across seeds; splitting side-by-side GT/generated videos; parallelizing FVD jobs across GPUs; or regenerating result tables and plots in the fid-metrics-scripts repository.
---

# Run Stratified FVD

Use the repository-level `scripts/evaluate_long_rollout_fvd_stratified.py`.
Run it from the `fid-metrics-scripts` repository root; it discovers the root
automatically.

## Prepare inputs

1. Split side-by-side rollout videos into separate GT and generated videos first.
2. Keep one video per episode in each directory. The script sorts paths and uses
   the first `--episodes` files, matching `VideoDataset` behavior.
3. Pass frame bounds as an inclusive interval. For each clip length, the script
   drops frames from the beginning until the interval contains an exact multiple
   of the clip length. For example, frames 0-1000 become frames 1-1000 for
   8-frame clips and frames 9-1000 for 16-frame clips.
4. Ensure `fid_metrics.dataset.VideoDataset` supports
   `stratified_num_clips` and `stratified_seed`. Each episode must use an
   independent RNG stream and sample one clip from every equal temporal stratum.

## Plan before allocating GPUs

Run a dry run to validate paths, counts, adjusted frame windows, and commands:

```bash
python scripts/evaluate_long_rollout_fvd_stratified.py \
  --gt-glob 'outputs/my_eval/split/sampled_df/gt/*.mp4' \
  --generated-glob 'outputs/my_eval/split/sampled_df/generated/*.mp4' \
  --output-dir outputs/my_stratified_fvd \
  --clip-lengths 8,16 \
  --clip-counts 64,32,16,8,4 \
  --frame-start 0 \
  --frame-end-inclusive 1000 \
  --episodes 20 \
  --repeats 5 \
  --dry-run
```

Infeasible conditions are reported and skipped by default, such as 64 clips
when only 62 non-overlapping clips exist. Pass `--strict-counts` to fail instead.

## Run on an interactive allocation

Request no more GPUs than useful concurrent conditions. The script gives each
GPU one condition at a time and evaluates that condition's seeds sequentially.

```bash
srun --gpus=4 --reservation=interactive --time=00:40:00 \
  --ntasks=1 --cpus-per-task=32 \
  .venv/bin/python -u \
  scripts/evaluate_long_rollout_fvd_stratified.py \
  --gt-glob 'outputs/my_eval/split/sampled_df/gt/*.mp4' \
  --generated-glob 'outputs/my_eval/split/sampled_df/generated/*.mp4' \
  --output-dir outputs/my_stratified_fvd \
  --clip-lengths 8,16 \
  --clip-counts 64,32,16,8,4 \
  --frame-start 0 --frame-end-inclusive 1000 \
  --episodes 20 --repeats 5
```

Use `--gpus 0,2` to select devices explicitly. Re-running resumes completed
`(clip_length, clip_count, seed)` cells from the repeats CSV. Pass `--force` to
recompute them. Use `--plot-only` to regenerate plots and summaries without FVD.

## Run the four-run 10-frame experiment suite

Use the repository-level `scripts/evaluate_rollout_fvd_10f_experiments.py` for
the fixed comparison of `allctx-t50`, `allctx-p00`, `sampled-df`, and
`allctx-p01`. The launcher discovers the locally cached W&B videos and performs:

1. Full long-rollout FVD on frames 1-1000: 20 videos and 100 10-frame clips/video.
2. Long-rollout FVD on inclusive windows 0-249, 250-499, 500-749, and 750-999:
   20 videos and exactly 25 10-frame clips/video/window.
3. Full free-running FVD on frames 7-76: 32 videos and 7 10-frame clips/video.

Select long-rollout IDs `s000000` through `s000019`; exclude the highest four
IDs from 24-video runs. Use all 32 free-running IDs. Treat the left side of each
source as GT and the right side as generated. The launcher reuses the validated
long-rollout splits at its default `--long-split-root`, creates free-running
splits, validates frame count/FPS/dimensions, runs 24 FVD cells, and makes three
PNG/PDF plots.

Run the complete workflow inside a four-GPU interactive allocation:

```bash
cd /projects/u6ni/francelico/fid-metrics-scripts
srun --gpus=4 --reservation=interactive --time=00:40:00 \
  --ntasks=1 --cpus-per-task=32 \
  .venv/bin/python -u scripts/evaluate_rollout_fvd_10f_experiments.py \
  --phase all
```

The default output is
`outputs/rollout_fvd_10f_experiments_20260819`. Override it with `--output`
and override the reusable long split with `--long-split-root`. Use phases to
resume or regenerate only what is needed:

```bash
# Prepare and validate splits without FVD.
.venv/bin/python scripts/evaluate_rollout_fvd_10f_experiments.py --phase prepare

# Compute/reuse FVD inside an allocation, then write CSV/JSON results.
srun --gpus=4 --reservation=interactive --time=00:40:00 \
  --ntasks=1 --cpus-per-task=32 \
  .venv/bin/python -u scripts/evaluate_rollout_fvd_10f_experiments.py \
  --phase compute

# Recreate all plots from completed CSV files; no GPU is needed.
.venv/bin/python scripts/evaluate_rollout_fvd_10f_experiments.py --phase plot
```

Completed FVD logs are reused automatically. Pass `--force-fvd` to recompute
all cells or `--force-prepare` to overwrite the free-running split videos.
Inspect `fvd_all_results.csv`, `source_manifest.csv`, and all 24 `logs/fvd_*.log`
files before reporting. The plot artifacts are:

- `exp1_long_full/fvd_by_run.{png,pdf}`
- `exp2_long_quarters/fvd_by_window.{png,pdf}`
- `exp3_free_full/fvd_by_run.{png,pdf}`

Copy the complete experiment artifacts to the local machine with:

```bash
mkdir -p ~/Downloads/rollout_fvd_10f_experiments_20260819
scp -r \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/exp1_long_full \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/exp2_long_quarters \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/exp3_free_full \
  ~/Downloads/rollout_fvd_10f_experiments_20260819/
```

## Verify outputs

Inspect these files under `--output-dir`:

- `fvd_stratified_repeats.csv` and `.json`: every raw repeat.
- `fvd_stratified_summary.csv` and `.json`: mean, sample SD, variance, range,
  coefficient of variation, and 95% t interval.
- `clip_selections.json`: exact per-episode clip IDs for every seed.
- `fvd_stratified_variance.png` and `.pdf`: combined clip-length trends.
- `fvd_stratified_variance_len_<N>.png` and `.pdf`: one plot per clip length.
- `logs/`: full FVD logs for diagnosing failures.
- `analysis_config.json`: resolved inputs, frame windows, and parameters.

Confirm the expected number of repeat rows for every feasible condition and
scan logs for tracebacks or CUDA errors before reporting results. Report FVD as
mean plus-or-minus sample SD and state the repeats, episodes, and dropped-frame
rule. Preserve the CSV, selection manifest, plot, and logs together when copying
artifacts off the cluster.
