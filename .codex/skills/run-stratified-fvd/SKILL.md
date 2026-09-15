---
name: run-stratified-fvd
description: Run rollout FVD and per-frame FID analyses in fid-metrics-scripts, including temporal sampling studies, full and quarter-window long rollouts, and free-running comparisons. Use for preparing paired videos, running metric campaigns, and regenerating their tables and plots.
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

1. `lr-exp1`: full long-rollout FVD on frames 1-1000: 20 videos and 100 10-frame clips/video.
2. `lr-exp2`: long-rollout FVD on inclusive windows 0-249, 250-499, 500-749, and 750-999:
   20 videos and exactly 25 10-frame clips/video/window.
3. `fr-exp1`: full free-running FVD on frames 7-76: 32 videos and 7 10-frame clips/video.

`lr-exp3` is per-frame FID versus rollout time, run separately with
`scripts/evaluate_rollout_fid_per_frame.py`. Experiment IDs are stable across
campaigns; video counts are campaign parameters. The 20/32-video settings below
describe the original August campaign, not a cap for subsequent analyses.

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

- `lr-exp1/fvd_by_run.{png,pdf}`
- `lr-exp2/fvd_by_window.{png,pdf}`
- `fr-exp1/fvd_by_run.{png,pdf}`

Copy the complete experiment artifacts to the local machine with:

```bash
mkdir -p ~/Downloads/rollout_fvd_10f_experiments_20260819
scp -r \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/lr-exp1 \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/lr-exp2 \
  u6ni.aip2.isambard:/projects/u6ni/francelico/fid-metrics-scripts/outputs/rollout_fvd_10f_experiments_20260819/fr-exp1 \
  ~/Downloads/rollout_fvd_10f_experiments_20260819/
```

## Run a downloaded 256-video campaign on Verda

Use `~/fid-metrics-scripts/.venv/bin/python` on Verda. Check GPU processes and
memory before selecting explicit `--gpus`; use tmux because there is no scheduler.
Keep downloaded videos and outputs outside tracked source files.

Enumerate W&B runs by **display name** containing `256` in
`prefix-forcing/prefix-forcing`, verify their eval config, and save the run IDs,
configs, filenames, sizes, and MD5 checksums. Download each run into its own folder:

```bash
.venv/bin/python scripts/download_data_wandb.py \
  --run prefix-forcing/prefix-forcing/RUN_ID \
  --prefix media/videos --video_filenames .mp4 \
  --save-dir outputs/CAMPAIGN/downloads/RUN_ID
```

The general `--run-pattern` also matches IDs and flattens output; avoid it when
selection must be by display name and several runs reuse the same sample IDs.
Verify all downloads against W&B sizes/checksums. Retain duplicate uploads but
select one per episode only when the duplicates have identical checksums; resolve
conflicting duplicates before analysis. Stage IDs `s000000` through `s000255` as
`campaign/runs/<family>_s<step>/eval_outputs/step_<step>/long_rollout/sXXXXXX.mp4`.
Preserve a selection manifest and the original W&B run ID (the campaign discoverer
reads it from `runs/<key>/wandb/run-<timestamp>-<run_id>`).

For captions-disabled side-by-side videos, split left = GT, right = generated.
Validate all 256 pairs per run as 1001 frames, 20 FPS, 224×128 per half. The
launcher retains the original FVD preprocessing and 10-frame clip alignment.

```bash
.venv/bin/python -u scripts/evaluate_rollout_fvd_10f_experiments.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --long-only --prepare-workers 8 --phase prepare

.venv/bin/python -u scripts/evaluate_rollout_fvd_10f_experiments.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --long-only --gpus 6,7 --batch-size 256 --num-workers 2 \
  --phase compute

.venv/bin/python scripts/evaluate_rollout_fvd_10f_experiments.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --long-only --phase plot

.venv/bin/python -u scripts/evaluate_rollout_fid_per_frame.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --gpus 6,7 --batch-size 128 --num-workers 2
```

The example GPUs must be rechecked on each launch. `--long-only` omits `fr-exp1`.
For 16 runs, expect 80 FVD rows: 16 `lr-exp1` and 64 `lr-exp2`. Full FVD uses
25,600 clips/distribution; each quarter uses 6,400. This workload is larger than
the original 40-minute, four-run example above; measure progress before estimating
runtime. Provide the existing `weights/i3d_pretrained_400.pt` weights.
Use sequential FVD loading: each worker caches one decoded video window, and
shuffling would defeat that cache. Batch 256 is validated on a B300 and keeps
most of a video's 100 clips in one worker batch. Recheck memory before using it
on smaller GPUs; reduce `--batch-size` there without changing metric semantics.

`lr-exp3` uses the existing `fid_per_frame` config and InceptionV3 2048-dimensional
features. It evaluates all 256 episodes independently at every frame 0–1000,
including the conditioning frame at 0. Time is frame index / 20 (0–50 seconds).
It writes 1001 CSV rows per run and unsmoothed PNG/PDF plots by model family, in
both seconds and frame indices. No free-running video is needed. The plotter
uses Matplotlib fonts when LaTeX is unavailable. Completed, finite, contiguous
per-frame CSVs are reused; `--force` recomputes, and `--phase plot` regenerates plots.

Keep `lr-exp1/`, `lr-exp2/`, `lr-exp3/`, manifests, configs, and logs together.
Verify 16,016 per-frame rows for 16 runs, finite scores, successful exit codes,
and every expected PNG/PDF. FID at 256 samples has sampling bias; compare runs
with the same episode count and preprocessing. The new names apply to new
outputs; historical directories are not automatically renamed.

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

## Required plot deliverables for expanded campaigns

A CSV alone does not complete `lr-exp3`. After computing (or reusing) the
per-frame CSVs, explicitly run both plot phases:

```bash
.venv/bin/python scripts/evaluate_rollout_fvd_10f_experiments.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --long-only --phase plot
.venv/bin/python scripts/evaluate_rollout_fid_per_frame.py \
  --campaign-root outputs/CAMPAIGN/campaign --output outputs/CAMPAIGN/analysis \
  --videos 256 --phase plot
```

Plot-only FID needs no GPU argument. Verify `lr-exp3/plot_manifest.json` and
all listed nonempty PNG/PDF files: `fid_vs_frame_<family>.{png,pdf}` and
`fid_vs_time_<family>.{png,pdf}` for **every** discovered model family.
The launcher raises an error if the plotter fails to create an artifact.
Visually inspect plots for complete legends/panels and readable labels.
The quarter-window FVD panel layout must include every family, not just four.

Select completed runs with 256 distinct uploaded episodes. Keep failed or
interrupted attempts in the inventory but exclude their incomplete videos.
Do not silently merge repeated successful runs of one model/checkpoint:
retain distinct run identities or explicitly document any exact duplicates.
For N runs, require 5*N FVD rows and 1001*N per-frame FID rows. For F model
families, require 4 FVD plot files plus 4*F FID plot files. The earlier 16-run
counts are examples, not fixed campaign sizes. Cached metrics may be reused
only for the same W&B run IDs, inputs, preprocessing and metric settings.

On the current Verda setup, run as `francelico@86.38.238.210`, use
`/home/francelico/fid-metrics-scripts/.venv`, and recheck free GPUs. Historical
artifacts remain on CPU node `root@86.38.182.153`. Copy the completed PNG/PDF
plots, CSVs and provenance manifests to the user's local `~/Downloads/` and
verify the local file counts before reporting completion.

## FVD versus training compute (PFLOPs)

For the existing ctx1 Minecraft checkpoints, use the compute accounting and
method colors from the August 2026 `pflop_threshold_fvd_10f` campaign:
all-context variants cost 3.68965809340416 PFLOPs per optimizer step;
`df`, `sampled-df`, and `sampled-df-notail` cost 1.94158962671616.
The x coordinate is checkpoint step times that factor, not eval runtime or
threshold-group names. Revalidate factors for different training configurations.

```bash
.venv/bin/python scripts/plot_rollout_fvd_vs_pflops.py \
  --analysis outputs/CAMPAIGN/analysis
```

Require five PNG/PDF pairs under `analysis/fvd_vs_pflops/`: full long rollout
and windows 0-249, 250-499, 500-749, 750-999. Preserve the emitted data CSV and
plot config with the compute factors. Do not include free-running plots when
that evaluation was skipped. Include these ten files in the local download.
