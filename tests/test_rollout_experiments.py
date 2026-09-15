"""Campaign frame semantics and resume validation, without GPUs or datasets."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_rollout_fvd_10f_experiments as fvd
from evaluate_rollout_fid_per_frame import valid_csv
import evaluate_rollout_fid_per_frame as fid
import csv
from fid_metrics.dataset import VideoDataset


class TestRolloutExperiments(unittest.TestCase):
    def tearDown(self):
        fvd.configure(None, 20, False)

    def test_long_campaign_preserves_all_episodes_and_clip_windows(self):
        fvd.configure(None, 256, True)
        self.assertEqual(list(fvd.LONG_IDS), list(range(256)))
        self.assertEqual(len(fvd.EVALUATIONS), 5)
        self.assertEqual({e.experiment for e in fvd.EVALUATIONS},
                         {"lr-exp1", "lr-exp2"})
        full, *quarters = fvd.EVALUATIONS
        self.assertEqual((full.clip_start, full.clip_end, full.clips_per_video),
                         (1, 1001, 100))
        self.assertEqual([(e.clip_start, e.clip_end) for e in quarters],
                         [(0, 250), (250, 500), (500, 750), (750, 1000)])
        self.assertTrue(all(e.videos * e.clips_per_video == 6400 for e in quarters))

    def test_free_running_keeps_its_separate_frame_window(self):
        fvd.configure(None, 256, False)
        free = fvd.EVALUATIONS[-1]
        self.assertEqual((free.experiment, free.clip_start, free.clip_end,
                          free.videos, free.clips_per_video),
                         ("fr-exp1", 7, 77, 32, 7))

    def test_resume_requires_complete_finite_contiguous_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fid.csv"
            self.assertFalse(valid_csv(path, 2))
            for contents in ("frame_index,fid\n0,1\n",
                             "frame_index,fid\n0,1\n2,3\n",
                             "frame_index,fid\n0,1\n1,nan\n",
                             "frame_index,fid\n0,1\n1,\n"):
                path.write_text(contents)
                self.assertFalse(valid_csv(path, 2))
            path.write_text("frame_index,fid\n0,1\n1,2\n")
            self.assertTrue(valid_csv(path, 2))

    def test_quarter_plot_includes_more_than_four_families(self):
        import matplotlib.pyplot as plt
        fvd.configure(None, 256, True)
        runs = tuple(fvd.Run(f"family{family}_s{step}", f"family{family}-step{step}",
                             "id", Path("."), f"family{family}", step)
                     for family in range(7) for step in (1, 2))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fvd, "RUNS", runs):
            out = Path(tmp)
            (out / "lr-exp2").mkdir()
            with (out / "lr-exp2/fvd_results.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run_key", "window", "fvd"])
                writer.writeheader()
                for run in runs:
                    for evaluation in fvd.EVALUATIONS[1:]:
                        writer.writerow(dict(run_key=run.key, window=evaluation.window, fvd=10))
            with mock.patch.object(plt, "close"):
                fvd.plot_quarters(out)
                fig = plt.gcf()
                titles = {ax.get_title() for ax in fig.axes if ax.get_visible()}
            plt.close(fig)
            self.assertEqual(titles, {f"family{i}" for i in range(7)})
            self.assertTrue((out / "lr-exp2/fvd_by_window.png").stat().st_size)
            self.assertTrue((out / "lr-exp2/fvd_by_window.pdf").stat().st_size)

    def test_fid_plot_fails_if_plotter_produces_no_artifacts(self):
        run = fvd.Run("df_s1", "df-step1", "id", Path("."), "df", 1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fvd, "RUNS", (run,)):
            out = Path(tmp)
            folder = out / "lr-exp3/df_s1"
            folder.mkdir(parents=True)
            (folder / "fid_per_frame.csv").write_text(
                "frame_index,fid\n" + "".join(f"{i},1\n" for i in range(1001)))
            with mock.patch.object(fid.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "Missing FID plot artifact"):
                    fid.plot(out)

    def test_video_dataset_decodes_window_once_for_sequential_clips(self):
        class FakeCapture:
            opens = 0

            def __init__(self, _path):
                type(self).opens += 1

            def get(self, _property):
                return 20

            def set(self, _property, _value):
                return True

            def read(self):
                return True, np.zeros((4, 6, 3), dtype=np.uint8)

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.touch()
            with mock.patch("fid_metrics.dataset.cv2.VideoCapture", FakeCapture):
                dataset = VideoDataset(
                    str(video), sequence_length=5, start_frame=0, end_frame=20
                )
                dataset.transforms = lambda frames: frames
                opens_after_init = FakeCapture.opens
                self.assertEqual(len(dataset[0]), 5)
                self.assertEqual(len(dataset[1]), 5)
                self.assertEqual(FakeCapture.opens, opens_after_init + 1)


if __name__ == "__main__":
    unittest.main()
