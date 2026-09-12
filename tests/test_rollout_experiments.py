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
