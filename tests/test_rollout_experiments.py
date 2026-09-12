"""Campaign frame semantics and resume validation, without GPUs or datasets."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_rollout_fvd_10f_experiments as fvd
from evaluate_rollout_fid_per_frame import valid_csv


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


if __name__ == "__main__":
    unittest.main()
