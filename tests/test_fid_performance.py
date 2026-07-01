"""Performance + correctness tests for the FID computation.

Run directly (no pytest dependency required):

    export PYTHONPATH=`pwd`:$PYTHONPATH
    python tests/test_fid_performance.py

It times the two halves of ``calculate_fid`` — ``calculate_act_statistics``
(cov) and ``calculate_frechet_distance`` (matrix sqrt) — for the numpy/scipy
reference and the torch implementation, on every available device, and checks
the torch path agrees with the reference.

The reference numpy path is the original bottleneck: ``scipy.linalg.sqrtm`` on a
D×D matrix runs single-threaded on the CPU. The timing table makes the win from
the GPU eigendecomposition path obvious.
"""
import time
import unittest

import numpy as np
import torch

from fid_metrics.fid import (
    calculate_act_statistics,
    calculate_act_statistics_np,
    calculate_fid,
    calculate_frechet_distance,
    calculate_frechet_distance_np,
)

# Typical FID shape: D=2048 (InceptionV3 pool3), N ~ a few thousand frames.
N = 2000
D = 2048
SEED = 0


def _dummy_acts(n=N, d=D, seed=SEED):
    """Two activation sets with a genuine mean/covariance gap (non-zero FID)."""
    rng = np.random.default_rng(seed)
    a1 = rng.standard_normal((n, d)).astype(np.float32)
    # give the second distribution a shifted mean and a different scale
    a2 = (rng.standard_normal((n, d)) * 1.3 + 0.25).astype(np.float32)
    return a1, a2


def _time(fn, *, warmup=0, repeats=1, sync_device=None):
    for _ in range(warmup):
        fn()
    if sync_device is not None and sync_device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    if sync_device is not None and sync_device.type == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    return out, dt


def _devices():
    devs = [torch.device('cpu')]
    if torch.cuda.is_available():
        devs.append(torch.device('cuda'))
    return devs


class TestFidPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a1, cls.a2 = _dummy_acts()
        # scipy/numpy reference FID (float64) — the number everything is checked against.
        m1, s1 = calculate_act_statistics_np(cls.a1.astype(np.float64))
        m2, s2 = calculate_act_statistics_np(cls.a2.astype(np.float64))
        cls.ref_fid = calculate_frechet_distance_np(m1, s1, m2, s2)
        print(f'\n[setup] N={N} D={D}  reference FID = {cls.ref_fid:.6f}')

    def test_timing_and_correctness(self):
        print('\n{:<8} {:<10} {:>12} {:>12} {:>12}'.format(
            'device', 'backend', 'stats (ms)', 'frechet (ms)', 'total (ms)'))
        print('-' * 58)

        # ---- numpy / scipy reference (CPU only) -------------------------------
        def np_stats():
            m1, s1 = calculate_act_statistics_np(self.a1.astype(np.float64))
            m2, s2 = calculate_act_statistics_np(self.a2.astype(np.float64))
            return m1, s1, m2, s2

        (m1, s1, m2, s2), t_stats_np = _time(np_stats)
        _, t_frechet_np = _time(lambda: calculate_frechet_distance_np(m1, s1, m2, s2))
        print('{:<8} {:<10} {:>12.2f} {:>12.2f} {:>12.2f}'.format(
            'cpu', 'numpy', t_stats_np * 1e3, t_frechet_np * 1e3,
            (t_stats_np + t_frechet_np) * 1e3))

        # ---- torch path on every available device -----------------------------
        for dev in _devices():
            def torch_stats():
                mm1, ss1 = calculate_act_statistics(self.a1, device=dev)
                mm2, ss2 = calculate_act_statistics(self.a2, device=dev)
                return mm1, ss1, mm2, ss2

            (mm1, ss1, mm2, ss2), t_stats = _time(
                torch_stats, warmup=1 if dev.type == 'cuda' else 0, sync_device=dev)
            _, t_frechet = _time(
                lambda: calculate_frechet_distance(mm1, ss1, mm2, ss2),
                warmup=1 if dev.type == 'cuda' else 0, sync_device=dev)
            print('{:<8} {:<10} {:>12.2f} {:>12.2f} {:>12.2f}'.format(
                dev.type, 'torch', t_stats * 1e3, t_frechet * 1e3,
                (t_stats + t_frechet) * 1e3))

            fid = calculate_fid(self.a1, self.a2, device=dev)
            # torch eigendecomposition path must match the scipy reference.
            self.assertTrue(
                np.isclose(fid, self.ref_fid, rtol=1e-4, atol=1e-3),
                f'torch FID {fid} on {dev} != reference {self.ref_fid}',
            )

    def test_stats_match_reference(self):
        m1_np, s1_np = calculate_act_statistics_np(self.a1.astype(np.float64))
        m1_t, s1_t = calculate_act_statistics(self.a1, device='cpu')
        np.testing.assert_allclose(m1_t.numpy(), m1_np, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s1_t.numpy(), s1_np, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
