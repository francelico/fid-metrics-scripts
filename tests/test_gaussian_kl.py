"""Correctness checks for directional Gaussian KL in feature space."""
import unittest

import numpy as np
import torch

from fid_metrics.fid import calculate_forward_reverse_kl, calculate_gaussian_kls


class TestGaussianKL(unittest.TestCase):
    def test_closed_form_and_direction(self):
        mu1 = torch.tensor([0.0], dtype=torch.float64)
        mu2 = torch.tensor([1.0], dtype=torch.float64)
        sigma1 = torch.tensor([[1.0]], dtype=torch.float64)
        sigma2 = torch.tensor([[4.0]], dtype=torch.float64)
        forward, reverse = calculate_gaussian_kls(mu1, sigma1, mu2, sigma2, eps=1e-12)
        self.assertAlmostEqual(forward.item(), 0.5 * (0.5 - 1 + np.log(4)), places=10)
        self.assertAlmostEqual(reverse.item(), 0.5 * (5 - 1 - np.log(4)), places=10)

    def test_matches_numpy_for_full_covariances(self):
        mu1 = np.array([0.2, -0.3])
        mu2 = np.array([-0.5, 0.7])
        s1 = np.array([[2.0, 0.4], [0.4, 1.0]])
        s2 = np.array([[0.7, -0.1], [-0.1, 1.5]])

        def reference(a, sa, b, sb):
            diff = b - a
            return 0.5 * (
                np.trace(np.linalg.solve(sb, sa))
                + diff @ np.linalg.solve(sb, diff)
                - len(a) + np.linalg.slogdet(sb)[1] - np.linalg.slogdet(sa)[1]
            )

        forward, reverse = calculate_gaussian_kls(
            torch.from_numpy(mu1), torch.from_numpy(s1),
            torch.from_numpy(mu2), torch.from_numpy(s2), eps=1e-12,
        )
        self.assertAlmostEqual(forward.item(), reference(mu1, s1, mu2, s2), places=10)
        self.assertAlmostEqual(reverse.item(), reference(mu2, s2, mu1, s1), places=10)

    def test_singular_sample_covariance_is_finite(self):
        a = np.array([[0., 0., 0.], [1., 0., 0.]])
        b = np.array([[1., 1., 0.], [2., 1., 0.]])
        forward, reverse = calculate_forward_reverse_kl(a, b, device='cpu')
        self.assertTrue(np.isfinite(forward))
        self.assertTrue(np.isfinite(reverse))
        self.assertGreater(forward, 0)
        self.assertGreater(reverse, 0)

    def test_identical_distributions_are_zero(self):
        a = np.array([[0., 1.], [1., 0.], [2., 3.], [3., 2.]])
        forward, reverse = calculate_forward_reverse_kl(a, a, device='cpu')
        self.assertAlmostEqual(forward, 0, places=8)
        self.assertAlmostEqual(reverse, 0, places=8)

    def test_batched_moments(self):
        mu1 = torch.zeros(2, 2, dtype=torch.float64)
        mu2 = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
        sigma = torch.eye(2, dtype=torch.float64).expand(2, -1, -1)
        forward, reverse = calculate_gaussian_kls(mu1, sigma, mu2, sigma)
        np.testing.assert_allclose(forward.numpy(), [0.5, 2.0], rtol=1e-6)
        np.testing.assert_allclose(reverse.numpy(), [0.5, 2.0], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
