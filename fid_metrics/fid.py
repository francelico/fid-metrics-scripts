"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import csv

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from fid_metrics.inception import InceptionV3
from fid_metrics.inception3d import InceptionI3d
from fid_metrics.resnet3d import resnet50


def build_inception(dims):
    assert dims in list(InceptionV3.BLOCK_INDEX_BY_DIM)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    return model


# def build_inception3d(path):
#     return torch.jit.load(path)


def build_inception3d(type, path):
    if type == 'videogpt':
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load(path))
    elif type == 'styleganv':
        model = torch.jit.load(path)
    return model


def build_resnet3d(path, sample_duration=16):
    model = resnet50(
        num_classes=400,
        shortcut_type="B",
        sample_size=112,
        sample_duration=sample_duration,
        last_fc=False)
    model_sd = torch.load(path, map_location='cpu')
    model_sd_new = {}
    for k, v in model_sd['state_dict'].items():
        model_sd_new[k.replace('module.', '')] = v

    model.load_state_dict(model_sd_new)
    return model


# ---------------------------------------------------------------------------
# Numpy / scipy reference implementations (kept for parity checks & CPU-only
# fallback). The GPU path below is the default used by ``calculate_fid``.
# ---------------------------------------------------------------------------
def calculate_act_statistics_np(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f'fid calculation produces singular product, ',
            'adding {eps} to diagonal of cov estimates',
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    return np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ---------------------------------------------------------------------------
# GPU-capable torch implementations.
#
# All functions accept a leading batch dimension (act: ``[..., N, D]``,
# sigma: ``[..., D, D]``) so the per-frame case can be evaluated in one shot.
# ---------------------------------------------------------------------------
def _resolve_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _as_tensor(x, device, dtype):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


def _batched_trace(m):
    return torch.diagonal(m, dim1=-2, dim2=-1).sum(-1)


def calculate_act_statistics(act, device=None, dtype=torch.float32):
    """Mean and covariance of activations ``act`` (``[..., N, D]``).

    Runs on ``device`` (GPU by default when available). Memory: the covariance
    is formed as ``(XᵀX - N·μμᵀ)/(N-1)`` so the only D×D allocations are XᵀX and
    the outer product — the centred ``[..., N, D]`` matrix is never materialised.
    For the typical FID shape (N≈10³-10⁴, D=2048) peak extra memory is the
    2048×2048 covariance, independent of N.
    """
    device = _resolve_device(device)
    act = _as_tensor(act, device, dtype)
    n = act.shape[-2]
    mu = act.mean(dim=-2)                                   # [..., D]
    xtx = act.transpose(-1, -2) @ act                       # [..., D, D]
    sigma = (xtx - n * mu.unsqueeze(-1) * mu.unsqueeze(-2)) / (n - 1)
    return mu, sigma


def _trace_sqrt_product(sigma1, sigma2):
    """Tr((sigma1·sigma2)^{1/2}) for PSD covariances, batched over leading dims.

    Uses the identity Tr((AB)^{1/2}) = Σ sqrt(eig(AB)). We evaluate it through
    the symmetric matrix ``A^{1/2} B A^{1/2}`` (same eigenvalues, all real and
    non-negative) so only symmetric eigensolvers are needed — no complex
    ``sqrtm``, and it runs on the GPU and batches naturally.
    """
    s1 = 0.5 * (sigma1 + sigma1.transpose(-1, -2))          # symmetrise
    evals, evecs = torch.linalg.eigh(s1)
    sqrt_evals = evals.clamp_min(0).sqrt()
    sqrt_s1 = (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    m = sqrt_s1 @ sigma2 @ sqrt_s1
    m = 0.5 * (m + m.transpose(-1, -2))
    return torch.linalg.eigvalsh(m).clamp_min(0).sqrt().sum(-1)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Frechet distance ``||mu1-mu2||^2 + Tr(s1 + s2 - 2·(s1·s2)^{1/2})``.

    Torch/GPU implementation. Accepts leading batch dims (``mu: [..., D]``,
    ``sigma: [..., D, D]``) and returns a scalar (or ``[...]``) tensor.
    """
    diff = mu1 - mu2
    tr_cov = _trace_sqrt_product(sigma1, sigma2)
    return (diff * diff).sum(-1) + _batched_trace(sigma1) + _batched_trace(sigma2) - 2 * tr_cov


def calculate_fid(act1, act2, device=None, dtype=torch.float32):
    """FID between two activation sets ``act1``, ``act2`` (each ``[N, D]``).

    Accepts numpy arrays or torch tensors, runs the full computation
    (statistics + Frechet distance) on ``device`` — CUDA when available — and
    returns a python float.
    """
    device = _resolve_device(device)
    m1, s1 = calculate_act_statistics(act1, device=device, dtype=dtype)
    m2, s2 = calculate_act_statistics(act2, device=device, dtype=dtype)
    return calculate_frechet_distance(m1, s1, m2, s2).item()


def calculate_gaussian_kls(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Return ``(KL(P1 || P2), KL(P2 || P1))`` for Gaussian features.

    The inputs have shapes ``[..., D]`` and ``[..., D, D]``. A shared diagonal
    ridge is added to both sample covariances because they are singular when
    the number of clips is at most the feature dimension. If necessary, the
    ridge is increased until both Cholesky factorizations succeed. The returned
    KLs therefore describe the same regularized Gaussian pair in both
    directions. Use float64 moments for reliable results at high dimensions.
    """
    if eps <= 0:
        raise ValueError('eps must be positive')
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError('Gaussian means and covariances must have matching shapes')
    if sigma1.shape[-2:] != (mu1.shape[-1], mu1.shape[-1]):
        raise ValueError('Covariance dimensions must match the feature dimension')

    d = mu1.shape[-1]
    eye = torch.eye(d, device=sigma1.device, dtype=sigma1.dtype)
    s1 = 0.5 * (sigma1 + sigma1.transpose(-1, -2))
    s2 = 0.5 * (sigma2 + sigma2.transpose(-1, -2))
    ridge = eps
    for _ in range(12):
        l1, info1 = torch.linalg.cholesky_ex(s1 + ridge * eye)
        l2, info2 = torch.linalg.cholesky_ex(s2 + ridge * eye)
        if not (torch.any(info1 != 0) or torch.any(info2 != 0)):
            break
        ridge *= 10
    else:
        raise ValueError('Covariances are not positive definite after regularization')

    diff = (mu2 - mu1).unsqueeze(-1)
    # ||L2^-1 L1||_F^2 = tr(S2^-1 S1), including the shared ridge.
    forward_cov = torch.linalg.solve_triangular(l2, l1, upper=False).square().sum((-2, -1))
    reverse_cov = torch.linalg.solve_triangular(l1, l2, upper=False).square().sum((-2, -1))
    forward_mean = torch.linalg.solve_triangular(l2, diff, upper=False).square().sum((-2, -1))
    reverse_mean = torch.linalg.solve_triangular(l1, diff, upper=False).square().sum((-2, -1))
    logdet1 = 2 * torch.log(torch.diagonal(l1, dim1=-2, dim2=-1)).sum(-1)
    logdet2 = 2 * torch.log(torch.diagonal(l2, dim1=-2, dim2=-1)).sum(-1)
    forward = 0.5 * (forward_cov + forward_mean - d + logdet2 - logdet1)
    reverse = 0.5 * (reverse_cov + reverse_mean - d + logdet1 - logdet2)
    return forward, reverse


def calculate_forward_reverse_kl(act1, act2, device=None, eps=1e-6):
    """Gaussian KLs for two activation sets, in input order.

    Returns Python floats ``(KL(act1 || act2), KL(act2 || act1))``. Statistics
    are calculated in float64 to reduce covariance roundoff before Cholesky.
    """
    device = _resolve_device(device)
    m1, s1 = calculate_act_statistics(act1, device=device, dtype=torch.float64)
    m2, s2 = calculate_act_statistics(act2, device=device, dtype=torch.float64)
    forward, reverse = calculate_gaussian_kls(m1, s1, m2, s2, eps=eps)
    return forward.item(), reverse.item()


def calculate_fid_per_frame(act1, act2, start_frame=0, output_csv='fid_per_frame.csv',
                            device=None, dtype=torch.float32, frame_batch_size=32):
    """FID computed independently at each frame index (batched on the GPU).

    ``act1``, ``act2`` are ``[num_videos, num_frames, D]`` feature arrays (frame
    t of every video grouped together). Instead of the old python loop that
    called scipy's ``sqrtm`` once per frame, this stacks frames and runs a
    single batched cov + batched symmetric eigendecomposition per chunk.

    ``frame_batch_size`` bounds the number of frames processed at once so the
    ``[chunk, D, D]`` covariance tensors fit in memory. Several D×D tensors are
    live per frame during the eigendecomposition; measured peak at D=2048 is
    ~0.28 GB/frame in float64 (~0.14 GB/frame in float32), so a chunk of 16
    peaks near ~4.4 GB (float64) / ~2.2 GB (float32). Lower it if you OOM, raise
    it for more throughput. Note float64 ``eigh`` is very slow on pre-Volta GPUs
    (e.g. GTX 10-series) — pass ``dtype=torch.float32`` there for a ~3x speedup
    at ~1e-5 relative error.

    Writes ``(frame_index, fid)`` rows to ``output_csv`` and returns the list of
    per-frame FID scores.
    """
    device = _resolve_device(device)
    a1 = _as_tensor(act1, device, dtype)
    a2 = _as_tensor(act2, device, dtype)
    num_frames = min(a1.shape[1], a2.shape[1])

    scores = []
    for s in range(0, num_frames, frame_batch_size):
        e = min(s + frame_batch_size, num_frames)
        # [V, chunk, D] -> [chunk, V, D]; statistics are taken over the V videos.
        c1 = a1[:, s:e].transpose(0, 1)
        c2 = a2[:, s:e].transpose(0, 1)
        m1, sig1 = calculate_act_statistics(c1, device=device, dtype=dtype)
        m2, sig2 = calculate_act_statistics(c2, device=device, dtype=dtype)
        fids = calculate_frechet_distance(m1, sig1, m2, sig2)   # [chunk]
        scores.extend(fids.reshape(-1).tolist())

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'fid'])
        for t, score in enumerate(scores):
            writer.writerow([start_frame + t, float(score)])
    print(f'Wrote {num_frames} rows to {output_csv}')
    return scores


def postprocess_i2d_pred(pred):
    pred = pred[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze()
