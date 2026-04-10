from __future__ import annotations

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """
    Convert model outputs in [-1, 1] float range to uint8 [0, 255].
    """
    if images.dtype != torch.float32 and images.dtype != torch.float64 and images.dtype != torch.float16:
        images = images.float()
    images = images.detach().clamp(-1.0, 1.0)
    return ((images + 1.0) * 0.5 * 255.0).round().to(torch.uint8)


def build_fid_metric(device: torch.device) -> FrechetInceptionDistance:
    # feature=2048 is the canonical FID setting with Inception-v3 pool3 features.
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)


def build_kid_metric(device: torch.device) -> KernelInceptionDistance:
    # 50 subsets of size 100 as required by the evaluation protocol.
    return KernelInceptionDistance(
        feature=2048,
        subsets=50,
        subset_size=100,
        normalize=False,
    ).to(device)
