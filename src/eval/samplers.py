from __future__ import annotations

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_vae(model, num_samples: int, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    remaining = int(num_samples)
    while remaining > 0:
        n = min(batch_size, remaining)
        out.append(model.sample(n, device=device))
        remaining -= n
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_dcgan(model, num_samples: int, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.generator.eval()
    out = []
    remaining = int(num_samples)
    latent_dim = model.generator.latent_dim
    while remaining > 0:
        n = min(batch_size, remaining)
        z = torch.randn(n, latent_dim, device=device)
        out.append(model.generator(z))
        remaining -= n
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_pixel_unet(model, schedule, num_samples: int, device: torch.device, image_size: int, batch_size: int = 32) -> torch.Tensor:
    model.eval()
    out = []
    remaining = int(num_samples)
    while remaining > 0:
        n = min(batch_size, remaining)
        samples = model.sample(schedule, shape=(n, 3, image_size, image_size), device=device, vae=None)
        out.append(samples)
        remaining -= n
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_latent_denoiser(
    model,
    schedule,
    vae,
    num_samples: int,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    model.eval()
    vae.eval()
    out = []
    remaining = int(num_samples)
    while remaining > 0:
        n = min(batch_size, remaining)
        samples = model.sample(
            schedule,
            shape=(n, vae.latent_dim, 2, 2),
            device=device,
            vae=vae,
        )
        out.append(samples)
        remaining -= n
    return torch.cat(out, dim=0)
