from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.models.StyleGanNetworks import Discriminator, Generator


class StyleGAN(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        w_dim: int = 256,
        img_resolution: int = 32,
        img_channels: int = 3,
        mapping_layers: int = 6,
        name: str = "StyleGAN",
    ):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.c_dim = 0
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.generator = Generator(
            z_dim=z_dim,
            c_dim=0,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            mapping_kwargs={"num_layers": mapping_layers},
            synthesis_kwargs={},
        )
        self.discriminator = Discriminator(
            c_dim=0,
            img_resolution=img_resolution,
            img_channels=img_channels,
            architecture="resnet",
        )

        self.optimizer_G = torch.optim.Adam(
            list(self.generator.mapping.parameters()) + list(self.generator.synthesis.parameters()),
            lr=2e-3,
            betas=(0.0, 0.99),
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2e-3,
            betas=(0.0, 0.99),
        )

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device, truncation_psi: float = 1.0) -> torch.Tensor:
        self.generator.eval()
        z = torch.randn(num_samples, self.z_dim, device=device)
        c = torch.zeros(num_samples, 0, device=device)
        return self.generator(z, c, truncation_psi=truncation_psi)

    def generate_and_save_images(self, x: torch.Tensor, output_dir: str, epoch: int, num_samples: int = 8):
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.generator.parameters()).device

        with torch.no_grad():
            n = min(num_samples, x.size(0))
            real = x[:n].to(device)
            samples = self.sample(num_samples, device=device)

        def save_grid(tensors, path, title):
            tensors = (tensors.cpu().clamp(-1, 1) + 1) / 2
            n_cols = tensors.size(0)
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2), squeeze=False)
            for ax, img in zip(axes[0], tensors):
                ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
                ax.axis("off")
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)

        def save_comparison_grid(top_row, bottom_row, path, title):
            top_row = (top_row.cpu().clamp(-1, 1) + 1) / 2
            bottom_row = (bottom_row.cpu().clamp(-1, 1) + 1) / 2
            n = min(top_row.size(0), bottom_row.size(0))
            fig, axes = plt.subplots(2, n, figsize=(n * 2, 4), squeeze=False)
            for i in range(n):
                axes[0][i].imshow(top_row[i].permute(1, 2, 0).numpy(), interpolation="nearest")
                axes[0][i].axis("off")
                axes[1][i].imshow(bottom_row[i].permute(1, 2, 0).numpy(), interpolation="nearest")
                axes[1][i].axis("off")
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)

        save_grid(samples, os.path.join(output_dir, f"samples_epoch{epoch:04d}.png"), f"Generated Samples (epoch {epoch})")
        save_comparison_grid(
            real,
            samples[:n],
            os.path.join(output_dir, f"comparison_epoch{epoch:04d}.png"),
            f"Real (first row) vs Generated (second row) - Epoch {epoch}",
        )
