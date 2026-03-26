import os
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from src.helpers.debugger import DBG

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 256, num_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # we dont do batch normalization in the final layer because
        # it introduces aditional stochasticity beyound the sampling stochasticity
        self.enc_out_dim = base_channels * 8 * 2 * 2
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Adam optimizer (create after modules so parameters are registered)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def encode(self, x: torch.Tensor):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.base_channels * 8, 2, 2)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        # encode, reparameterize, decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Decode to get reconstruction
        recon = self.decode(z)
        return recon, mu, logvar

    def normal_log_pdf(self, sample: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # used to calculate log N(sample | mu, exp(logvar))
        log2pi = torch.log(torch.tensor(2 * torch.pi, device=sample.device))
        return -0.5*((sample - mu)**2*torch.exp(-logvar) + logvar + log2pi)
    
    def compute_loss(self, x, beta=1.0):
        # negative ELBO loss    
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # average over batch
        # DBG(f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
        return recon_loss + beta * kl_loss
    
    def generate_and_save_images(self, x: torch.Tensor, output_dir: str, epoch: int, num_samples: int = 8):
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.parameters()).device

        with torch.no_grad():
            # Reconstructions
            n = min(num_samples, x.size(0))
            original = x[:n].to(device)
            recon, _, _ = self.forward(x[:n])
            # Random samples
            samples = self.sample(num_samples, device)

        def save_grid(tensors, path, title):
            # tensors shape: (N, C, H, W), values in [-1, 1]
            tensors = (tensors.cpu().clamp(-1, 1) + 1) / 2  # to [0, 1]
            n = tensors.size(0)
            fig, axes = plt.subplots(1, n, figsize=(n * 2, 2), squeeze=False)
            for ax, img in zip(axes[0], tensors):
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis('off')
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)
        

        save_grid(recon, os.path.join(output_dir, f"reconstructions_epoch{epoch:04d}.png"), f"Reconstructions (epoch {epoch})")
        save_grid(samples, os.path.join(output_dir, f"samples_epoch{epoch:04d}.png"), f"Samples (epoch {epoch})")

        comparison = torch.cat([original.cpu(), recon.cpu()], dim=0)
        save_grid(comparison, os.path.join(output_dir, f"comparison_epoch{epoch:04d}.png"),
                  f"Original (top row) vs Reconstructed (bottom row) - Epoch {epoch}")

    def train_step(self, x, beta=1.0):
        self.optimizer.zero_grad()
        loss = self.compute_loss(x, beta)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        



