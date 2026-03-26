import torch
import torch.nn as nn
import math
from src.helpers.debugger import DBG
from src.helpers.diffusion_helpers import SinusoidalPosEmb, ResnetBlock, GaussianDiffusion
from torch.nn import functional as F

class LatentDenoiseNetwork(nn.Module):
    """
    Denoising Network operating on latent tensors of shape [B, latent_channels, H_lat, W_lat].
    For MNIST latents from VAE, shape may be [B, 4, 4, 4].
    """
    def __init__(self, latent_channels=4, model_channels=64, num_res_blocks=3, name="LatentDenoiserNetwork"):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        self.init_conv = nn.Conv2d(latent_channels, model_channels, 3, padding=1)

        self.res_blocks = nn.ModuleList([
            ResnetBlock(model_channels, model_channels * 4)
            for _ in range(num_res_blocks)
        ])

        self.out_conv = nn.Conv2d(model_channels, latent_channels, 3, padding=1)

        #Optimizer (should be initialized only during training)
        self.optimizer = None

        # Nome do modelo, para não ter de o escrever quando o tiver de guardar
        self.name = name

    def forward(self, x, t):
        # t is shape [B]
        t_emb = self.time_embed(t)
        h = self.init_conv(x)
        for block in self.res_blocks:
            h = block(h, t_emb)
        return self.out_conv(h)

    # Scheduler é o Diffusion model utilizado (GaussianDiffusion)
    # Shape: (num_amostras, dimensões do espaço latente) -> Preciso ainda de descobrir quais são
    # Nota: num_amostrar convém ser quadrado perfeito (16, 25, ...) para ficar bem numa grid
    def sample(self, schedule, shape, vae=None):
        self.eval()
        x = torch.randn(shape, device=device)

        for step in reversed(range(schedule.num_timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t)

            alpha_t = schedule.alphas[step]
            alpha_bar_t = schedule.alphas_cumprod[step]
            beta_t = schedule.betas[step]

            if step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (1.0 / torch.sqrt(alpha_t))
                * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
                + torch.sqrt(beta_t) * noise
            )

        #Se for utilizado espaço latente para treinar, tem se ser utilizado espaço latente para amostrar
        if vae is not None:
            x = vae.decode(x)

        return x

    def generate_and_save_images(self, x: torch.Tensor, output_dir: str, epoch: int, shape: tuple[int, int, int, int], schedule, vae=None):
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.parameters()).device

        with torch.no_grad():
            # Random samples
            samples = self.sample(schedule, shape, device, vae)

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

        save_grid(samples, os.path.join(output_dir, f"samples_epoch{epoch:04d}.png"), f"Samples (epoch {epoch})")

    def compute_loss(self, x, schedule):
        # 1) Determine batch size and sample random diffusion steps.
        # 2) Use q_sample to obtain x_t and the target noise.
        # 3) Predict the noise with model(x_t, t).
        # 4) Compute the MSE loss against the sampled noise.
        t = torch.randint(0, schedule.num_timesteps, (x.shape[0],), device=device)
        eps = torch.normal(mean=0.0, std=1.0, size=x.shape, device=device)
        x_t = schedule.q_sample(x, t, noise=eps)
        eps_pred = self.forward(x_t, t)
        loss = F.mse_loss(eps_pred, eps)
        return loss

    def train_step(self, x, schedule):
        # 5) Zero gradients, backpropagate, and step the optimizer.
        # 6) Update the running loss and batch counter.
        loss = self.compute_loss(x, schedule)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# --- PIXEL UNET ---

class PixelUNet(nn.Module):
    """
    Standard UNet for Diffusion on image space.
    Fits 28x28 MNIST images.
    """
    def __init__(self, in_channels=1, model_channels=64, name="PixelUNet"):
        super().__init__()
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        time_dim = model_channels * 4

        # Initial Conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down 1: 28 -> 14
        self.down1_res = ResnetBlock(model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)

        # Down 2: 14 -> 7
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # Middle
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)

        # Up 2: 7 -> 14
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1) # 7 -> 14
        # Skip connection from down2_res is model_channels * 2
        # After concat: model_channels (up) + model_channels*2 (skip) = model_channels * 3
        self.up2_res = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels)

        # Up 1: 14 -> 28
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1) # 14 -> 28
        # Skip connection from down1_res is model_channels
        # After concat: model_channels (up) + model_channels (skip) = model_channels * 2
        self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)

        # Out
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)

        #Optimizer (should be initialized only during training)
        self.optimizer = None

        # Nome do modelo, para não ter de o escrever quando o tiver de guardar
        self.name = name

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Initial
        h_init = self.init_conv(x) # [B, C, 28, 28]

        # Down 1
        h1 = self.down1_res(h_init, t_emb) # [B, C, 28, 28]
        h1_pool = self.down1_pool(h1)      # [B, C, 14, 14]

        # Down 2
        h2 = self.down2_res(h1_pool, t_emb) # [B, 2C, 14, 14]
        h2_pool = self.down2_pool(h2)       # [B, 2C, 7, 7]

        # Middle
        h_mid = self.mid_res1(h2_pool, t_emb) # [B, 2C, 7, 7]
        h_mid = self.mid_res2(h_mid, t_emb)   # [B, 2C, 7, 7]

        # Up 2
        h_up2 = self.up2_conv(h_mid) # [B, C, 14, 14]
        h_up2 = torch.cat([h_up2, h2], dim=1) # [B, 3C, 14, 14]
        h_up2 = self.up2_res(h_up2, t_emb)   # [B, C, 14, 14]

        # Up 1
        h_up1 = self.up1_conv(h_up2) # [B, C, 28, 28]
        h_up1 = torch.cat([h_up1, h1], dim=1) # [B, 2C, 28, 28]
        h_up1 = self.up1_res(h_up1, t_emb)   # [B, C, 28, 28]

        # Out
        return self.out_conv(h_up1)

    # Scheduler é o Diffusion model utilizado (GaussianDiffusion)
    # Shape: (num_amostras, dimensões do espaço latente) -> Preciso ainda de descobrir quais são
    # Nota: num_amostrar convém ser quadrado perfeito (16, 25, ...) para ficar bem numa grid
    def sample(schedule, shape: tuple[int, int, int, int], device: torch.device, vae=None):
        if vae is not None:
            print("Warning: Latent Space not implemented for this Denoiser Network (PixelUNet)",
                "If a VAE was used to generate Latent Space for trainig this model, the sampling might given weird results")

        self.eval()
        x = torch.randn(shape, device=device)

        for step in reversed(range(schedule.num_timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t)

            alpha_t = schedule.alphas[step]
            alpha_bar_t = schedule.alphas_cumprod[step]
            beta_t = schedule.betas[step]

            if step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (1.0 / torch.sqrt(alpha_t))
                * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
                + torch.sqrt(beta_t) * noise
            )

        return x

    def generate_and_save_images(self, x: torch.Tensor, output_dir: str, epoch: int, shape: tuple[int, int, int, int], schedule, vae=None):
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.parameters()).device

        with torch.no_grad():
            # Random samples
            samples = self.sample(schedule, shape, device, vae)

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

        save_grid(samples, os.path.join(output_dir, f"samples_epoch{epoch:04d}.png"), f"Samples (epoch {epoch})")

    def compute_loss(self, x, schedule):
        # 1) Determine batch size and sample random diffusion steps.
        # 2) Use q_sample to obtain x_t and the target noise.
        # 3) Predict the noise with model(x_t, t).
        # 4) Compute the MSE loss against the sampled noise.
        t = torch.randint(0, schedule.num_timesteps, (x.shape[0],), device=device)
        eps = torch.normal(mean=0.0, std=1.0, size=x.shape, device=device)
        x_t = schedule.q_sample(x, t, noise=eps)
        eps_pred = self.forward(x_t, t)
        loss = F.mse_loss(eps_pred, eps)
        return loss


    def train_step(self, x, schedule):
        # 5) Zero gradients, backpropagate, and step the optimizer.
        # 6) Update the running loss and batch counter.
        loss = self.compute_loss(x, schedule)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
