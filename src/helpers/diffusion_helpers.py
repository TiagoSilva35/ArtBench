import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Position Embedding for time steps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # Handle odd dimension by padding if necessary, but dim should be even
        return emb

class ResnetBlock(nn.Module):
    """
    Residual Block with Time Embedding projection.
    Supports channel dimension changes with short-cut projection.
    """
    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, self.out_dim)
        )
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        # GroupNorm tends to work better for diffusion than BatchNorm
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, self.out_dim)
        self.act = nn.SiLU()

        # Shortcut for residual if dims don't match
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # Add time embedding
        time_emb = self.mlp(time_emb)
        # Expand time_emb to match spatial dimensions [B, C, 1, 1]
        h = h + time_emb[:, :, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class GaussianDiffusion:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Scheduler.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=torch.device('cpu')):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta scheduler (can be swapped for Cosine for better efficiency)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # alphas_cumprod_prev starts with 1.0 (no noise)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # posterior_variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: Add noise to x_0 at step t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_prod)*x_0, (1-alpha_prod)*I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse diffusion step: Sample x_{t-1} given x_t and the model.
        """
        betas_t = self._get_index(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))

        # Predict noise
        predicted_noise = model(x, t)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Clip step to be safe, or just add variance
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Sample all steps from pure noise to reconstruct an image in latent space.
        """
        model.eval()
        x = torch.randn(shape).to(self.device)
        # Reverse loop from T-1 back to 0
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i)
        return x

    def _get_index(self, tensor, t, x_shape):
        """Get value at index t and expand to match x_shape."""
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))
