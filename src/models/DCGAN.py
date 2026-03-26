import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps*2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )        
    
    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

class DCDescriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1)
    
class DCGAN(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super().__init__()
        self.generator = DCGenerator(latent_dim, img_channels, feature_maps)
        self.latent_dim = latent_dim
        self.discriminator = DCDescriminator(img_channels, feature_maps)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
    def generate_and_save_images(self, x: torch.Tensor, output_dir: str, epoch: int, num_samples: int = 8):
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.generator.parameters()).device

        with torch.no_grad():
            n = min(num_samples, x.size(0))
            real = x[:n].to(device)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.generator(z)

        def save_grid(tensors, path, title):
            # tensors shape: (N, C, H, W), values in [-1, 1]
            tensors = (tensors.cpu().clamp(-1, 1) + 1) / 2  # to [0, 1]
            n_cols = tensors.size(0)
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2), squeeze=False)
            for ax, img in zip(axes[0], tensors):
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis('off')
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)

        save_grid(samples, os.path.join(output_dir, f"samples_epoch{epoch:04d}.png"), f"Generated Samples (epoch {epoch})")

        comparison = torch.cat([real.cpu(), samples[:n].cpu()], dim=0)
        save_grid(
            comparison,
            os.path.join(output_dir, f"comparison_epoch{epoch:04d}.png"),
            f"Real (first row) vs Generated (second row) - Epoch {epoch}"
        )
        
    
def init_DCGAN_weights(model):
    classname = model.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def plot_DCGAN_losses(history, title="DCGAN Losses"):
    plt.figure(figsize=(7, 4))
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

    
