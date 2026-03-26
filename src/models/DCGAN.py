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
        self.discriminator = DCDescriminator(img_channels, feature_maps)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
    
    def train(self, loader, device, epochs=20):
        assert self.generator.training and self.discriminator.training, "Make sure to call model.train() before training"
        history = {'d_loss': [], 'g_loss': []}
        self.discriminator.train()
        self.generator.train()
        for epoch in range(epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            n_batches = 0
            for real_imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                
                valid = torch.ones(batch_size, device=device) # the real targets
                fake_targets = torch.zeros(batch_size, device=device) 
                
                assert self.discriminator is not None and self.generator is not None, "Make sure to initialize the generator and discriminator before training"

                # Train Discriminator
                self.optimizer_D.zero_grad()
                d_real_loss = self.criterion(self.discriminator(real_imgs), valid)
                z = torch.randn(batch_size, self.generator.latent_dim, device=device)
                gen_imgs = self.generator(z) # generate fake images
                d_fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake_targets)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                n_batches += 1
            history['d_loss'].append(d_loss_epoch / n_batches)
            history['g_loss'].append(g_loss_epoch / n_batches)
            print(f"Epoch {epoch+1}/{epochs} - D Loss: {history['d_loss'][-1]:.4f}, G Loss: {history['g_loss'][-1]:.4f}")
        return history
        
    
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
    

    
