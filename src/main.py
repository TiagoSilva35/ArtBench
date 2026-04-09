from __future__ import annotations

import sys
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from collections import Counter
from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import SEED, PROJECT_ROOT, KAGGLE_ROOT, SCRIPTS_DIR, IMAGE_SIZE, TRAIN_FRACTION, BATCH_SIZE, TRAINING_CSV_PATH, INDEX_COLUMN
from src.helpers.debugger import DBG
from src.dataset_manager.HFloader import HFDatasetTorch
from src.helpers.utils import make_subset_indices, show_batch_grid
from src.helpers.csv_handler import load_ids_from_training_csv, export_split_to_folder
from src.helpers.diffusion_helpers import GaussianDiffusion
from src.train import train_vae, train_DCGAN, train_diffusion
from src.models.vae import VAE
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import PixelUNet, LatentDenoiseNetwork

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


set_seed(42)
device = get_device()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if __name__ == '__main__':

    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds["train"]
    label_feature = train_hf.features["label"]
    class_names = list(label_feature.names)
    num_classes = len(class_names)
    train_counts = Counter(train_hf["label"])

    DBG("\nTrain class distribution:")
    for cid, name in enumerate(class_names):
        DBG(f"  {cid:2d} | {name:>15s} | {train_counts.get(cid, 0):6d}")

    transform = T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ])

    train_ids_from_csv = load_ids_from_training_csv(TRAINING_CSV_PATH, index_column=INDEX_COLUMN)
    DBG(f'Loaded ids: {len(train_ids_from_csv)}')
    DBG(f'First 10 ids: {train_ids_from_csv[:10]}')

    if train_ids_from_csv:
        max_idx = len(train_hf) - 1
        train_indices = [i for i in train_ids_from_csv if 0 <= i <= max_idx]
        if len(train_indices) != len(train_ids_from_csv):
            DBG(
                f'Filtered {len(train_ids_from_csv) - len(train_indices)} ids outside [0, {max_idx}]'
            )
        DBG(f'Using CSV subset size: {len(train_indices)}')
    else:
        train_indices = make_subset_indices(len(train_hf), TRAIN_FRACTION, seed=SEED)
        DBG(f'Using random subset size: {len(train_indices)} (fraction={TRAIN_FRACTION})')

    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    #show_batch_grid(train_loader, class_names, n_images=36, nrow=6, title='ArtBench-10 Train Samples')

    EXPORT_ROOT = Path('exported_data')
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    export_split_to_folder(train_loader, class_names, EXPORT_ROOT / 'train_subset', max_images=500)


    vae_model = VAE(latent_dim=16, num_channels=3, base_channels=32)
    dcgan_model = DCGAN(latent_dim=256, img_channels=3, feature_maps=32)

    gaussianDiffusion_model = GaussianDiffusion(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device)
    pixelUNet_model = PixelUNet(in_channels=3, model_channels=64)
    latentDenoiseNetwork_model = LatentDenoiseNetwork(latent_channels=vae_model.latent_dim, model_channels=64, num_res_blocks=3)

    keep_last_vae = False
    device = get_device()
    if os.path.exists('vae_results/vae_final.pt') and keep_last_vae:
        model_parameters = torch.load('vae_results/vae_final.pt', map_location=device, weights_only=True)
        vae_model.load_state_dict(model_parameters)
    else:
        trained_vae, history = train_vae(
            vae_model,
            train_loader,
            device=device,
            val_loader=None,
            epochs=50,
            lr=1e-3,
            beta=1e-5,
            save_dir='vae_results',
            checkpoint_freq=10
        )
    trained_DCGAN, history_DCGAN = train_DCGAN(
        dcgan_model,
        train_loader,
        device=device,
        val_loader=None,
        epochs=50,
        save_dir='dcgan_results',
        checkpoint_freq=10
    )
    trained_PixelUNet, history_PixelUNet = train_diffusion(
        pixelUNet_model,
        train_loader,
        gaussianDiffusion_model,
        device,
        val_loader=None,
        epochs=50,
        lr=2e-4,
        vae=None,
        save_dir="PixelUNet_results",
        checkpoint_freq=10
    )
    for p in vae_model.parameters():
        p.requires_grad = False
    vae_model.eval()
    trained_LatentDenoiseNetwork, history_LatentDenoiseNetwork = train_diffusion(
        latentDenoiseNetwork_model,
        train_loader,
        gaussianDiffusion_model,
        device,
        val_loader=None,
        epochs=50,
        lr=1e-4,
        vae=vae_model,
        save_dir="LatentDenoiseNetwork_results",
        checkpoint_freq=10
    )
