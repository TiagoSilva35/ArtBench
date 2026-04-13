from __future__ import annotations

import argparse
import subprocess
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import (
    SEED,
    PROJECT_ROOT,
    KAGGLE_ROOT,
    SCRIPTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAINING_CSV_PATH,
    INDEX_COLUMN,
    EXPERIMENT_STAGE,
    VALID_EXPERIMENT_STAGES,
)
from src.helpers.debugger import DBG
from src.dataset_manager.HFloader import HFDatasetTorch
from src.helpers.utils import show_batch_grid, set_seed, get_device
from src.helpers.data_utils import build_image_transform, resolve_train_indices
from src.helpers.csv_handler import export_split_to_folder
from src.helpers.diffusion_helpers import GaussianDiffusion
from src.train import train_vae, train_DCGAN, train_diffusion, train_stylegan
from src.models.vae import VAE
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import PixelUNet, LatentDenoiseNetwork
from src.models.StyleGAN import StyleGAN


def load_state_dict(model, checkpoint_path: Path, device: torch.device, key: str | None = None):
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint[key] if key is not None else checkpoint
    model.load_state_dict(state_dict)
    DBG(f"Loaded checkpoint: {checkpoint_path}")
    return True


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str):
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def resolve_run_dir(base_dir: str, stage: str) -> str:
    if stage == 'dev20':
        return str(Path(base_dir) / 'dev20')
    return base_dir


def should_run_hpo(stage: str, hpo_mode: str) -> bool:
    if hpo_mode == 'on':
        return True
    if hpo_mode == 'off':
        return False
    return stage == 'dev20'


def run_hpo_subprocess(args) -> None:
    cmd = [
        sys.executable,
        '-m',
        'src.hpo_optuna',
        '--hpo-stage',
        args.stage,
        '--models',
        *args.hpo_models,
        '--trials',
        str(args.hpo_trials),
        '--epochs',
        str(args.hpo_epochs),
    ]
    if args.hpo_timeout is not None:
        cmd.extend(['--timeout', str(args.hpo_timeout)])
    if args.hpo_wandb_project:
        cmd.extend(['--wandb-project', args.hpo_wandb_project])

    DBG('Starting Optuna HPO before training...')
    DBG(f'HPO command: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def load_vae(model, train_loader, device, save_dir='vae_results', resume=True):
    final_path = Path(save_dir) / 'vae_final.pt'
    if resume and load_state_dict(model, final_path, device):
        return model, None
    latest_path = find_latest_checkpoint(Path(save_dir) / 'checkpoints', 'vae_epoch_*.pt')
    if resume and latest_path is not None and load_state_dict(model, latest_path, device):
        return model, None
    return train_vae(
        model,
        train_loader,
        device=device,
        val_loader=None,
        epochs=50,
        lr=1e-3,
        beta=1e-5,
        save_dir=save_dir,
        checkpoint_freq=10
    )


def load_dcgan(model, train_loader, device, save_dir='dcgan_results', resume=True):
    final_path = Path(save_dir) / 'dcgan_final.pt'
    checkpoint_path = final_path if resume else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        checkpoint_path = find_latest_checkpoint(Path(save_dir) / 'checkpoints', 'dcgan_epoch_*.pt')

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'optimizer_G_state_dict' in checkpoint:
            model.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if 'optimizer_D_state_dict' in checkpoint:
            model.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        DBG(f"Loaded full DCGAN checkpoint: {checkpoint_path}")
        return model, None

    return train_DCGAN(
        model,
        train_loader,
        device=device,
        val_loader=None,
        epochs=50,
        save_dir=save_dir,
        checkpoint_freq=10
    )


def load_diffusion(model, train_loader, gaussian_diffusion, device, vae, save_dir, lr, resume=True):
    final_path = Path(save_dir) / f"{model.name}_final.pt"
    if resume and load_state_dict(model, final_path, device):
        return model, None
    latest_path = find_latest_checkpoint(Path(save_dir) / 'checkpoints', f'{model.name}_epoch_*.pt')
    if resume and latest_path is not None and load_state_dict(model, latest_path, device):
        return model, None
    return train_diffusion(
        model,
        train_loader,
        gaussian_diffusion,
        device,
        val_loader=None,
        epochs=50,
        lr=lr,
        vae=vae,
        save_dir=save_dir,
        checkpoint_freq=10
    )


def load_stylegan(model, train_loader, device, save_dir='stylegan_results', resume=True):
    final_path = Path(save_dir) / "StyleGAN_final.pt"
    checkpoint_path = final_path if resume else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        checkpoint_path = find_latest_checkpoint(Path(save_dir) / "checkpoints", "StyleGAN_epoch_*.pt")

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.generator.load_state_dict(checkpoint["generator_state_dict"])
        model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if "optimizer_G_state_dict" in checkpoint:
            model.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        if "optimizer_D_state_dict" in checkpoint:
            model.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        DBG(f"Loaded full StyleGAN checkpoint: {checkpoint_path}")
        return model, None

    return train_stylegan(
        model,
        train_loader,
        device=device,
        val_loader=None,
        epochs=50,
        lr=2e-3,
        save_dir=save_dir,
        checkpoint_freq=10,
    )


set_seed(42)
device = get_device()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generative models on ArtBench-10.')
    parser.add_argument(
        '--stage',
        choices=sorted(VALID_EXPERIMENT_STAGES),
        default=EXPERIMENT_STAGE,
        help='dev20 uses training_20_percent.csv, final100 uses full training split.',
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Ignore existing checkpoints and always start training from scratch.',
    )
    parser.add_argument(
        '--hpo-mode',
        choices=['auto', 'on', 'off'],
        default='auto',
        help='Run HPO before training: auto runs only for dev20, on always runs, off never runs.',
    )
    parser.add_argument(
        '--hpo-trials',
        type=int,
        default=20,
        help='Number of Optuna trials when HPO is enabled.',
    )
    parser.add_argument(
        '--hpo-epochs',
        type=int,
        default=8,
        help='Epochs per HPO trial when HPO is enabled.',
    )
    parser.add_argument(
        '--hpo-timeout',
        type=int,
        default=None,
        help='Optional timeout in seconds for HPO.',
    )
    parser.add_argument(
        '--hpo-wandb-project',
        type=str,
        default='',
        help='Optional W&B project name for HPO logging.',
    )
    parser.add_argument(
        '--hpo-models',
        nargs='+',
        default=['all'],
        help='Models to tune before training: vae dcgan stylegan pixelunet latentdenoiser or all.',
    )
    args = parser.parse_args()
    stage = args.stage
    resume = not args.force_retrain

    DBG(f'Experiment stage: {stage}')
    DBG(f'Resume from checkpoints: {resume}')

    if should_run_hpo(stage=stage, hpo_mode=args.hpo_mode):
        run_hpo_subprocess(args)

    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds["train"]
    label_feature = train_hf.features["label"]
    class_names = list(label_feature.names)
    num_classes = len(class_names)
    train_counts = Counter(train_hf["label"])

    DBG("\nTrain class distribution:")
    for cid, name in enumerate(class_names):
        DBG(f"  {cid:2d} | {name:>15s} | {train_counts.get(cid, 0):6d}")

    transform = build_image_transform(IMAGE_SIZE)

    train_indices = resolve_train_indices(
        train_hf,
        stage=stage,
        training_csv_path=TRAINING_CSV_PATH,
        index_column=INDEX_COLUMN,
        debug_fn=DBG,
    )

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

    vae_save_dir = resolve_run_dir('vae_results', stage)
    dcgan_save_dir = resolve_run_dir('dcgan_results', stage)
    stylegan_save_dir = resolve_run_dir('stylegan_results', stage)
    pixel_save_dir = resolve_run_dir('PixelUNet_results', stage)
    latent_save_dir = resolve_run_dir('LatentDenoiseNetwork_results', stage)


    vae_model = VAE(latent_dim=16, num_channels=3, base_channels=32)
    dcgan_model = DCGAN(latent_dim=256, img_channels=3, feature_maps=32)
    stylegan_model = StyleGAN(
        z_dim=256,
        w_dim=256,
        img_resolution=IMAGE_SIZE,
        img_channels=3,
        mapping_layers=6,
    )

    gaussianDiffusion_model = GaussianDiffusion(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=device)
    
    pixelUNet_model = PixelUNet(in_channels=3, model_channels=64)
    
    latentDenoiseNetwork_model = LatentDenoiseNetwork(latent_channels=vae_model.latent_dim, model_channels=64, num_res_blocks=3)
    
    trained_vae, history = load_vae(vae_model, train_loader, device, save_dir=vae_save_dir, resume=resume)
    
    trained_DCGAN, history_DCGAN = load_dcgan(dcgan_model, train_loader, device, save_dir=dcgan_save_dir, resume=resume)
    trained_StyleGAN, history_StyleGAN = load_stylegan(stylegan_model, train_loader, device, save_dir=stylegan_save_dir, resume=resume)
    
    trained_PixelUNet, history_PixelUNet = load_diffusion(
        pixelUNet_model,
        train_loader,
        gaussianDiffusion_model,
        device,
        vae=None,
        save_dir=pixel_save_dir,
        lr=2e-4,
        resume=resume,
    )
    
    for p in vae_model.parameters():
        p.requires_grad = False
    
    vae_model.eval()
    
    trained_LatentDenoiseNetwork, history_LatentDenoiseNetwork = load_diffusion(
        latentDenoiseNetwork_model,
        train_loader,
        gaussianDiffusion_model,
        device,
        vae=vae_model,
        save_dir=latent_save_dir,
        lr=1e-4,
        resume=resume,
    )
