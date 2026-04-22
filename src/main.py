from __future__ import annotations

import argparse
import json
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
from src.train import train_vae, train_DCGAN, train_diffusion
from src.models.vae import VAE
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import PixelUNet, LatentDenoiseNetwork

TRAINABLE_MODELS = ['vae', 'dcgan', 'stylegan', 'pixelunet', 'latentdenoiser']


def resolve_train_models(raw_models: list[str]) -> list[str]:
    tokens = [m.strip().lower() for m in raw_models if m.strip()]
    if not tokens or 'all' in tokens:
        return list(TRAINABLE_MODELS)

    invalid = [m for m in tokens if m not in TRAINABLE_MODELS]
    if invalid:
        raise ValueError(f"Invalid models: {invalid}. Valid: {TRAINABLE_MODELS} or 'all'")

    deduped: list[str] = []
    seen = set()
    for m in tokens:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def load_state_dict(model, checkpoint_path: Path, device: torch.device, key: str | None = None):
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint[key] if key is not None else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        DBG(f"Skipping incompatible checkpoint {checkpoint_path}: {exc}")
        return False
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


def load_best_hpo_params(model_key: str, preferred_stage: str | None = 'dev20') -> dict:
    best_path = Path('hpo_results') / f'best_{model_key}_params.json'
    if not best_path.exists():
        return {}

    try:
        payload = json.loads(best_path.read_text(encoding='utf-8'))
    except Exception as exc:
        DBG(f'Failed to read HPO params from {best_path}: {exc}')
        return {}

    if not isinstance(payload, dict):
        DBG(f'Invalid HPO params format in {best_path}: expected JSON object')
        return {}

    payload_stage = str(payload.get('hpo_stage', '')).strip().lower()
    if preferred_stage is not None and payload_stage and payload_stage != preferred_stage:
        DBG(
            f'Skipping {best_path}: expected hpo_stage={preferred_stage}, '
            f'found {payload_stage}'
        )
        return {}

    params = payload.get('best_params', payload)
    if not isinstance(params, dict):
        DBG(f'Invalid best_params in {best_path}: expected JSON object')
        return {}

    DBG(f'Using HPO params for {model_key} from {best_path}')
    return params


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


def load_vae(
    model,
    train_loader,
    device,
    save_dir='vae_results',
    resume=True,
    epochs=50,
    lr=1e-3,
    beta=1e-5,
):
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
        epochs=epochs,
        lr=lr,
        beta=beta,
        save_dir=save_dir,
        checkpoint_freq=10
    )


def load_dcgan(
    model,
    train_loader,
    device,
    save_dir='dcgan_results',
    resume=True,
    epochs=50,
    lr=2e-4,
    beta1=0.5,
    real_label=0.9,
):
    model.optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    model.optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    final_path = Path(save_dir) / 'dcgan_final.pt'
    checkpoint_path = final_path if resume else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        checkpoint_path = find_latest_checkpoint(Path(save_dir) / 'checkpoints', 'dcgan_epoch_*.pt')

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        try:
            model.generator.load_state_dict(checkpoint['generator_state_dict'])
            model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        except RuntimeError as exc:
            DBG(f"Skipping incompatible DCGAN checkpoint {checkpoint_path}: {exc}")
            checkpoint = None
        if checkpoint is None:
            checkpoint_path = None
        else:
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
        epochs=epochs,
        save_dir=save_dir,
        checkpoint_freq=10,
        lr=lr,
        beta1=beta1,
        real_label=real_label,
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
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to train: vae dcgan stylegan pixelunet latentdenoiser or all.',
    )
    parser.add_argument(
        '--hpo-best-stage',
        choices=['dev20', 'final100', 'match-stage', 'none'],
        default='dev20',
        help='Which HPO best params to apply to training. dev20 uses params searched on the 20% subset.',
    )
    args = parser.parse_args()
    stage = args.stage
    resume = not args.force_retrain
    selected_models = resolve_train_models(args.models)

    DBG(f'Experiment stage: {stage}')
    DBG(f'Resume from checkpoints: {resume}')
    DBG(f'Models selected for training: {selected_models}')

    if should_run_hpo(stage=stage, hpo_mode=args.hpo_mode):
        run_hpo_subprocess(args)

    preferred_hpo_stage = args.hpo_best_stage
    if preferred_hpo_stage == 'match-stage':
        preferred_hpo_stage = stage
    if preferred_hpo_stage == 'none':
        preferred_hpo_stage = None

    hpo_vae_params = load_best_hpo_params('vae', preferred_hpo_stage)
    hpo_dcgan_params = load_best_hpo_params('dcgan', preferred_hpo_stage)
    hpo_stylegan_params = load_best_hpo_params('stylegan', preferred_hpo_stage)
    hpo_pixel_params = load_best_hpo_params('pixelunet', preferred_hpo_stage)
    hpo_latent_params = load_best_hpo_params('latentdenoiser', preferred_hpo_stage)

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

    def make_train_loader(batch_size: int) -> DataLoader:
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_train_loader(BATCH_SIZE)

    #show_batch_grid(train_loader, class_names, n_images=36, nrow=6, title='ArtBench-10 Train Samples')

    EXPORT_ROOT = Path('exported_data')
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    export_split_to_folder(train_loader, class_names, EXPORT_ROOT / 'train_subset', max_images=500)

    vae_save_dir = resolve_run_dir('vae_results', stage)
    dcgan_save_dir = resolve_run_dir('dcgan_results', stage)
    stylegan_save_dir = resolve_run_dir('stylegan_results', stage)
    pixel_save_dir = resolve_run_dir('PixelUNet_results', stage)
    latent_save_dir = resolve_run_dir('LatentDenoiseNetwork_results', stage)


    vae_latent_dim = int(hpo_vae_params.get('latent_dim', 16))
    vae_base_channels = int(hpo_vae_params.get('base_channels', 32))
    vae_batch_size = int(hpo_vae_params.get('batch_size', BATCH_SIZE))
    vae_lr = float(hpo_vae_params.get('lr', 1e-3))
    vae_beta = float(hpo_vae_params.get('beta', 1e-5))

    dcgan_latent_dim = int(hpo_dcgan_params.get('latent_dim', 256))
    dcgan_feature_maps = int(hpo_dcgan_params.get('feature_maps', 32))
    dcgan_batch_size = int(hpo_dcgan_params.get('batch_size', BATCH_SIZE))
    dcgan_lr = float(hpo_dcgan_params.get('lr', 2e-4))
    dcgan_beta1 = float(hpo_dcgan_params.get('beta1', 0.5))
    dcgan_real_label = float(hpo_dcgan_params.get('real_label', 0.9))

    stylegan_z_dim = int(hpo_stylegan_params.get('z_dim', 256))
    stylegan_w_dim = int(hpo_stylegan_params.get('w_dim', 256))
    stylegan_mapping_layers = int(hpo_stylegan_params.get('mapping_layers', 6))
    stylegan_batch_size = int(hpo_stylegan_params.get('batch_size', BATCH_SIZE))
    stylegan_lr = float(hpo_stylegan_params.get('lr', 2e-3))

    pixel_model_channels = int(hpo_pixel_params.get('model_channels', 64))
    pixel_batch_size = int(hpo_pixel_params.get('batch_size', BATCH_SIZE))
    pixel_lr = float(hpo_pixel_params.get('lr', 2e-4))
    pixel_num_timesteps = int(hpo_pixel_params.get('num_timesteps', 1000))

    latent_model_channels = int(hpo_latent_params.get('model_channels', 64))
    latent_num_res_blocks = int(hpo_latent_params.get('num_res_blocks', 3))
    latent_batch_size = int(hpo_latent_params.get('batch_size', BATCH_SIZE))
    latent_lr = float(hpo_latent_params.get('lr', 1e-4))
    latent_num_timesteps = int(hpo_latent_params.get('num_timesteps', 1000))

    DBG(f"HPO stage filter for params: {preferred_hpo_stage}")
    DBG(f"VAE config -> latent_dim={vae_latent_dim}, base_channels={vae_base_channels}, batch_size={vae_batch_size}, lr={vae_lr}, beta={vae_beta}")
    DBG(f"DCGAN config -> latent_dim={dcgan_latent_dim}, feature_maps={dcgan_feature_maps}, batch_size={dcgan_batch_size}, lr={dcgan_lr}, beta1={dcgan_beta1}, real_label={dcgan_real_label}")
    DBG(f"StyleGAN config -> z_dim={stylegan_z_dim}, w_dim={stylegan_w_dim}, mapping_layers={stylegan_mapping_layers}, batch_size={stylegan_batch_size}, lr={stylegan_lr}")
    DBG(f"PixelUNet config -> model_channels={pixel_model_channels}, batch_size={pixel_batch_size}, lr={pixel_lr}, timesteps={pixel_num_timesteps}")
    DBG(f"LatentDenoiser config -> model_channels={latent_model_channels}, num_res_blocks={latent_num_res_blocks}, batch_size={latent_batch_size}, lr={latent_lr}, timesteps={latent_num_timesteps}")

    vae_model = VAE(latent_dim=vae_latent_dim, num_channels=3, base_channels=vae_base_channels)
    dcgan_model = DCGAN(latent_dim=dcgan_latent_dim, img_channels=3, feature_maps=dcgan_feature_maps)

    pixel_schedule = GaussianDiffusion(
        num_timesteps=pixel_num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
    )

    latent_schedule = GaussianDiffusion(
        num_timesteps=latent_num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
    )

    pixelUNet_model = PixelUNet(in_channels=3, model_channels=pixel_model_channels)

    latentDenoiseNetwork_model = LatentDenoiseNetwork(
        latent_channels=vae_model.latent_dim,
        model_channels=latent_model_channels,
        num_res_blocks=latent_num_res_blocks,
    )

    vae_loader = make_train_loader(vae_batch_size)
    dcgan_loader = make_train_loader(dcgan_batch_size)
    stylegan_loader = make_train_loader(stylegan_batch_size)
    pixel_loader = make_train_loader(pixel_batch_size)
    latent_loader = make_train_loader(latent_batch_size)

    trained_vae, history = None, None
    trained_DCGAN, history_DCGAN = None, None
    trained_StyleGAN, history_StyleGAN = None, None
    trained_PixelUNet, history_PixelUNet = None, None
    trained_LatentDenoiseNetwork, history_LatentDenoiseNetwork = None, None

    if 'vae' in selected_models or 'latentdenoiser' in selected_models:
        trained_vae, history = load_vae(
            vae_model,
            vae_loader,
            device,
            save_dir=vae_save_dir,
            resume=resume,
            epochs=50,
            lr=vae_lr,
            beta=vae_beta,
        )
    else:
        DBG('Skipping VAE training.')

    if 'dcgan' in selected_models:
        trained_DCGAN, history_DCGAN = load_dcgan(
            dcgan_model,
            dcgan_loader,
            device,
            save_dir=dcgan_save_dir,
            resume=resume,
            epochs=50,
            lr=dcgan_lr,
            beta1=dcgan_beta1,
            real_label=dcgan_real_label,
        )
    else:
        DBG('Skipping DCGAN training.')

    if 'pixelunet' in selected_models:
        trained_PixelUNet, history_PixelUNet = load_diffusion(
            pixelUNet_model,
            pixel_loader,
            pixel_schedule,
            device,
            vae=None,
            save_dir=pixel_save_dir,
            lr=pixel_lr,
            resume=resume,
        )
    else:
        DBG('Skipping PixelUNet training.')

    if 'latentdenoiser' in selected_models:
        if trained_vae is None:
            raise RuntimeError('LatentDenoiser training requires VAE to be loaded or trained.')
        for p in trained_vae.parameters():
            p.requires_grad = False
        trained_vae.eval()

        trained_LatentDenoiseNetwork, history_LatentDenoiseNetwork = load_diffusion(
            latentDenoiseNetwork_model,
            latent_loader,
            latent_schedule,
            device,
            vae=trained_vae,
            save_dir=latent_save_dir,
            lr=latent_lr,
            resume=resume,
        )
    else:
        DBG('Skipping LatentDenoiser training.')
