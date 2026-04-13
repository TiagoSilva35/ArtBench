# Hyperparameter Search Workflow (All Models)

This file documents the current HPO implementation in `src/hpo_optuna.py`.

## Scope

HPO now supports all available models:

- `vae`
- `dcgan`
- `stylegan`
- `pixelunet`
- `latentdenoiser`

Search is implemented with Optuna (local, sequential trials).

## Stage Policy

- `dev20`: use 20% subset from `training_20_percent.csv` for fast iteration.
- `final100`: use full training split.

Recommended policy:

1. Tune in `dev20`.
2. Train final selected model in `final100`.

## Model-by-Model Search

## VAE

Objective minimized:

- Best validation ELBO-style loss (`reconstruction + beta * KL`).

Hyperparameters searched:

- `latent_dim` in `[16, 32, 64]`
- `base_channels` in `[16, 32, 64]`
- `batch_size` in `[32, 64, 128]`
- `lr` in `[1e-4, 3e-3]` (log scale)
- `beta` in `[1e-6, 1e-3]` (log scale)

Why these:

- `latent_dim`, `base_channels` control representation capacity.
- `lr` is the main convergence/stability knob.
- `beta` controls reconstruction vs latent regularization balance.
- `batch_size` affects gradient noise and optimization dynamics.

## DCGAN

Objective minimized:

- `val_d_loss + val_g_loss` (validation adversarial proxy).

Hyperparameters searched:

- `latent_dim` in `[128, 256, 512]`
- `feature_maps` in `[32, 64]`
- `batch_size` in `[32, 64, 128]`
- `lr` in `[1e-4, 5e-4]` (log scale)
- `beta1` in `[0.3, 0.7]`
- `real_label` in `[0.85, 1.0]` (label smoothing)

Why these:

- `latent_dim`, `feature_maps` control generator/discriminator capacity.
- `lr`, `beta1` strongly impact GAN stability.
- `real_label` controls discriminator confidence and helps avoid overconfidence.

## StyleGAN

Objective minimized:

- `val_d_loss + val_g_loss` computed from StyleGAN adversarial terms.

Hyperparameters searched:

- `z_dim` in `[128, 256]`
- `w_dim` in `[128, 256]`
- `mapping_layers` in `[4, 6, 8]`
- `batch_size` in `[16, 32, 64]`
- `lr` in `[1e-4, 3e-3]` (log scale)
- `style_mixing_prob` in `[0.5, 0.95]`
- `r1_gamma` in `[1.0, 10.0]`
- `pl_weight` in `[0.5, 2.5]`

Why these:

- `z_dim`, `w_dim`, `mapping_layers` tune latent expressiveness and mapping depth.
- `lr`, `batch_size` are key for adversarial stability.
- `style_mixing_prob`, `r1_gamma`, `pl_weight` are major regularization/quality controls in StyleGAN2-style training.

## PixelUNet

Objective minimized:

- Best validation denoising MSE.

Hyperparameters searched:

- `model_channels` in `[32, 64, 96]`
- `batch_size` in `[16, 32, 64]`
- `lr` in `[1e-4, 5e-4]` (log scale)
- `num_timesteps` in `[500, 1000]`

Why these:

- `model_channels` controls denoiser capacity.
- `lr` and `batch_size` affect optimization quality.
- `num_timesteps` controls diffusion schedule granularity and compute/quality tradeoff.

## LatentDenoiser

Objective minimized:

- Best validation latent denoising MSE.

Hyperparameters searched:

- `model_channels` in `[32, 64, 96]`
- `num_res_blocks` in `[2, 3, 4]`
- `batch_size` in `[16, 32, 64]`
- `lr` in `[1e-4, 5e-4]` (log scale)
- `num_timesteps` in `[500, 1000]`

Why these:

- `model_channels`, `num_res_blocks` set latent denoiser depth/width.
- `lr` and `batch_size` define optimization behavior.
- `num_timesteps` affects diffusion quality and runtime.

Notes:

- Latent-denoiser HPO uses a frozen VAE encoder.
- If no VAE checkpoint exists, a short warmup VAE is trained automatically.

## Commands

## Run HPO directly

Tune all models on dev20:

```bash
python -m src.hpo_optuna --models all --hpo-stage dev20 --trials 20 --epochs 8
```

Tune a subset:

```bash
python -m src.hpo_optuna --models vae stylegan --hpo-stage dev20 --trials 30 --epochs 10
```

Use W&B logging:

```bash
python -m src.hpo_optuna --models all --hpo-stage dev20 --trials 20 --epochs 8 --wandb-project your_project
```

Fast debug run (limit batches):

```bash
python -m src.hpo_optuna --models all --hpo-stage dev20 --trials 5 --epochs 2 --max-batches-per-epoch 20
```

Run final full-data training after search (for selected model list):

```bash
python -m src.hpo_optuna --models vae --hpo-stage dev20 --trials 30 --epochs 10 --train-final --final-epochs 50
```

## Run HPO through main training entrypoint

Automatic HPO in dev stage:

```bash
python -m src.main --stage dev20 --hpo-mode auto --hpo-models all
```

Choose models and budget:

```bash
python -m src.main --stage dev20 --hpo-mode on --hpo-models vae stylegan --hpo-trials 25 --hpo-epochs 8
```

Disable HPO:

```bash
python -m src.main --stage dev20 --hpo-mode off
```

## CLI Parameters

### `src/hpo_optuna.py`

- `--models`: model list (`all` or any of `vae dcgan stylegan pixelunet latentdenoiser`)
- `--trials`: trials per model
- `--epochs`: epochs per trial
- `--timeout`: max seconds per model study
- `--hpo-stage`: `dev20` or `final100`
- `--val-fraction`: validation split fraction
- `--max-batches-per-epoch`: batch cap per epoch for faster HPO
- `--study-name`: Optuna study prefix
- `--db-path`: SQLite path for study storage
- `--train-final`: run final full-data training after HPO
- `--final-epochs`: epochs for final training
- `--latent-vae-warmup-epochs`: warmup epochs if latent-denoiser needs VAE features and no checkpoint exists
- `--wandb-project`: optional W&B project

### `src/main.py` HPO-related flags

- `--hpo-mode`: `auto`, `on`, `off`
- `--hpo-models`: model list passed to HPO script
- `--hpo-trials`: trials passed to HPO script
- `--hpo-epochs`: epochs per trial passed to HPO script
- `--hpo-timeout`: timeout passed to HPO script
- `--hpo-wandb-project`: W&B project passed to HPO script

## Output Files

- Shared Optuna DB: `hpo_results/optuna_studies.db`
- Per-model best configs:
  - `hpo_results/best_vae_params.json`
  - `hpo_results/best_dcgan_params.json`
  - `hpo_results/best_stylegan_params.json`
  - `hpo_results/best_pixelunet_params.json`
  - `hpo_results/best_latentdenoiser_params.json`
- Aggregate summary:
  - `hpo_results/best_all_models_summary.json`
