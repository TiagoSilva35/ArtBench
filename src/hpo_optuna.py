from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import BATCH_SIZE, IMAGE_SIZE, INDEX_COLUMN, KAGGLE_ROOT, SEED, TRAINING_CSV_PATH
from src.dataset_manager.HFloader import HFDatasetTorch
from src.helpers.data_utils import build_image_transform, resolve_train_indices, unpack_images
from src.helpers.debugger import DBG
from src.helpers.diffusion_helpers import GaussianDiffusion
from src.helpers.utils import get_device, set_seed
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import LatentDenoiseNetwork, PixelUNet
from src.models.vae import VAE
from src.train import train_DCGAN, train_diffusion, train_stylegan, train_vae

AVAILABLE_MODELS = ["vae", "dcgan", "stylegan", "pixelunet", "latentdenoiser"]


def resolve_models(raw_models: list[str]) -> list[str]:
    tokens = [m.strip().lower() for m in raw_models if m.strip()]
    if not tokens or "all" in tokens:
        return list(AVAILABLE_MODELS)
    invalid = [m for m in tokens if m not in AVAILABLE_MODELS]
    if invalid:
        raise ValueError(f"Invalid models: {invalid}. Valid: {AVAILABLE_MODELS} or 'all'")
    deduped: list[str] = []
    seen = set()
    for m in tokens:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def build_hf_train_split():
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    return hf_ds["train"]


def split_indices(indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if len(indices) < 2:
        raise ValueError("Need at least 2 samples to split train/validation.")

    rng = np.random.RandomState(seed)
    shuffled = np.array(indices, dtype=np.int64)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)

    val_idx = shuffled[:n_val].tolist()
    train_idx = shuffled[n_val:].tolist()
    return train_idx, val_idx


def make_loaders(train_hf, transform, train_indices: list[int], val_indices: list[int], batch_size: int):
    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)
    val_ds = HFDatasetTorch(train_hf, transform=transform, indices=val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def to_latents(vae: VAE, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mu, logvar = vae.encode(images)
        return vae.reparameterize(mu, logvar)


def maybe_log(wandb_run, payload: dict) -> None:
    if wandb_run is not None:
        wandb_run.log(payload)


def train_eval_vae(params: dict, context: dict, wandb_run=None) -> float:
    train_loader, val_loader = make_loaders(
        context["train_hf"],
        context["transform"],
        context["train_indices"],
        context["val_indices"],
        int(params["batch_size"]),
    )

    device = context["device"]
    max_batches = context["max_batches"]

    model = VAE(
        latent_dim=int(params["latent_dim"]),
        num_channels=3,
        base_channels=int(params["base_channels"]),
    ).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

    best_val = float("inf")
    for epoch in range(1, context["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for bidx, batch in enumerate(train_loader):
            if max_batches is not None and bidx >= max_batches:
                break
            images = unpack_images(batch).to(device)
            loss = model.train_step(images, beta=float(params["beta"]))
            train_loss += float(loss) * images.size(0)
            train_count += images.size(0)
        avg_train = train_loss / max(train_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                if max_batches is not None and bidx >= max_batches:
                    break
                images = unpack_images(batch).to(device)
                loss = model.compute_loss(images, beta=float(params["beta"]))
                val_loss += float(loss.item()) * images.size(0)
                val_count += images.size(0)
        avg_val = val_loss / max(val_count, 1)
        best_val = min(best_val, avg_val)

        maybe_log(
            wandb_run,
            {
                "model": "vae",
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "best_val_loss": best_val,
            },
        )

    return best_val


def train_eval_dcgan(params: dict, context: dict, wandb_run=None) -> float:
    train_loader, val_loader = make_loaders(
        context["train_hf"],
        context["transform"],
        context["train_indices"],
        context["val_indices"],
        int(params["batch_size"]),
    )

    device = context["device"]
    max_batches = context["max_batches"]

    model = DCGAN(
        latent_dim=int(params["latent_dim"]),
        img_channels=3,
        feature_maps=int(params["feature_maps"]),
    ).to(device)

    model.optimizer_G = torch.optim.Adam(
        model.generator.parameters(),
        lr=float(params["lr"]),
        betas=(float(params["beta1"]), 0.999),
    )
    model.optimizer_D = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=float(params["lr"]),
        betas=(float(params["beta1"]), 0.999),
    )

    best_obj = float("inf")

    for epoch in range(1, context["epochs"] + 1):
        model.train()
        train_d = 0.0
        train_g = 0.0
        train_n = 0

        for bidx, batch in enumerate(train_loader):
            if max_batches is not None and bidx >= max_batches:
                break

            real_imgs = unpack_images(batch).to(device)
            batch_size = real_imgs.size(0)

            valid = torch.full((batch_size,), float(params["real_label"]), device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            model.optimizer_D.zero_grad()
            d_real_loss = model.criterion(model.discriminator(real_imgs).view(-1), valid)

            z = torch.randn(batch_size, model.generator.latent_dim, device=device)
            gen_imgs = model.generator(z)
            d_fake_loss = model.criterion(model.discriminator(gen_imgs.detach()).view(-1), fake_targets)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            model.optimizer_D.step()

            model.optimizer_G.zero_grad()
            g_targets = torch.ones(batch_size, device=device)
            g_loss = model.criterion(model.discriminator(gen_imgs).view(-1), g_targets)
            g_loss.backward()
            model.optimizer_G.step()

            train_d += float(d_loss.item())
            train_g += float(g_loss.item())
            train_n += 1

        avg_train_d = train_d / max(train_n, 1)
        avg_train_g = train_g / max(train_n, 1)

        model.eval()
        val_d = 0.0
        val_g = 0.0
        val_n = 0
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                if max_batches is not None and bidx >= max_batches:
                    break

                real_imgs = unpack_images(batch).to(device)
                batch_size = real_imgs.size(0)
                valid = torch.full((batch_size,), float(params["real_label"]), device=device)
                fake_targets = torch.zeros(batch_size, device=device)

                d_real_loss = model.criterion(model.discriminator(real_imgs).view(-1), valid)
                z = torch.randn(batch_size, model.generator.latent_dim, device=device)
                gen_imgs = model.generator(z)
                d_fake_loss = model.criterion(model.discriminator(gen_imgs).view(-1), fake_targets)
                g_targets = torch.ones(batch_size, device=device)
                g_loss = model.criterion(model.discriminator(gen_imgs).view(-1), g_targets)

                val_d += float((d_real_loss + d_fake_loss).item())
                val_g += float(g_loss.item())
                val_n += 1

        avg_val_d = val_d / max(val_n, 1)
        avg_val_g = val_g / max(val_n, 1)
        objective = avg_val_d + avg_val_g
        best_obj = min(best_obj, objective)

        maybe_log(
            wandb_run,
            {
                "model": "dcgan",
                "epoch": epoch,
                "train_d_loss": avg_train_d,
                "train_g_loss": avg_train_g,
                "val_d_loss": avg_val_d,
                "val_g_loss": avg_val_g,
                "objective": objective,
                "best_objective": best_obj,
            },
        )

    return best_obj


def train_eval_pixelunet(params: dict, context: dict, wandb_run=None) -> float:
    train_loader, val_loader = make_loaders(
        context["train_hf"],
        context["transform"],
        context["train_indices"],
        context["val_indices"],
        int(params["batch_size"]),
    )

    device = context["device"]
    max_batches = context["max_batches"]

    model = PixelUNet(in_channels=3, model_channels=int(params["model_channels"])).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

    schedule = GaussianDiffusion(
        num_timesteps=int(params["num_timesteps"]),
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
    )

    best_val = float("inf")
    for epoch in range(1, context["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for bidx, batch in enumerate(train_loader):
            if max_batches is not None and bidx >= max_batches:
                break
            images = unpack_images(batch).to(device)
            loss = model.train_step(images, schedule, device)
            train_loss += float(loss)
            train_n += 1
        avg_train = train_loss / max(train_n, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                if max_batches is not None and bidx >= max_batches:
                    break
                images = unpack_images(batch).to(device)
                loss = model.compute_loss(images, schedule, device)
                val_loss += float(loss.item())
                val_n += 1
        avg_val = val_loss / max(val_n, 1)
        best_val = min(best_val, avg_val)

        maybe_log(
            wandb_run,
            {
                "model": "pixelunet",
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "best_val_loss": best_val,
            },
        )

    return best_val


def get_or_prepare_vae_for_latent(context: dict) -> VAE:
    cached = context.get("latent_vae")
    if cached is not None:
        return cached

    device = context["device"]
    train_hf = context["train_hf"]
    transform = context["transform"]

    vae = VAE(latent_dim=16, num_channels=3, base_channels=32).to(device)
    candidate_paths = [
        Path("vae_results") / "dev20" / "vae_final.pt",
        Path("vae_results") / "vae_final.pt",
    ]

    loaded = False
    for ckpt in candidate_paths:
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            vae.load_state_dict(state)
            loaded = True
            DBG(f"Loaded VAE checkpoint for latent HPO: {ckpt}")
            break

    if not loaded:
        DBG("No VAE checkpoint found; training a short warmup VAE for latent HPO.")
        train_loader, _ = make_loaders(
            train_hf,
            transform,
            context["train_indices"],
            context["val_indices"],
            BATCH_SIZE,
        )
        train_vae(
            vae,
            train_loader,
            device=device,
            val_loader=None,
            epochs=max(1, int(context["latent_vae_warmup_epochs"])),
            lr=1e-3,
            beta=1e-5,
            save_dir=str(Path("hpo_results") / "warmup_latent_vae"),
            checkpoint_freq=1,
        )

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()
    context["latent_vae"] = vae
    return vae


def train_eval_latentdenoiser(params: dict, context: dict, wandb_run=None) -> float:
    train_loader, val_loader = make_loaders(
        context["train_hf"],
        context["transform"],
        context["train_indices"],
        context["val_indices"],
        int(params["batch_size"]),
    )

    device = context["device"]
    max_batches = context["max_batches"]
    vae = get_or_prepare_vae_for_latent(context)

    model = LatentDenoiseNetwork(
        latent_channels=vae.latent_dim,
        model_channels=int(params["model_channels"]),
        num_res_blocks=int(params["num_res_blocks"]),
    ).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

    schedule = GaussianDiffusion(
        num_timesteps=int(params["num_timesteps"]),
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
    )

    best_val = float("inf")
    for epoch in range(1, context["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for bidx, batch in enumerate(train_loader):
            if max_batches is not None and bidx >= max_batches:
                break
            images = unpack_images(batch).to(device)
            latents = to_latents(vae, images)
            loss = model.train_step(latents, schedule, device)
            train_loss += float(loss)
            train_n += 1
        avg_train = train_loss / max(train_n, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                if max_batches is not None and bidx >= max_batches:
                    break
                images = unpack_images(batch).to(device)
                latents = to_latents(vae, images)
                loss = model.compute_loss(latents, schedule, device)
                val_loss += float(loss.item())
                val_n += 1
        avg_val = val_loss / max(val_n, 1)
        best_val = min(best_val, avg_val)

        maybe_log(
            wandb_run,
            {
                "model": "latentdenoiser",
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "best_val_loss": best_val,
            },
        )

    return best_val


def suggest_params(trial, model_key: str) -> dict:
    if model_key == "vae":
        return {
            "latent_dim": trial.suggest_categorical("latent_dim", [16, 32, 64]),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "beta": trial.suggest_float("beta", 1e-6, 1e-3, log=True),
        }

    if model_key == "dcgan":
        return {
            "latent_dim": trial.suggest_categorical("latent_dim", [128, 256, 512]),
            "feature_maps": trial.suggest_categorical("feature_maps", [32, 64]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-4, log=True),
            "beta1": trial.suggest_float("beta1", 0.3, 0.7),
            "real_label": trial.suggest_float("real_label", 0.85, 1.0),
        }

    if model_key == "stylegan":
        return {
            "z_dim": trial.suggest_categorical("z_dim", [128, 256]),
            "w_dim": trial.suggest_categorical("w_dim", [128, 256]),
            "mapping_layers": trial.suggest_categorical("mapping_layers", [4, 6, 8]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "style_mixing_prob": trial.suggest_float("style_mixing_prob", 0.5, 0.95),
            "r1_gamma": trial.suggest_float("r1_gamma", 1.0, 10.0),
            "pl_weight": trial.suggest_float("pl_weight", 0.5, 2.5),
        }

    if model_key == "pixelunet":
        return {
            "model_channels": trial.suggest_categorical("model_channels", [32, 64, 96]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-4, log=True),
            "num_timesteps": trial.suggest_categorical("num_timesteps", [500, 1000]),
        }

    if model_key == "latentdenoiser":
        return {
            "model_channels": trial.suggest_categorical("model_channels", [32, 64, 96]),
            "num_res_blocks": trial.suggest_categorical("num_res_blocks", [2, 3, 4]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-4, log=True),
            "num_timesteps": trial.suggest_categorical("num_timesteps", [500, 1000]),
        }

    raise ValueError(f"Unsupported model_key={model_key}")


def objective_factory(model_key: str, context: dict, wandb_module, args):
    def objective(trial):
        trial_seed = SEED + trial.number
        set_seed(trial_seed)

        params = suggest_params(trial, model_key=model_key)
        wandb_run = None
        if wandb_module is not None:
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                group=f"{args.study_name}_{model_key}",
                name=f"{model_key}-trial-{trial.number:04d}",
                config={**params, "trial": trial.number, "stage": args.hpo_stage, "model": model_key},
                reinit=True,
            )

        try:
            if model_key == "vae":
                score = train_eval_vae(params, context=context, wandb_run=wandb_run)
            elif model_key == "dcgan":
                score = train_eval_dcgan(params, context=context, wandb_run=wandb_run)
            elif model_key == "pixelunet":
                score = train_eval_pixelunet(params, context=context, wandb_run=wandb_run)
            elif model_key == "latentdenoiser":
                score = train_eval_latentdenoiser(params, context=context, wandb_run=wandb_run)
            else:
                raise ValueError(f"Unsupported model_key={model_key}")

            trial.set_user_attr("params", params)
            return float(score)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    return objective


def train_final_model(model_key: str, best_params: dict, context: dict, args) -> None:
    device = context["device"]
    train_hf = context["train_hf"]
    transform = context["transform"]
    full_indices = resolve_train_indices(
        train_hf,
        stage="final100",
        training_csv_path=TRAINING_CSV_PATH,
        index_column=INDEX_COLUMN,
        debug_fn=DBG,
    )

    batch_size = int(best_params.get("batch_size", BATCH_SIZE))
    final_loader = DataLoader(
        HFDatasetTorch(train_hf, transform=transform, indices=full_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    if model_key == "vae":
        model = VAE(
            latent_dim=int(best_params.get("latent_dim", 16)),
            num_channels=3,
            base_channels=int(best_params.get("base_channels", 32)),
        )
        train_vae(
            model,
            final_loader,
            device=device,
            val_loader=None,
            epochs=args.final_epochs,
            lr=float(best_params.get("lr", 1e-3)),
            beta=float(best_params.get("beta", 1e-5)),
            save_dir="vae_results",
            checkpoint_freq=10,
        )
        return

    if model_key == "dcgan":
        model = DCGAN(
            latent_dim=int(best_params.get("latent_dim", 256)),
            img_channels=3,
            feature_maps=int(best_params.get("feature_maps", 32)),
        )
        train_DCGAN(
            model,
            final_loader,
            device=device,
            val_loader=None,
            epochs=args.final_epochs,
            save_dir="dcgan_results",
            checkpoint_freq=10,
            lr=float(best_params.get("lr", 2e-4)),
            beta1=float(best_params.get("beta1", 0.5)),
            real_label=float(best_params.get("real_label", 0.9)),
        )
        return

    if model_key == "pixelunet":
        model = PixelUNet(in_channels=3, model_channels=int(best_params.get("model_channels", 64)))
        schedule = GaussianDiffusion(
            num_timesteps=int(best_params.get("num_timesteps", 1000)),
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        train_diffusion(
            model,
            final_loader,
            schedule,
            device,
            val_loader=None,
            epochs=args.final_epochs,
            lr=float(best_params.get("lr", 2e-4)),
            vae=None,
            save_dir="PixelUNet_results",
            checkpoint_freq=10,
        )
        return

    if model_key == "latentdenoiser":
        vae_ckpt = Path("vae_results") / "vae_final.pt"
        if vae_ckpt.exists():
            vae = VAE(latent_dim=16, num_channels=3, base_channels=32).to(device)
            state = torch.load(vae_ckpt, map_location=device, weights_only=True)
            vae.load_state_dict(state)
            for p in vae.parameters():
                p.requires_grad = False
            vae.eval()
            DBG(f"Loaded VAE checkpoint for final latent denoiser training: {vae_ckpt}")
        else:
            vae = get_or_prepare_vae_for_latent(context)
        model = LatentDenoiseNetwork(
            latent_channels=vae.latent_dim,
            model_channels=int(best_params.get("model_channels", 64)),
            num_res_blocks=int(best_params.get("num_res_blocks", 3)),
        )
        schedule = GaussianDiffusion(
            num_timesteps=int(best_params.get("num_timesteps", 1000)),
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        train_diffusion(
            model,
            final_loader,
            schedule,
            device,
            val_loader=None,
            epochs=args.final_epochs,
            lr=float(best_params.get("lr", 1e-4)),
            vae=vae,
            save_dir="LatentDenoiseNetwork_results",
            checkpoint_freq=10,
        )
        return

    raise ValueError(f"Unsupported model_key={model_key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for ArtBench models. Supports VAE, DCGAN, StyleGAN, PixelUNet, and LatentDenoiser."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Models to tune: {AVAILABLE_MODELS} or 'all'.",
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per model.")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per trial.")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds per model study.")
    parser.add_argument(
        "--hpo-stage",
        choices=["dev20", "final100"],
        default="dev20",
        help="Dataset stage used for search; dev20 is recommended for tuning.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Optional batch cap per epoch for faster trials.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="artbench_hpo",
        help="Study name prefix; each model uses <prefix>_<model>.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("hpo_results") / "optuna_studies.db",
        help="SQLite path for Optuna storage.",
    )
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="After search, train final model(s) on full dataset using best params.",
    )
    parser.add_argument("--final-epochs", type=int, default=50, help="Epochs for final full-data training.")
    parser.add_argument(
        "--latent-vae-warmup-epochs",
        type=int,
        default=3,
        help="Warmup VAE epochs if latent-denoiser tuning needs latent features and no VAE checkpoint exists.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="",
        help="Optional W&B project name. If empty, W&B logging is disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)

    set_seed(SEED)
    device = get_device()

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required. Install with: pip install optuna") from exc

    use_wandb = bool(args.wandb_project.strip())
    wandb_module = None
    if use_wandb:
        try:
            import wandb as wandb_module  # type: ignore[no-redef]
        except Exception as exc:
            raise RuntimeError("W&B logging requested, but wandb is not installed. Install with: pip install wandb") from exc

    train_hf = build_hf_train_split()
    transform = build_image_transform(IMAGE_SIZE)
    stage_indices = resolve_train_indices(
        train_hf,
        stage=args.hpo_stage,
        training_csv_path=TRAINING_CSV_PATH,
        index_column=INDEX_COLUMN,
        debug_fn=DBG,
    )
    dev_train_idx, dev_val_idx = split_indices(stage_indices, val_fraction=args.val_fraction, seed=SEED)

    DBG(
        f"HPO stage={args.hpo_stage} | total={len(stage_indices)} | train={len(dev_train_idx)} | val={len(dev_val_idx)}"
    )
    DBG(f"Models selected for HPO: {models}")

    context = {
        "train_hf": train_hf,
        "transform": transform,
        "train_indices": dev_train_idx,
        "val_indices": dev_val_idx,
        "device": device,
        "epochs": args.epochs,
        "max_batches": args.max_batches_per_epoch,
        "latent_vae_warmup_epochs": args.latent_vae_warmup_epochs,
    }

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.db_path.as_posix()}"

    out_dir = Path("hpo_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for model_key in models:
        study_name = f"{args.study_name}_{model_key}"
        DBG(f"Starting study: {study_name}")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        objective = objective_factory(model_key=model_key, context=context, wandb_module=wandb_module, args=args)
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

        best_payload = {
            "model": model_key,
            "study_name": study_name,
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "hpo_stage": args.hpo_stage,
            "trials_completed": len(study.trials),
        }

        best_path = out_dir / f"best_{model_key}_params.json"
        best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        DBG(f"Saved best params: {best_path}")
        summary[model_key] = best_payload

        if args.train_final:
            DBG(f"Running final full-data training for model={model_key}")
            train_final_model(model_key, best_payload["best_params"], context=context, args=args)

    summary_path = out_dir / "best_all_models_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    DBG(f"Saved multi-model summary: {summary_path}")


if __name__ == "__main__":
    main()
