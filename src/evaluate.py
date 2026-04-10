from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

# Ensure project root is importable when running: python src\evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import IMAGE_SIZE, KAGGLE_ROOT
from src.dataset_manager.HFloader import HFDatasetTorch
from src.eval.metrics import build_fid_metric, build_kid_metric, images_to_uint8
from src.eval.samplers import (
    sample_dcgan,
    sample_latent_denoiser,
    sample_pixel_unet,
    sample_vae,
    set_global_seed,
)
from src.helpers.diffusion_helpers import GaussianDiffusion
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import LatentDenoiseNetwork, PixelUNet
from src.models.vae import VAE


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_real_images(
    image_size: int,
    n_samples: int,
    pool_seed: int,
    batch_size: int,
) -> torch.Tensor:
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds["train"]
    transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    full_ds = HFDatasetTorch(train_hf, transform=transform)
    if n_samples > len(full_ds):
        raise ValueError(f"Requested {n_samples} real images but dataset has only {len(full_ds)}.")

    rng = np.random.default_rng(pool_seed)
    indices = rng.choice(len(full_ds), size=n_samples, replace=False).tolist()
    loader = DataLoader(
        Subset(full_ds, indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    out = []
    for batch in loader:
        if len(batch) == 2:
            x, _ = batch
        else:
            x, _, _ = batch
        out.append(x)
    return torch.cat(out, dim=0)


def save_qualitative_grid(images: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = (images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis[: nrow * nrow], nrow=nrow)
    save_image(grid, out_path)


def load_vae_from_checkpoint(device: torch.device, checkpoint_path: Path) -> VAE:
    model = VAE(latent_dim=16, num_channels=3, base_channels=32).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_dcgan_from_checkpoint(device: torch.device, checkpoint_path: Path) -> DCGAN:
    model = DCGAN(latent_dim=256, img_channels=3, feature_maps=32).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.generator.load_state_dict(state["generator_state_dict"])
    model.discriminator.load_state_dict(state["discriminator_state_dict"])
    model.eval()
    return model


def load_pixel_unet_from_checkpoint(device: torch.device, checkpoint_path: Path) -> PixelUNet:
    model = PixelUNet(in_channels=3, model_channels=64).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_latent_denoiser_from_checkpoint(device: torch.device, checkpoint_path: Path) -> LatentDenoiseNetwork:
    model = LatentDenoiseNetwork(latent_channels=16, model_channels=64, num_res_blocks=3).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_single_eval(
    model_key: str,
    seed: int,
    real_images: torch.Tensor,
    device: torch.device,
    num_samples: int,
    gen_batch_size: int,
    metric_batch_size: int,
    metrics_device: torch.device,
    output_dir: Path,
    vae_model: VAE | None = None,
) -> tuple[float, float, float]:
    set_global_seed(seed)

    fid_metric = build_fid_metric(metrics_device)
    kid_metric = build_kid_metric(metrics_device)
    n_vis = 64
    generated_vis_chunks: list[torch.Tensor] = []

    def update_metrics_with_generated_batch(gen_batch: torch.Tensor, start_idx: int) -> None:
        local_offset = 0
        while local_offset < gen_batch.size(0):
            n = min(metric_batch_size, gen_batch.size(0) - local_offset)
            real_batch = real_images[start_idx + local_offset : start_idx + local_offset + n]
            gen_chunk = gen_batch[local_offset : local_offset + n]
            real_u8 = images_to_uint8(real_batch.to(metrics_device))
            gen_u8 = images_to_uint8(gen_chunk.to(metrics_device))
            fid_metric.update(real_u8, real=True)
            fid_metric.update(gen_u8, real=False)
            kid_metric.update(real_u8, real=True)
            kid_metric.update(gen_u8, real=False)
            local_offset += n

    processed = 0
    if model_key == "vae":
        if vae_model is None:
            raise ValueError("VAE model is required for model_key='vae'")
        while processed < num_samples:
            n = min(gen_batch_size, num_samples - processed)
            generated = sample_vae(vae_model, num_samples=n, device=device, batch_size=n)
            update_metrics_with_generated_batch(generated, processed)
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
    elif model_key == "dcgan":
        dcgan = load_dcgan_from_checkpoint(device, Path("dcgan_results") / "dcgan_final.pt")
        while processed < num_samples:
            n = min(gen_batch_size, num_samples - processed)
            generated = sample_dcgan(dcgan, num_samples=n, device=device, batch_size=n)
            update_metrics_with_generated_batch(generated, processed)
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
        del dcgan
    elif model_key == "pixelunet":
        pixel = load_pixel_unet_from_checkpoint(device, Path("PixelUNet_results") / "PixelUNet_final.pt")
        schedule = GaussianDiffusion(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        while processed < num_samples:
            n = min(min(gen_batch_size, 32), num_samples - processed)
            generated = sample_pixel_unet(
                pixel,
                schedule,
                num_samples=n,
                device=device,
                image_size=IMAGE_SIZE,
                batch_size=n,
            )
            update_metrics_with_generated_batch(generated, processed)
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
        del pixel
        del schedule
    elif model_key == "latentdenoiser":
        if vae_model is None:
            raise ValueError("VAE model is required for model_key='latentdenoiser'")
        latent = load_latent_denoiser_from_checkpoint(
            device, Path("LatentDenoiseNetwork_results") / "LatentDenoiserNetwork_final.pt"
        )
        schedule = GaussianDiffusion(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        while processed < num_samples:
            n = min(min(gen_batch_size, 32), num_samples - processed)
            generated = sample_latent_denoiser(
                latent,
                schedule,
                vae=vae_model,
                num_samples=n,
                device=device,
                batch_size=n,
            )
            update_metrics_with_generated_batch(generated, processed)
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
        del latent
        del schedule
    else:
        raise ValueError(f"Unknown model_key '{model_key}'")

    fid = float(fid_metric.compute().item())
    fid_metric.reset()

    kid_mean, kid_std = kid_metric.compute()
    kid_mean_value = float(kid_mean.item())
    kid_std_value = float(kid_std.item())
    kid_metric.reset()

    run_dir = output_dir / model_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_vis = torch.cat(generated_vis_chunks, dim=0) if generated_vis_chunks else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE)
    save_qualitative_grid(generated_vis, run_dir / "generated_grid.png")
    save_qualitative_grid(real_images, run_dir / "real_grid.png")

    del generated_vis
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fid, kid_mean_value, kid_std_value


def aggregate(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generative models with FID/KID protocol.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vae", "dcgan", "pixelunet", "latentdenoiser"],
        choices=["vae", "dcgan", "pixelunet", "latentdenoiser"],
        help="Models to evaluate.",
    )
    parser.add_argument("--num-samples", type=int, default=5000, help="Generated and real sample count.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random-seed repetitions.")
    parser.add_argument("--seed-start", type=int, default=42, help="Starting seed for repetitions.")
    parser.add_argument("--real-pool-seed", type=int, default=1234, help="Seed for fixed real-image pool sampling.")
    parser.add_argument("--real-batch-size", type=int, default=256, help="Batch size for loading real images.")
    parser.add_argument("--gen-batch-size", type=int, default=128, help="Batch size for generators.")
    parser.add_argument(
        "--metric-batch-size",
        type=int,
        default=64,
        help="Chunk size used when updating FID/KID (lower this to reduce memory usage).",
    )
    parser.add_argument(
        "--metrics-device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device used by FID/KID computation. Use cpu to avoid CUDA OOM.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"), help="Output directory.")
    args = parser.parse_args()

    device = get_device()
    if args.metrics_device == "auto":
        metrics_device = device
    elif args.metrics_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --metrics-device cuda, but CUDA is not available.")
        metrics_device = torch.device("cuda")
    else:
        metrics_device = torch.device("cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Metrics device: {metrics_device}")
    print(f"Preparing fixed real pool with {args.num_samples} images...")
    real_images = load_real_images(
        image_size=IMAGE_SIZE,
        n_samples=args.num_samples,
        pool_seed=args.real_pool_seed,
        batch_size=args.real_batch_size,
    )

    vae_model = None
    if "vae" in args.models or "latentdenoiser" in args.models:
        vae_path = Path("vae_results") / "vae_final.pt"
        if not vae_path.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {vae_path}")
        vae_model = load_vae_from_checkpoint(device, vae_path)

    summary_rows: list[dict[str, str | float | int]] = []
    per_run_rows: list[dict[str, str | float | int]] = []
    eval_seeds = [args.seed_start + i for i in range(args.seeds)]

    for model_key in args.models:
        print(f"\nEvaluating model: {model_key}")
        fid_values: list[float] = []
        kid_mean_values: list[float] = []
        kid_std_values: list[float] = []

        for seed in eval_seeds:
            print(f"  - seed={seed}")
            fid, kid_mean, kid_std = run_single_eval(
                model_key=model_key,
                seed=seed,
                real_images=real_images,
                device=device,
                num_samples=args.num_samples,
                gen_batch_size=args.gen_batch_size,
                metric_batch_size=args.metric_batch_size,
                metrics_device=metrics_device,
                output_dir=args.output_dir,
                vae_model=vae_model,
            )
            fid_values.append(fid)
            kid_mean_values.append(kid_mean)
            kid_std_values.append(kid_std)
            per_run_rows.append(
                {
                    "model": model_key,
                    "seed": seed,
                    "fid": fid,
                    "kid_mean": kid_mean,
                    "kid_std": kid_std,
                }
            )
            print(f"    FID={fid:.4f} | KID={kid_mean:.6f} ± {kid_std:.6f}")

        fid_mean, fid_std = aggregate(fid_values)
        kid_mean_of_means, kid_std_across_seeds = aggregate(kid_mean_values)
        summary_rows.append(
            {
                "model": model_key,
                "num_samples": args.num_samples,
                "num_seeds": args.seeds,
                "fid_mean": fid_mean,
                "fid_std": fid_std,
                "kid_mean": kid_mean_of_means,
                "kid_std_across_seeds": kid_std_across_seeds,
                "kid_subset_std_mean": float(np.mean(kid_std_values)),
            }
        )
        print(
            f"  Final ({model_key}): "
            f"FID={fid_mean:.4f} ± {fid_std:.4f} | "
            f"KID={kid_mean_of_means:.6f} ± {kid_std_across_seeds:.6f}"
        )

    per_run_csv = args.output_dir / "per_run_metrics.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "fid", "kid_mean", "kid_std"])
        writer.writeheader()
        writer.writerows(per_run_rows)

    summary_csv = args.output_dir / "summary_metrics.csv"
    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "num_samples",
                "num_seeds",
                "fid_mean",
                "fid_std",
                "kid_mean",
                "kid_std_across_seeds",
                "kid_subset_std_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved per-run metrics to: {per_run_csv}")
    print(f"Saved summary metrics to: {summary_csv}")


if __name__ == "__main__":
    main()
