from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from prdc import compute_prdc
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

# Ensure project root is importable when running: python src\evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import IMAGE_SIZE, KAGGLE_ROOT
from src.dataset_manager.HFloader import HFDatasetTorch
from src.eval.metrics import build_fid_metric, build_kid_metric, images_to_uint8, lerp, slerp
from src.eval.samplers import (
    sample_dcgan,
    sample_latent_denoiser,
    sample_pixel_unet,
    sample_stylegan,
    sample_vae,
    set_global_seed,
)
from src.helpers.diffusion_helpers import GaussianDiffusion
from src.helpers.data_utils import build_image_transform, unpack_images
from src.helpers.utils import get_device
from src.models.DCGAN import DCGAN
from src.models.DenoiserNetworks import LatentDenoiseNetwork, PixelUNet
from src.models.StyleGAN import StyleGAN
from src.models.vae import VAE


def load_real_images(
    image_size: int,
    n_samples: int,
    pool_seed: int,
    batch_size: int,
) -> torch.Tensor:
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds["train"]
    transform = build_image_transform(image_size)
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
        out.append(unpack_images(batch))
    return torch.cat(out, dim=0)


def save_qualitative_grid(images: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = (images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis[: nrow * nrow], nrow=nrow)
    save_image(grid, out_path)


def export_images_to_folder(images: torch.Tensor, out_dir: Path, start_idx: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = (images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    for i in range(vis.size(0)):
        save_image(vis[i], out_dir / f"img_{start_idx + i:06d}.png")


def compute_prdc_metrics(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    nearest_k: int,
) -> tuple[float, float, float, float]:
    # PRDC expects 2D feature arrays; here we use flattened image features.
    real_features = real_images.detach().cpu().float().reshape(real_images.size(0), -1).numpy()
    fake_features = generated_images.detach().cpu().float().reshape(generated_images.size(0), -1).numpy()
    metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)
    return (
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["density"]),
        float(metrics["coverage"]),
    )


def plot_coverage_vs_quality(results: dict[str, tuple[float, float]], output_path: Path) -> None:
    """
    results: {model_name: (quality, coverage)} where quality=precision.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "vae": "#e41a1c",
        "dcgan": "#377eb8",
        "stylegan": "#ff7f00",
        "pixelunet": "#4daf4a",
        "latentdenoiser": "#984ea3",
    }
    markers = {
        "vae": "o",
        "dcgan": "s",
        "stylegan": "P",
        "pixelunet": "^",
        "latentdenoiser": "D",
    }

    for model, (quality, coverage) in results.items():
        ax.scatter(
            coverage,
            quality,
            c=colors.get(model, "gray"),
            marker=markers.get(model, "o"),
            s=150,
            label=model,
            edgecolors="black",
            linewidths=1.5,
        )
        ax.annotate(
            model,
            (coverage, quality),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Quality (Precision)", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Quality vs Coverage Trade-off")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _count_indexed_blocks(state: dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices: set[int] = set()
    for key in state:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return (max(indices) + 1) if indices else 0


def infer_vae_architecture(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    if "fc_mu.weight" not in state:
        raise KeyError("Missing key 'fc_mu.weight' in VAE checkpoint.")
    latent_dim = int(state["fc_mu.weight"].shape[0])
    enc_out_dim = int(state["fc_mu.weight"].shape[1])
    if enc_out_dim % 8 != 0:
        raise ValueError(f"Cannot infer VAE base_channels from encoder dim {enc_out_dim}.")
    base_channels = enc_out_dim // 8
    return latent_dim, base_channels


def infer_dcgan_architecture(state: dict[str, dict[str, torch.Tensor]]) -> tuple[int, int, int]:
    g_state = state["generator_state_dict"]
    first_weight = g_state["net.0.weight"]
    last_weight = g_state["net.9.weight"]
    latent_dim = int(first_weight.shape[0])
    feature_maps = int(first_weight.shape[1]) // 4
    img_channels = int(last_weight.shape[1])
    return latent_dim, img_channels, feature_maps


def infer_stylegan_architecture(state: dict[str, dict[str, torch.Tensor]]) -> tuple[int, int, int, int, int]:
    g_state = state["generator_state_dict"]

    fc_keys = [k for k in g_state if re.fullmatch(r"mapping\.fc\d+\.weight", k)]
    if not fc_keys:
        raise KeyError("Missing mapping FC weights in StyleGAN checkpoint.")
    mapping_layers = len(fc_keys)
    z_dim = int(g_state["mapping.fc0.weight"].shape[1])
    w_dim = int(g_state["mapping.fc0.weight"].shape[0])

    torgb_keys = [k for k in g_state if k.endswith("torgb.weight")]
    if not torgb_keys:
        raise KeyError("Missing ToRGB weights in StyleGAN checkpoint.")
    img_channels = int(g_state[torgb_keys[-1]].shape[0])

    noise_keys = [k for k in g_state if k.endswith("noise_const")]
    if not noise_keys:
        raise KeyError("Missing noise_const buffers in StyleGAN checkpoint.")
    img_resolution = max(int(g_state[k].shape[-1]) for k in noise_keys)

    return z_dim, w_dim, img_resolution, img_channels, mapping_layers


def infer_pixel_unet_architecture(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    init_weight = state["init_conv.weight"]
    model_channels = int(init_weight.shape[0])
    in_channels = int(init_weight.shape[1])
    return in_channels, model_channels


def infer_latent_denoiser_architecture(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    init_weight = state["init_conv.weight"]
    model_channels = int(init_weight.shape[0])
    latent_channels = int(init_weight.shape[1])
    num_res_blocks = _count_indexed_blocks(state, "res_blocks")
    if num_res_blocks == 0:
        raise ValueError("Cannot infer num_res_blocks from latent denoiser checkpoint.")
    return latent_channels, model_channels, num_res_blocks


def load_vae_from_checkpoint(device: torch.device, checkpoint_path: Path) -> VAE:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim, base_channels = infer_vae_architecture(state)
    model = VAE(latent_dim=latent_dim, num_channels=3, base_channels=base_channels).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_dcgan_from_checkpoint(device: torch.device, checkpoint_path: Path) -> DCGAN:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim, img_channels, feature_maps = infer_dcgan_architecture(state)
    model = DCGAN(latent_dim=latent_dim, img_channels=img_channels, feature_maps=feature_maps).to(device)
    model.generator.load_state_dict(state["generator_state_dict"])
    model.discriminator.load_state_dict(state["discriminator_state_dict"])
    model.eval()
    return model


def load_stylegan_from_checkpoint(device: torch.device, checkpoint_path: Path) -> StyleGAN:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    z_dim, w_dim, img_resolution, img_channels, mapping_layers = infer_stylegan_architecture(state)
    model = StyleGAN(
        z_dim=z_dim,
        w_dim=w_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        mapping_layers=mapping_layers,
    ).to(device)
    model.generator.load_state_dict(state["generator_state_dict"])
    model.discriminator.load_state_dict(state["discriminator_state_dict"])
    model.eval()
    return model


def load_pixel_unet_from_checkpoint(device: torch.device, checkpoint_path: Path) -> PixelUNet:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    in_channels, model_channels = infer_pixel_unet_architecture(state)
    model = PixelUNet(in_channels=in_channels, model_channels=model_channels).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_latent_denoiser_from_checkpoint(device: torch.device, checkpoint_path: Path) -> LatentDenoiseNetwork:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_channels, model_channels, num_res_blocks = infer_latent_denoiser_architecture(state)
    model = LatentDenoiseNetwork(
        latent_channels=latent_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
    ).to(device)
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
    real_images_dir: Path,
    prdc_nearest_k: int,
    vae_model: VAE | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    set_global_seed(seed)

    fid_metric = build_fid_metric(metrics_device)
    kid_metric = build_kid_metric(metrics_device)
    n_vis = 64
    generated_vis_chunks: list[torch.Tensor] = []
    generated_prdc_chunks: list[torch.Tensor] = []
    run_dir = output_dir / model_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir = run_dir / "generated_images"
    if generated_images_dir.exists():
        shutil.rmtree(generated_images_dir)
    generated_images_dir.mkdir(parents=True, exist_ok=True)

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
            export_images_to_folder(generated, generated_images_dir, start_idx=processed)
            generated_prdc_chunks.append(generated.detach().cpu())
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
    elif model_key == "dcgan":
        dcgan = load_dcgan_from_checkpoint(device, Path("dcgan_results") / "dcgan_final.pt")
        while processed < num_samples:
            n = min(gen_batch_size, num_samples - processed)
            generated = sample_dcgan(dcgan, num_samples=n, device=device, batch_size=n)
            update_metrics_with_generated_batch(generated, processed)
            export_images_to_folder(generated, generated_images_dir, start_idx=processed)
            generated_prdc_chunks.append(generated.detach().cpu())
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
        del dcgan
    elif model_key == "stylegan":
        stylegan = load_stylegan_from_checkpoint(device, Path("stylegan_results") / "StyleGAN_final.pt")
        while processed < num_samples:
            n = min(gen_batch_size, num_samples - processed)
            generated = sample_stylegan(stylegan, num_samples=n, device=device, batch_size=n)
            update_metrics_with_generated_batch(generated, processed)
            export_images_to_folder(generated, generated_images_dir, start_idx=processed)
            generated_prdc_chunks.append(generated.detach().cpu())
            if sum(x.size(0) for x in generated_vis_chunks) < n_vis:
                generated_vis_chunks.append(generated[: n_vis - sum(x.size(0) for x in generated_vis_chunks)].detach().cpu())
            processed += n
        del stylegan
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
            export_images_to_folder(generated, generated_images_dir, start_idx=processed)
            generated_prdc_chunks.append(generated.detach().cpu())
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
            export_images_to_folder(generated, generated_images_dir, start_idx=processed)
            generated_prdc_chunks.append(generated.detach().cpu())
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

    generated_prdc = torch.cat(generated_prdc_chunks, dim=0)
    precision, recall, density, coverage = compute_prdc_metrics(
        real_images=real_images,
        generated_images=generated_prdc,
        nearest_k=prdc_nearest_k,
    )

    generated_vis = torch.cat(generated_vis_chunks, dim=0) if generated_vis_chunks else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE)
    save_qualitative_grid(generated_vis, run_dir / "generated_grid.png")
    save_qualitative_grid(real_images, run_dir / "real_grid.png")

    del generated_vis
    del generated_prdc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fid, kid_mean_value, kid_std_value, precision, recall, density, coverage


def aggregate(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def make_noise_batch(shape: tuple[int, ...], batch_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((batch_size, *shape), generator=generator, device=device)
    return noise.to(device)

def build_interpolation_paths(noise_a: torch.Tensor, noise_b: torch.Tensor, t_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lerp_path = torch.stack([lerp(noise_a, noise_b, float(t)) for t in t_values], dim=0)
    slerp_path = torch.stack([slerp(noise_a, noise_b, float(t)) for t in t_values], dim=0)
    return lerp_path, slerp_path

def show_interpolation_comparison(lerp_images: torch.Tensor, slerp_images: torch.Tensor, t_values: torch.Tensor, output_dir: Path, title: str):
    lerp_images = lerp_images.detach().cpu().float().clamp(0, 1)
    slerp_images = slerp_images.detach().cpu().float().clamp(0, 1)
    n = len(t_values)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.0))
    for row_idx, (row_axes, images, label) in enumerate(zip(axes, [lerp_images, slerp_images], ['lerp', 'slerp'])):
        for col_idx, (ax, img, t) in enumerate(zip(row_axes, images, t_values)):
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap='gray')
            else:
                ax.imshow(img.permute(1, 2, 0))
            if row_idx == 0:
                ax.set_title(f't={float(t):.2f}')
            if col_idx == 0:
                ax.set_ylabel(label)
            ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{title}.png")

def interpolation(model_name:str, interpolation_steps: int, seed_a: int, seed_b: int, output_dir: str, device: torch.device):
    run_dir = output_dir / model_name / f"seed_{seed_a}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t_values = torch.linspace(0.0, 1.0, steps=interpolation_steps, device=device)

    model = None
    if model_name == "vae":
        vae = load_vae_from_checkpoint(device, Path("vae_results") / "vae_final.pt")
        noise_a = make_noise_batch(shape=(vae.latent_dim, 2, 2), batch_size=1, seed=seed_a, device=device)[0]
        noise_b = make_noise_batch(shape=(vae.latent_dim, 2, 2), batch_size=1, seed=seed_b, device=device)[0]
        lerp_path, slerp_path = build_interpolation_paths(noise_a, noise_b, t_values)
        generated_lerp = sample_vae(vae, num_samples=interpolation_steps, device=device, batch_size=128, noise=lerp_path)
        generated_slerp = sample_vae(vae, num_samples=interpolation_steps, device=device, batch_size=128, noise=slerp_path)
        del vae
    elif model_name == "dcgan":
        dcgan = load_dcgan_from_checkpoint(device, Path("dcgan_results") / "dcgan_final.pt")
        noise_a = make_noise_batch(shape=(dcgan.generator.latent_dim,), batch_size=1, seed=seed_a, device=device)[0]
        noise_b = make_noise_batch(shape=(dcgan.generator.latent_dim,), batch_size=1, seed=seed_b, device=device)[0]
        lerp_path, slerp_path = build_interpolation_paths(noise_a, noise_b, t_values)
        generated_lerp = sample_dcgan(dcgan, num_samples=interpolation_steps, device=device, batch_size=128, noise=lerp_path)
        generated_slerp = sample_dcgan(dcgan, num_samples=interpolation_steps, device=device, batch_size=128, noise=slerp_path)
        del dcgan
    elif model_name == "stylegan":
        stylegan = load_stylegan_from_checkpoint(device, Path("stylegan_results") / "StyleGAN_final.pt")
        noise_a = make_noise_batch(shape=(stylegan.z_dim,), batch_size=1, seed=seed_a, device=device)[0]
        noise_b = make_noise_batch(shape=(stylegan.z_dim,), batch_size=1, seed=seed_b, device=device)[0]
        lerp_path, slerp_path = build_interpolation_paths(noise_a, noise_b, t_values)
        generated_lerp = sample_stylegan(stylegan, num_samples=interpolation_steps, device=device, batch_size=64, noise=lerp_path)
        generated_slerp = sample_stylegan(stylegan, num_samples=interpolation_steps, device=device, batch_size=64, noise=slerp_path)
        del stylegan
    elif model_name == "pixelunet":
        pixel = load_pixel_unet_from_checkpoint(device, Path("PixelUNet_results") / "PixelUNet_final.pt")
        schedule = GaussianDiffusion(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        noise_a = make_noise_batch(shape=(3, IMAGE_SIZE, IMAGE_SIZE), batch_size=1, seed=seed_a, device=device)[0]
        noise_b = make_noise_batch(shape=(3, IMAGE_SIZE, IMAGE_SIZE), batch_size=1, seed=seed_b, device=device)[0]
        lerp_path, slerp_path = build_interpolation_paths(noise_a, noise_b, t_values)
        generated_lerp = sample_pixel_unet(
            pixel,
            schedule,
            num_samples=interpolation_steps,
            device=device,
            image_size=IMAGE_SIZE,
            batch_size=32,
            noise = lerp_path
        )
        generated_slerp = sample_pixel_unet(
            pixel,
            schedule,
            num_samples=interpolation_steps,
            device=device,
            image_size=IMAGE_SIZE,
            batch_size=32,
            noise = slerp_path
        )
        del pixel
        del schedule
    elif model_name == "latentdenoiser":
        vae_model = load_vae_from_checkpoint(device, Path("vae_results") / "vae_final.pt")
        latent = load_latent_denoiser_from_checkpoint(
            device, Path("LatentDenoiseNetwork_results") / "LatentDenoiserNetwork_final.pt"
        )
        schedule = GaussianDiffusion(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            device=device,
        )
        noise_a = make_noise_batch(shape=(vae_model.latent_dim, 2, 2), batch_size=1, seed=seed_a, device=device)[0]
        noise_b = make_noise_batch(shape=(vae_model.latent_dim, 2, 2), batch_size=1, seed=seed_b, device=device)[0]
        lerp_path, slerp_path = build_interpolation_paths(noise_a, noise_b, t_values)
        generated_lerp = sample_latent_denoiser(
            latent,
            schedule,
            vae=vae_model,
            num_samples=interpolation_steps,
            device=device,
            batch_size=32,
            noise=lerp_path
        )
        generated_slerp = sample_latent_denoiser(
            latent,
            schedule,
            vae=vae_model,
            num_samples=interpolation_steps,
            device=device,
            batch_size=32,
            noise=slerp_path
        )
        del latent
        del schedule
    else:
        raise ValueError(f"Unknown model name '{model_name}'")

    show_interpolation_comparison(lerp_images=generated_lerp, slerp_images=generated_slerp, t_values=t_values.cpu(), output_dir=run_dir, title=f"{model_name}_Lerp_vs_Slerp")

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
    parser.add_argument(
        "--prdc-nearest-k",
        type=int,
        default=5,
        help="Nearest-neighbor k used by PRDC metrics (quality/coverage).",
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

    real_images_dir = args.output_dir / "_torch_fidelity" / f"real_{args.num_samples}"
    if real_images_dir.exists():
        shutil.rmtree(real_images_dir)
    export_images_to_folder(real_images, real_images_dir)

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
        precision_values: list[float] = []
        recall_values: list[float] = []
        density_values: list[float] = []
        coverage_values: list[float] = []

        for seed in eval_seeds:
            print(f"  - seed={seed}")
            fid, kid_mean, kid_std, precision, recall, density, coverage = run_single_eval(
                model_key=model_key,
                seed=seed,
                real_images=real_images,
                device=device,
                num_samples=args.num_samples,
                gen_batch_size=args.gen_batch_size,
                metric_batch_size=args.metric_batch_size,
                metrics_device=metrics_device,
                output_dir=args.output_dir,
                real_images_dir=real_images_dir,
                prdc_nearest_k=args.prdc_nearest_k,
                vae_model=vae_model,
            )
            fid_values.append(fid)
            kid_mean_values.append(kid_mean)
            kid_std_values.append(kid_std)
            precision_values.append(precision)
            recall_values.append(recall)
            density_values.append(density)
            coverage_values.append(coverage)
            per_run_rows.append(
                {
                    "model": model_key,
                    "seed": seed,
                    "fid": fid,
                    "kid_mean": kid_mean,
                    "kid_std": kid_std,
                    "precision": precision,
                    "recall": recall,
                    "density": density,
                    "coverage": coverage,
                }
            )
            print(
                f"    FID={fid:.4f} | KID={kid_mean:.6f} ± {kid_std:.6f} | "
                f"P={precision:.4f} R={recall:.4f} D={density:.4f} C={coverage:.4f}"
            )
            interpolation(model_name=model_key, interpolation_steps=10, seed_a=seed, seed_b=seed*2, output_dir=args.output_dir, device=device)

        fid_mean, fid_std = aggregate(fid_values)
        kid_mean_of_means, kid_std_across_seeds = aggregate(kid_mean_values)
        precision_mean, precision_std = aggregate(precision_values)
        recall_mean, recall_std = aggregate(recall_values)
        density_mean, density_std = aggregate(density_values)
        coverage_mean, coverage_std = aggregate(coverage_values)
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
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
                "density_mean": density_mean,
                "density_std": density_std,
                "coverage_mean": coverage_mean,
                "coverage_std": coverage_std,
            }
        )
        print(
            f"  Final ({model_key}): "
            f"FID={fid_mean:.4f} ± {fid_std:.4f} | "
            f"KID={kid_mean_of_means:.6f} ± {kid_std_across_seeds:.6f} | "
            f"P={precision_mean:.4f}±{precision_std:.4f} "
            f"R={recall_mean:.4f}±{recall_std:.4f} "
            f"D={density_mean:.4f}±{density_std:.4f} "
            f"C={coverage_mean:.4f}±{coverage_std:.4f}"
        )

    per_run_csv = args.output_dir / "per_run_metrics.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "seed", "fid", "kid_mean", "kid_std", "precision", "recall", "density", "coverage"],
        )
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
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "density_mean",
                "density_std",
                "coverage_mean",
                "coverage_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    coverage_vs_quality: dict[str, tuple[float, float]] = {
        str(row["model"]): (float(row["precision_mean"]), float(row["coverage_mean"])) for row in summary_rows
    }
    plot_path = args.output_dir / "quality_vs_coverage.png"
    plot_coverage_vs_quality(coverage_vs_quality, plot_path)

    print(f"\nSaved per-run metrics to: {per_run_csv}")
    print(f"Saved summary metrics to: {summary_csv}")
    print(f"Saved quality/coverage plot to: {plot_path}")


if __name__ == "__main__":
    main()
