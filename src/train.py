import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from pathlib import Path
from src.helpers.debugger import DBG

def train_vae(
    model,
    train_loader,
    device,
    val_loader=None,
    epochs=50,
    lr=1e-4,
    beta=1.0,
    save_dir='vae_results',
    checkpoint_freq=10
):
    run_dir = Path(save_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
    }

    DBG(f"Iniciando treino do VAE por {epochs} épocas")
    DBG(f"Dispositivo: {device}")
    DBG(f"Beta: {beta}")
    DBG(f"Resultados em: {run_dir}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f'Época {epoch}/{epochs} [Treino]')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                images, _ = batch
            else:
                images, _, _ = batch
            images = images.to(device)
            batch_size = images.size(0)

            loss = model.train_step(images, beta=beta)

            train_loss_accum += loss * batch_size
            train_samples += batch_size

            pbar.set_postfix({'loss': loss})

        avg_train_loss = train_loss_accum / train_samples
        history['train_loss'].append(avg_train_loss)
        DBG(f'Época {epoch} - Loss de Treino: {avg_train_loss:.4f}')
        if val_loader is not None:
            model.eval()
            val_loss_accum = 0.0
            val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Época {epoch}/{epochs} [Validação]'):
                    if len(batch) == 2:
                        images, _ = batch
                    else:
                        images, _, _ = batch
                    images = images.to(device)
                    batch_size = images.size(0)

                    loss = model.compute_loss(images, beta=beta).item()

                    val_loss_accum += loss * batch_size
                    val_samples += batch_size

            avg_val_loss = val_loss_accum / val_samples
            history['val_loss'].append(avg_val_loss)
            DBG(f'Época {epoch} - Loss de Validação: {avg_val_loss:.4f}')
        else:
            history['val_loss'].append(None)

        # store the final model
        final_path = os.path.join(run_dir, f"vae_final.pt")
        torch.save(model.state_dict(), final_path)
        DBG(f"Modelo final salvo em: {final_path}")

        if epoch % checkpoint_freq == 0 or epoch == epochs:
            sample_batch = next(iter(train_loader))
            if len(sample_batch) == 2:
                sample_batch, _ = sample_batch
            else:
                sample_batch, _, _ = sample_batch
            sample_batch = sample_batch.to(device)
            model.generate_and_save_images(
                sample_batch,
                output_dir=os.path.join(run_dir, "samples"),
                epoch=epoch,
            )

    DBG(f"Treino completo. Resultados salvos em: {run_dir}")
    return model, history

def train_diffusion(model, loader, schedule, epochs=20, lr=2e-4, vae=None, save_dir=None, checkpoint_freq=10):
    if save_dir is None:
        print("Error: a saving directory needs to be given")
        return

    run_dir = Path(save_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    model = model.to(device)

    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss" = [],
        "val_loss" = []
    }

    DBG(f"Iniciando treino do VAE por {epochs} épocas")
    DBG(f"Dispositivo: {device}")
    DBG(f"Resultados em: {run_dir}")

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f'Época {epoch}/{epochs} [Treino]', leave=False)
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                images, _ = batch
            else:
                images, _, _ = batch

            images = images.to(device)

            # Criar espaço latente, caso uma vae for dada
            if vae is not None:
                with torch.no_grad():
                    mu, logvar = vae.encoder(images)
                    images = vae.reparameterize(mu, logvar)

            train_loss_accum += model.train_step(image, schedule)
            n_batches += 1

        avg_train_loss = train_loss_accum / max(n_batches, 1)
        history["train_loss"].append(avg)
        DBG(f'Época {epoch} - Loss de Treino: {avg_train_loss:.4f}')

        if val_loader is not None:
            model.eval()
            val_loss_accum = 0.0
            n_batches = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Época {epoch}/{epochs} [Validação]'):
                    if len(batch) == 2:
                        images, _ = batch
                    else:
                        images, _, _ = batch
                    images = images.to(device)

                    # Criar espaço latente, caso uma vae for dada
                    if vae is not None:
                        with torch.no_grad():
                            mu, logvar = vae.encoder(images)
                            images = vae.reparameterize(mu, logvar)


                    loss = model.compute_loss(images, schedule).item()

                    val_loss_accum += loss
                    n_batches += 1

            avg_val_loss = val_loss_accum / max(n_batches, 1)
            history["val_loss"].append(avg_val_loss)
            DBG(f'Época {epoch} - Loss de Validação: {avg_val_loss:.4f}')
        else:
            history['val_loss'].append(None)

        # store the final model
        final_path = os.path.join(run_dir, f"{model.name}_final.pt")
        torch.save(model.state_dict(), final_path)
        DBG(f"Modelo final salvo em: {final_path}")

        if epoch % checkpoint_freq == 0 or epoch == epochs:
            sample_batch = next(iter(train_loader))
            if len(sample_batch) == 2:
                sample_batch, _ = sample_batch
            else:
                sample_batch, _, _ = sample_batch
            sample_batch = sample_batch.to(device)

            # Criar espaço latente, caso uma vae for dada
            if vae is not None:
                with torch.no_grad():
                    mu, logvar = vae.encoder(sample_batch)
                    sample_batch = vae.reparameterize(mu, logvar)

            shape = (8, 4, 4, 4)
            model.generate_and_save_images(
                sample_batch,
                output_dir=os.path.join(run_dir, "samples"),
                epoch=epoch,
                shape,
                schedule,
                vae
            )

    DBG(f"Treino completo. Resultados salvos em: {run_dir}")
    return model, history
