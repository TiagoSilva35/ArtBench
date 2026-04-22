import torch
import torch.optim as optim
import os
from tqdm import tqdm
from pathlib import Path
from src.helpers.debugger import DBG
from src.helpers.data_utils import unpack_images
    
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
    # Keep optimizer on the model object because model.train_step uses model.optimizer.
    model.optimizer = optim.Adam(model.parameters(), lr=lr)

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
            images = unpack_images(batch).to(device)
            batch_size = images.size(0)

            loss = model.train_step(images, beta=beta)

            # 6) Update the running loss and batch counter.
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
                    images = unpack_images(batch).to(device)
                    batch_size = images.size(0)

                    loss = model.compute_loss(images, beta=beta).item()

                    val_loss_accum += loss * batch_size
                    val_samples += batch_size

            avg_val_loss = val_loss_accum / val_samples
            history['val_loss'].append(avg_val_loss)
            DBG(f'Época {epoch} - Loss de Validação: {avg_val_loss:.4f}')
        else:
            history['val_loss'].append(None)


        if epoch % checkpoint_freq == 0 or epoch == epochs:
            checkpoint_path = run_dir / "checkpoints" / f"vae_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

            sample_batch = unpack_images(next(iter(train_loader))).to(device)
            model.generate_and_save_images(
                sample_batch,
                output_dir=os.path.join(run_dir, "samples"),
                epoch=epoch,
            )

    # store the final model
    final_path = os.path.join(run_dir, f"vae_final.pt")
    torch.save(model.state_dict(), final_path)
    DBG(f"Modelo final salvo em: {final_path}")
    DBG(f"Treino completo. Resultados salvos em: {run_dir}")
    return model, history

def train_DCGAN(
    model,
    train_loader,
    device,
    val_loader=None,
    epochs=50,
    checkpoint_freq=10,
    save_dir='dcgan_results',
    lr=2e-4,
    beta1=0.5,
    real_label=0.9,
):

    # 1. Setup Directories
    run_dir = Path(save_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    os.makedirs(run_dir / "samples", exist_ok=True)

    # Ensure all DCGAN modules are on the same device as input tensors.
    model = model.to(device)

    # Keep optimizer construction in one place so HPO/final training uses the same path.
    model.optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    model.optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "DCGAN params -> "
        f"G: {g_params:,} | D: {d_params:,} | "
        f"Total: {total_params:,} | Trainable: {trainable_params:,}"
    )

    # 2. Fixed latent vectors for consistent image samples over epochs
    fixed_z = torch.randn(16, model.generator.latent_dim, device=device)


    history = {'d_loss': [], 'g_loss': [], 'fid': []}

    # Set networks to training mode
    model.generator.train()
    model.discriminator.train()

    # 3. Main Training Loop
    for epoch in range(1, epochs + 1):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            real_imgs = unpack_images(batch).to(device)
            batch_size = real_imgs.size(0)

            valid = torch.full((batch_size,), real_label, device=device)
            fake_targets = torch.full((batch_size,), 0.0, device=device)

            model.optimizer_D.zero_grad()

            # Loss on real images
            d_real_loss = model.criterion(model.discriminator(real_imgs).view(-1), valid)

            # Loss on fake images
            z = torch.randn(batch_size, model.generator.latent_dim, device=device)
            gen_imgs = model.generator(z)

            # Use .detach() so we don't backprop through the Generator here
            d_fake_loss = model.criterion(model.discriminator(gen_imgs.detach()).view(-1), fake_targets)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            model.optimizer_D.step()

            model.optimizer_G.zero_grad()

            # Generator wants Discriminator to think fake images are real (target = 1.0)
            # Note: We do NOT use label smoothing for the Generator's targets, just strict 1.0
            g_targets = torch.full((batch_size,), 1.0, device=device)
            g_loss = model.criterion(model.discriminator(gen_imgs).view(-1), g_targets)

            g_loss.backward()
            model.optimizer_G.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()
            n_batches += 1

            # Update progress bar description
            pbar.set_postfix({"D Loss": f"{d_loss.item():.4f}", "G Loss": f"{g_loss.item():.4f}"})

        # Calculate epoch averages
        avg_d_loss = d_loss_epoch / n_batches
        avg_g_loss = g_loss_epoch / n_batches
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)

        print(f"\nEnd of Epoch {epoch}/{epochs} - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

        model.generator.eval()
        with torch.no_grad():
            sample_imgs = model.generator(fixed_z)
            save_path = run_dir / "samples" / f"epoch_{epoch:03d}.png"
            model.generate_and_save_images(sample_imgs, save_path, epoch)

        fid_metric = None  # TODO: FID METRIC

        if val_loader is not None and fid_metric is not None:
            # For a proper FID, we accumulate statistics over multiple batches
            fid_metric.reset()
            with torch.no_grad():
                # Get a batch of real images
                real_batch = unpack_images(next(iter(val_loader))).to(device)

                # Get a batch of fake images
                z_val = torch.randn(real_batch.size(0), model.generator.latent_dim, device=device)
                fake_batch = model.generator(z_val)

                # Convert from [-1, 1] to [0, 255] byte tensors for torchmetrics
                real_byte = ((real_batch + 1) / 2 * 255).byte()
                fake_byte = ((fake_batch + 1) / 2 * 255).byte()

                fid_metric.update(real_byte, real=True)
                fid_metric.update(fake_byte, real=False)

                fid_score = fid_metric.compute().item()
                history['fid'].append(fid_score)
                print(f"FID Score: {fid_score:.4f}")
        else:
            history['fid'].append(None)

        if epoch % checkpoint_freq == 0 or epoch == epochs:
            checkpoint_path = run_dir / "checkpoints" / f"dcgan_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'optimizer_G_state_dict': model.optimizer_G.state_dict(),
                'optimizer_D_state_dict': model.optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

        # Switch back to train mode for the next epoch
        model.generator.train()

    final_path = run_dir / "dcgan_final.pt"
    torch.save({
        'epoch': epochs,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
    }, final_path)
    print(f"Final checkpoint saved at: {final_path}")

    print(f"\nTraining complete. Results saved in: {run_dir}")
    return model, history

def train_diffusion(model, train_loader, schedule, device, val_loader=None, epochs=20, lr=2e-4, vae=None, save_dir=None, checkpoint_freq=10):
    if save_dir is None:
        run_dir = Path(model.name + "_results")
    else:
        run_dir = Path(save_dir)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    model = model.to(device)

    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss":[],
        "val_loss":[]
    }

    DBG(f"Iniciando treino de difusão por {epochs} épocas")
    DBG(f"Dispositivo: {device}")
    DBG(f"Resultados em: {run_dir}")

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Época {epoch}/{epochs} [Treino]', leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = unpack_images(batch).to(device)

            # Criar espaço latente, caso uma vae for dada
            if vae is not None:
                with torch.no_grad():
                    mu, logvar = vae.encode(images)
                    images = vae.reparameterize(mu, logvar)

            train_loss_accum += model.train_step(images, schedule, device)
            n_batches += 1

        avg_train_loss = train_loss_accum / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        DBG(f'Época {epoch} - Loss de Treino: {avg_train_loss:.4f}')

        if val_loader is not None:
            model.eval()
            val_loss_accum = 0.0
            n_batches = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Época {epoch}/{epochs} [Validação]'):
                    images = unpack_images(batch).to(device)

                    # Criar espaço latente, caso uma vae for dada
                    if vae is not None:
                        with torch.no_grad():
                            mu, logvar = vae.encode(images)
                            images = vae.reparameterize(mu, logvar)


                    loss = model.compute_loss(images, schedule, device).item()

                    val_loss_accum += loss
                    n_batches += 1

            avg_val_loss = val_loss_accum / max(n_batches, 1)
            history["val_loss"].append(avg_val_loss)
            DBG(f'Época {epoch} - Loss de Validação: {avg_val_loss:.4f}')
        else:
            history['val_loss'].append(None)

        if epoch % checkpoint_freq == 0 or epoch == epochs:
            checkpoint_path = run_dir / "checkpoints" / f"{model.name}_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

            model.generate_and_save_images(
                schedule,
                output_dir=os.path.join(run_dir, "samples"),
                epoch=epoch,
                num_samples=8,
                vae=vae
            )

    # store the final model
    final_path = os.path.join(run_dir, f"{model.name}_final.pt")
    torch.save(model.state_dict(), final_path)
    DBG(f"Modelo final salvo em: {final_path}")
    DBG(f"Treino completo. Resultados salvos em: {run_dir}")
    return model, history
