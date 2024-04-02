import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(
    model,
    ema,
    diffusion_scheduler,
    train_loader,
    val_loader,
    epochs,
    device,
    optimizer,
    criterion,
    tensorboard_path="./runs/diffusion",
):
    print("Start training...")
    tb_writer = SummaryWriter(tensorboard_path)
    train_losses = []

    for epoch in tqdm(range(epochs)):
        model.train()
        
        for i, (X, y) in enumerate(train_loader):
            batch_size = X.size(0)

            timesteps = torch.randint(0, diffusion_scheduler.T, (batch_size,)).long()
            noise = torch.randn_like(X)

            noisy_images = diffusion_scheduler.add_noise(X, timesteps, noise)

            noise = noise.to(device)
            noisy_images = noisy_images.to(device)

            predicted_noise = model(noisy_images, timesteps.to(device))

            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ema.update(model)

            if i % 50 == 0:
                print(f"Epoch {epoch} Iteration {i} Loss {loss.item()}")
                train_losses.append(loss.item())

                tb_writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(train_loader) + i
                )

                break
        
        evaluate(model, diffusion_scheduler, val_loader, device, criterion, epoch)

    print("Sampling final image")
    samples = diffusion_scheduler.sample_image(model)

    for i, sample in enumerate(samples):
        sample = (sample.clamp(-1, 1) + 1) / 2
        sample = (sample * 255).type(torch.uint8)

        tb_writer.add_image("Final sample image", sample, i)

@torch.no_grad()
def evaluate(
    model, diffusion_scheduler, val_loader, device, criterion, epoch, tensorboard_path="./runs/diffusion"
):
    tb_writer = SummaryWriter(tensorboard_path)
    model.eval()

    total_loss = 0

    for i, (X, y) in enumerate(val_loader):
        batch_size = X.size(0)

        timesteps = torch.randint(0, diffusion_scheduler.T, (batch_size,)).long()
        noise = torch.randn_like(X)

        noisy_images = diffusion_scheduler.add_noise(X, timesteps, noise)

        noise = noise.to(device)
        noisy_images = noisy_images.to(device)

        predicted_noise = model(noisy_images, timesteps.to(device))

        loss = criterion(noise, predicted_noise)
        total_loss += loss.item()

    tb_writer.add_scalar("Loss/val", total_loss / len(val_loader), epoch)

    print(f"Epoch {epoch} Validation Loss {total_loss / len(val_loader)}")

    if epoch % 3 == 0:
        print("Sampling image")
        samples = diffusion_scheduler.sample_image(model)
        sample = (samples[-1].clamp(-1, 1) + 1) / 2
        sample = (sample * 255).type(torch.uint8)

        tb_writer.add_image("Sample image", sample, epoch)
