import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    scheduler,
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
            noise = torch.randn_like(y)

            noisy_images = diffusion_scheduler.add_noise(y, timesteps, noise)
            noisy_images = torch.cat([noisy_images, X], dim=1).to(device)
                
            noise = noise.to(device)

            predicted_noise = model(noisy_images, timesteps.to(device))

            loss = criterion(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #ema.update(model)

            if i % 50 == 0:
                print(f"Epoch {epoch} Iteration {i} Loss {loss.item()}")
                train_losses.append(loss.item())

                tb_writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(train_loader) + i
                )
        
        evaluate(model, diffusion_scheduler, val_loader, scheduler, device, criterion, epoch, tensorboard_path)

    print("Sampling final image")
    tb_writer = SummaryWriter(tensorboard_path)
    X, y = next(iter(val_loader))
    X = X[0].unsqueeze(0).to(device)
    y = y[0].to(device)

    samples = diffusion_scheduler.generate(unet, X)

    X[0] = (X[0] + 1) / 2
    y = (y + 1) / 2

    for i, sample in enumerate(samples):
        grid_images = torchvision.utils.make_grid([X[0].cpu(), sample[0].cpu(), y.cpu()], nrow=3)

        tb_writer.add_image("Final sample image", grid_images, i)

@torch.no_grad()
def evaluate(
    model, diffusion_scheduler, val_loader, scheduler, device, criterion, epoch, tensorboard_path="./runs/diffusion"
):
    tb_writer = SummaryWriter(tensorboard_path)
    model.eval()

    total_loss = 0

    for i, (X, y) in enumerate(val_loader):
        batch_size = X.size(0)

        timesteps = torch.randint(0, diffusion_scheduler.T, (batch_size,)).long()
        noise = torch.randn_like(X)

        noisy_images = diffusion_scheduler.add_noise(X, timesteps, noise)
        noisy_images = torch.cat([noisy_images, X], dim=1).to(device)

        noise = noise.to(device)
        noisy_images = noisy_images.to(device)

        predicted_noise = model(noisy_images, timesteps.to(device))

        loss = criterion(noise, predicted_noise) 
        total_loss += loss.item()

    tb_writer.add_scalar("Loss/val", total_loss / len(val_loader), epoch)
    scheduler.step(total_loss / len(val_loader))

    print(f"Epoch {epoch} Validation Loss {total_loss / len(val_loader)}")

    if epoch % 1 == 0:
        print("Sampling image")
        X, y = next(iter(val_loader))
        X = X[0].unsqueeze(0).to(device)
        y = y[0].to(device)
        
        samples = diffusion_scheduler.generate(model, X)[-1]
        sample = samples[0]

        X = (X + 1) / 2
        y = (y + 1) / 2
        
        grid_images = torchvision.utils.make_grid([X[0].cpu(), sample.cpu(), y.cpu()], nrow=3)
        tb_writer.add_image("Sample image", grid_images, epoch)
        