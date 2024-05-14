import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

from simple_diffusion.scheduler import DDIMScheduler
from srd import SuperResolutionDataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import copy
import cv2
import os
import json
import time

from PIL import Image
#from model import UNet
from simple_diffusion.model import UNet 
from dsr import DSRDataset
from pathlib import Path

def train(
    model,
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
    
    print('half2')

    for epoch in tqdm(range(epochs)):
        model.train()
        
        for i, (X, y) in enumerate(train_loader):
            batch_size = X.size(0)
            
            X = X.half()
            y = y.half()
            
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

    samples = diffusion_scheduler.generate(model, X)

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

        X = X.to(device)
        y = y.to(device)

        upsamples = diffusion_scheduler.generate(model, X, batch_size=X.shape[0], image_size=X.shape[1])[-1]
        upsamples = 2 * upsamples - 1
        
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(y.cpu(), upsamples)
        
        tb_writer.add_scalar("Metrics/PSNR", psnr_value.item(), epoch)

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        lpips_value = lpips(y.cpu(), upsamples)
        
        tb_writer.add_scalar("Metrics/LPIPS", lpips_value.item(), epoch)

        ssim = StructuralSimilarityIndexMeasure()
        ssim_value = ssim(y.cpu(), upsamples)
        
        tb_writer.add_scalar("Metrics/SSIM", ssim_value.item(), epoch)

        X = (X + 1) / 2
        y = (y + 1) / 2
        upsamples = (upsamples + 1) / 2

        examples = torch.cat((X[:3].cpu(), y[:3].cpu(), upsamples[:3].cpu()), dim=0)

        grid_images = torchvision.utils.make_grid(examples, nrow=3)
        tb_writer.add_image("Sample image", grid_images, epoch)
        

if __name__ == "__main__": 
    root = Path('./dsr/')

    with open(root / 'train_valid_test_split.json', 'r') as f:
        split = json.load(f)
    
    train_dataset = DSRDataset(root, split['train'], resolution=256)
    val_dataset = DSRDataset(root, split['test'], resolution=256)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #unet = UNet(channel_mults=(1, 2, 4, 8, 8), image_size=64).to(device)
    unet = UNet(hidden_dims=[16, 32, 64, 128], image_size=256).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(),lr=1e-4,betas=(0.9, 0.99),weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()
    epochs = 15
    T = 2000
    diffusion_scheduler = DDIMScheduler(beta_schedule="cosine")
    tensorboard_path="./runs/diffusion-2M-dsr-256R-cosine-1e-4"


    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("Number of parameters:", num_params, device)
    
    train(unet, diffusion_scheduler, train_loader, val_loader, epochs, device, optimizer, scheduler, criterion, tensorboard_path)
    torch.save(unet.state_dict(), 'dsr_sr.pth')
    