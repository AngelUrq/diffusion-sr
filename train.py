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

from diffusers import UNet2DModel, DDIMScheduler
from srd import SuperResolutionDataset
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

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
from dsr import DSRDataset
from pathlib import Path
from accelerate import Accelerator

from samplers import DDIMPipeline 

def train(
    model,
    diffusion_scheduler,
    train_loader,
    val_loader,
    epochs,
    device,
    optimizer,
    lr_scheduler,
    criterion,
    tensorboard_path="./runs/diffusion",
    num_accumulation_steps=1
):
    print("Starting training for", epochs, "epochs")
    print("Accumulation steps:", num_accumulation_steps)
    print("Logging to:", tensorboard_path)
    tb_writer = SummaryWriter(tensorboard_path)
    train_losses = []
    

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=num_accumulation_steps
    )
    
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    
    for epoch in tqdm(range(epochs)):
        model.train()
        
        for i, (X, y) in enumerate(train_loader):
            batch_size = X.size(0)
            
            timesteps = torch.randint(0, len(diffusion_scheduler.timesteps), (batch_size,)).long()
            noise = torch.randn_like(y)

            noisy_images = diffusion_scheduler.add_noise(y, noise, timesteps)
            noisy_images = torch.cat([noisy_images, X], dim=1).to(device)

            noise = noise.to(device)
            
            with accelerator.accumulate(model):
                predicted_noise = model(noisy_images, timesteps.to(device))[0]
                loss = criterion(noise, predicted_noise)

                accelerator.backward(loss)
                #loss.backward()

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if i % 50 == 0:
                print(f"Epoch {epoch} Iteration {i} Loss {loss.item()}")
                train_losses.append(loss.detach().item())

                tb_writer.add_scalar(
                    "Loss/train", loss.detach().item(), epoch * len(train_loader) + i
                )
                
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    tb_writer.add_scalar(
                        "Learning Rate", current_lr, epoch * len(train_loader) + i
                    )
        
        evaluate(model, diffusion_scheduler, val_loader, lr_scheduler, device, criterion, epoch, tensorboard_path)

    print("Sampling final images")
    X, y = next(iter(val_loader))
    X = X.to(device)
    y = y.to(device)

    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=diffusion_scheduler)
    generated_images = pipeline(
        X,
        batch_size=X.shape[0],
        num_inference_steps=100
    )
    
    X = (X + 1) / 2
    y = (y + 1) / 2
    
    examples = torch.cat((X[:10].cpu(), y[:10].cpu(), generated_images[:10]), dim=0)
    
    grid_images = torchvision.utils.make_grid(examples, nrow=10)
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

        timesteps = torch.randint(0, len(diffusion_scheduler.timesteps), (batch_size,)).long()
        noise = torch.randn_like(X)

        noisy_images = diffusion_scheduler.add_noise(X, noise, timesteps)
        noisy_images = torch.cat([noisy_images, X], dim=1).to(device)

        noise = noise.to(device)
        noisy_images = noisy_images.to(device)

        predicted_noise = model(noisy_images, timesteps.to(device))[0]

        loss = criterion(noise, predicted_noise)
        
        total_loss += loss.item()

    tb_writer.add_scalar("Loss/val", total_loss / len(val_loader), epoch)
    scheduler.step(total_loss / len(val_loader))

    print(f"Epoch {epoch} Validation Loss {total_loss / len(val_loader)}")

    if epoch % 5 == 0:
        print("Sampling image")
        pipeline = DDIMPipeline(unet=model, scheduler=diffusion_scheduler)
        
        X, y = next(iter(val_loader))

        X = X.to(device)
        y = y.to(device)
        
        generated_images = pipeline(
            X,
            batch_size=X.shape[0],
            num_inference_steps=100
        )
        
        X = (X + 1) / 2
        y = (y + 1) / 2
        
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(y.cpu(), generated_images)
        
        tb_writer.add_scalar("Metrics/PSNR", psnr_value.item(), epoch)

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        lpips_value = lpips(y.cpu(), generated_images)
        
        tb_writer.add_scalar("Metrics/LPIPS", lpips_value.item(), epoch)

        ssim = StructuralSimilarityIndexMeasure()
        ssim_value = ssim(y.cpu(), generated_images)
        
        tb_writer.add_scalar("Metrics/SSIM", ssim_value.item(), epoch)

        examples = torch.cat((X[:3].cpu(), y[:3].cpu(), generated_images[:3]), dim=0)

        grid_images = torchvision.utils.make_grid(examples, nrow=3)
        tb_writer.add_image("Sample image", grid_images, epoch)
        

if __name__ == "__main__": 
    root = Path('./dsr/')

    with open(root / 'train_valid_test_split.json', 'r') as f:
        split = json.load(f)
    
    train_dataset = DSRDataset(root, split['train'], resolution=256, real_lr=True)
    val_dataset = DSRDataset(root, split['test'], resolution=256, real_lr=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = UNet2DModel(
        sample_size=128,  # the target image resolution
        in_channels=6,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=3,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 1024, 1024),  # the number of output channels for each UNet block
        norm_num_groups=1,
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        ),
    ).to(device)
    
    epochs = 50
    T = 2000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    diffusion_scheduler = DDIMScheduler(num_train_timesteps=T, beta_schedule='squaredcos_cap_v2')
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=(len(train_loader) * epochs),
    )
    criterion = nn.MSELoss()
    tensorboard_path="./runs/diffusion-525M-dsr-false-nogroup-256R-cosine-1e-4"

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params, device)
    
    train(model, diffusion_scheduler, train_loader, val_loader, epochs, device, optimizer, lr_scheduler, criterion, tensorboard_path, num_accumulation_steps=16)
    torch.save(model.state_dict(), 'models/dsr_sr.pth')
    