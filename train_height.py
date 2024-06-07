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
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

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
from dsr_height import DSRDataset
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
    num_accumulation_steps=1,
    altitude_conditioning=False,
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
        
        for i, (X, y, altitudes) in enumerate(train_loader):
            batch_size = X.size(0)
            
            timesteps = torch.randint(0, len(diffusion_scheduler.timesteps), (batch_size,)).long()
            noise = torch.randn_like(y)

            noisy_images = diffusion_scheduler.add_noise(y, noise, timesteps)
            noisy_images = torch.cat([noisy_images, X], dim=1).to(device)

            noise = noise.to(device)
            altitudes = altitudes.to(device)
            timesteps = timesteps.to(device)
            
            with accelerator.accumulate(model):
                if altitude_conditioning:
                    predicted_noise = model(noisy_images, timesteps, class_labels=altitudes)[0]
                else:
                    predicted_noise = model(noisy_images, timesteps)[0]
                    
                loss = criterion(noise, predicted_noise)

                accelerator.backward(loss)
                
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
            tb_writer.add_scalar("Learning Rate", current_lr, epoch)
            break
        
        evaluate(accelerator.unwrap_model(model), diffusion_scheduler, val_loader, lr_scheduler, device, criterion, epoch, tensorboard_path, altitude_conditioning=altitude_conditioning)

    print("Sampling final images")
    X, y, altitudes = next(iter(val_loader))
    X = X.to(device)
    y = y.to(device)
    altitudes = altitudes.to(device)

    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=diffusion_scheduler)
    if altitude_conditioning:
        generated_images = pipeline(
            X,
            altitudes=altitudes,
            batch_size=X.shape[0],
            num_inference_steps=100
        )
    else:
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
    model, diffusion_scheduler, val_loader, lr_scheduler, device, criterion, epoch, tensorboard_path="./runs/diffusion", altitude_conditioning=False
):
    tb_writer = SummaryWriter(tensorboard_path)
    model.eval()

    total_loss = 0

    for i, (X, y, altitudes) in enumerate(val_loader):
        batch_size = X.size(0)

        timesteps = torch.randint(0, len(diffusion_scheduler.timesteps), (batch_size,)).long()
        noise = torch.randn_like(X)

        noisy_images = diffusion_scheduler.add_noise(X, noise, timesteps)
        noisy_images = torch.cat([noisy_images, X], dim=1).to(device)

        noise = noise.to(device)
        noisy_images = noisy_images.to(device)

        if altitude_conditioning:
            altitudes = altitudes.to(device)
            predicted_noise = model(noisy_images, timesteps.to(device), class_labels=altitudes)[0]
        else:
            predicted_noise = model(noisy_images, timesteps.to(device))[0]

        loss = criterion(noise, predicted_noise)
        
        total_loss += loss.item()

    tb_writer.add_scalar("Loss/val", total_loss / len(val_loader), epoch)
    lr_scheduler.step(total_loss / len(val_loader))

    print(f"Epoch {epoch} Validation Loss {total_loss / len(val_loader)}")

    if epoch % 5 == 0:
        print("Sampling image")
        pipeline = DDIMPipeline(unet=model, scheduler=diffusion_scheduler)
        
        X, y, altitudes = next(iter(val_loader))
        X = X.to(device)
        y = y.to(device)
        altitudes = altitudes.to(device)
        
        if altitude_conditioning:
            generated_images = pipeline(
                X,
                altitudes=altitudes,
                batch_size=X.shape[0],
                num_inference_steps=100
            )
        else:
            generated_images = pipeline(
                X,
                batch_size=X.shape[0],
                num_inference_steps=100
            )
            from simple_diffusion.scheduler import DDIMScheduler
            diffusion_scheduler = DDIMScheduler(beta_schedule="cosine")

            generated_images = diffusion_scheduler.generate(model, X, batch_size=X.shape[0], image_size=X.shape[2])[-1]
            generated_images = 2 * generated_images - 1
        
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
    
    train_dataset = DSRDataset(root, split['train'], resolution=128, real_lr=False)
    val_dataset = DSRDataset(root, split['test'], resolution=128, real_lr=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True,
        prefetch_factor=2
    )
    
    altitude_conditioning = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_classes = len(train_dataset.ALTITUDES)
    
    model = UNet2DModel(
        sample_size=128,  # the target image resolution
        in_channels=6,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256),  # the number of output channels for each UNet block
        norm_num_groups=1,
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
        ),
        class_embed_type=None,
        num_class_embeds=num_classes
    ).to(device)
    
    epochs = 50
    T = 2000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    diffusion_scheduler = DDIMScheduler(num_train_timesteps=T, beta_schedule='squaredcos_cap_v2')
    lr_scheduler = get_constant_schedule(
        optimizer=optimizer
    )
    
    criterion = nn.MSELoss()
    tensorboard_path="./runs/diffusion-14M-dsr-false-128R-height-cosine-1e-4"

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params, device)
    
    train(model, diffusion_scheduler, train_loader, val_loader, epochs, device, optimizer, lr_scheduler, criterion, tensorboard_path, num_accumulation_steps=4, altitude_conditioning=altitude_conditioning)
    torch.save(model.state_dict(), 'models/dsr_sr_128_height.pth')
