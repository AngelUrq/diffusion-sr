import torch
import math

import numpy as np
import torch
from tqdm import tqdm


class DiffusionScheduler:

    def __init__(self, T, schedule_type='linear', beta_start=0.0001, beta_end=0.02, s=0.008):
        self.T = T
        self.schedule_type = schedule_type

        if schedule_type == 'linear':
            self.betas = self.linear_diffusion_schedule(T, beta_start, beta_end)
        elif schedule_type == 'cosine':
            self.betas = self.cosine_diffusion_schedule(T, s)
            
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)

        self.sqrt_alphas_bar= torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def linear_diffusion_schedule(self, timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def cosine_diffusion_schedule(self, timesteps, s=0.008):
        t = torch.linspace(0, timesteps, timesteps + 1)
            
        def f(t):
            return torch.cos((((t / timesteps) + s) / (1 + s)) * torch.pi * 0.5) ** 2
        
        alphas_bar = f(t) / f(torch.zeros_like(t))

        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        return betas

    def add_noise(self, x0, t, noise):
        batch_size = x0.shape[0]

        sqrt_alphas_bar_t = self.sqrt_alphas_bar.gather(-1, t).reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar.gather(-1, t).reshape(batch_size, 1, 1, 1)

        return sqrt_alphas_bar_t * x0 + sqrt_one_minus_alphas_bar_t * noise
    
    def remove_noise_step(self, predicted_noise, sample, t, device='cuda'):
        z = torch.randn(1, 3, 64, 64).to(device)

        t = t.cpu().item()
 
        alpha = self.alphas[t]
        beta = self.betas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t]

        sample_mean = (1 / ((alpha)**(1/2))) * (sample - ((1 - alpha) / (sqrt_one_minus_alpha_bar)) * predicted_noise)

        if t == 0:
            sample = sample_mean
        else:
            sample = sample_mean + z * (beta ** (1/2))
            
        return sample

    @torch.no_grad()
    def generate(self, model, device='cuda'):
        model.eval()

        sample = torch.randn(1, 3, 64, 64).to(device)

        samples = []

        for step in range(self.T - 1, -1, -1):
            t = torch.ones(1).long().to(device) * step
            predicted_noise = model(sample, t)
            sample = self.remove_noise_step(predicted_noise, sample, t)
            samples.append(sample[0].cpu())

        return samples
