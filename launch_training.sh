#!/bin/bash

echo "Starting training..."

pip install torchmetrics
pip install diffusers

tensorboard --logdir /scratch/students/2024-spring-sp-azenteno/diffusion-sr/runs/ &

echo "$(nvidia-smi)"

python train.py
