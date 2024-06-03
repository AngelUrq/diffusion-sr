#!/bin/bash

echo "Starting training..."

pip install torchmetrics
pip install diffusers[training]
sudo pip uninstall -y transformer-engine

tensorboard --logdir /scratch/students/2024-spring-sp-azenteno/diffusion-sr/runs/ &

echo "$(nvidia-smi)"

python train.py
