#!/bin/bash

echo "Starting training..."

pip install torchmetrics

tensorboard --logdir /scratch/students/2024-spring-sp-azenteno/diffusion-sr/runs/ &

python train.py
