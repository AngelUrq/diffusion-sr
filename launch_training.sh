#!/bin/bash

echo "Starting training..."

export WANDB_API_KEY=e7c0a6928e86657f120e972e1a3179aa50641d2d

pip install -r requirement.txt

pip install wandb

python sr.py -p train -c config/sr_sr3_16_128.json -enable_wandb
