
# Diffusion-SR

This repository contains the implementation of a diffusion model for super-resolution. The project aims to enhance image resolution using diffusion techniques.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running Training](#running-training)
  - [Changing Hyperparameters](#changing-hyperparameters)
- [Inference](#inference)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AngelUrq/diffusion-sr.git
    cd diffusion-sr
    ```
2. Download the DSR data and place it in the root folder of the repository.

## Usage

### Files Description

- `ddpm.ipynb`: Jupyter notebook for Denoising Diffusion Probabilistic Models (DDPM) implementation.
- `diffusion.py`: Implementation of the diffusion scheduling.
- `dsr.py`, `dsr_height.py`: Pytorch datasets for DSR.
- `eda.ipynb`: Exploratory Data Analysis notebook.
- `eval.ipynb`: Notebook for model evaluation.
- `launch_tensorboard.sh`: Script to launch TensorBoard for monitoring training.
- `launch_training.sh`: Script to launch the training process.
- `model.py`: U-net implementation, not used in the final version. We adopted Diffusers implementation.
- `samplers.py`: Contains sampling methods for the diffusion process.
- `srd.py`: Pytorch dataset for flowers.
- `super_resolution.ipynb`: Notebook demonstrating the super resolution process.
- `train.py`, `train_height.py`: Training scripts.

### Running Training

To run the training process, execute the following script:
```bash
bash launch_training.sh
```
This script will initiate the training using the default settings specified in `train.py`.

### Changing Hyperparameters

To change hyperparameters, modify the `train.py` file. Here are some key hyperparameters you might want to adjust:
- `learning_rate`: Adjust the learning rate for the optimizer.
- `batch_size`: Change the number of samples per batch.
- `num_epochs`: Set the number of training epochs.
- `data_path`: Specify the path to your training dataset.

Example:
```python
# train.py
learning_rate = 0.001
batch_size = 32
num_epochs = 100
data_path = 'path/to/your/dataset'
```

## Inference

Pre-trained models are available in the `models` folder. To perform inference using these models, run the following notebook:
```bash
jupyter notebook super_resolution.ipynb
```
This notebook contains code to load the pre-trained models and apply them to your images for super-resolution. Note that this method works for models trained with artificial images.

For inference with models trained on real data, we use the [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) GitHub repository. We chose this approach to validate our model by testing it with different codebases. In this case, you need to modify `config/sr_sr3_16_128.json`. We have already adjusted this config to load our latest checkpoint. Instructions for running the model can be found in the README of that repository.
