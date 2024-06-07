
# Diffusion-SR

This repository contains the implementation of a diffusion model for super-resolution. The project aims to enhance image resolution using diffusion techniques.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running Training](#running-training)
  - [Changing Hyperparameters](#changing-hyperparameters)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AngelUrq/diffusion-sr.git
    cd diffusion-sr
    ```

## Usage

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
This notebook contains code to load the pre-trained models and apply them to your images for super-resolution.
