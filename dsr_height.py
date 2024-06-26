import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import json
import os
import cv2
import random

from pathlib import Path
from PIL import Image


class DSRDataset(torch.utils.data.Dataset):

    ALTITUDES = (10, 20, 30, 40, 50, 70, 80, 100, 120, 140)
    DEFAULT_IMAGE = 'color_correction.png'
    DEFAULT_TARGET = 'tele.png'
    
    def __init__(self, root, scenes, height=None, transform=None, resolution=128, real_lr=True):
        self.root = root
        self.transform = transform
        self.scenes = scenes
        self.resolution = resolution
        self.real_lr = real_lr

        self.pairs = []
        
        self.altitude_to_class = {altitude: i for i, altitude in enumerate(self.ALTITUDES)}
        self.class_to_altitude = {i: altitude for i, altitude in enumerate(self.ALTITUDES)}

        for scene in scenes:
            if height is not None:
                pairs_height = self.get_pairs(root, scene, height)

                if len(pairs_height) > 0:
                    self.pairs.extend(pairs_height)
            else:
                for altitude in self.ALTITUDES:
                    pairs_height = self.get_pairs(root, scene, altitude)

                    if len(pairs_height) > 0:
                        self.pairs.extend(pairs_height)
                
        print(f'Loaded {len(self.pairs)} pairs for scenes {scenes} and height {height}')
                        
    def get_pairs(self, root, scene, height):
        pairs = []
        filepath = root / scene / str(height)
    
        if not filepath.exists():
            return pairs

        directories = [name for name in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, name))]
        directories.sort()

        for directory in directories:
            pairs.append((filepath / directory / self.DEFAULT_IMAGE, filepath / directory / self.DEFAULT_TARGET, height))
 
        return pairs
        
    def load_image(self, path):
        return np.array(Image.open(path))
        
    def __len__(self):
        return len(self.pairs)
    
    
    def __getitem__(self, idx):
        if self.real_lr:
            image_path, target_path, altitude = self.pairs[idx]
            image = self.load_image(image_path) / 255.0
            target = self.load_image(target_path) / 255.0

            h, w, _ = image.shape
            target_size = self.resolution
            ratio = target.shape[0] / image.shape[0]

            crop_size = int(target_size / ratio)

            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            crop = image[y:y+crop_size, x:x+crop_size]
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)  

            x = int(x * ratio)
            y = int(y * ratio)

            target = target[y:y+target_size, x:x+target_size]

            crop = torch.tensor((crop - 0.5) / 0.5).permute(2, 0, 1).float()
            crop_resized = torch.tensor((crop_resized - 0.5) / 0.5).permute(2, 0, 1).float()
            target = torch.tensor((target - 0.5) / 0.5).permute(2, 0, 1).float()
            
            return crop_resized, target, self.altitude_to_class[altitude]
        else:
            image_path, target_path, altitude = self.pairs[idx]
            target = self.load_image(target_path) / 255.0

            image = target[:512, :512]
            target = target[:512, :512]

            target = cv2.resize(target, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            image_size = target.shape[0]

            blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
            downscaled_image = cv2.resize(blurred_image, (32, 32), interpolation=cv2.INTER_AREA)

            upscaled_image = cv2.resize(downscaled_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            upscaled_image = np.clip(upscaled_image, 0, 1)

            upscaled_image = (upscaled_image - 0.5) / 0.5
            target = (target - 0.5) / 0.5

            upscaled_image = torch.tensor(upscaled_image).permute(2, 0, 1).float()
            target = torch.tensor(target).permute(2, 0, 1).float()
                     
            return upscaled_image, target, self.altitude_to_class[altitude]
