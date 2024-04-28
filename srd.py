import torch
import numpy as np
import cv2

class SuperResolutionDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, resolution=128):
        self.data_folder = data_path
        self.image_paths = self.get_image_paths()
        self.resolution = resolution

    def get_image_paths(self):
        image_paths = []
        for class_folder in os.listdir(self.data_folder):
            class_path = os.path.join(self.data_folder, class_folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    if 'ipynb' in image_path:
                        print(image_path)
                    image_paths.append(image_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))

        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image_size = image.shape[0]
        
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        downscaled_image = cv2.resize(blurred_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        upscaled_image = cv2.resize(downscaled_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        upscaled_image = np.clip(upscaled_image, 0, 1)
        
        upscaled_image = (upscaled_image - 0.5) / 0.5
        image = (image - 0.5) / 0.5

        upscaled_image = torch.tensor(upscaled_image).permute(2, 0, 1).float()
        image = torch.tensor(image).permute(2, 0, 1).float()

        return upscaled_image, image
    