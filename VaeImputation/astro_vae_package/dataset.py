import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset

class CreateTrainingImagesDataset(Dataset):
    def __init__(self, input_images, target_images, img_edge_size=256):
        """
        Args:
            input_images (list of nparray): list of reference images masked by mask_ring.
            target_images (list of nparray): list of reference images masked by mask_impute.
        """
        self.input_images = input_images
        self.target_images = target_images
        self.img_edge_size = img_edge_size
        
        # Ensure the images and masks match
        if len(self.input_images) != len(self.target_images):
            raise ValueError("The number of target images and masked images do not match.")

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to fetch.
        
        Returns:
            Tuple: (target_image, masked_image)
                - target_image: Loaded target image.
                - masked_image: Loaded masked image.
        """
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]
        
        # Convert to torch tensors if needed (for example, if using PyTorch)
        target_image = torch.tensor(target_image, dtype=torch.float32).reshape((1, self.img_edge_size, self.img_edge_size))
        input_image = torch.tensor(input_image, dtype=torch.float32).reshape((1, self.img_edge_size, self.img_edge_size))
        
        return input_image, target_image
