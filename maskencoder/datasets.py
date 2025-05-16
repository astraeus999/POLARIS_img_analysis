import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image



class UnsupervisedImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to RGB if needed, convert to grayscale now, add
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

def getdataset(batch_size, image_size=256, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to the desired image size
        transforms.ToTensor(),
    ])

    train_dataset = UnsupervisedImageDataset(root="./images", transform=preprocess)
    test_dataset = UnsupervisedImageDataset(root="./labeled_images", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader



def mask_image(data_loader, mask_params, patch_size):
    height = mask_params['height']
    width = mask_params['width']
    mask_id = mask_params['mask_id']
    mask_data_loader = []
    copy_data_loader = []
    # Get the mask positions
    mask_positions = [(i // (height // patch_size), i % (width // patch_size)) for i in mask_id]

    for data in data_loader:
        masked_image = data[0].clone()
        for row, col in mask_positions:
            masked_image[
                :, 
                :, 
                row * patch_size:(row + 1) * patch_size, 
                col * patch_size:(col + 1) * patch_size
            ] = 0
        mask_data_loader.append((masked_image))
        copy_data_loader.append((data[0]))

    return mask_data_loader, copy_data_loader