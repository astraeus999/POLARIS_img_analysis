import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from networks.resnet_big_old import SupConResNet
from tqdm import tqdm


class TestImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        # Filter to include only .png files and ensure sorted order
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, img_path


# Function to Load the Trained Model
def load_model(checkpoint_path, model_name='resnet18'):
    # Initialize the model (use SupConResNet as in training)
    model = SupConResNet(name=model_name, feat_dim=16)
    model = model.cuda() if torch.cuda.is_available() else model

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = value

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)  # Ensure the key matches the saved state_dict
    # Set the model to evaluation mode
    model.eval()
    return model

# Function to Prepare the Testing DataLoader
def prepare_test_loader(test_folder, batch_size=4):
    mean = (0.5,)
    std = (0.5,)
    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=1.0),  # Ensure the image remains grayscale
        transforms.ToTensor(),
        normalize,
    ])
  ### there is no need for using twocroptransform for testing data
    test_dataset = TestImageDataset(test_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to Run Inference and Save Features
def save_testing_results(model, test_loader, output_path):
    model.eval()
    features = []
    image_paths = []
    
    print("start testing....")
    with torch.no_grad():  # Disable gradient computation for inference
        for images, img_paths in tqdm(test_loader, desc="Processing Batches", unit="batch"):
            images = images.cuda() if torch.cuda.is_available() else images
            outputs = model(images)  # Extract features
            features.append(outputs.cpu().numpy())
            image_paths.extend(img_paths)  # Save image paths for reference
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save features and image paths to a file
    features = np.concatenate(features, axis=0)
    print(f"Features shape: {features.shape}")  # Print the shape of the features
    np.save(output_path, features)  # Save features as .npy file
    with open(output_path.replace('.npy', '_paths.txt'), 'w') as f:
        for path in image_paths:
            filename = os.path.basename(path)
            f.write(f"{filename}\n")  # Save image paths as a text file
    
    
# Main Testing Function
def main_test():
    # Define paths
    test_folder = './labeled_images_example'  # Path to testing images
    checkpoint_path = './ckpt/last.pth'  # Path to trained model
    output_path = './simCLR_features_200.npy'  # Path to save features

    # Load the trained model
    model = load_model(checkpoint_path, model_name='resnet18')

    # Prepare the testing DataLoader
    test_loader = prepare_test_loader(test_folder, batch_size=4)

    # Run inference and save features
    save_testing_results(model, test_loader, output_path)
    print(f"Testing features saved to {output_path}")

# Run the testing script
if __name__ == '__main__':
    main_test()