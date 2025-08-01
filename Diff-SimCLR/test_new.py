import os
import torch
import csv
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from networks.resnet_big import SupConResNet
from tqdm import tqdm


class CSVTestDataset(Dataset):
    def __init__(self, root,transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.feature_paths = []
        csv_path = os.path.join(root, "v_feat_mapping.csv")
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                img_name, feat_name = row
                self.image_paths.append(os.path.join(root, img_name))
                self.feature_paths.append(os.path.join(root, 'features', feat_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        feature = np.load(self.feature_paths[idx])
        feature = torch.tensor(feature, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, feature, img_path

# Prepare DataLoader
def prepare_test_loader(root,batch_size=4):
    normalize = transforms.Normalize((0.5,), (0.5,))
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CSVTestDataset(root=root,transform=test_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)





# Function to Load the Trained Model
def load_model(checkpoint_path, model_name='resnet18'):
    # Initialize the model (use SupConResNet as in training)
    model = SupConResNet(name=model_name)
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


def save_testing_results(model, test_loader, output_path):
    model.eval()
    features = []
    image_paths = []
    
    print("start testing....")
    with torch.no_grad():  # Disable gradient computation for inference
        for images, feats, img_paths in tqdm(test_loader, desc="Processing Batches", unit="batch"):
            images = images.cuda() if torch.cuda.is_available() else images
            feats = feats.cuda() if torch.cuda.is_available() else feats
            outputs = model(images, feats)  # Extract features
            features.append(outputs.cpu().numpy())
            image_paths.extend(img_paths)  # Save image paths for reference
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save features and image paths to a file
    features = np.concatenate(features, axis=0)
    print(f"Features shape: {features.shape}")  # Print the shape of the features
    np.save(output_path, features)  # Save features as .npy file
    with open(output_path.replace('.npy', '_paths.txt'), 'w') as f:
        for path in image_paths:
            filename = os.path.basename(path)  # Extract filename from path
            f.write(f"{filename}\n")  # Save image paths as a text file
    
    
# Main Testing Function
def main_test():
    # Define paths
    test_folder = './labeled_images_example'  # Path to testing images
    checkpoint_path = './ckpt/last.pth'  # Path to trained model
    output_path = './test_features_200.npy'  # Path to save features

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