import torch
from resnet_big_old import SupConResNet, LinearClassifier
from torchvision import transforms
from PIL import Image
import os

def load_checkpoint(checkpoint_path, model, classifier, optimizer=None):
    """Load the checkpoint and restore the model, classifier, and optimizer states."""
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

import numpy as np

def test_model(checkpoint_path, image_folder, output_folder,  transform, num_classes=2):
    """Load the checkpoint and test the classifier on a sample dataset."""
    # Initialize the model and classifier
    model = SupConResNet(name='resnet18', head='mlp', feat_dim=32).cuda()
    classifier = LinearClassifier(name='resnet18', num_classes=num_classes).cuda()

    # Load the checkpoint
    load_checkpoint(checkpoint_path, model, classifier)

    # Set the model and classifier to evaluation mode
    model.eval()
    classifier.eval()

    # Load and preprocess the images
    image_paths = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, fname)) and fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
    ]
    print(f"Found {len(image_paths)} images in '{image_folder}'")

    # Prepare storage for results
    predicted_classes = []
    features_list = []

    # Open a text file to store image paths
    img_path_file = os.path.join(output_folder, "image_paths_cluster_32.txt")
    with open(img_path_file, 'w') as f:
        for img_path in image_paths:
            # Load and preprocess the image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension

            # Pass the image through the model and classifier
            with torch.no_grad():
                features = model(input_tensor)
                output = classifier(features)
                predicted_class = torch.argmax(output, dim=1).item()

            # Save the results
            predicted_classes.append(predicted_class)
            features_list.append(features.cpu().numpy().squeeze())  # Convert to NumPy array
            f.write(f"{img_path}\n")  # Write image path to the text file

            print(f"Image: {img_path} | Predicted Class: {predicted_class}")

    # Save predicted classes and features as .npy files
    np.save(os.path.join(output_folder, "predicted_classes_cluster_32.npy"), np.array(predicted_classes))
    np.save(os.path.join(output_folder, "features_cluster_32.npy"), np.array(features_list))

    print(f"Image paths saved to: {img_path_file}")
    print(f"Predicted classes saved to: {os.path.join(output_folder, 'predicted_classes_cluster_32.npy')}")
    print(f"Features saved to: {os.path.join(output_folder, 'features_cluster_32.npy')}, shape: {np.array(features_list).shape}")

if __name__ == "__main__":
    # Path to the checkpoint
    checkpoint_path = "./ckpt/cluster_method_new/ckpt_epoch_200.pt"

    # Path to the folder containing test images
    image_folder = "./labeled_images"
    output_folder = "./downstream_tasks"

    # Define the image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match training image size
        transforms.ToTensor(),
    ])

    # Test the model
    test_model(checkpoint_path, image_folder, output_folder, transform)