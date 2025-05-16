import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import MNISTDiffusion  # Replace with your model class
from train import UnsupervisedImageDataset  # Replace with your dataset class
from train import create_custom_dataloaders  # If needed
import os
import numpy as np 
from utils import ExponentialMovingAverage
import matplotlib.pyplot as plt

def test_model(checkpoint_path, test_images_path, output_dir, image_size=256, batch_size=4, device="cuda"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize the model
    model = MNISTDiffusion(
        timesteps=500,  # Adjust based on your model
        image_size=image_size,
        in_channels=1,  # Adjust based on your input (1 for grayscale, 3 for RGB)
        base_dim=4,  # Adjust based on your model
        dim_mults=[8, 16]
    ).to(device)
    model.load_state_dict(checkpoint["model"])  # Load model weights

    adjust = 1* 16 * 10 / 200
    alpha = 1.0 - 0.995
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    # Preprocessing for the test images
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5])
    ])

    # Load the test dataset
    test_dataset = UnsupervisedImageDataset(root=test_images_path, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) ## no shuffle

    # Run the model on the test images
    os.makedirs(output_dir, exist_ok=True)
    import csv
    mapping_file = os.path.join(output_dir, "v_feat_mapping.csv")
    with open(mapping_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["original_filename", "v_feat_path"])
        
        with torch.no_grad():
            for i, (images, filenames) in enumerate(test_loader):
                images = images.to(device)
                ### 1 is empty, no need a condition here, but need change it to 4? need to match the batch
                samples=model_ema.module.cond_sampling(batch_size, images, clipped_reverse_diffusion=True,device=device) # Forward pass through the model
                v_feat = model_ema.module.v
                print(f"v_feat shape: {v_feat.shape}")
                v_feat_numpy = v_feat.cpu().numpy()
                for j in range(v_feat_numpy.shape[0]):  # Loop through the batch
                    feat_path = f"{output_dir}/v_feat_image_{i * batch_size + j}.npy"
                    np.save(feat_path, v_feat_numpy[j])
                    csv_writer.writerow([filenames[j], os.path.basename(feat_path)])
                    print(f"Saved v_feat for image {i * batch_size + j} to {output_dir}/v_feat_image_{i * batch_size + j}.npy")
                # Save the output (latent representations or reconstructed images)
                save_image(samples, f"{output_dir}/output_batch_{i}.png", normalize=True ,value_range=(0, 1))
                #save_image(v_feat, f"{output_dir}/latent_batch_{i}.png")
                #  v_feat as a heatmap
                plt.imshow(v_feat[0].cpu().numpy(), cmap='PuOr', interpolation='nearest')  ## plot the first batch
                plt.savefig(f"{output_dir}/heatmap_batch_{i}.png")
                plt.close()

                print(f"Saved output for batch {i} to {output_dir}/output_batch_{i}.png")

                # Stop after processing 8 images
                # if (i + 1) * batch_size >= 8:
                #     break

# Example usage
test_model(
    checkpoint_path="./results/new_results/steps_00030651.pt",
    test_images_path="./labeled_images",  # Path to your test images
    output_dir="./results/test_outputs_new_20451/labels",
    image_size=256,
    batch_size=4,
    device= torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
)