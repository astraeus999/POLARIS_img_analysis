import torch
from model import AutoEncoder
from datasets import getdataset
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Test Encoder')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint', default='./ckpt/maskencoder_128/150epoch.pth')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use (e.g., cuda:0 or cpu)')
    args = parser.parse_args()
    return args

def test_encoder():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = AutoEncoder( x = 8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Load the test dataset
    _, test_loader = getdataset(args.batch_size)

    # Extract encoder outputs
    encoder_outputs = []
    data_order = []
    with torch.no_grad():
        for data, names in test_loader:
            data = data.to(device)
            encoder, _ = model(data)  # Extract the encoder output
            # encoder_outputs.append(encoder.cpu())  # Store on CPU for further processing
            encoder = encoder.view(encoder.size(0), -1)  # Flatten to [batch, features]
            encoder_outputs.append(encoder.cpu()) 
            # print(f"Reshaped encoder outputs shape: {encoder_outputs.shape}")
            data_order.extend(names)
    # Combine all encoder outputs into a single tensor
    encoder_outputs = torch.cat(encoder_outputs, dim=0)
    print(f"Encoder outputs shape: {encoder_outputs.shape}")

    # Save the encoder outputs to a file (optional)
    np.save('maskencoder_128_features_150.npy', encoder_outputs.numpy())
    print("Encoder outputs saved to 'maskencoder_128_features_150.npy'")
    # torch.save(encoder_outputs, 'encoder_outputs_new.pt')
    # print("Encoder outputs saved to 'encoder_outputs_new.pt'")
    with open('data_order_new_16_150.txt', 'w') as f:
        for name in data_order:
            f.write(f"{name}\n")
    print("Data order saved to 'data_order_new.txt'")

if __name__ == "__main__":
    test_encoder()