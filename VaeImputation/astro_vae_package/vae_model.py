import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset

class AstroConvVae(nn.Module):
    def __init__(self, in_channel=10, latent_dim=16):
        super().__init__()

        # Encoder with MaxPool2d for spatial downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, padding=1),  # (1 img, 256px, 256px)-> (32 channel, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # -> (32, 128, 128)

            nn.Conv2d(32, 64, 3, padding=1),          # -> (64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # -> (64, 64, 64)

            nn.Conv2d(64, 128, 3, padding=1),         # -> (128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # -> (128, 32, 32)

        )

        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_bias = nn.Linear(128 * 32 * 32, 1)
        self.fc_cf = nn.Linear(128 * 32 * 32, 1)
        self.fc_decode = nn.Linear(latent_dim, 128 * 32 * 32)

        # Decoder: Upsample with ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # (32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # (1, 256, 256)
            nn.Sigmoid()  # Match input normalization [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.flatten()
        return self.fc_mu(h_flat), self.fc_logvar(h_flat), self.fc_cf(h_flat), self.fc_bias(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cf, bias):
        h = self.fc_decode(z).view(-1, 128, 32, 32)
        return (self.decoder(h) - 0.5) * cf + bias

    def forward(self, x):
        mu, logvar, cf, bias = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cf, bias) 
        return x_hat, mu, logvar