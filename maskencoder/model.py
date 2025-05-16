from torch import nn


# ----------------------------------------------------------------
# define the model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
#             nn.ReLU(True),
#             ResidualBlock(16),
#             nn.Conv2d(16, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             ResidualBlock(16),
#             nn.Conv2d(16, 12, 3, stride=1, padding=1),  # b, 12, 5, 5
#             nn.ReLU(True),
#             nn.Conv2d(12, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             ResidualBlock(8),
#             nn.ConvTranspose2d(8, 12, 3, stride=2, padding=1),  # b, 12, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(12, 16, 2, stride=2),  # b, 16, 10, 10
#             nn.ReLU(True),
#             ResidualBlock(16),
#             nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),  # b, 1, 28, 28
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
    
    
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=2, padding=1),     # → [16, 128, 128]
#             nn.ReLU(True),
#             ResidualBlock(16),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),    # → [32, 64, 64]
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),                    # → [32, 32, 32]
#             ResidualBlock(32),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),    # → [64, 16, 16]
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),   # → [128, 8, 8]
#             nn.ReLU(True),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1),  # → [256, 4, 4]
#             nn.ReLU(True),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # → [128, 8, 8]
#             nn.ReLU(True),
#             ResidualBlock(128),
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # → [64, 16, 16]
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # → [32, 32, 32]
#             nn.ReLU(True),
#             ResidualBlock(32),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),    # → [16, 64, 64]
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),     # → [8, 128, 128]
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),      # → [1, 256, 256]
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#          encoded = self.encoder(x)
#          decoded = self.decoder(encoded)
#          return encoded, decoded



import torch.nn as nn

# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, 3, stride=2, padding=1),      # [8, 128, 128]
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, 3, stride=2, padding=1),     # [16, 64, 64]
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),    # [32, 32, 32]
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),    # [64, 16, 16]
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),   # [128, 8, 8]
#             nn.ReLU(True),
#         )
        
        # self.encoder_fc = nn.Linear(128 * 8 * 8, 128)

        # # Decoder
        # self.decoder_fc = nn.Linear(128, 128 * 8 * 8)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # [64, 16, 16]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # [32, 32, 32]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),    # [16, 64, 64]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),     # [8, 128, 128]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),      # [1, 256, 256]
        #     nn.Sigmoid()
        # )
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(  #[32, 1, 256 , 256]
            nn.Conv2d(1, 8, 4, stride=4),      # [8, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, stride=4),     # [16, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=4),    # [32, 4, 4]
            nn.ReLU(True),
            nn.Conv2d(32, 32, 2, stride=2),    # [32, 2, 2]
            nn.ReLU(True),
            nn.Conv2d(32, 32, 2, stride=2),    # [32, 1, 1]
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),      # [32, 2, 2]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),      # [32, 4, 4]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=4),      # [16, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=4),       # [8, 64, 64]
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=4),        # [1, 256, 256]
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)                         # [B, 128, 8, 8]
        # x_flat = x.view(x.size(0), -1)              # Flatten: [B, 128*8*8]
        # encoded = self.encoder(x)           # [B, 128]
        # print(f'encodedshape:',encoded.shape)
        # x = self.decoder_fc(encoded)                # [B, 128*8*8]
        # x = x.view(x.size(0), 128, 8, 8)            # Reshape to [B, 128, 8, 8]
        decoded = self.decoder(encoded)                   # [B, 1, 256, 256]
        
        return encoded, decoded






class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Classifier, self).__init__()
        dim = 8*2*2 # 8*2*2 is the output shape of encoder
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),  
            nn.ReLU(True),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x