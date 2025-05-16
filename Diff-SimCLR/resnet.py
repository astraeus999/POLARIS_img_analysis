import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

### make the in_channel to be 2

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, zero_init_residual=False):  ## changed in_channel = 1 because it's greyscale
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        #print('x shape:', x.shape)
        x = self.conv1(x)
        #print('x shape:', x.shape)
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
## this net is too big....

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

### handle the feature dimension####
def flatten_feat(tensor):
    # Input: shape [B, C, H] or [B, C, H, W]
    # Output: shape [B, C*H] or [B, C*H*W]
    return tensor.view(tensor.size(0), -1)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SupConResNet(nn.Module):
    """
    SupConResNet with:
      - configurable image input channels (img_in_channels)
      - flexible prior: either image-shaped (prior_in_channels) or flat (prior_dim)
      - still takes name=opt.model so your set_model call stays the same
    """
    def __init__(
        self,
        name: str = 'resnet18',        # e.g. 'resnet18' or 'resnet50'
        head: str = 'mlp',             # 'linear' or 'mlp'
        feat_dim: int = 128,           # output embedding size
        pretrained: bool = False,      
        img_in_channels: int = 3,      # #channels in your main images (you have 1)
        prior_in_channels: int = None, # if your prior is an image: its #channels
        prior_dim: int = None          # if your prior is flat: its flattened size
    ):
        super().__init__()

        # 1) Main image encoder
        backbone = getattr(models, name)(pretrained=pretrained)
        # Patch its first conv to accept img_in_channels
        if img_in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                img_in_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False
            )
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # 2) Prior encoder: either CNN or MLP
        if prior_in_channels is not None:
            # CNN prior branch
            prior_cnn = getattr(models, name)(pretrained=pretrained)
            # patch its first conv for prior_in_channels
            prior_cnn.conv1 = nn.Conv2d(
                prior_in_channels,
                prior_cnn.conv1.out_channels,
                kernel_size=prior_cnn.conv1.kernel_size,
                stride=prior_cnn.conv1.stride,
                padding=prior_cnn.conv1.padding,
                bias=False
            )
            prior_cnn.fc = nn.Identity()
            self.prior_encoder = prior_cnn
            self._use_cnn_prior = True

        elif prior_dim is not None:
            # MLP prior branch
            self.prior_encoder = nn.Sequential(
                nn.Linear(prior_dim, num_feats),
                nn.ReLU(inplace=True),
                nn.Linear(num_feats, num_feats)
            )
            self._use_cnn_prior = False

        else:
            raise ValueError(
                "Must specify either prior_in_channels (for image priors) or prior_dim (for flat priors)."
            )

        # 3) Projection head on [img_feat ∥ prior_feat]
        combined_dim = num_feats * 2
        if head == 'linear':
            self.head = nn.Linear(combined_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.ReLU(inplace=True),
                nn.Linear(combined_dim, feat_dim)
            )
        else:
            raise ValueError(f"Unsupported head type: {head!r}")

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        x     : [B, img_in_channels, H, W]
        prior : either
                  - [B, prior_in_channels, H, W]  (if _use_cnn_prior)
                  - [B, N, D]                      (if flat; we'll reshape→[B, N*D])
        """
        # encode image
        img_feat = self.encoder(x)              # → [B, num_feats]

        # encode prior
        if self._use_cnn_prior:
            prior_feat = self.prior_encoder(prior)  # → [B, num_feats]
        else:
            B = prior.size(0)
            flat = prior.reshape(B, -1)             # → [B, prior_dim]
            prior_feat = self.prior_encoder(flat)   # → [B, num_feats]

        # fuse & project
        combined = torch.cat([img_feat, prior_feat], dim=1)  # [B, 2*num_feats]
        z = self.head(combined)                              # [B, feat_dim]
        return F.normalize(z, dim=1)

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
