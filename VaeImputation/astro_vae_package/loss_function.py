# =======================
# Loss Function (center-only circle reconstruction)
# =======================

import torch
import torch.nn.functional as F
def get_pattern_loss(recon_x, target_x, kernel):
    conv_recon = F.conv2d(recon_x, kernel)
    conv_target = F.conv2d(target_x, kernel)
    return F.mse_loss(conv_recon, conv_target, reduction='none')

def loss_function_circle_center(recon_x, target_x, mu, logvar, 
                                kl_weight=1.0, boundary_weight=1.0,
                                circle_radius=50, exclude_radius=1, 
                                boundary_width=5):
    """
    Computes VAE loss with reconstruction constrained to a circular region, averaged per channel.
    """
    batch_size, channels, height, width = recon_x.size()
    device = recon_x.device
    
    # Create circular and boundary masks once (shape: 1x1xHxW)
    Y, X = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    dist = torch.sqrt((X - width // 2) ** 2 + (Y - height // 2) ** 2)

    mask_center = ((dist > exclude_radius) & (dist <= circle_radius)).float()
    mask_boundary = ((dist >= circle_radius -1) & (dist <= circle_radius + boundary_width)).float()

    # Expand masks to shape (1, C, H, W) to match recon_x and target_x
    mask_center = mask_center.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
    mask_boundary = mask_boundary.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)

    # Compute masked MSE per channel
    recon_loss_per_channel = F.mse_loss(recon_x * mask_center, target_x * mask_center, reduction='none')
    boundary_loss_per_channel = F.mse_loss(recon_x * mask_boundary, target_x * mask_boundary, reduction='none')
    
    # Sum over spatial dimensions, average over batch and channels
    recon_loss = recon_loss_per_channel.sum(dim=(2, 3)) / mask_center.sum(dim=(2, 3))
    boundary_loss = boundary_loss_per_channel.sum(dim=(2, 3)) / mask_boundary.sum(dim=(2, 3))

    recon_loss = recon_loss.mean()  # Average over batch and channels
    boundary_loss = boundary_loss.mean()

    # KL divergence (averaged over batch)
    kl =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    # kl = 0

    # Pattern loss
    kernel_1 = torch.tensor([[-1., -1., -1.],
                        [0., 0., 0.],
                        [1., 1., 1.]]).view(1, 1, 3, 3).repeat(batch_size, channels, 1, 1).to(device)
    kernel_2 = torch.tensor([[-1., 0., 1.],
                        [-1., 0., 1.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(batch_size, channels, 1, 1).to(device)
    kernel_3 = torch.tensor([[2., 1., 0.],
                        [1., 0., -1.],
                        [0., -1., -2.]]).view(1, 1, 3, 3).repeat(batch_size, channels, 1, 1).to(device)
    kernel_4 = torch.tensor([[0., 1., 2.],
                        [-1., 0., 1.],
                        [-2., -1., 0.]]).view(1, 1, 3, 3).repeat(batch_size, channels, 1, 1).to(device)

    pattern_loss_1 = get_pattern_loss(recon_x, target_x, kernel_1).mean()
    pattern_loss_2 = get_pattern_loss(recon_x, target_x, kernel_2).mean()
    pattern_loss_3 = get_pattern_loss(recon_x, target_x, kernel_3).mean()
    pattern_loss_4 = get_pattern_loss(recon_x, target_x, kernel_4).mean()
    
    pattern_loss = pattern_loss_1 + pattern_loss_2 + pattern_loss_3 + pattern_loss_4
    total_loss = recon_loss + kl_weight * kl + boundary_weight * boundary_loss + pattern_loss 

    return total_loss, recon_loss.item(), kl.item(), boundary_loss.item(), pattern_loss_1.item(), pattern_loss_2.item(), pattern_loss_3.item(), pattern_loss_4.item(), 