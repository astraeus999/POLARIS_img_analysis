# =======================
# Training Function
# =======================
import os
import torch
from .loss_function import loss_function_circle_center
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.colors as colors

def train_vae(model, train_loader, val_loader, optimizer, device, 
              epochs=50, kl_max=1.0, kl_warmup_epochs=20, boundary_weight=1.0, circle_radius=45, boundary_width = 5,
              print_every=2, save_every=2, 
              checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    
    # Store losses for tracking
    loss_hist = {
        'train_total': [], 'train_recon': [], 'train_kl': [], 'train_boundary': [], 
        'train_pattern_h': [], 'train_pattern_v': [], 'train_pattern_dia': [], 'train_pattern_adia': [],
        'val_total': [], 'val_recon': [], 'val_kl': [], 'val_boundary': [], 
        'val_pattern_h': [], 'val_pattern_v': [], 'val_pattern_dia': [], 'val_pattern_adia': []
    }

    for epoch in range(epochs):
        # Gradually increase kl_weight during warm-up period
        kl_weight = min(kl_max, kl_max * (epoch + 1) / kl_warmup_epochs)
        
        total_loss, total_recon, total_kl, total_boundary, total_pattern_h, total_pattern_v, total_pattern_dia, total_pattern_adia  = 0, 0, 0, 0, 0, 0, 0, 0
        
        # Training loop
        for masked_batch, target_batch in train_loader:
            masked_batch = masked_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(masked_batch)
            
            # Calculate loss for the current batch
            loss, recon_l, kl_l, boundary_l, pattern_l_h, pattern_l_v, pattern_l_dia, pattern_l_adia = loss_function_circle_center(recon_batch, 
                                                                 target_batch, mu, 
                                                                 logvar, kl_weight,
                                                                 boundary_weight, circle_radius, boundary_width=boundary_width)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l
            total_kl += kl_l
            total_boundary += boundary_l
            total_pattern_h += pattern_l_h
            total_pattern_v += pattern_l_v
            total_pattern_dia += pattern_l_dia
            total_pattern_adia += pattern_l_adia
            
        # Log training losses for the epoch
        loss_hist['train_total'].append(total_loss / len(train_loader))
        loss_hist['train_recon'].append(total_recon / len(train_loader))
        loss_hist['train_kl'].append(total_kl / len(train_loader))
        loss_hist['train_boundary'].append(total_boundary / len(train_loader))
        loss_hist['train_pattern_h'].append(total_pattern_h / len(train_loader))
        loss_hist['train_pattern_v'].append(total_pattern_v / len(train_loader))
        loss_hist['train_pattern_dia'].append(total_pattern_dia / len(train_loader))
        loss_hist['train_pattern_adia'].append(total_pattern_adia / len(train_loader))   

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1} | Train Total Loss: {total_loss / len(train_loader):.4f} "
                  f"| Recon Loss: {total_recon / len(train_loader):.4f} | KL Loss: {total_kl / len(train_loader):.4f} | "
                  f"KL Weight: {kl_weight:.4f}"
                  f"Pattern Loss: {(total_pattern_h+total_pattern_v+total_pattern_dia+total_pattern_adia) / len(train_loader):.4f}")

        # Save model at intervals
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch+1:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
        
        # Validation loop
        model.eval()
        val_loss, val_recon, val_kl, val_boundary, val_pattern_h, val_pattern_v, val_pattern_dia, val_pattern_adia = 0, 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for val_masked, val_target in val_loader:
                val_masked = val_masked.to(device)
                val_target = val_target.to(device)
                
                # Get model output
                val_recon_out, val_mu, val_logvar = model(val_masked)
                
                # Calculate validation loss
                val_l, val_r, val_k, val_b, val_h, val_v, val_dia, val_adia = loss_function_circle_center(val_recon_out, 
                                                                     val_target, 
                                                                     val_mu, 
                                                                     val_logvar, 
                                                                     kl_weight, 
                                                                     boundary_weight, 
                                                                     circle_radius)
                val_loss += val_l.item()
                val_recon += val_r
                val_kl += val_k
                val_boundary += val_b
                val_pattern_h += val_h
                val_pattern_v += val_v
                val_pattern_dia += val_dia
                val_pattern_adia += val_adia
                

        # Log validation losses for the epoch
        loss_hist['val_total'].append(val_loss / len(val_loader))
        loss_hist['val_recon'].append(val_recon / len(val_loader))
        loss_hist['val_kl'].append(val_kl / len(val_loader))
        loss_hist['val_boundary'].append(val_boundary / len(val_loader))
        loss_hist['val_pattern_h'].append(val_pattern_h / len(val_loader))
        loss_hist['val_pattern_v'].append(val_pattern_v / len(val_loader))
        loss_hist['val_pattern_dia'].append(val_pattern_dia / len(val_loader))
        loss_hist['val_pattern_adia'].append(val_pattern_adia / len(val_loader))       


        model.train()

        # Print validation losses at intervals
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1} | Val Total Loss: {val_loss / len(val_loader):.4f} "
                  f"| Val Recon Loss: {val_recon / len(val_loader):.4f} | Val KL Loss: {val_kl / len(val_loader):.4f}"
                  f"| Val Boundary Loss: {val_boundary / len(val_loader):.4f}"
                  f"| Val Pattern Loss: {((val_pattern_h + val_pattern_v + val_pattern_dia + val_pattern_adia)/ len(val_loader)):.4f}")

    return loss_hist


# =======================
# Loss Plotting
# =======================
def plot_train_val_loss(loss_hist):
    epochs = range(1, len(loss_hist['train_total']) + 1)
    fig, axs = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

    axs[0].plot(epochs, loss_hist['train_total'], label='Train Total', color='black')
    axs[0].plot(epochs, loss_hist['val_total'], label='Val Total', color='gray', linestyle='--')

    axs[1].plot(epochs, loss_hist['train_recon'], label='Train Recon', color='blue')
    axs[1].plot(epochs, loss_hist['val_recon'], label='Val Recon', color='skyblue', linestyle='--')

    axs[2].plot(epochs, loss_hist['train_kl'], label='Train KL', color='red')
    axs[2].plot(epochs, loss_hist['val_kl'], label='Val KL', color='salmon', linestyle='--')

    axs[3].plot(epochs, loss_hist['train_boundary'], label='Train Boundary', color='darkgreen')
    axs[3].plot(epochs, loss_hist['val_boundary'], label='Val Boundary', color='lightgreen', linestyle='--')

    axs[4].plot(epochs, loss_hist['train_pattern_h'], label='Train Pattern', color='orange')
    axs[4].plot(epochs, loss_hist['val_pattern_h'], label='Val Pattern', color='gold', linestyle='--')

    for ax in axs:
        ax.legend()
        ax.grid(True)

    axs[2].set_xlabel('Epoch')
    axs[0].set_ylabel('Total Loss')
    axs[1].set_ylabel('Reconstruction Loss')
    axs[2].set_ylabel('KL Loss')
    axs[3].set_ylabel('Boundary Loss')
    axs[4].set_ylabel('Pattern Loss')
    plt.tight_layout()
    plt.show()