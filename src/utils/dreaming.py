import torch
import numpy as np
from src.utils.trajectory import integrate_path, get_unique_path
import matplotlib.pyplot as plt

def sample_from_mdn(pi, mu, sigma):
    """
    Samples a latent vector z from the GMM parameters output by the MDN.
    pi: (Batch, Seq, K_Gaussians)
    mu: (Batch, Seq, K, Latent_Dim)
    sigma: (Batch, Seq, K, Latent_Dim)
    """
    # 1. Select which Gaussian to use (based on pi probabilities)
    # For a smoother trajectory, you can just pick the argmax (most likely)
    # For strict generative sampling, use torch.multinomial
    
    # Let's use the most likely Gaussian (argmax) for trajectory stability
    k_best = torch.argmax(pi, dim=2) # Shape: (Batch, Seq)
    
    # Gather the mu and sigma corresponding to k_best
    # We need to gather across the K dimension (dim=2)
    # mu shape: (B, S, K, L)
    # k_best shape: (B, S) -> unsqueeze to (B, S, 1, 1) -> expand to (B, S, 1, L)
    
    batch_size, seq_len, k_gaussians, latent_dim = mu.shape
    
    k_expanded = k_best.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, latent_dim)
    
    mu_k = torch.gather(mu, 2, k_expanded).squeeze(2)       # (B, S, L)
    sigma_k = torch.gather(sigma, 2, k_expanded).squeeze(2) # (B, S, L)
    
    # 2. Sample from that Gaussian (reparameterization)
    z_sample = torch.normal(mu_k, sigma_k)
    
    return z_sample # Shape: (Batch, Seq, Latent)

def evaluate_closed_loop(model, vae, dataset, device, save_dir, seq_id, limit=1000):
    model.eval()
    vae.eval()
    
    # 1. INITIALIZATION (t=0)
    # We only take the VERY FIRST frame from the dataset
    imgs, poses = dataset[0] 
    
    # Ground Truth for plotting comparison
    true_deltas_all = []
    
    # Get the initial z from the REAL first image
    img_t0 = imgs[0].unsqueeze(0).to(device)
    current_z = vae.encode(img_t0).unsqueeze(1) # z_0 (Batch=1, Seq=1, Latent)
    
    hidden = None
    pred_deltas = []
    
    print(f"    [-] Starting Dreaming (Closed Loop) for Sequence {seq_id}...")
    
    with torch.no_grad():
        for i in range(limit):
            # A. PREDICT NEXT STEP
            # We feed the *previous* z (which might be real or imagined)
            pi, mu, sigma, pred_pose, hidden = model(current_z, hidden)
            
            # Store the movement prediction
            pred_d = pred_pose[0, 0].cpu().numpy()
            pred_deltas.append(pred_d)
            
            # B. CRITICAL STEP: FEEDBACK
            # We do NOT load a new image from the dataset.
            # We sample z_(t+1) from our own prediction.
            current_z = sample_from_mdn(pi, mu, sigma)
            
            # (Optional) Collect Ground Truth for this step to compare later
            # We have to grab it from the dataset manually since we aren't iterating it
            if i < len(dataset):
                _, true_poses = dataset[i]
                true_deltas_all.append(true_poses[0].cpu().numpy())
                
    # Integrate
    path_pred = integrate_path(pred_deltas)
    path_true = integrate_path(true_deltas_all)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(
        path_true[:, 0], path_true[:, 2], "k-", label="Ground Truth", linewidth=2
    )
    plt.plot(
        path_pred[:, 0], path_pred[:, 2], "b--", label="Dreaming (Closed Loop)", linewidth=2
    )

    plt.title(f"Dreaming Result (Seq {seq_id})")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = get_unique_path(
        save_dir / f"dreaming_seq{seq_id}.png"
    )
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved dreaming plot to {save_path}")
    
    return save_path
