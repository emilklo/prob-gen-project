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
    if vae is not None:
        vae.eval()
    
    # 1. WARMUP (t=0 to t=warmup_steps)
    # Access full sequence directly if possible (LatentSequenceDataset loads all into memory)
    if hasattr(dataset, 'all_mus') and len(dataset.all_mus) > 0:
        # Assuming dataset was initialized with select_sequences=[seq_id], so index 0 is correct
        full_z = dataset.all_mus[0].to(device) # (Total_Frames, Latent)
        full_poses = dataset.all_poses[0].to(device)
    else:
        raise ValueError("Dataset does not have 'all_mus' attribute or is empty.")

    warmup_steps = 5
    hidden = None
    pred_deltas = []
    
    print(f"    [-] Warming up RNN with first {warmup_steps} frames...")
    
    # Feed warmup sequence
    # We want to feed z_0, z_1, ..., z_4 to update hidden state
    warmup_z = full_z[:warmup_steps].unsqueeze(0) # (1, Warmup, Latent)
    
    with torch.no_grad():
        # Pass the entire warmup sequence at once (RNN handles sequence)
        pi, mu, sigma, pred_pose, hidden = model(warmup_z, hidden)
        
        # Collect warmup ground truth deltas for plotting continuity
        # (Optional: we could use predicted deltas during warmup too)
        warmup_deltas = full_poses[:warmup_steps, :].cpu().numpy()
        for i in range(warmup_steps):
             # For plotting, we can use GT or Pred. Let's use GT for "History" context.
             # Note: pred_pose is (1, Warmup, 6)
             pred_deltas.append(warmup_deltas[i])

        # The last output of the warmup (t=4) predicts t=5.
        # We sample z_5 from this prediction to start the dream.
        last_pi = pi[:, -1, :].unsqueeze(1)
        last_mu = mu[:, -1, :].unsqueeze(1)
        last_sigma = sigma[:, -1, :].unsqueeze(1)
        
        current_z = sample_from_mdn(last_pi, last_mu, last_sigma)
        
    print(f"    [-] Starting Dreaming (Closed Loop) for Sequence {seq_id}...")
    
    with torch.no_grad():
        for i in range(limit - warmup_steps):
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
            # We can grab it from full_poses
            # i goes from 0 to limit-warmup.
            # The current step corresponds to t = warmup + i
            # But we are predicting for t+1.
            # Let's just collect ALL ground truth at the end or slice it now.
            pass
                
    # Collect ALL ground truth for the duration
    # We simulated 'limit' steps total (warmup + dream)
    # Actually, loop ran (limit - warmup) times.
    # Total steps = warmup + (limit - warmup) = limit.
    # We need GT for 0 to limit.
    true_deltas_all = full_poses[:limit].cpu().numpy()
                
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
