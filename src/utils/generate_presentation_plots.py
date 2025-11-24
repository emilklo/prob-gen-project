import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.latent_loader import LatentSequenceDataset
from src.models.conv_vea import ConvVAE
from src.models.mdnrnn_pose import DreamerMDRNN
from src.utils.device import get_compute_device, Config
from src.data.odometry_loader import KITTIOdometryDataset
from src.utils.visualization import (
    save_reconstruction_comparison,
    save_latent_samples,
    plot_training_curves,
)
from src.utils.trajectory import evaluate_and_plot_test_sequences
from src.utils.dreaming import evaluate_closed_loop


def generate_plots():
    device = get_compute_device()
    print(f"Using device: {device}")

    # Paths
    vae = None
    vae_checkpoint_path = Path(
        "outputs/vae_z128_img128x416_ep100/checkpoints/vae_epoch_40.pth"
    )
    rnn_checkpoint_path = Path(
        "outputs/FAST_RNN_1l_h256_bs512/rnn_checkpoints/rnn_best.pth"
    )

    output_dir = Path("outputs/presentation_assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. VAE Visualizations ---
    print("\n--- Generating VAE Visualizations ---")
    if vae_checkpoint_path.exists():
        checkpoint = torch.load(
            vae_checkpoint_path, map_location=device, weights_only=False
        )
        vae_config_dict = checkpoint["config"]
        # Reconstruct Config object from dict if necessary, or just access dict
        # The Config class might have a from_dict method or we can just use the dict for parameters

        # Use Config.from_dict to handle potential legacy structures
        vae_cfg = Config.from_dict(vae_config_dict)

        latent_dim = vae_cfg.vae.latent_dim
        img_height = vae_cfg.data.img_height
        img_width = vae_cfg.data.img_width

        vae = ConvVAE(
            latent_dim=latent_dim, img_height=img_height, img_width=img_width
        ).to(device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        vae.eval()

        # Load a small batch of data for reconstruction
        transform = transforms.Compose(
            [transforms.Resize((img_height, img_width)), transforms.ToTensor()]
        )
        # We need the data path. Assuming it's in the config or standard location
        data_path = vae_cfg.data.path
        dataset = KITTIOdometryDataset(
            root_dir=data_path, seq_len=1, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        try:
            batch = next(iter(dataloader))
            # KITTIOdometryDataset returns (images, poses) or just images depending on mode?
            # Looking at odometry_loader.py, __getitem__ returns (image_seq, pose_seq)
            # image_seq shape is (Seq, 3, H, W). Here Seq=1.
            images = batch[0].squeeze(1).to(device)  # (B, 3, H, W)

            with torch.no_grad():
                recon, _, _ = vae(images)

            save_reconstruction_comparison(images, recon, 40, output_dir, num_samples=4)
            print(f"Saved reconstruction comparison to {output_dir}")

            save_latent_samples(vae, 16, output_dir, 40, device)
            print(f"Saved latent samples to {output_dir}")

            # Plot training curves if history exists
            if "loss_history" in checkpoint:
                plot_training_curves(
                    checkpoint["loss_history"], output_dir / "vae_training_curves.png"
                )
                print(f"Saved VAE training curves to {output_dir}")

        except Exception as e:
            print(f"Error generating VAE plots: {e}")
    else:
        print(f"VAE checkpoint not found at {vae_checkpoint_path}")

    # --- 2. RNN Trajectory Plots ---
    print("\n--- Generating RNN Trajectory Plots ---")
    if rnn_checkpoint_path.exists():
        if vae is None:
            print("VAE model not loaded. Skipping RNN trajectory plots.")
            return

        checkpoint = torch.load(
            rnn_checkpoint_path, map_location=device, weights_only=False
        )
        # RNN config might be nested or flat, let's check how it was saved
        # In train_mdnrnn.py: "config": cfg (which is a Config object)
        if "config" in checkpoint:
            rnn_cfg = checkpoint["config"]
        else:
            # lets load the json config file if exists
            config_path = rnn_checkpoint_path.parent.parent / "config.json"
            if config_path.exists():
                rnn_cfg = Config.from_file(config_path)
            else:
                print("Loading RNN configuration from config.json", config_path)
                raise ValueError(
                    "RNN configuration not found in checkpoint or as config.json"
                )

        # Ensure we have the VAE for the RNN evaluation
        # We already loaded 'vae' above, assuming it's the compatible one.

        rnn = DreamerMDRNN(
            latent_dim=rnn_cfg.vae.latent_dim,
            hidden_size=rnn_cfg.rnn.hidden_size,
            num_layers=rnn_cfg.rnn.num_layers,
        ).to(device)
        rnn.load_state_dict(checkpoint["model_state_dict"])
        rnn.eval()

        test_sequences = getattr(rnn_cfg.data, "test_sequences", None)
        if not test_sequences:
            test_sequences = [
                "00",
                "01",
                "02",
                "09",
                "10",
            ]  # Default test sequences if none specified

        evaluate_and_plot_test_sequences(
            model=rnn,
            vae=None,
            test_sequences=test_sequences,
            cfg=rnn_cfg,
            save_dir=output_dir,
            epoch=999,
            device=device,
            unique_plots=False,
        )
        print(f"Saved trajectory plots to {output_dir}")

        # --- 3. Dreaming (Closed Loop) ---
        print("\n--- Generating Dreaming Plots (Closed Loop) ---")

        # Path to where you saved the .npz files in the preprocessing step
        # Assuming standard location:
        processed_data_dir = "data/processed_latents"

        for seq_id in test_sequences:
            try:
                # Initialize the FAST loader for this specific sequence
                # We pass sequences=[seq_id] so it only loads the one we want to test
                dataset = LatentSequenceDataset(
                    processed_dir=processed_data_dir,
                    seq_len=rnn_cfg.rnn.sequence_length,
                    select_sequences=[seq_id],
                )

                if len(dataset) > 0:
                    evaluate_closed_loop(rnn, vae, dataset, device, output_dir, seq_id)
            except Exception as e:
                print(f"Failed to dream on sequence {seq_id}: {e}")
                import traceback

                traceback.print_exc()

    else:
        print(f"RNN checkpoint not found at {rnn_checkpoint_path}")


if __name__ == "__main__":
    generate_plots()
