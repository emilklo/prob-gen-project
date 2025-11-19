import argparse
import yaml
from src.utils.device import get_compute_device

def train_vae(cfg):
    """Training loop for VAE."""
    device = get_compute_device()
    print(f"Training VAE on {device}")
    # Implementation goes here
    pass

def train_rnn(cfg):
    """Training loop for RNN (requires trained VAE)."""
    device = get_compute_device()
    print(f"Training RNN on {device}")
    # Implementation goes here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model components")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["vae", "rnn"], required=True, help="Component to train")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if args.mode == "vae":
        train_vae(cfg)
    elif args.mode == "rnn":
        train_rnn(cfg)
