import torch

def get_compute_device() -> torch.device:
    """
    Selects the best available compute device.
    Returns:
        torch.device: 'cuda' (H100), 'mps' (M4 Max), or 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
