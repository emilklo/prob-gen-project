from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml
from pathlib import Path
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


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int


@dataclass
class VaeConfig:
    latent_dim: int


@dataclass
class RnnConfig:
    hidden_size: int
    num_layers: int


@dataclass
class ModelConfig:
    vae: VaeConfig
    rnn: RnnConfig


@dataclass
class DataConfig:
    path: str
    img_height: int
    img_width: int
    vea_sequence_length: int
    rnn_sequence_length: int


@dataclass
class VisualizationConfig:
    save_every: int
    num_samples: int


@dataclass
class Config:
    run_name: str
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    visualization: VisualizationConfig
    device: str = field(default_factory=lambda: str(get_compute_device()))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries.
    Values in 'override' overwrite values in 'base'.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_dict(config_dir: str = "config") -> Dict[str, Any]:
    """
    Loads the default configuration and overrides it with device-specific settings.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / config_dir
    default_config_path = config_path / "default.yaml"

    if not default_config_path.exists():
        config_path = Path(config_dir).resolve()
        default_config_path = config_path / "default.yaml"

    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_config_path}")

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_compute_device().type
    device_config_path = config_path / f"{device}.yaml"

    if device_config_path.exists():
        print(f"Loading device-specific config from {device_config_path}")
        with open(device_config_path, "r") as f:
            device_config = yaml.safe_load(f)
        config = deep_merge(config, device_config)
    else:
        print(f"No device-specific config found for {device}. Using default.")

    return config


def get_config() -> Config:
    """
    Loads the configuration and returns a typed Config object.
    """
    data = load_config_dict()

    return Config(
        run_name=data.get("run_name", "default_run"),
        training=TrainingConfig(**data["training"]),
        model=ModelConfig(
            vae=VaeConfig(**data["model"]["vae"]), rnn=RnnConfig(**data["model"]["rnn"])
        ),
        data=DataConfig(**data["data"]),
        visualization=VisualizationConfig(**data["visualization"]),
    )
