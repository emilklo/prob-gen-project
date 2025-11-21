from dataclasses import dataclass, field
import json

from typing import Any
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
    sequence_length: int
    training: TrainingConfig


@dataclass
class RnnConfig:
    hidden_size: int
    num_layers: int
    sequence_length: int
    training: TrainingConfig


@dataclass
class DataConfig:
    path: str
    pose_path: str
    test_sequences: list[str]
    img_height: int
    img_width: int
    # Sequence lengths moved to Model Configs


@dataclass
class VisualizationConfig:
    save_every: int
    num_samples: int


# --- LEGACY SUPPORT START -----------------------------------------
# This class MUST exist for torch.load to unpickle old checkpoints.
# It is not used in new code, but the unpickler looks for it.
@dataclass
class ModelConfig:
    vae: VaeConfig
    rnn: RnnConfig


# --- LEGACY SUPPORT END -------------------------------------------


@dataclass
class Config:
    run_name: str
    vae: VaeConfig
    rnn: RnnConfig
    data: DataConfig
    visualization: VisualizationConfig
    device: str = field(default_factory=lambda: str(get_compute_device()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Factory method to create a Config object.
        Handles backward compatibility for old config structures.
        """
        # ---------------------------------------------------------
        # MIGRATION LOGIC: Handle Old Config Structure
        # ---------------------------------------------------------
        if "model" in data:
            print("Detected legacy configuration format. Migrating to new structure...")

            # 1. Extract global training config (used for both VAE and RNN in old format)
            global_training = data.get("training", {})

            # 2. Extract Model Params
            old_model_vae = data["model"].get("vae", {})
            old_model_rnn = data["model"].get("rnn", {})

            # 3. Extract Data Params and Sequence Lengths
            data_cfg = data.get("data", {})
            # Handle the specific typo 'vea' from old config if present
            vae_seq_len = data_cfg.pop(
                "vea_sequence_length", data_cfg.pop("vae_sequence_length", 1)
            )
            rnn_seq_len = data_cfg.pop("rnn_sequence_length", 5)

            # 4. Construct VAE Config (New Format)
            vae_config = VaeConfig(
                latent_dim=old_model_vae.get("latent_dim", 64),
                sequence_length=vae_seq_len,
                training=TrainingConfig(**global_training),
            )

            # 5. Construct RNN Config (New Format)
            rnn_config = RnnConfig(
                hidden_size=old_model_rnn.get("hidden_size", 256),
                num_layers=old_model_rnn.get("num_layers", 1),
                sequence_length=rnn_seq_len,
                training=TrainingConfig(**global_training),
            )

            # 6. Construct Data Config
            # Ensure test_sequences exists for robustness
            if "test_sequences" not in data_cfg:
                data_cfg["test_sequences"] = ["00"]

            data_obj = DataConfig(**data_cfg)

            return cls(
                run_name=data.get("run_name", "legacy_run"),
                vae=vae_config,
                rnn=rnn_config,
                data=data_obj,
                visualization=VisualizationConfig(**data.get("visualization", {})),
            )

        # ---------------------------------------------------------
        # STANDARD LOGIC: Handle New Config Structure
        # ---------------------------------------------------------
        return cls(
            run_name=data.get("run_name", "default_run"),
            vae=VaeConfig(
                latent_dim=data["vae"]["latent_dim"],
                sequence_length=data["vae"]["sequence_length"],
                training=TrainingConfig(**data["vae"]["training"]),
            ),
            rnn=RnnConfig(
                hidden_size=data["rnn"]["hidden_size"],
                num_layers=data["rnn"]["num_layers"],
                sequence_length=data["rnn"]["sequence_length"],
                training=TrainingConfig(**data["rnn"]["training"]),
            ),
            data=DataConfig(**data["data"]),
            visualization=VisualizationConfig(**data["visualization"]),
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Config":
        """
        Loads configuration from a YAML or JSON file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            if path.suffix == ".json":
                data = json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls.from_dict(data)

    # --- MAGIC METHOD FOR PICKLE MIGRATION ---
    def __setstate__(self, state):
        """
        This magic method is called when torch.load (pickle) reconstructs the object.
        We use it to migrate old checkpoint data to the new class structure.
        """
        if "model" in state:
            # We detected an OLD config format in the pickle file!
            # Structure was: {model: ModelConfig(vae, rnn), training: TrainingConfig, ...}

            print("Migrating legacy checkpoint config to new structure...")

            # 1. Extract the old components
            old_model = state["model"]  # This is the ModelConfig instance
            old_training = state["training"]  # This is the global TrainingConfig

            # 2. Setup VAE (Inject missing fields)
            self.vae = old_model.vae
            self.vae.training = old_training  # Duplicate global training to VAE
            # Fallback for sequence length if missing (old value was in data)
            if not hasattr(self.vae, "sequence_length"):
                self.vae.sequence_length = 1

            # 3. Setup RNN (Inject missing fields)
            self.rnn = old_model.rnn
            self.rnn.training = old_training  # Duplicate global training to RNN
            if not hasattr(self.rnn, "sequence_length"):
                self.rnn.sequence_length = 5

            # 4. Map remaining fields
            self.data = state["data"]
            self.visualization = state["visualization"]
            self.run_name = state.get("run_name", "legacy_run")
            self.device = state.get("device", "cpu")

        else:
            # Standard loading for new checkpoints
            self.__dict__.update(state)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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


def load_config_dict(config_dir: str = "config") -> dict[str, Any]:
    """
    Loads the default configuration and overrides it with device-specific settings.
    Returns a raw dictionary.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / config_dir
    default_config_path = config_path / "default.yaml"

    # Fallback logic if running from different relative paths
    if not default_config_path.exists():
        config_path = Path(config_dir).resolve()
        default_config_path = config_path / "default.yaml"

    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_config_path}")

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device-specific overrides (e.g., config/cuda.yaml or config/mps.yaml)
    device = get_compute_device().type
    device_config_path = config_path / f"{device}.yaml"

    if device_config_path.exists():
        print(f"Loading device-specific config from {device_config_path}")
        with open(device_config_path, "r") as f:
            device_config = yaml.safe_load(f)
        config = deep_merge(config, device_config)

    return config


def get_config() -> Config:
    """
    Loads the configuration and returns a typed Config object.
    """
    data = load_config_dict()
    return Config.from_dict(data)
