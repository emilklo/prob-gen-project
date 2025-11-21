from pathlib import Path
import json
from dataclasses import asdict
from typing import Any


def get_unique_path(path_str: str | Path) -> Path:
    """
    Takes a path string. If the path exists, appends an incrementing number
    (e.g., _1, _2) to the filename until a unique path is found.
    """
    path = Path(path_str) if isinstance(path_str, str) else path_str
    # make sure folder exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # If the file/folder doesn't exist, return original path
    if not path.exists():
        return path

    # Breakdown the path parts
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    counter = 1

    while True:
        # Create new filename: name_1.ext
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name

        # Check if this new variation exists
        if not new_path.exists():
            return new_path

        counter += 1


def setup_run_directory(base_output_dir: Path, run_name: str, cfg: Any) -> Path:
    """
    Sets up the run directory.
    - If directory exists and config matches: Resumes (returns existing path).
    - If directory exists and config differs: Creates unique path (returns new path).
    - If directory doesn't exist: Creates it.

    Args:
        base_output_dir: Base path for outputs (e.g. "outputs")
        run_name: Proposed run name
        cfg: Configuration object (must be dataclass or have asdict)

    Returns:
        Path to the run directory
    """
    target_dir = base_output_dir / run_name

    if target_dir.exists():
        config_path = target_dir / "config.json"
        should_resume = False

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)

                # Convert current config to dict/json-compatible format for comparison
                current_config = json.loads(json.dumps(asdict(cfg)))

                if existing_config == current_config:
                    should_resume = True
                    print(
                        f"[-] Config matches existing run at {target_dir}. Resuming..."
                    )
                else:
                    print(
                        f"[-] Config mismatch at {target_dir}. Creating new unique run."
                    )
            except Exception as e:
                print(
                    f"[!] Error reading existing config: {e}. Creating new unique run."
                )

        if should_resume:
            run_dir = target_dir
        else:
            # Create unique path (e.g. run_name_1)
            run_dir = Path(get_unique_path(str(target_dir)))
    else:
        run_dir = target_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    # Update config run_name and save metadata
    # Note: We modify the cfg object in place if it has a run_name attribute
    if hasattr(cfg, "run_name"):
        cfg.run_name = run_dir.name

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    return run_dir
