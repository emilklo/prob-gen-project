# Visual Odometry Pipeline

A world model approach to visual odometry using VAE + MDN-RNN on the KITTI dataset.

## What This Project Does

Predicts vehicle ego-motion (how the camera moves) from driving images:

```
Camera Image → VAE Encoder → Latent z → MDN-RNN → 6-DOF Pose
                (128-d)                         [dx, dy, dz, roll, pitch, yaw]
```

The predicted poses can be integrated into a full trajectory and compared against GPS ground truth.

---

## Architecture

### Stage 1: VAE (Variational Autoencoder)
- **Input**: RGB image (192 × 640)
- **Output**: 128-dimensional latent vector
- **Purpose**: Compress visual information into compact representation

### Stage 2: MDN-RNN (Mixture Density Network RNN)
- **Input**: Sequence of latent vectors
- **Output**:
  - Pose prediction: `[dx, dy, dz, roll, pitch, yaw]`
  - Future latent distribution (5 Gaussians)
- **Purpose**: Learn temporal dynamics and predict motion

---

## Current Configuration

**Config file**: `config/cuda.yaml`

| Parameter | Value |
|-----------|-------|
| run_name | model_emil_z128 |
| latent_dim | 128 |
| hidden_size | 512 |
| num_layers | 2 |
| batch_size | 128 |
| learning_rate | 0.0001 |
| epochs | 100 |
| image_size | 192 × 640 |
| seq_length | 5 |

---

## Training Pipeline

### Improvements Implemented

- **Data augmentation**: ColorJitter, GaussianBlur
- **Normalization**: ImageNet mean/std
- **Validation split**: Last sequence held out
- **Early stopping**: Patience of 10 epochs
- **LR scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Dropout**: 0.1 between LSTM layers

### Step 1: Train VAE

```bash
sbatch train_vae.slurm
```

**What it does**:
1. Loads KITTI images from all sequences except the last (validation)
2. Trains VAE to reconstruct images
3. Saves best model based on validation loss
4. Early stops if no improvement for 10 epochs

**Output**: `outputs/model_emil_z128/checkpoints/vae_best.pth`

### Step 2: Train MDN-RNN

```bash
sbatch train_rnn.slurm
```

**What it does**:
1. Loads frozen VAE encoder
2. Encodes image sequences to latent space
3. Trains RNN to predict pose deltas and future latents
4. Validates on held-out sequence

**Output**: `outputs/model_emil_z128/rnn_checkpoints/rnn_best.pth`

### Step 3: Evaluate Trajectory

```bash
python -m src.utils.trajectory
```

**What it does**:
1. Loads trained VAE + RNN
2. Processes evaluation sequence (currently seq 03)
3. Predicts pose deltas frame-by-frame
4. Integrates into full trajectory
5. Plots predicted vs ground truth path

**Output**: `trajectory_result.png`

---

## File Structure

```
prob-gen-project/
├── config/
│   ├── default.yaml          # Base configuration
│   └── cuda.yaml              # GPU-specific overrides
├── src/
│   ├── models/
│   │   ├── world_model.py     # ConvVAE
│   │   └── mdnrnn_pose.py     # DreamerMDRNN
│   ├── data/
│   │   ├── loaders.py         # KITTIDataset (images only)
│   │   └── odometry_loader.py # KITTIOdometryDataset (images + poses)
│   ├── train.py               # VAE training
│   ├── train_mdnrnn.py        # RNN training
│   └── utils/
│       └── trajectory.py      # Evaluation & plotting
├── train_vae.slurm            # SLURM job for VAE
├── train_rnn.slurm            # SLURM job for RNN
└── outputs/
    └── model_emil_z128/
        ├── checkpoints/       # VAE checkpoints
        └── rnn_checkpoints/   # RNN checkpoints
```

---

## Data Format

### KITTI Odometry Structure

```
data/kitti/
├── 00/
│   └── image_02/data/*.png    # Left camera images
├── 01/
├── 02/
├── 03/
└── poses/
    ├── 00.txt                 # Ground truth (N × 12 matrix)
    ├── 01.txt
    └── ...
```

### Pose Format

Each pose file line contains a flattened 3×4 transformation matrix:
```
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

The data loader converts these to relative 6-DOF deltas between consecutive frames.

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/vae_<job_id>.out
tail -f logs/rnn_<job_id>.out

# Check GPU usage
ssh <node> nvidia-smi
```

---

## Key Hyperparameters to Tune

| Parameter | Location | Effect |
|-----------|----------|--------|
| `pose_weight` | train_mdnrnn.py:221 | Balance MDN vs pose loss (currently 1000) |
| `patience` | train.py, train_mdnrnn.py | Early stopping threshold (currently 10) |
| `dropout` | mdnrnn_pose.py:17 | Regularization (currently 0.1) |
| `num_gaussians` | mdnrnn_pose.py:16 | MDN mixture components (currently 5) |

---

## Expected Results

After training:
- **VAE**: PSNR > 20 dB on validation
- **RNN**: Pose MSE < 0.001 on validation
- **Trajectory**: Visual alignment with ground truth path

The trajectory plot shows:
- **Black line**: Ground truth GPS trajectory
- **Red dashed**: Model predicted trajectory

Good alignment indicates the model learned meaningful motion patterns.
