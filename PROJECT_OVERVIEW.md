# World Model for Visual Odometry

A deep learning project that learns to predict vehicle ego-motion from camera images using a world model architecture inspired by [Ha & Schmidhuber's World Models](https://worldmodels.github.io/) and the [Dreamer](https://arxiv.org/abs/1912.01603) architecture.

## Project Goal

**Predict how a vehicle moves through the world using only camera images.**

Given a sequence of driving images, the model outputs 6-DOF pose deltas:
- **Translation**: `[dx, dy, dz]` - movement in meters
- **Rotation**: `[roll, pitch, yaw]` - orientation changes in radians

This is the core task of **Visual Odometry** - estimating camera motion from visual input alone.

---

## Architecture Overview

The system uses a two-stage approach:

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Camera     │ --> │    VAE      │ --> │   MDN-RNN    │ --> Pose Prediction
│  Image      │     │  (Encoder)  │     │  (Dynamics)  │     [dx,dy,dz,r,p,y]
│ (640x192)   │     │  z (128-d)  │     │              │
└─────────────┘     └─────────────┘     └──────────────┘
```

### Stage 1: Variational Autoencoder (VAE)

**Purpose**: Compress high-dimensional images into compact latent representations.

**File**: `src/models/world_model.py` - `ConvVAE`

**Architecture**:
- **Encoder**: 4 convolutional layers (3→32→64→128→256 channels)
- **Latent space**: 128-dimensional vector
- **Decoder**: 4 transposed convolutions to reconstruct image

**Input**: RGB image `(B, 3, 192, 640)`
**Output**: Latent vector `z` of shape `(B, 128)`

**Training** (`src/train.py`):
- Loss: Reconstruction (MSE) + β-weighted KL divergence
- KL annealing over first 50% of epochs
- Monitors PSNR (Peak Signal-to-Noise Ratio) for reconstruction quality

### Stage 2: Mixture Density Network RNN (MDN-RNN)

**Purpose**: Learn temporal dynamics in latent space and predict ego-motion.

**File**: `src/models/mdnrnn_pose.py` - `DreamerMDRNN`

**Architecture**:
- **Memory**: 2-layer LSTM (hidden size 512)
- **Two prediction heads**:
  1. **Vision Head (MDN)**: Predicts next latent state probabilistically
  2. **Pose Head**: Predicts 6-DOF motion deterministically

**Input**: Sequence of latent vectors `(B, Seq_Len, 128)`
**Outputs**:
- `pi`: Mixture coefficients `(B, Seq, 5)` - which Gaussian to use
- `mu`: Gaussian means `(B, Seq, 5, 128)` - predicted latent centers
- `sigma`: Gaussian stds `(B, Seq, 5, 128)` - uncertainty
- `pose`: Motion prediction `(B, Seq, 6)` - **this is what you submit to KITTI**

**Training** (`src/train_mdnrnn.py`):
- Vision loss: MDN negative log-likelihood
- Pose loss: MSE on 6-DOF deltas
- Combined: `loss = loss_mdn + 1000 * loss_pose`
- The 1000x weight balances the scale (poses are tiny ~0.01, MDN loss is ~10)

---

## What the MDN-RNN Outputs Mean

### The Mixture Density Network (Vision Head)

Instead of predicting a single "next latent", the MDN predicts a **mixture of 5 Gaussians**:

```python
pi, mu, sigma, pose, hidden = rnn(z_sequence)
```

- `pi[i]` = probability of using Gaussian i
- `mu[i]` = mean of Gaussian i (where that future "could" be)
- `sigma[i]` = uncertainty of Gaussian i

**Why?** The future is uncertain. At an intersection, you might go left OR right. A single prediction can't capture both possibilities, but a mixture of Gaussians can.

### The Pose Head

```python
pose_out = self.fc_pose(lstm_hidden)  # Shape: (B, Seq, 6)
```

This is a simple linear layer that predicts:
- `pose[0:3]` = `[dx, dy, dz]` - translation
- `pose[3:6]` = `[roll, pitch, yaw]` - rotation

**This is deterministic** (not probabilistic) because for odometry evaluation, we need a single best estimate.

---

## Trajectory Evaluation

**File**: `src/utils/trajectory.py`

### What it Does

1. **Load trained models** (VAE + RNN)
2. **Process frames sequentially**:
   ```python
   for each frame:
       z = vae.encode(image)           # Compress to latent
       _, _, _, pose, hidden = rnn(z, hidden)  # Predict motion
       pred_deltas.append(pose)
   ```
3. **Integrate pose deltas** into global trajectory:
   ```python
   # Each delta is a local transformation
   # Accumulate: Global_New = Global_Old @ Local_Step
   path = integrate_path(pred_deltas)  # Returns XYZ coordinates
   ```
4. **Plot bird's-eye view** comparing predicted vs ground truth

### Path Integration Math

Each 6-DOF delta `[dx, dy, dz, roll, pitch, yaw]` is converted to a 4x4 transformation matrix:

```python
T_local = | R  t |    # R = 3x3 rotation from Euler angles
          | 0  1 |    # t = [dx, dy, dz] translation

current_pose = current_pose @ T_local  # Chain transformations
```

---

## Training Pipeline

### 1. Train VAE
```bash
python -m src.train --mode vae
```
- Learns to compress KITTI images to 128-d latent space
- Outputs: `outputs/{run_name}/checkpoints/vae_final.pth`

### 2. Train MDN-RNN
```bash
python -m src.train_mdnrnn
```
- Uses frozen VAE to encode images
- Learns dynamics and pose prediction
- Outputs: `outputs/{run_name}/rnn_checkpoints/rnn_best.pth`

### 3. Evaluate Trajectory
```bash
python -m src.utils.trajectory
```
- Generates `trajectory_result.png` comparing predicted vs ground truth path

---

## Configuration

**File**: `config/cuda.yaml`

```yaml
run_name: "cuda_z128_full"

training:
  batch_size: 128
  learning_rate: 0.0001
  epochs: 100

model:
  vae:
    latent_dim: 128
  rnn:
    hidden_size: 512
    num_layers: 2

data:
  img_height: 192
  img_width: 640
```

---

## Data Format

### KITTI Odometry Dataset

```
data/kitti/
├── 00/
│   └── image_02/
│       └── data/
│           ├── 000000.png
│           ├── 000001.png
│           └── ...
├── 01/
├── 02/
└── poses/
    ├── 00.txt    # Ground truth poses (N x 12 matrix)
    ├── 01.txt
    └── ...
```

**Pose format**: Each line is a flattened 3x4 transformation matrix (12 floats).

The data loader (`src/data/odometry_loader.py`) converts these to relative 6-DOF deltas between consecutive frames.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/train.py` | VAE training loop |
| `src/train_mdnrnn.py` | RNN training loop |
| `src/models/world_model.py` | ConvVAE architecture |
| `src/models/mdnrnn_pose.py` | DreamerMDRNN architecture |
| `src/utils/trajectory.py` | Trajectory evaluation & plotting |
| `src/data/odometry_loader.py` | KITTI dataset loading & preprocessing |

---

## Why This Architecture?

### World Models Approach

Traditional visual odometry uses:
- Feature extraction (ORB, SIFT)
- Feature matching
- Geometric constraints (epipolar geometry)

This project uses a **learned world model**:
- VAE learns a compressed representation of "what the world looks like"
- RNN learns "how the world changes over time"
- Combined, they can predict motion without hand-crafted features

### Benefits

1. **End-to-end learning**: No manual feature engineering
2. **Uncertainty modeling**: MDN captures multiple possible futures
3. **Temporal context**: LSTM maintains memory of past observations
4. **Generative capability**: Can "dream" future scenarios (via `sample_dream`)

---

## Improving Generalization

If the model performs well on training sequences but poorly on unseen data:

1. **Train on more sequences**: Use all KITTI sequences 00-10
2. **Data augmentation**: ColorJitter, GaussianBlur (already added)
3. **Proper train/val split**: Don't train on evaluation sequences
4. **Increase model capacity**: Larger latent dim, more LSTM layers
5. **Regularization**: Dropout, weight decay

---

## References

- [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/)
- [Dream to Control: Learning Behaviors by Latent Imagination (Dreamer, 2019)](https://arxiv.org/abs/1912.01603)
- [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
