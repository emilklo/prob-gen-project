# MDN-RNN and Trajectory Evaluation - Technical Deep Dive

This document explains the Mixture Density Network RNN (MDN-RNN) architecture and how it's used in `trajectory.py` for visual odometry.

---

## Table of Contents

1. [MDN-RNN Architecture](#mdn-rnn-architecture)
2. [The Mixture Density Network](#the-mixture-density-network)
3. [Loss Function](#loss-function)
4. [Training Process](#training-process)
5. [Trajectory Evaluation](#trajectory-evaluation)
6. [Path Integration Mathematics](#path-integration-mathematics)

---

## MDN-RNN Architecture

The MDN-RNN (`src/models/mdnrnn_pose.py`) combines an LSTM with two prediction heads:

```
                    ┌─────────────────┐
                    │   LSTM Memory   │
                    │  (2 layers,     │
                    │   512 hidden)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │   π     │    │   μ     │    │   σ     │
        │ (5 mix) │    │(5×128)  │    │(5×128)  │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                   MDN (Vision) Head

                            │
                            ▼
                    ┌─────────────┐
                    │  Pose Head  │
                    │   (6-DOF)   │
                    └─────────────┘
```

### Components

#### 1. LSTM Memory

```python
self.lstm = nn.LSTM(
    latent_dim,      # 128 input features
    hidden_size,     # 512 hidden units
    num_layers=2,    # 2 stacked layers
    batch_first=True,
    dropout=0.1,     # Regularization between layers
)
```

**Purpose**: Maintains temporal context across the sequence. The hidden state `h_t` encodes all past observations.

#### 2. MDN Head (Vision Prediction)

Three linear layers that parameterize a Mixture of Gaussians:

```python
self.fc_pi = nn.Linear(hidden_size, num_gaussians)      # π: mixing coefficients
self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_dim)   # μ: means
self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_dim) # σ: stds
```

#### 3. Pose Head (Odometry Prediction)

```python
self.fc_pose = nn.Linear(hidden_size, 6)  # [dx, dy, dz, roll, pitch, yaw]
```

**Purpose**: Direct regression of 6-DOF camera motion.

---

## The Mixture Density Network

### Why Use an MDN?

Standard neural networks predict a single output. But the future is often **multimodal** - at an intersection, you might turn left OR right. A single prediction can't capture both possibilities.

An MDN predicts a **probability distribution** over possible outputs using a mixture of Gaussians:

```
p(z_next | z_current, h) = Σᵢ πᵢ · N(z_next | μᵢ, σᵢ²)
```

### MDN Parameters

For `num_gaussians=5` and `latent_dim=128`:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| π (pi) | (batch, seq, 5) | Mixing coefficients (sum to 1) |
| μ (mu) | (batch, seq, 5, 128) | Mean of each Gaussian |
| σ (sigma) | (batch, seq, 5, 128) | Std of each Gaussian |

### Forward Pass

```python
def forward(self, z, hidden=None):
    # z shape: (Batch, Seq_Len, 128)

    # 1. LSTM processes sequence
    lstm_out, next_hidden = self.lstm(z, hidden)  # (B, S, 512)

    # 2. MDN predictions
    pi = F.softmax(self.fc_pi(lstm_out), dim=-1)  # (B, S, 5)
    mu = self.fc_mu(lstm_out).view(B, S, 5, 128)  # (B, S, 5, 128)
    sigma = torch.exp(self.fc_sigma(lstm_out))    # (B, S, 5, 128)
    sigma = torch.clamp(sigma, min=1e-5, max=10.0)  # Stability

    # 3. Pose prediction
    pose = self.fc_pose(lstm_out)  # (B, S, 6)

    return pi, mu, sigma, pose, next_hidden
```

---

## Loss Function

The total loss combines two components:

```python
total_loss = loss_mdn + (pose_weight × loss_pose)
```

### MDN Loss (Negative Log-Likelihood)

We want to maximize the probability of the true next latent under the predicted mixture:

```python
# For each Gaussian i, compute log probability of true z
log_prob_i = -0.5 * (log(2π) + 2*log(σᵢ) + (z_true - μᵢ)²/σᵢ²)

# Sum across latent dimensions
log_prob_i = sum(log_prob_i, dim=latent)  # (B, S, 5)

# Weighted by mixing coefficients (log-sum-exp for stability)
log_prob = logsumexp(log(πᵢ) + log_prob_i, dim=gaussians)

# Negative log-likelihood
loss_mdn = -mean(log_prob)
```

**Intuition**: Penalize the model if the true next latent is unlikely under any of the predicted Gaussians.

### Pose Loss (MSE)

Simple mean squared error between predicted and true pose:

```python
loss_pose = MSE(pred_pose, true_pose)
```

### Pose Weight

```python
pose_weight = 1000.0
```

**Why so high?**
- Pose values are tiny (translations ~0.01-0.1 meters, rotations ~0.001 radians)
- MDN loss is typically ~1-10
- Without weighting, optimizer would ignore pose loss
- 1000× brings pose loss to same scale as MDN loss

---

## Training Process

### Input Preparation

Given a sequence of 5 frames:

```python
images: (B, 5, 3, H, W)
poses:  (B, 5, 6)
```

1. **Encode all frames** with frozen VAE:
```python
z_sequence = vae.encode(images)  # (B, 5, 128)
```

2. **Split into input/target**:
```python
rnn_input = z_sequence[:, :-1, :]   # z₀, z₁, z₂, z₃  (B, 4, 128)
z_target = z_sequence[:, 1:, :]     # z₁, z₂, z₃, z₄  (B, 4, 128)
pose_target = poses[:, 1:, :]       # δ₁, δ₂, δ₃, δ₄  (B, 4, 6)
```

3. **Forward pass**:
```python
pi, mu, sigma, pred_pose, _ = rnn(rnn_input)
# pred_pose: (B, 4, 6) - predictions for each transition
```

### What the Model Learns

For each timestep t, given z_t:
- **Predict z_{t+1}** as a mixture of Gaussians (where will the visual scene be?)
- **Predict Δpose_{t→t+1}** as a 6-DOF vector (how did the camera move?)

---

## Trajectory Evaluation

`trajectory.py` uses the trained model to predict a driving trajectory and compare against ground truth.

### Step-by-Step Process

#### 1. Load Models

```python
vae = ConvVAE(latent_dim=128, ...).to(device)
rnn = DreamerMDRNN(latent_dim=128, hidden_size=512, num_layers=2).to(device)

vae.load_state_dict(checkpoint['model_state_dict'])
rnn.load_state_dict(checkpoint['model_state_dict'])

vae.eval()
rnn.eval()
```

#### 2. Process Each Sequence Window

```python
for i in range(num_samples):
    imgs, poses = dataset[i]  # (5, 3, H, W), (5, 6)

    # Encode all 5 frames
    imgs_batch = imgs.unsqueeze(0).to(device)  # (1, 5, 3, H, W)
    z_sequence = vae.encode(imgs_batch)        # (1, 5, 128)

    # Feed first 4 latents to RNN
    rnn_input = z_sequence[:, :-1, :]          # (1, 4, 128)
    _, _, _, pred_pose, hidden = rnn(rnn_input, hidden)

    # Take the last prediction (most context)
    pred_delta = pred_pose[0, -1]  # [dx, dy, dz, roll, pitch, yaw]
    true_delta = poses[-1]

    pred_deltas.append(pred_delta)
    true_deltas.append(true_delta)
```

#### 3. Understanding the Prediction

```
Window i:  [frame_0, frame_1, frame_2, frame_3, frame_4]
                ↓        ↓        ↓        ↓
Latents:   [  z_0,     z_1,     z_2,     z_3,     z_4  ]
                ↓        ↓        ↓        ↓
RNN input: [  z_0,     z_1,     z_2,     z_3  ]
                ↓        ↓        ↓        ↓
Predicts:  [ δ₀→₁,    δ₁→₂,    δ₂→₃,    δ₃→₄ ]
                                            ↑
                                    We use this one
```

The last prediction has seen the most context (all 4 previous latents).

---

## Path Integration Mathematics

### From Deltas to Global Trajectory

Each 6-DOF delta `[dx, dy, dz, roll, pitch, yaw]` describes motion in the **local frame**. To build a global trajectory, we must transform and accumulate these.

### Coordinate System (KITTI)

```
      Z (forward)
      ↑
      │
      │
      └──────→ X (right)
     /
    /
   Y (down)
```

### Euler Angles to Rotation Matrix

```python
def euler_to_matrix(roll, pitch, yaw):
    # Rotation around X-axis (roll)
    Rx = [[1,    0,         0      ],
          [0,  cos(roll), -sin(roll)],
          [0,  sin(roll),  cos(roll)]]

    # Rotation around Y-axis (pitch)
    Ry = [[ cos(pitch), 0, sin(pitch)],
          [    0,       1,    0      ],
          [-sin(pitch), 0, cos(pitch)]]

    # Rotation around Z-axis (yaw)
    Rz = [[cos(yaw), -sin(yaw), 0],
          [sin(yaw),  cos(yaw), 0],
          [   0,         0,     1]]

    return Rz @ Ry @ Rx
```

### Transformation Matrix

Each delta becomes a 4×4 transformation matrix:

```python
T_local = | R₃ₓ₃   t₃ₓ₁ |
          | 0₁ₓ₃    1   |

where:
  R = euler_to_matrix(roll, pitch, yaw)
  t = [dx, dy, dz]ᵀ
```

### Path Integration

```python
def integrate_path(deltas):
    current_pose = np.eye(4)  # Start at origin
    path = [[0, 0, 0]]

    for delta in deltas:
        dx, dy, dz, roll, pitch, yaw = delta

        # Create local transformation
        T_local = np.eye(4)
        T_local[0:3, 0:3] = euler_to_matrix(roll, pitch, yaw)
        T_local[0:3, 3] = [dx, dy, dz]

        # Chain transformations
        # "Move from current pose by local step"
        current_pose = current_pose @ T_local

        # Extract global position
        position = current_pose[0:3, 3]  # [X, Y, Z]
        path.append(position)

    return np.array(path)
```

### Why Matrix Multiplication?

Consider two consecutive movements:
1. Move forward 1m, turn right 90°
2. Move forward 1m

If we just added translations: `[0,0,1] + [0,0,1] = [0,0,2]` ❌

With proper transformation:
1. After step 1: position=[0,0,1], facing right
2. After step 2: position=[1,0,1] ✓

The rotation from step 1 affects how step 2's translation is applied in global coordinates.

### Plotting

```python
# Bird's eye view: X (right) vs Z (forward)
plt.plot(path_true[:, 0], path_true[:, 2], 'k-', label='Ground Truth')
plt.plot(path_pred[:, 0], path_pred[:, 2], 'r--', label='Predicted')
```

---

## Summary

### MDN-RNN Purpose

1. **Temporal modeling**: LSTM remembers past observations
2. **Uncertainty modeling**: MDN predicts distribution over possible futures
3. **Odometry prediction**: Pose head directly regresses camera motion

### Inference Pipeline

```
Images → VAE Encoder → Latent Sequence → RNN → Pose Deltas → Integration → Trajectory
```

### Key Equations

**MDN Probability**:
```
p(z|π,μ,σ) = Σᵢ πᵢ · N(z|μᵢ,σᵢ²)
```

**Pose Integration**:
```
T_global(t+1) = T_global(t) × T_local(t→t+1)
```

### Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_gaussians` | 5 | More = captures more modes, but slower |
| `hidden_size` | 512 | Larger = more capacity |
| `num_layers` | 2 | Deeper = better temporal modeling |
| `pose_weight` | 1000 | Higher = prioritize odometry accuracy |
| `dropout` | 0.1 | Higher = more regularization |
