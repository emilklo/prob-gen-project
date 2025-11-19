# Generative Models for Autonomous Vehicle Perception: Comprehensive Project Proposals and Technical Implementation Guide

## 1. Executive Summary
This report articulates a comprehensive technical framework for a graduate-level final project in Generative Modeling, specifically designed for a two-person team operating under constrained timelines (approximately 40 combined man-hours) and a heterogeneous hardware environment consisting of Apple Silicon (M4 Max) for development and NVIDIA Hopper (H100) for high-performance training.

The central theme of this research proposal is "**Understanding Self-Driving Mechanics via Generative Image Processing**," a domain that sits at the intersection of computer vision, robotics, and probabilistic modeling.

The autonomous driving (AD) industry is currently undergoing a paradigm shift from purely discriminative models—which classify and detect objects—to generative foundation models that simulate, predict, and reason about the driving environment. Companies like Waymo and startups utilizing end-to-end learning are leveraging Generative AI (GenAI) not merely for content creation, but for "imagining" potential futures (predictive world modeling) and synthesizing rare training data (domain adaptation). This report translates these industrial trends into three rigorous, academically viable project pathways suitable for Master’s students.

The proposed pathway leverages **Variational Autoencoders (VAEs) coupled with Recurrent Neural Networks (RNNs)** for predictive world modeling.

This pathway is scoped to fit the 20-hour-per-student constraint while adhering to the requirement of training or modifying substantial models rather than utilizing pre-packaged API wrappers. Crucially, the report details a hybrid workflow optimizing the interplay between local Apple Metal Performance Shaders (MPS) and remote CUDA acceleration, ensuring efficient resource utilization.

## 2. Introduction: The Generative Turn in Autonomous Driving

### 2.1 The Limitations of Discriminative Perception
For the past decade, the perception stack of autonomous vehicles has been dominated by discriminative artificial intelligence. These systems are engineered to map high-dimensional sensory inputs—primarily camera images, LiDAR point clouds, and radar returns—onto low-dimensional semantic labels. Convolutional Neural Networks (CNNs) and, more recently, Vision Transformers (ViTs) have achieved remarkable success in tasks such as 2D/3D object detection, semantic segmentation, and lane tracking [1]. However, these systems are fundamentally reactive and bounded by their training distributions.

A critical failure mode for discriminative systems is the "corner case" or "long-tail" scenario. Driving data follows a Zipfian distribution where mundane driving (highway cruising in clear weather) is overrepresented, while critical safety events (e.g., a child running onto a snowy road at night) are statistically rare. Collecting sufficient real-world data to train robust classifiers for every conceivable combination of weather, lighting, and obstacle is logistically and economically infeasible [2]. Discriminative models, when faced with out-of-distribution (OOD) data, often fail unpredictably, leading to safety disengagements or accidents.

### 2.2 The Promise of Generative AI
Generative AI offers a complementary approach by modeling the joint probability distribution of the sensor data itself. Rather than reducing an image to a label, generative models learn to reconstruct, synthesize, and predict the visual world. This capability unlocks three transformative applications for autonomous driving, which form the theoretical basis of the proposed projects:

1.  **Predictive World Modeling**: Perhaps the most profound application is the "World Model," which posits that an intelligent agent should learn a compressed spatial-temporal representation of its environment. By predicting future sensor states based on current actions (e.g., "dreaming" the result of a steering maneuver), the vehicle can plan trajectories in a learned latent space, effectively simulating the consequences of its actions without physical risk [7].

### 2.3 Contextualizing the Project Within Industry Trends
The relevance of this project is underscored by recent industrial developments. Waymo has developed "Foundation Models" for driving that integrate sensor data from multiple sources to predict the behavior of road users and simulate scenarios, functioning similarly to Large Language Models (LLMs) but for physical dynamics [8]. Similarly, end-to-end driving models (E2E), which map raw pixels directly to control commands, rely heavily on generative pre-training to understand scene dynamics [2]. By engaging with these concepts, the students will not only complete a course requirement but also gain exposure to the state-of-the-art methodologies currently reshaping the automotive AI sector.

## 3. Computational Infrastructure and Workflow Strategy
A unique constraint of this project is the heterogeneous hardware profile: a MacBook Pro M4 Max for local development and an NVIDIA H100 for training. Optimizing this workflow is critical to meeting the 40-hour time budget.

### 3.1 The Apple Silicon (M4 Max) Development Environment
The M4 Max, built on ARM64 architecture, represents a potent local development platform. Its Unified Memory Architecture (UMA) allows the GPU to access system memory directly, bypassing the PCIe bottleneck common in x86/NVIDIA setups. However, PyTorch support for Apple Silicon relies on the Metal Performance Shaders (MPS) backend, which differs significantly from CUDA [10].

**MPS Optimization and Limitations:**
The MPS backend accelerates PyTorch operations using the Metal graph framework. While it supports most standard layers (Conv2d, Linear, BatchNorm), certain advanced operators required for specific generative architectures (e.g., complex number operations in some spectral analysis layers or specific sparse tensor operations) may lack implementation, triggering CPU fallbacks that degrade performance [10].

To ensure code portability between the M4 Max and the H100, the students must adopt a rigorous device-agnostic coding standard. Hardcoding `.cuda()` calls is strictly prohibited. Instead, dynamic device selection is mandatory:

```python
def get_compute_device():
    if torch.cuda.is_available():
        # NVIDIA H100: Optimized for high-throughput training
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple M4 Max: Optimized for local interactive debugging
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

**Local Workflow:** The M4 Max should be utilized for:
*   **Data Pipeline Validation**: Verifying that custom Dataset and DataLoader classes correctly resize, normalize, and augment BDD100K/KITTI images without blocking the GPU.
*   **Architecture Debugging**: Running "overfit tests" on a single batch of data. If the model cannot overfit 8 images to zero loss on the M4, it will not train on the H100.
*   **Inference and Visualization**: Running trained models to generate qualitative outputs (images/videos) for the final report.

### 3.2 The NVIDIA H100 Training Environment
The H100 is a massive accelerator designed for Transformer and large-scale CNN workloads. To fully leverage its capability, the students must employ specific optimization techniques that differ from local development.

*   **Mixed Precision Training**: The H100 excels at lower-precision arithmetic, specifically BFloat16 (Brain Floating Point) and FP8. Standard FP32 training is an inefficient use of this hardware. Students should implement `torch.cuda.amp` (Automatic Mixed Precision) to cast gradients and activations to BF16/FP16, which can double or triple throughput compared to FP32 [12].
*   **Data Loading Bottlenecks**: The H100 consumes data faster than standard hard drives can provide. A common pitfall is reading thousands of small image files (like BDD100K frames) from a network file system (NFS). The recommended strategy is to pre-process the selected subset of data into a contiguous format (e.g., LMDB or WebDataset) or to unzip the raw dataset directly into the compute node's local NVMe scratch space at the start of the job.

### 3.3 Cross-Platform Compatibility Table

| Feature | Apple M4 Max (Local) | NVIDIA H100 (Remote) | Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **Backend** | `torch.device("mps")` | `torch.device("cuda")` | Use conditional device assignment logic. |
| **Precision** | FP32 (preferred for stability) | BF16 / FP16 (essential for speed) | Use `torch.cuda.amp.GradScaler` only when `device.type == 'cuda'`. |
| **Batch Size** | Small (1-8) for debugging | Large (32-128) for convergence | Parameterize batch size in config files. |
| **Data Path** | Local SSD (Relative Path) | Cluster Scratch (Absolute Path) | Use environment variables to set data root. |
| **Visualization** | Matplotlib/OpenCV (Interactive) | TensorBoard/W&B (Logged) | Implement logging callbacks that save images to disk. |

## 4. Data Ecosystem: Selection and Engineering
Generative models are notoriously data-hungry. However, attempting to download and process the entirety of industry-standard datasets like BDD100K (hundreds of gigabytes) or Waymo Open Dataset (terabytes) is infeasible within a 20-hour constraint. The project strategy relies on "Smart Subsetting."



### 4.2 KITTI: The Temporal Benchmark
For projects involving temporal prediction (World Models), KITTI is superior to BDD100K. BDD100K clips are often short (10 seconds) and discontinuous. KITTI provides long, stable odometry sequences where the ego-motion is smooth, making it easier for an RNN to learn the physics of motion [16].

**Subset Strategy for Project B (World Model):**
*   **Target**: Sequences 00-05 from the Odometry benchmark.
*   **Data Type**: The project requires the "Rectified Color Images" (Camera 2).
*   **Preprocessing**: Resize images to $64 \times 64$ or $128 \times 128$. While low resolution, this is standard for academic World Model experiments (e.g., the original Ha & Schmidhuber paper used $64 \times 64$) to ensure the VAE converges quickly [7].





## 5. Project Option: "The Dreamer" (Predictive World Modeling)
**Objective**: Implement a simplified "World Model" that learns to predict future driving states from current visual observations, establishing a foundation for model-based planning.

### 6.1 Theoretical Basis: Latent Dynamics
This project draws on the "World Models" paper (Ha & Schmidhuber) and recent applications in AD like "GenAD" [7]. The hypothesis is that an autonomous agent need not predict every pixel of the future (which is computationally expensive and stochastic) but rather the *concept* of the future.

The architecture comprises three decoupled components:
1.  **Vision Model (V)**: A Variational Autoencoder (VAE) that compresses high-dimensional images ($x_t$) into a low-dimensional latent vector ($z_t$).
2.  **Memory Model (M)**: An RNN (LSTM or MDN-RNN) that learns the transition dynamics $P(z_{t+1} | z_t, a_t)$.
3.  **Controller (C)**: (Optional for this project) A policy network that acts on $z_t$.

### 6.2 Implementation Strategy
**Step 1: The VAE (Spatial Compression)**
*   **Architecture**: A standard Convolutional VAE.
*   **Loss**: Reconstruction Loss (MSE) + KL Divergence ($D_{KL}[q(z|x) || p(z)]$).
*   **Challenge**: VAEs tend to produce blurry reconstructions. For AD, preserving lane lines is critical. Students should experiment with **KL-Annealing** (slowly introducing the KL term) to prevent "posterior collapse," where the decoder ignores the latent code and outputs an average image [26].
*   **Hardware**: Train on H100 using KITTI individual frames.

**Step 2: The RNN (Temporal Prediction)**
*   **Data Prep**: Once the VAE is trained, the entire KITTI dataset is passed through it to generate a dataset of latent sequences. This reduces the dataset size by orders of magnitude (e.g., from 100GB of images to 100MB of vectors), allowing the RNN to be trained in minutes even on the M4 Max [7].
*   **Prediction**: The RNN is trained to minimize the log-likelihood of the next latent vector.

### 6.3 Research Nuances & Insights
*   **The "Dreaming" Process**: Once trained, the system can "dream" by feeding the RNN's output back into itself ($z_{t+1} \rightarrow \text{RNN} \rightarrow z_{t+2}$). Decoding this sequence visualizes the model's understanding of physics.
*   **Insight - Object Permanence**: A key qualitative metric is whether the model remembers objects. If a car passes behind a truck, does the "dreamt" future remember it exists? This tests the memory capacity of the RNN component [7].
*   **Comparison to Video Diffusion**: Unlike video diffusion models (e.g., DrivingDiffusion), which generate high-fidelity pixels but are slow, VAE-RNNs are fast and suitable for real-time planning, though they sacrifice visual sharpness [29].



## 8. Implementation Strategy & Code Architecture
To ensure success within the 40-hour limit, the codebase must be modular and leverage existing open-source repositories where permitted.

### 8.1 Repository Structure
The project should be organized to facilitate the hybrid M4/H100 workflow:
```text
/project_root
  /configs            # YAML files for hyperparameters (batch_size, lr)
  /data
    /loaders.py       # PyTorch Dataset classes (BDD100K, KITTI)
    /preprocess.py    # Resizing and subsetting scripts
  /models
    /vae_rnn.py       # World Model Architecture
  /utils
    /device.py        # The MPS/CUDA selector logic
    /metrics.py       # FID, SSIM calculations
  train.py            # Main entry point (CLI args)
  eval.py             # Inference and visualization
  requirements.txt
```



## 9. Evaluation Methodologies

### 9.1 Quantitative Metrics

*   **SSIM (Structural Similarity Index)**: For World Models, SSIM measures the structural fidelity of the predicted future frame against the ground truth. It is more robust to pixel-level noise than Mean Squared Error (MSE) [42].



## 10. Conclusion
This research proposal outlines a rigorous pathway for Master’s students to interrogate the mechanics of autonomous driving through the lens of Generative AI. By moving beyond passive analysis and into active synthesis—whether by translating lighting domains, dreaming future states, or simulating sensor inputs—students engage with the core challenges of modern robotics: robustness, prediction, and domain adaptation.

The recommended path, **"The Dreamer" (Predictive World Modeling)**, offers the optimal balance of feasibility and pedagogical value. It utilizes the provided hardware effectively (M4 for architecture dev, H100 for training), engages with the complex KITTI dataset, and directly addresses the critical industrial problem of predictive planning.

Through this project, students will demonstrate that generative models are not merely creative tools but essential infrastructure for the safe deployment of autonomous systems, capable of simulating the infinite tail of rare events that define the challenge of self-driving.

## Citations Table

| ID | Source Context |
| :--- | :--- |
| [1] | Overview of Generative AI in AD applications (MDPI) |
| [2] | GenAI for vehicle autonomy and end-to-end models (WEF) |
| [3] | Prompting strategies for adverse weather simulation |
| [4] | Comprehensive review of GenAI metrics (FID) and models (ArXiv) |
| [5] | BYOL-Drive and representation learning in AD (MDPI) |
| [6] | CycleGAN specifically for Day-to-Night translation |
| [7] | World Models foundational theory (Ha & Schmidhuber) |
| [8] | Waymo's Foundation Model and generative simulators |
| [9] | Waymo CEO discussion on GenAI simulation layers |
| [10] | PyTorch MPS backend documentation |
| [11] | Apple Metal Performance Shaders documentation |
| [12] | Performance comparison: M1/M3 vs NVIDIA GPUs |
| [13] | Detailed comparison of AD datasets (BDD100K, Cityscapes) |
| [14] | Detailed comparison of AD datasets (BDD100K, Cityscapes) |
| [15] | BDD100K validation set details |
| [16] | KITTI dataset specifications and download |
| [17] | KITTI benchmarking and sensor setup |
| [18] | CycleGAN official project page and loss functions |
| [19] | Technical implementation details of CycleGAN |
| [20] | CycleGAN TensorFlow tutorial and architecture |
| [21] | PatchGAN architecture explanation |
| [22] | PyTorch CycleGAN repository and identity loss |
| [23] | Limitations of current datasets for night driving |
| [24] | Udacity Self-Driving Car dataset stats |
| [26] | KL-Annealing to prevent posterior collapse in VAEs |
| [27] | Deep dive into VAE KL divergence and latent space |
| [28] | VAE-RNN architectures for simpler environments |
| [29] | DrivingDiffusion for multi-view video generation |
| [31] | ControlNet guide for Stable Diffusion |
| [32] | Explanation of ControlNet architecture |
| [33] | ControlNet segmentation tutorial |
| [34] | Training ControlNet on small datasets |
| [35] | Hyperparameters for ControlNet training |
| [36] | Sensor cleaning and prompt engineering issues |
| [37] | ControlNet classifier-free guidance and settings |
| [38] | Synthetic rain generation using physics vs. diffusion |
| [39] | Physics-based rain rendering library |
| [40] | Generative models for sensor cleaning and map creation |
| [42] | SSIM metric explanation for image quality |
| [43] | BDD100K dataset download and subsetting |
| [44] | BDD100K diversity vs. other datasets |
| [45] | Installing PyTorch on Apple Silicon |