# Deep Q-Network for Atari Ms. Pac-Man

This project implements a Deep Q-Network (DQN) agent to play Atari Ms. Pac-Man using PyTorch. The agent learns directly from visual input and improves its performance through interaction with the environment.

In addition to training a baseline agent, this project includes an ablation study to evaluate the role of temporal information via frame stacking.

---

## Overview

The goal is to train an agent that can navigate the Ms. Pac-Man environment, collect rewards, and avoid negative outcomes. The model processes raw pixel input using a convolutional neural network and learns action-value estimates through reinforcement learning.

The project focuses on:
- training a DQN agent under limited computational resources  
- analyzing the importance of temporal context in decision-making  

---

## Method

The implementation follows the standard DQN framework:

- Convolutional Neural Network for Q-value approximation  
- Experience Replay to reduce correlation between samples  
- Target Network for training stability  
- Epsilon-greedy exploration  
- Gradient clipping for stable updates  

---

## State Representation

Each state is constructed as:

- RGB frames converted to grayscale  
- Resized to 64 × 64  
- Stacked across 4 frames → shape: (4, 64, 64)  

Frame stacking allows the agent to capture motion and environment dynamics.

---

## Training

The agent was trained using Google Colab under constrained resources.

**Key settings:**

- Episodes: 50–100  
- Replay Buffer Size: 10,000  
- Batch Size: 16  
- Discount Factor (γ): 0.99  
- Learning Frequency: every 4 steps  
- Reward Clipping: sign(reward)  

**Exploration:**

- Initial ε = 1.0  
- Final ε ≈ 0.05  
- Linear decay during training  

---

## Results (Baseline Training)

The agent demonstrates gradual learning:

- Early episodes show random exploration  
- Mid-training shows occasional high-reward trajectories  
- Later episodes show improved reward trends  

Performance remains somewhat noisy due to limited training duration.

---

## Ablation Study: Frame Stacking

To understand the importance of temporal information, an ablation study was conducted by removing frame stacking while keeping all other components fixed.

### Experimental Setup

Two configurations were compared:

- Single-frame input (no temporal context)  
- 4-frame stacked input (temporal context)  

Both models were trained independently under identical settings.

---

### Results

- No Frame Stacking: Mean = 17.45, Std = 3.70  
- 4-Frame Stacking: Mean = 18.32, Std = 3.56  
- Improvement: +0.87  

Frame stacking leads to a consistent but modest improvement in performance.

---

### Analysis

The stacked model achieves slightly higher rewards and more stable learning behavior. Temporal information helps the agent better capture movement dynamics, leading to more consistent decisions.

Without frame stacking, the agent struggles to:
- infer motion from static observations  
- maintain stable movement patterns  
- learn consistent policies  

The relatively small improvement is likely due to:
- short training duration  
- reduced model capacity  
- computational constraints  

---

## Visualizations

### Training Performance

![Reward Curve](results/training/reward_curve.png)

![Loss Curve](results/training/loss_curve.png)

![Q-Value Evolution](results/training/q_value.png)

---

### Ablation Study (Frame Stacking)

![Raw Reward Comparison](results/ablation/avg_rewards.png)

![Smoothed Reward Comparison](results/ablation/smoothed_rewards.png)
---

## Gameplay

A sample gameplay recording of the trained agent:

- `videos/pacman.mp4`

---

## Project Structure

```
DQN-Pacman/
├── notebooks/
│   ├── dqn_training.ipynb                  # Main DQN training
│   └── dqn_ablation_frame_stacking.ipynb   # Ablation study
│
├── models/
│   └── dqn_pacman.pth
│
├── results/
│   ├── training/
│   │   ├── reward_curve.png
│   │   └── q_value.png
│   │
│   └── ablation/
│       ├── avg_rewards.png
│       └── smoothed_rewards.png
│
├── videos/
│   └── pacman.mp4
│
├── data/
│   └── episode_rewards.npy
│
├── README.md
└── requirements.txt
```
---

## Technologies Used

- Python  
- PyTorch  
- Gymnasium (Atari environments)  
- OpenCV  
- NumPy  
- Matplotlib  

---

## Limitations

- Limited training due to compute constraints  
- High variance in rewards  
- Results are not fully converged  

---

## Future Work

- Double DQN  
- Dueling Networks  
- Prioritized Experience Replay  
- Longer training  

---

## Author

Aastha Khatri