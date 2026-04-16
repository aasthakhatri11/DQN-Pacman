# Deep Q Network for Atari MsPacman

Tested and developed on Google Colab.

This project implements a Deep Q Network (DQN) agent to play Atari MsPacman using PyTorch. The agent learns directly from raw pixel input and improves its performance through interaction with the environment.

## Overview

The objective is to train an agent that can navigate the MsPacman environment, collect rewards, and avoid negative outcomes. The model processes visual input using a convolutional neural network and learns action-value estimates through reinforcement learning.

This project focuses on understanding the practical challenges of training deep reinforcement learning agents from visual input under limited computational resources.

## Approach

The implementation follows the standard DQN framework with the following components:

* A convolutional neural network to estimate Q-values from image input
* Experience replay to reduce correlation between samples
* A target network updated periodically to stabilize learning
* Frame stacking (4 frames) to incorporate temporal information
* An epsilon-greedy policy for exploration
* Gradient clipping to improve training stability

The input to the model consists of four consecutive grayscale frames (84×84), allowing the agent to capture motion and environment dynamics.

## State Representation

Each state is constructed as:

* RGB frames converted to grayscale
* Resized to 84 × 84
* Stacked across 4 frames → shape: (4, 84, 84)

This enables the agent to capture temporal information such as movement direction and object dynamics.

## Training

The agent was trained under constrained computational settings using Google Colab.

### Key Settings

* Episodes: 100
* Replay Buffer Size: 10,000 (limited by memory constraints)
* Batch Size: 32
* Discount Factor (γ): 0.99
* Target Network Update: every 500 steps
* Learning Frequency: every 4 environment steps

## Exploration Strategy

An epsilon-greedy policy is used:

* Initial epsilon: 1.0
* Minimum epsilon: 0.1
* Decay applied per step: 0.9999

This allows sustained exploration in early training and gradual transition to exploitation.

## Reward Design

* Rewards are clipped using `reward = sign(reward)` to stabilize learning

## Results

The agent demonstrates clear learning progression over time.

### Performance Improvement

* Early Average Reward: 397.5
* Late Average Reward: 613.0
* Improvement: 54.21%

This indicates that the agent successfully learned better policies during training.

### Training Behavior

* Early episodes show random and exploratory behavior
* Mid-training introduces high-reward trajectories
* Later episodes show improved reward trends, though performance remains variable

Peak episode scores exceeded **1400**, indicating that the agent learned non-trivial navigation and reward collection strategies.

## Training Visualizations

### Reward Curve

![Reward Curve](images/reward_curve.png)

### Loss Curve

![Loss Curve](images/loss_curve.png)

### Q-Value Evolution

![Q Values](images/q_values.png)

## Final Evaluation

The trained agent was evaluated using a fully greedy policy (no exploration).

* Environment: ALE/MsPacman-v5
* Frame stacking: 4
* Evaluation episodes: 20

The agent achieves average scores in the range of **500–700**, with high variability due to the stochastic nature of the environment and limited training duration.

## Gameplay

A sample gameplay recording generated after training is available at:

[Watch Gameplay Video](videos/pacman.mp4)

## Ablation Study

To better understand the importance of temporal information, an ablation study was conducted comparing:

* Single-frame input (no temporal context)
* 4-frame stacked input (with temporal context)

Results show that frame stacking significantly improves performance.

Without temporal context, the agent struggles to:

* anticipate ghost movement
* maintain consistent direction
* learn stable behaviour

Detailed experiments can be found in:
`notebooks/dqn_pacman_ablation.ipynb`

## Project Structure

```
DQN-Pacman/
├── notebooks/     # Training and ablation notebooks
├── models/        # Saved model weights
├── videos/        # Gameplay recordings
├── images/        # Training plots
├── data/          # Stored metrics
└── README.md
```

## Technologies Used

* Python
* PyTorch
* Gymnasium (Atari environments)
* OpenCV
* NumPy
* Matplotlib

## Limitations

* Limited training duration due to computational constraints
* Smaller replay buffer compared to standard DQN implementations
* High variance in performance across episodes
  
## Possible Improvements

* Double DQN to reduce overestimation bias
* Dueling network architecture
* Prioritized experience replay
* Longer training for improved performance

## Author

Aastha Khatri
