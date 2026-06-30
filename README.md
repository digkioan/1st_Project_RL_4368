# Deep Reinforcement Learning for CarRacing-v3

A Deep Reinforcement Learning project that investigates the performance of two Deep Q-Network (DQN) architectures on the **CarRacing-v3** environment from Gymnasium.

The project compares a simple fully-connected DQN baseline against a Convolutional Neural Network (CNN-DQN), evaluates their learning behavior, and analyzes the impact of image preprocessing and hyperparameter tuning on autonomous driving performance.

---

## Overview

This project implements and compares two reinforcement learning agents:

- **Flatten DQN** – a fully connected neural network operating directly on flattened RGB images.
- **CNN-DQN** – a convolutional architecture with image preprocessing and frame stacking.

The objective is to train an autonomous racing agent capable of completing a full lap around the track while maximizing cumulative reward.

---

## Environment

| Property | Value |
|----------|-------|
| Environment | CarRacing-v3 |
| Framework | Gymnasium |
| Observation | RGB Image |
| Input Size | 96 × 96 × 3 |
| Action Space | Discrete (5 actions) |
| Algorithm | Deep Q-Network (DQN) |

---

## Discrete Action Space

The continuous control problem was converted into five discrete actions:

| ID | Action |
|----|--------|
| 0 | No Action |
| 1 | Turn Left |
| 2 | Turn Right |
| 3 | Accelerate |
| 4 | Brake |

---

# Deep Q-Network

The project follows the standard DQN algorithm.

The Q-network approximates the action-value function

\[
Q(s,a)
\]

using a neural network.

For every state:

- Input: environment observation
- Output: Q-value for every possible action
- Selected action: action with maximum Q-value

Training uses the Bellman target

\[
y=r+\gamma\max_{a'}Q_{target}(s',a')
\]

---

# Replay Memory & Target Network

To stabilize training, two standard DQN techniques were implemented.

### Replay Memory

Stores transitions

```
(state, action, reward, next_state)
```

Advantages:

- random mini-batch sampling
- reduces correlation between samples
- improves stability

---

### Target Network

A separate target network is periodically synchronized with the policy network.

Benefits:

- stable target Q-values
- reduced oscillations
- improved convergence

---

# Architecture 1 — Flatten DQN

The baseline model directly flattens the RGB image.

Pipeline

```
RGB Image
      ↓
Normalize
      ↓
Flatten
      ↓
Fully Connected (256)
      ↓
Output (5 Q-values)
```

### Advantages

- Very simple
- Fast training

### Disadvantages

- Ignores spatial relationships
- Treats neighboring pixels independently

---

# Architecture 2 — CNN-DQN

The second architecture exploits spatial information.

Pipeline

```
RGB Image
      ↓
Grayscale
      ↓
Resize (84×84)
      ↓
Frame Stack (4 frames)
      ↓
Conv2D
      ↓
Conv2D
      ↓
Conv2D
      ↓
Flatten
      ↓
Linear (512)
      ↓
Output (5 Q-values)
```

Input shape

```
4 × 84 × 84
```

The frame stack provides temporal information and helps the agent understand motion.

---

# Hyperparameter Tuning

Both models were tuned independently.

The following parameters were evaluated:

- Learning Rate
- Batch Size
- Discount Factor (γ)
- Replay Memory Size
- Epsilon Decay
- Target Network Synchronization Rate

A two-stage tuning strategy was followed:

1. Fast screening
2. Retraining of the best candidates

---

# Final Hyperparameters

## Flatten DQN

| Parameter | Value |
|------------|-------|
| Learning Rate | 5e-5 |
| Discount Factor | 0.99 |
| Replay Memory | 50,000 |
| Batch Size | 128 |
| Epsilon Decay | 0.995 |
| Target Sync | 3000 |
| Loss | MSE |

---

## CNN-DQN

| Parameter | Value |
|------------|-------|
| Learning Rate | 1e-4 |
| Discount Factor | 0.97 |
| Replay Memory | 50,000 |
| Batch Size | 64 |
| Epsilon Decay | 0.995 |
| Target Sync | 500 |
| Loss | Smooth L1 |
| Frame Stack | 4 |

---

# Training Results

## Flatten DQN

- Fast initial improvement
- Mean reward reached approximately **700**
- Required fewer training episodes

---

## CNN-DQN

- Slower but more stable learning
- Better suited for image-based reinforcement learning
- Achieved higher final performance

---

# Evaluation

The agents were evaluated **without rendering**.

Evaluation criteria:

- Number of successful laps
- Success rate
- Mean reward
- Mean lap time
- Best lap time

---

## Final Results

| Model | Successes | Success Rate | Mean Reward | Mean Lap Time | Best Lap Time |
|---------|------------|---------------|---------------|----------------|----------------|
| Flatten DQN | 4 / 5 | 80% | 572.14 | 21.4 s | 11.0 s |
| CNN-DQN | 5 / 5 | **100%** | **856.82** | **13.8 s** | **9.7 s** |

---

# CNN Progress

The CNN agent was periodically evaluated.

| Episodes | Success Rate | Mean Reward |
|-----------|---------------|--------------|
| 250 | 0% | -574.82 |
| 500 | 20% | 124.34 |
| 750 | 100% | 698.00 |
| 1000 | 100% | 825.82 |

The CNN agent began consistently completing the track after approximately **600–750 episodes**.

---

# Behavioral Observations

## Flatten DQN

Pros

- Surprisingly good performance for a simple architecture
- Fast learning

Cons

- Occasionally loses track
- Performs unnecessary rotations
- Less stable driving

---

## CNN-DQN

Pros

- Smooth and stable driving
- Better exploitation of image information
- More consistent lap completion

Cons

- Higher computational cost
- Longer training time

---

# Conclusions

This project demonstrates that:

- A simple fully-connected DQN can provide a strong baseline.
- CNN-based DQN significantly improves autonomous driving performance.
- Frame stacking provides useful temporal information.
- Convolutional architectures outperform fully connected networks on image-based reinforcement learning tasks.

The final CNN-DQN achieved:

- ✅ 100% success rate
- ✅ Mean reward: **856.82**
- ✅ Best lap time: **9.7 s**

---

# Future Work

Possible improvements include:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy Networks
- More systematic hyperparameter optimization
- PPO and SAC comparison
- Curriculum learning

---

# Technologies

- Python
- PyTorch
- Gymnasium
- NumPy
- OpenCV
- Matplotlib

---

# Project Structure

```
.
├── models/
├── checkpoints/
├── training/
├── evaluation/
├── videos/
├── utils/
├── plots/
├── README.md
└── requirements.txt
```

