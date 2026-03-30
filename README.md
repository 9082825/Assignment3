# CSCN8020 Assignment 3: Deep Q-Network for Atari Pong

**Student Name:** Jose George  
**Student ID:** 9082825  

---

## Overview

This assignment implements a **Deep Q-Network (DQN)** agent to train an Atari Pong player using reinforcement learning. The project demonstrates how deep learning and reinforcement learning can be combined to solve high-dimensional control problems, where the agent learns directly from raw image input. The notebook follows a structured workflow: **Implementation → Validation → Experimentation → Analysis → Demo**. [web:10][web:14]

---

## Objectives

The main objectives of this assignment are:

- Implement a working DQN agent using PyTorch.
- Apply image preprocessing and frame stacking for Atari input.
- Evaluate the impact of training duration and hyperparameters.
- Analyze learning behavior using reward trends and plots.
- Demonstrate the trained agent through a final gameplay simulation.

---

## Methodology

### Environment Setup

- **Environment:** Atari Pong using Gymnasium and ALE.
- **Input:** Raw RGB frames from the game.
- **Output:** A discrete action space.  
The current Gymnasium/ALE equivalent for the older deterministic Pong setup is `ALE/Pong-v5` with appropriate arguments such as `frameskip=4` and `repeat_action_probability=0.0`. [web:2][web:4][web:9]

### Preprocessing Pipeline

To make learning more efficient, the following preprocessing steps are applied:

- Crop irrelevant screen regions.
- Downsample the image resolution.
- Convert frames to grayscale.
- Normalize pixel values to the range `[-1, 1]`.
- Stack multiple frames to capture motion information.

---

## DQN Architecture

The implementation includes the following core components:

- **Convolutional Neural Network (CNN):** Extracts visual features from stacked frames.
- **Experience Replay Buffer:** Improves learning stability by sampling past transitions.
- **Target Network:** Reduces instability during training.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation.  
These design choices follow the standard DQN approach introduced by DeepMind. [web:10][web:14]

### Core Components

- **DQN:** Neural network model that estimates action values.
- **DQNAgent:** Handles learning and action selection.
- **PongTrainer:** Manages the training loop and experiments.

---

## Experiments Conducted

### Debug Run (5 Episodes)

**Purpose:** Validate pipeline correctness.  
**Observation:** Rewards remain highly negative, with no meaningful learning trend.  
**Conclusion:** Useful for debugging only.

### Training Duration Experiment

| Episodes | Observation |
|---|---|
| 250 | Early improvement, unstable |
| 500 | Clear upward trend and better rewards |

**Conclusion:** Longer training is required to observe meaningful learning.

### Batch Size Comparison

| Batch Size | Observation |
|---|---|
| 8 | Faster short-term improvement, higher variance |
| 16 | Slightly smoother but similar performance |

**Conclusion:** Batch size has a limited impact under the current training scale.

### Target Network Update Frequency

| Update Frequency | Observation |
|---|---|
| Every 3 episodes | Faster early learning but more fluctuation |
| Every 10 episodes | More stable learning and competitive final performance |

---

## Extended Training

After identifying the better configuration, an additional experiment was conducted:

- **1000 episodes**
- **Batch size = 8**
- **Target update = every 10 episodes**

### Purpose

- Improve policy quality for the final demonstration.
- Validate whether longer training improves performance.

This extended run is not part of the main comparison, but it serves as a refinement step for the demo.

---

## Results and Key Insights

- Very short runs, such as 5 episodes, are only useful for debugging.
- Learning becomes visible only after sufficient training duration.
- 500 episodes show meaningful improvement, but not full convergence.
- Target network update frequency significantly affects stability.
- Batch size has a comparatively smaller effect.
- Reinforcement learning exhibits high variance and stochastic behavior.

---

## Final Demo

The final demo loads the trained model from the extended 1000-episode run and performs a full gameplay episode using a greedy policy.

- **Model file:** `Trained_model.pth`
- **Example result:** reward approximately `20–21`

**Note:** Small reward variations are expected due to environment stochasticity.

---

## Limitations

- The agent does not fully converge within 1000 episodes.
- Reward variance remains high.
- Only standard DQN is implemented.

---

## Future Improvements

- **Double DQN:** Reduce overestimation bias.
- **Dueling DQN:** Improve value estimation.
- **Longer training:** 2000+ episodes.
- **Hyperparameter tuning:** Learning rate and epsilon decay.

---

## Why DQN Works for Pong

- Pong has high-dimensional image input, so a CNN is appropriate.
- Traditional Q-learning does not scale well, so function approximation is needed.
- Experience replay improves sample efficiency.
- The target network stabilizes learning. [web:10][web:14]

---

## Main File

- `CSCN8020_Assignment3.ipynb` — main submission notebook

---

## How to Run

### Recommended Python Version

This project was developed and tested with **Python 3.11.3**.

### Setup

```bash
git clone https://github.com/9082825/Assignment3.git
cd Assignment3
pip install -r requirements.txt
jupyter notebook
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Run Notebook

- Open `CSCN8020_Assignment3.ipynb`.
- Run all cells sequentially.

The notebook will:

- Validate the pipeline.
- Run experiments (`250`, `500` episodes).
- Compare hyperparameters.
- Train the extended model (`1000` episodes).
- Run the final demo animation.

---

## Reproducibility

- Random seeds are set where applicable.
- Results are generally reproducible, but small variations may occur due to stochastic environment dynamics and GPU/CPU differences.

---

## Conclusion

This assignment successfully implements a **Deep Q-Network (DQN)** to solve Atari Pong. The work shows that training duration is the most critical factor in reinforcement learning, controlled experimentation is necessary for meaningful comparison, and target network update frequency plays a key role in stability. The additional 1000-episode run confirms that extended training improves policy quality and supports a strong final gameplay demonstration. [web:10][web:14]

---
