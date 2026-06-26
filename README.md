# Flappy Bird Deep Q-Network (DQN) Agent

A PyTorch-based Reinforcement Learning project that trains an artificial intelligence to play Flappy Bird. This repository implements a Deep Q-Network (DQN) with a target network and experience replay to master the `flappy_bird_gymnasium` environment.

---

## Key Features

* **Deep Q-Learning Architecture:** Utilizes a primary policy network and a synchronized target network to stabilize training.
* **Experience Replay Buffer:** Stores past state transitions in a `deque` memory buffer to break correlation between consecutive samples and improve learning efficiency.
* **Custom Reward Shaping:** Implements immediate episode termination upon pipe collision (intercepting the default falling frames) to prevent the agent from training on "dead" states.
* **Hardware Agnostic:** Automatically detects and pipelines tensor operations to Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), or fallback to `cpu`.
* **Smart Network Initialization:** Employs Kaiming Uniform weight initialization and LeakyReLU activations to prevent exploding/vanishing gradients and stabilize coordinate processing.

---

## Project Structure

* `agent.py`: The main execution script containing the `Agent` class, training loop, evaluation logic, and terminal controller mapping.
* `dqn.py`: Defines the neural network architecture (Multi-Layer Perceptron) using PyTorch.
* `experience_replay.py`: Contains the `ReplayMemory` class for storing and sampling agent transitions.
* `parameters.yaml`: Configuration file for easily tuning hyperparameters without altering the codebase.

---

## Installation & Setup

**1. Clone the repository and navigate to the project directory:**
*(Assuming you have downloaded the source files into a folder)*

**2. Install the required dependencies:**
Ensure you have Python 3.8+ installed. Install the required packages using `pip`:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml

```

> **Note:** If you are using a specific hardware accelerator (like CUDA), ensure you install the appropriate PyTorch build from the [official PyTorch website](https://www.google.com/search?q=https.pytorch.org).

---

## Usage

The project is controlled via a command-line interface (CLI) provided in `agent.py`.

### Training the Agent

To train the agent from scratch using the default configuration, simply run:

```bash
python agent.py

```

*During training, the agent will automatically save the best-performing model weights to the `models/` directory and log metrics to the `logs/` directory.*

### Evaluating the Agent

To watch your trained agent play, run the script in evaluation mode and enable rendering:

```bash
python agent.py --evaluate --render

```

### CLI Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | `str` | `flappybirdv0` | The key name of the parameter set to load from `parameters.yaml`. |
| `--evaluate` | `flag` | `False` | Disables training and loads the best saved `.pt` weights for evaluation. |
| `--render` | `flag` | `False` | Enables PyGame graphical rendering to watch the agent play. |

---

## Hyperparameters & Configuration

The training process is governed by `parameters.yaml`. The default `flappybirdv0` configuration is tuned as follows:

| Parameter | Value | Description |
| --- | --- | --- |
| **Environment ID** | `FlappyBird-v0` | The target gymnasium environment. |
| **Max Reward Thresh** | `200` | The target score condition to halt training. |
| **Epsilon Init** | `1.0` | Initial exploration rate (100% random actions). |
| **Epsilon Min** | `0.01` | Minimum exploration rate bounds. |
| **Epsilon Decay** | `0.99952` | Multiplier applied to epsilon per episode. |
| **Replay Memory** | `50000` | Maximum transitions stored in the buffer. |
| **Mini Batch Size** | `64` | Number of samples pulled for network optimization. |
| **Target Sync Rate** | `1500` | Number of *steps* between target network updates. |
| **Alpha (LR)** | `0.00025` | Learning rate for the Adam optimizer. |
| **Gamma** | `0.99` | Discount factor for future rewards. |

---

## 📊 Neural Network Architecture

The DQN (`dqn.py`) evaluates a continuous 12-dimensional state space and outputs Q-values for 2 discrete actions (Flap or Do Nothing).

* **Input Layer:** 12 dimensions
* **Hidden Layer 1:** 512 dimensions + LeakyReLU(0.1)
* **Hidden Layer 2:** 512 dimensions + LeakyReLU(0.1)
* **Output Layer:** 2 dimensions (Linear)
