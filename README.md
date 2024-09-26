# Deep Q-Network (DQN) for Lunar Lander Environment

## Overview

This project implements a **Deep Q-Network (DQN)** to solve the **Lunar Lander** problem using **reinforcement learning (RL)**. The goal is to train an agent to land a lunar module safely on a designated landing pad using a value-based RL approach. We use **TensorFlow** to build the deep neural networks that approximate the Q-function and OpenAI's **Gym** to simulate the environment.

## Key Concepts

### Reinforcement Learning (RL)

**Reinforcement Learning (RL)** is a machine learning technique where an agent learns to make decisions by interacting with an environment. Through these interactions, the agent receives **rewards** or **penalties** based on the outcome of its actions, and the goal is to maximize the cumulative reward over time.

### The Lunar Lander Environment

In the **Lunar Lander** environment, the goal is to safely land the lunar module on a designated landing pad, located at the center of the environment. The environment is part of OpenAI’s **Gym** and is commonly used as a testbed for reinforcement learning algorithms.

Key attributes of the environment:
- The landing pad is located at **coordinates (0, 0)**, but landing is allowed outside the pad with reduced rewards.
- The agent (lander) starts at the top center with a random initial force.
- The agent has **infinite fuel** for the training process.

### Markov Decision Process (MDP)

The Lunar Lander environment can be modeled as a **Markov Decision Process (MDP)**, which consists of:

- **State**: The current position, velocity, and other relevant variables of the lander.
- **Action**: The agent's choice of applying force to the lander’s engines (e.g., firing the main, right, or left engine).
- **Reward**: The feedback the agent receives, such as a positive reward for a safe landing or negative rewards for crashes or fuel wastage.
- **Transition Function**: The dynamics that determine how the state changes based on the agent’s action.

### Deep Q-Learning (DQN)

**Deep Q-Learning** is a reinforcement learning algorithm that combines **Q-learning**, a value-based method, with deep neural networks. The goal of DQN is to approximate the Q-function, which predicts the future cumulative reward for each state-action pair. The agent selects the action with the highest predicted reward (Q-value).

Key components:
- **Q-Function**: Approximates future rewards.
- **Experience Replay**: Stores past experiences to train the model and break correlations between consecutive samples.
- **Target Network**: Stabilizes training by keeping a fixed target network for a number of iterations.

### TensorFlow

We use **TensorFlow**, an open-source machine learning framework, to build and train the deep neural network that estimates the Q-values for each state-action pair.

### Agent Actions

The agent has 4 discrete actions available:
1. **Do nothing** (Action = 0)
2. **Fire right engine** (Action = 1)
3. **Fire main engine** (Action = 2)
4. **Fire left engine** (Action = 3)

### Observation Space & Action Space

- **Observation Space**: The state vector consists of 8 variables, including the lander’s position, velocity, angle, and angular velocity.
- **Action Space**: There are 4 discrete actions the agent can choose from, as mentioned above.

The number of input neurons to the neural network corresponds to the size of the state vector (8), and the output neurons correspond to the number of possible actions (4).

## Conclusion

In this project, the DQN model was able to progressively improve its performance in the Lunar Lander environment. As training progressed, the agent learned to maximize the rewards by improving its landing strategy. However, some fluctuation in the reward signal indicates the potential need for further tuning of hyperparameters or exploration-exploitation balance.

Although the agent's performance showed a positive trend (please refer to the jupyter notebook file), further refinements could lead to more stable landings and consistent rewards over time.
