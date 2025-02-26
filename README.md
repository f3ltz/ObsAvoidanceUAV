# ObsAvoidanceUAV
IEEE Executive Proj

# Obstacle Collision Avoidance Using Deep Reinforcement Learning (DRL)

This project demonstrates the implementation of deep reinforcement learning (DRL) algorithms to solve the obstacle collision avoidance problem. The goal is to train an agent to navigate through an environment with obstacles while avoiding collisions, using various DRL algorithms, including:

-   Proximal Policy Optimization (PPO)
-   Deep Q-Network (DQN)
-   Deep Deterministic Policy Gradient (DDPG)
-   Twin Delayed Deep Deterministic Policy Gradient (TD3)


## Algorithms Implemented

### 1. **Proximal Policy Optimization (PPO)**

PPO is a policy gradient method that uses a clipped objective function to ensure that the agent's policy update does not change drastically, thus preventing instability. This algorithm was chosen for its stability and simplicity in training.

### 2. **Deep Q-Network (DQN)**

DQN is a value-based reinforcement learning algorithm that combines Q-learning with deep neural networks. It learns an action-value function to evaluate the optimal policy.

### 3. **Deep Deterministic Policy Gradient (DDPG)**

DDPG is an off-policy, model-free DRL algorithm that works well with continuous action spaces. It uses actor-critic methods and target networks to stabilize learning and improve performance.

### 4. **Twin Delayed Deep Deterministic Policy Gradient (TD3)**

TD3 is an improvement over DDPG, addressing some of its shortcomings. It uses techniques such as target smoothing, delayed updates, and deterministic target policies to improve stability and performance.

## Environment

The project uses a custom-made environment to simulate the obstacle collision avoidance task. The agent must learn how to navigate through the environment without colliding with obstacles.

### Key Features:

-   **Agent**: The agent can perform actions like moving forward, backward, left, and right.
-   **Obstacles**: The environment contains randomly placed obstacles that the agent needs to avoid.
-   **Reward System**: The agent is rewarded for avoiding collisions and completing tasks (e.g., reaching a target). Penalties are applied for collisions.

The environment is based on OpenAI’s `gym` library, making it easy to experiment with different DRL algorithms.

## Results

During training, the following results will be generated:

-   Training logs for each episode
-   Model checkpoints for the trained agents
-   Graphs showing the agent's performance over time

You can visualize the training progress and the agent's learning behavior by analyzing the logs and models.
