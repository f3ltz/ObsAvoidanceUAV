import numpy as np
import random
import torch as T
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.distributions.categorical import Categorical
import os
import time

# Check if GPU is available and set the device
device = "cuda:0" if T.cuda.is_available() else "cpu"
print(device)

# Helper function to plot the running average of scores
def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')


# Memory class for storing transitions during training
class BufferMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    # Generate mini-batches for training
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    # Store transition in memory
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # Clear memory after training
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

# Actor network for the PPO algorithm
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()


        # Neural network architecture
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        # Optimizer and device setup
        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward pass
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


# Critic network for the PPO algorithm
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        # Neural network architecture
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        # Optimizer and device setup
        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward pass
    def forward(self, state):
        value = self.critic(state)
        return value


# PPO Agent class
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.001, gae_lambda=0.95,
            policy_clip=0.1, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        # Initialize actor, critic, and memory
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = BufferMemory(batch_size)

    # Convert grid position to one-hot input
    def converter(self, posi):
        x, y = posi
        inp = T.zeros(100)  # 10x10 grid = 100 states
        inp[10 * x + y] = 1
        return inp

    # Store transition in memory
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    # Choose action using the actor network
    def choose_action(self, observation):
        state = self.converter(observation).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = (dist.log_prob(action)).item()
        action = (action).item()
        value = (value).item()

        return action, probs, value

    # Train the agent
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate advantages
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(np.array([self.converter(state) for state in state_arr[batch]]), dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

# Environment class for the grid world game
class Environment():
    # Initializing the class with initial position and grid
    def __init__(self):
        self.pos = (0, 0)
        self.grid = np.zeros((10, 10), dtype=int)
        self.obs = set()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.cmap = ListedColormap(["red", "white", "blue", "green", "yellow"])
        self.plot = None

        # Initialize the plot
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.set_xticks(np.arange(0.5, 10, 1))
        self.ax.set_xticklabels(range(10))
        self.ax.set_yticks(np.arange(0.5, 10, 1))
        self.ax.set_yticklabels(range(10))
        self.ax.set_title("CurrentPosition(Yellow), Start(Blue), Obstacles(Red), End(Green)", y=-0.1)

    # Function to reset the grid and add obstacles
    def reset(self):
        self.grid = np.zeros((10, 10), dtype=int)
        self.pos = (0, 0)
        self.obs = set()

        # Add predefined obstacles
        self.obs.add((3,2))
        self.obs.add((2,2))
        self.obs.add((2,4))
        self.obs.add((1,2))
        self.obs.add((1,1))
        self.obs.add((5,4))
        self.obs.add((4,4))
        self.obs.add((4,6))
        self.obs.add((3,5))
        self.obs.add((3,3))
        self.obs.add((7,6))
        self.obs.add((6,6))
        self.obs.add((6,8))
        self.obs.add((5,6))
        self.obs.add((4,5))
        for obs in self.obs:
            self.grid[obs] = -1
        self.grid[0, 0] = 1  # Start position
        self.grid[9, 9] = 2  # End position

    # Function to move the agent given an action
    def step(self, action):
        y, x = self.pos
        if action == 0 and y > 0:
            newpos = (y - 1, x)  # Move up
        elif action == 1 and y < 9:
            newpos = (y + 1, x)  # Move down
        elif action == 2 and x > 0:
            newpos = (y, x - 1)  # Move left
        elif action == 3 and x < 9:
            newpos = (y, x + 1)  # Move right
        else:
            newpos = (y, x)  # Invalid move

        if self.grid[newpos] == -1:  # Obstacle
            return self.pos, -15, False

        self.pos = newpos
        reward = -1
        terminated = False
        if self.pos == (9,9):  # Reached the goal
            reward = 20
            terminated = True
        return self.pos, reward, terminated

    # Render the grid world
    def render(self):
        # Update the grid with the current position
        grid = self.grid.copy()
        grid[self.pos] = 4  # 4 corresponds to Yellow in cmap

        if self.plot is None:
            # First-time plot initialization
            self.plot = self.ax.pcolormesh(grid, cmap=self.cmap, edgecolors='k', linewidth=0.5, shading='auto')
        else:
            # Update the plot data
            self.plot.set_array(grid.ravel())

        # Redraw the plot
        plt.draw()
        plt.pause(0.05)  # Pause for a short duration to allow updates

if __name__ == '__main__':
    env = Environment()
    agent = Agent(n_actions=4, input_dims=100)
    n_games =50

    N = 100;
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        env.reset()
        observation = env.pos
        done = False
        score = 0
        while not done:
            env.render()
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        """if avg_score > best_score:
            best_score = avg_score
            agent.save_models()"""
        if i % 1 == 0 and i > 0:

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    T.save(ActorNetwork.state_dict(), "grid.pt")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(score_history)
    ax.set_title("Rewards Per Episode")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")