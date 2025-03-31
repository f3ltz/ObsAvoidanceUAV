import subprocess
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import torch
from env import SimEnv
from agent import DDPGAgent
from memory import ReplayBuffer

class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """
        Initialize the Ornstein-Uhlenbeck (OU) noise process.

        Args:
        - mu (array-like): The mean value towards which the noise will decay (target mean).
        - sigma (float): The standard deviation of the noise, controlling its magnitude.
        - theta (float): The rate at which the noise reverts to the mean (`mu`).
        - dt (float): The timestep for each noise update.
        - x0 (array-like, optional): Initial value of the noise. Defaults to zeros.

        Attributes:
        - theta: Controls how strongly the process is pulled toward `mu`.
        - sigma: Magnitude of the random fluctuations.
        - dt: Timestep for each update.
        - x0: Initial state of the noise process.
        - x_prev: Tracks the previous noise value for generating the next one.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()  # Initialize the noise state

    def __call__(self):
        """
        Generate the next noise value using the Ornstein-Uhlenbeck process.

        Formula:
        x_next = x_prev + theta * (mu - x_prev) * dt + sigma * sqrt(dt) * N(0, 1)

        Returns:
        - x (array-like): The updated noise value.
        """
        # Calculate the next value based on the OU formula
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +  # Drift term
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))  # Random fluctuation term
        x = np.clip(x,0,10)
        self.x_prev = x  # Update the previous value
        
        return x  # Return the current noise value

    def reset(self):
        """
        Reset the noise process to its initial state.
        If `x0` is provided, it resets to `x0`. Otherwise, it resets to zero.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def StartSimHeadless():
    coppelia_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\CoppeliaSim.exe"
    scene_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\scenes\pathfollowerbot.ttt"
    subprocess.Popen([coppelia_path, "-h", "-f", scene_path])

def StartSimGUI():
    coppelia_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\CoppeliaSim.exe"
    scene_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\scenes\pathfollowerbot.ttt"
    subprocess.Popen([coppelia_path, "-f", scene_path])


# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#StartSimGUI()

# Connect to CoppeliaSim via the ZeroMQ Remote API
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")

# Retrieve the drone handle (adjust the name based on your scene)
drone_handle = sim.getObject("/Quadcopter")
joint_handles = [sim.getObject(f'/propeller[{i}]/respondable/body/joint')for i in range(4)]
target_handle = sim.getObject("/target")
target = sim.getObjectPosition(target_handle)  # Define the target position for the drone
#sim.setObjectParent(target_handle,-1,True)
sim.setObjectPosition(target_handle,-1,sim.getObjectPosition(sim.getObject("/Sphere[4]")))

# Create our environment wrapper
env = SimEnv(sim, drone_handle, joint_handles, target)
state_dim = 10  # [position (3) + velocity (6) + distance to targ]
action_dim = 8  # [velocity command in x, y, z]
max_action = 3  # Maximum allowed value for each action component
noise = OUActionNoise(mu = np.zeros(action_dim), dt = sim.getSimulationTimeStep())

# Initialize the DDPG agent and replay buffer
agent = DDPGAgent(state_dim, action_dim, max_action, device)
agent.load_models()
replay_buffer = ReplayBuffer(max_size=100000)

num_episodes = 10000
max_steps = 100
batch_size = 128
best_reward = -500000

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Select action with exploration noise
        action = agent.select_action(state)
        #noise = np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action + noise(), -max_action, max_action)
        
        # Step simulation and obtain next state and reward
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        # Train the agent using experiences from the replay buffer
        agent.train(replay_buffer, batch_size)
        
        if done:
            break
    if(episode_reward>best_reward):
        best_reward = episode_reward
        agent.actor.save_checkpoint()
        agent.actor_target.save_checkpoint()
        agent.critic.save_checkpoint()
        agent.critic_target.save_checkpoint()

    print(f"Episode {episode+1}: Total Reward = {episode_reward:.3f}")

# Optionally, stop the simulation after training
sim.stopSimulation()

