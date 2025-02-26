import numpy as np
import random
import torch
from torch import nn
from collections import deque
import matplotlib.pyplot as plt


#class created for replay memory buffer
class ReplayMem():
    #initializing class
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    #function to append data to buffer
    def append(self, transition):
        self.memory.append(transition)

    #function for sampling minibatches from buffer
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)

class Agent():
    #initializing class
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.actions = ['U', 'D', 'L', 'R']

    #function that returns one out of 4 possible actions
    def sample(self):
        return random.randint(0, 3)
    
    #function to convert int input to tensor
    def convert(self, posi):
        inp = torch.zeros(25)  # 5x5 grid = 25 states
        inp[posi] = 1
        return inp

    #function to convert index input to tensor
    def converter(self, posi):
        x, y = posi
        inp = torch.zeros(25)  # 5x5 grid = 25 states
        inp[5 * x + y] = 1
        return inp


    #function to train agent
    def train(self, episodes):
        #set epsilon to 1 and declare memory buffer
        self.epsilon = 1
        memory = ReplayMem(maxlen=4000)
        
        # Define policy and target networks
        policy_DQN = nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10 , 4)
        )
        
        target_DQN = nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10 , 4)
        )
       
        #copy policy network onto target network
        target_DQN.load_state_dict(policy_DQN.state_dict())

        #declare optimizer and print policy
        self.optimizer = torch.optim.Adam(policy_DQN.parameters(), lr=self.alpha)
        self.print_dqn(policy_DQN)

        #declaring arrays for rewards and epsilon history
        rewards_per_episode = np.zeros(episodes)
        epsilon_hist = []

        
        #episode loop
        for i in range(episodes):
            #reset variables
            env.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            #while loop to navigate through grid
            while not terminated and step_count<1000:
                step_count+=1

                #taking actions according to epsilon greedy policy
                if random.random() < self.epsilon:
                    action = self.sample()
                else:
                    with torch.no_grad():
                        action = policy_DQN(self.converter(env.pos)).argmax().item()

                
                #taking step and storing data in memory buffer
                init_state = env.pos
                new_state, reward, terminated = env.step(action)
                total_reward += reward
                memory.append((init_state, action, new_state, reward, terminated))

            rewards_per_episode[i] = (total_reward)
            
            #optimizing neural network in minibatches of size 64
            if len(memory) > 64:
                mini_batch = memory.sample(64)
                self.optimize(mini_batch, policy_DQN, target_DQN,i)
                if  i%100==0:
                    self.print_dqn(policy_DQN)

            # Decay epsilon
                self.epsilon = max(self.epsilon *0.9975, 0.1)
                epsilon_hist.append(self.epsilon)

            # Update target network
                target_DQN.load_state_dict(policy_DQN.state_dict())

        # Save the trained model
        torch.save(policy_DQN.state_dict(), "grid.pt")


        # Plot results
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.title("Rewards Per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.subplot(122)
        plt.plot(epsilon_hist)
        plt.title("Epsilon Decay")
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")
        plt.show()

    def test(self, episodes):
        env.reset()

        policy_DQN = nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10 , 4)
        )

        #loading trained model
        policy_DQN.load_state_dict(torch.load("grid.pt"))
        policy_DQN.eval()
        print('Policy (trained):')
        self.print_dqn(policy_DQN)

        #simulate episodes
        for i in range(episodes):
            env.reset()
            terminated = False
            step = 0
            #taking actions according to trained policy
            while not terminated and step<200:
                step +=1
                init_state = env.pos
                with torch.no_grad():
                    action = policy_DQN(self.converter(env.pos)).argmax().item()

                new_state , reward, terminated = env.step(action=action)
                if env.pos == (4,4):
                    print("End Reached!!")
                env.render()
        


    #function to optimize neural net
    def optimize(self, mini_batch, policy_DQN, target_DQN, i):

        current_q_list = []
        target_q_list = []
        
        #loops through minibatch
        for init_state, action, pos, reward, terminated in mini_batch:
            
            #updates target according to bellman equation with alpha set to 1
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor((reward + self.gamma * target_DQN(self.converter(pos)).max()))

            #storing q values of policy and target network
            current_q = policy_DQN(self.converter(init_state))
            current_q_list.append(current_q)

            target_q = target_DQN(self.converter(init_state))
            target_q[action] = target
            target_q_list.append(target_q)

       
        #calculating loss
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        
        #optimizing the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #function to print policy 
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = 25

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.convert(s)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.actions[dqn(self.convert(s)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%5==0:
                print() # Print a newline every 4 states

class Environment():
    #initalizing class with inital position and making grid
    def __init__(self):
        self.pos = (0, 0)
        self.grid = np.zeros((5, 5), dtype=int)
        self.obs = set()

    #function to reset grid and add obstacles
    def reset(self):
        self.grid = np.zeros((5, 5), dtype=int)
        self.pos = (0, 0)
        self.obs = set()

        self.obs.add((3,2))
        self.obs.add((2,2))
        self.obs.add((2,4))
        self.obs.add((1,2))
        self.obs.add((1,1))
        for obs in self.obs:
            self.grid[obs] = -1
        self.grid[0, 0] = 1
        self.grid[4, 4] = 2

    #function to move agent given an input action
    def step(self, action):
        y, x = self.pos
        if action == 0 and y > 0:
            newpos = (y - 1, x)
        elif action == 1 and y < 4:
            newpos = (y + 1, x)
        elif action == 2 and x > 0:
            newpos = (y, x - 1)
        elif action == 3 and x < 4:
            newpos = (y, x + 1)
        else:
            newpos = (y, x)

        if self.grid[newpos] == -1:
            return self.pos, -10, False

        self.pos = newpos
        reward = -1
        terminated = False
        if self.pos == (4,4):
            reward = 10
            terminated = True
        return self.pos, reward, terminated


    #function to render the grid
    def render(self):
        grid = self.grid.copy()
        grid[self.pos] = 3
        print(grid)
    
    

env = Environment()

env.reset()
env.render()

agent = Agent(0.01, 0.99)
agent.train(500)
agent.test(10)
env.render()

