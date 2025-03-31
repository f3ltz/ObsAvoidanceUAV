import torch
import torch.nn as nn
import torch.optim as optim
import os

##########################
# Neural Network Models  #
##########################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, name, chkpt_dir = 'tmp/ddpg'):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = ((torch.tanh(self.fc3(x))+1) * self.max_action )
        return action
    
    def save_checkpoint(self):
        print("....Saving Checkpoint....")
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("....Loading Checkpoint....")
        checkpoint = torch.load(self.checkpoint_file, map_location=torch.device('cpu'))  # Load the checkpoint
        self.load_state_dict(checkpoint)  # Update the model with the loaded state_dict

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, name, chkpt_dir = 'tmp/ddpg'):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_val = self.fc3(x)
        return q_val
    
    def save_checkpoint(self):
        print("....Saving Checkpoint....")
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("....Loading Checkpoint....")
        checkpoint = torch.load(self.checkpoint_file, map_location=torch.device('cpu'))  # Load the checkpoint
        self.load_state_dict(checkpoint)  # Update the model with the loaded state_dict