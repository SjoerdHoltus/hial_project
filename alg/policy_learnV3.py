import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from feature_func import feature_function
from trajectory_utils import generate_trajectories_from_files


DESIRED_GOAL = np.array([ 0.  , -0.2 ,  0.02])

def load_feature_weights(file_path):
    with open(file_path, 'r') as f:
        weights_str = f.read()
        weights = np.array([float(x) for x in weights_str.strip('[]').split()], dtype=np.float32)
    return weights

feature_weights = load_feature_weights('final_feature_weights.csv')

print(f"Loaded feature weights: {feature_weights}")

class RewardCalculator:
    def __init__(self, weights):
        self.weights = weights

    def calculate_reward(self, state, next_state=None):
        if next_state is None:
            return 0.0
        current_features = feature_function([(state, None)])
        next_features = feature_function([(next_state, None)])
        feature_delta = next_features - current_features
        reward = np.dot(self.weights, feature_delta)
        if self.is_success(next_state):
            reward += 10.0
        return float(reward)
    
    def is_success(self, state_dict):
        """Check if the state represents a successful grasp"""
        distance = np.linalg.norm(state_dict[7:10] - DESIRED_GOAL)
        return distance < 0.05

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming actions are bounded [-1, 1]

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
    
    def add(self, state, action, reward, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        
        batch_states = torch.tensor(self.states[idxs], dtype=torch.float32)
        batch_actions = torch.tensor(self.actions[idxs], dtype=torch.float32)
        batch_rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32)
        batch_next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states

# Advantage Calculation
def compute_advantage(critic, states, actions):
    Q_values = critic(states, actions).detach()
    V_values = Q_values.mean(dim=0, keepdim=True)
    return (Q_values - V_values).detach()

# AWAC Actor Loss
def awac_actor_loss(actor, critic, states, actions, beta):
    advantage = compute_advantage(critic, states, actions)
    weights = torch.exp(advantage / beta).clamp(max=100.0)
    print(f"Weights: {weights}")
    log_probs = torch.log(actor(states).gather(1, actions.long()))
    return -torch.mean(weights * log_probs)

# Sample Training Step
def train_awac(actor, critic, actor_optimizer, critic_optimizer, batch, gamma=0.99, beta=0.1):
    states, actions, rewards, next_states = batch
    
    with torch.no_grad():
        next_actions = actor(next_states)
        target_Q = rewards + gamma * critic(next_states, next_actions)
    
    critic_loss = torch.mean((critic(states, actions) - target_Q) ** 2)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    actor_loss = awac_actor_loss(actor, critic, states, actions, beta)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    return actor_loss.item(), critic_loss.item()


trajectories, _ = generate_trajectories_from_files()
state_dim = 22
action_dim = 4

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

buffer = ReplayBuffer(state_dim, action_dim, max_size=100000)

reward_calculator = RewardCalculator(feature_weights)

for demo in trajectories:
    for i in range(len(demo)-1):
        state = demo[i][0]
        action = demo[i][1]
        next_state = demo[i+1][0]
        reward = reward_calculator.calculate_reward(state, next_state)
        buffer.add(state, action, reward, next_state)

print(f"Buffer Size: {buffer.size}")

num_train_steps = 10_000 
batch_size = 64

for step in range(num_train_steps):
    batch = buffer.sample(batch_size)
    actor_loss, critic_loss = train_awac(actor, critic, actor_optimizer, critic_optimizer, batch)
    
    if step % 100 == 0:  # Print loss every 100 steps
        print(f"Step {step}: Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")
