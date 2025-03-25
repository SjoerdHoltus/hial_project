import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Setup paths and imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from init_env import init_env
from feature_func import feature_function
from trajectory_utils import generate_trajectories_from_files

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 150
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
HIDDEN_DIM = 256
AWAC_LAMBDA = 1.0

class RewardCalculator:
    def __init__(self, weights_file='final_feature_weights.csv'):
        # Load learned weights
        with open(os.path.join(CURRENT_DIR, weights_file), 'r') as f:
            weights_str = f.read()
            self.weights = np.array([float(x) for x in weights_str.strip('[]').split()], 
                                  dtype=np.float32)
        print(f"Loaded feature weights: {self.weights}")

    def calculate_reward(self, state, next_state):
        # Calculate features and feature differences
        current_features = feature_function([(state, None)])
        next_features = feature_function([(next_state, None)])
        feature_delta = next_features - current_features
        
        # Calculate reward
        reward = float(np.dot(self.weights, feature_delta))
        
        # Add bonus for task completion
        if self.is_success(next_state):
            reward += 10.0
            
        return reward

    def is_success(self, state):
        achieved_goal = state[7:10]
        desired_goal = np.array([0.0, -0.2, 0.02])
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.is_demo = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done, is_demo=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.is_demo[self.ptr] = is_demo
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Sample both demonstrations and policy data
        demo_batch_size = batch_size // 4
        policy_batch_size = batch_size - demo_batch_size
        
        demo_idx = np.random.choice(np.where(self.is_demo == 1)[0], size=demo_batch_size)
        policy_idx = np.random.choice(np.where(self.is_demo == 0)[0], size=policy_batch_size)
        idx = np.concatenate([demo_idx, policy_idx])
        
        return (
            torch.FloatTensor(self.state[idx]).to(DEVICE),
            torch.FloatTensor(self.action[idx]).to(DEVICE),
            torch.FloatTensor(self.next_state[idx]).to(DEVICE),
            torch.FloatTensor(self.reward[idx]).to(DEVICE),
            torch.FloatTensor(self.done[idx]).to(DEVICE),
            torch.FloatTensor(self.is_demo[idx]).to(DEVICE)
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)
        
    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        return mean, log_std.exp()

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        action = normal.rsample()
        return torch.tanh(action)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

def train_awac():
    # Initialize environment and get dimensions
    env = init_env(render=False)
    state_dim = env.observation_space['observation'].shape[0] + env.observation_space['achieved_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize components
    reward_calculator = RewardCalculator()
    buffer = ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    
    # Load demonstrations
    print("Loading expert demonstrations...")
    demos, _ = generate_trajectories_from_files()
    for demo in demos:
        for i in range(len(demo)-1):
            state = demo[i][0]
            next_state = demo[i+1][0]
            action = demo[i][1]
            reward = reward_calculator.calculate_reward(state, next_state)
            done = float(i == len(demo)-2)
            buffer.add(state, action, next_state, reward, done, is_demo=1)
    print(f"Buffer size: {buffer.size}")
    # Training metrics
    episode_rewards = []
    success_rates = []
    
    # Training loop
    for episode in range(MAX_EPISODES):
        state = env.reset()
        state_array = np.concatenate([state['observation'], state['achieved_goal']])
        episode_reward = 0
        
        # Episode rollout
        for step in range(MAX_STEPS_PER_EPISODE):
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(DEVICE)
                action = actor.sample(state_tensor).cpu().numpy()[0]
            
            # Take step in environment
            next_state, _, done, info = env.step(action)
            next_state_array = np.concatenate([next_state['observation'], next_state['achieved_goal']])
            
            # Calculate reward
            reward = reward_calculator.calculate_reward(state_array, next_state_array)
            episode_reward += reward
            
            # Store transition
            buffer.add(state_array, action, next_state_array, reward, float(done))
            
            if buffer.size > BATCH_SIZE:
                # Sample batch
                state_batch, action_batch, next_state_batch, reward_batch, done_batch, is_demo_batch = buffer.sample(BATCH_SIZE)
                
                # Update critic
                with torch.no_grad():
                    next_action = actor.sample(next_state_batch)
                    target_q1, target_q2 = critic_target(next_state_batch, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = reward_batch + (1 - done_batch) * GAMMA * target_q
                
                current_q1, current_q2 = critic(state_batch, action_batch)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Update actor using AWAC
                q1, q2 = critic(state_batch, action_batch)
                q = torch.min(q1, q2)
                
                # Calculate advantages
                with torch.no_grad():
                    policy_action = actor.sample(state_batch)
                    policy_q1, policy_q2 = critic(state_batch, policy_action)
                    policy_q = torch.min(policy_q1, policy_q2)
                    advantage = (q - policy_q).unsqueeze(-1)
                    weights = torch.exp(advantage / AWAC_LAMBDA)
                    weights = weights * (1 + is_demo_batch)  # Extra weight for demos
                
                actor_loss = -torch.mean(weights * q)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Update target network
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            if done:
                break
                
            state = next_state
            state_array = next_state_array
        
        # Track metrics
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            success_rate = evaluate_policy(env, actor)
            success_rates.append(success_rate)
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Success Rate: {success_rate:.2f}")
            
            # Plot progress
            plot_training_progress(episode_rewards, success_rates)

def evaluate_policy(env, actor, n_episodes=10):
    successes = 0
    for _ in range(n_episodes):
        state = env.reset()
        state_array = np.concatenate([state['observation'], state['achieved_goal']])
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(DEVICE)
                action = actor.sample(state_tensor).cpu().numpy()[0]
            next_state, _, done, info = env.step(action)
            state_array = np.concatenate([next_state['observation'], next_state['achieved_goal']])
            
            if info.get('is_success', False):
                successes += 1
                break
    
    return successes / n_episodes

def plot_training_progress(rewards, success_rates):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(rewards), 10), success_rates)
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    train_awac()