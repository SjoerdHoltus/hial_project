import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import os
import sys
from torch.distributions import Normal
import copy



# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from init_env import init_env
from trajectory_utils import generate_trajectories_from_files
from feature_func import feature_function

# Set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enhanced CUDA setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

# Constants
MAX_ENV_STEPS = 500000
SAVE_INTERVAL = 1000
EVAL_INTERVAL = 1000
MAX_EPISODE_STEPS = 300
T = 500
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.01
HIDDEN_DIM = 256
BUFFER_SIZE = 1000000
AWAC_LAMBDA = 2.0
LR = 1e-5

DESIRED_GOAL = np.array([ 0.  , -0.2 ,  0.02])

# Directory for saving models and results
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load feature weights
def load_feature_weights(file_path):
    with open(file_path, 'r') as f:
        weights_str = f.read()
        # Strip brackets and convert to numpy array
        weights = np.array([float(x) for x in weights_str.strip('[]').split()], dtype=np.float32)
    return weights

feature_weights = load_feature_weights(os.path.join(CURRENT_DIR, 'final_feature_weights.csv'))
print(f"Loaded feature weights: {feature_weights}")

# Reward calculation
class RewardCalculator:
    def __init__(self, weights):
        self.weights = weights

    def calculate_reward(self, state, next_state=None):
        """Calculate reward using feature function and weights"""
        # For terminal state
        if next_state is None:
            return 0.0
        
        # Calculate features for current and next state
        current_features = feature_function([(state, None)])
        next_features = feature_function([(next_state, None)])
        
        # Calculate change in features
        feature_delta = next_features - current_features
        
        # Calculate reward (negative rewards for reducing distance)
        reward = np.dot(self.weights, feature_delta)
        
        # Scale the regular reward
        reward = reward * 0.1  # Scale down regular rewards
        
        # Add sparse reward for task completion
        if self.is_success(next_state):
            reward += 10.0
        return float(reward)
    
    def is_success(self, state_dict):
        """Check if the state represents a successful grasp"""
        distance = np.linalg.norm(state_dict[-3:] - DESIRED_GOAL)
        return distance < 0.05

# Replay buffer for AWAC
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
        
        # Demo buffer indicator (1 if from demo, 0 if from policy)
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
        # Ensure we sample some demos in each batch
        demo_batch_size = batch_size // 2 
        policy_batch_size = batch_size - demo_batch_size
        
        # Sample separately from demos and policy experience
        demo_ind = np.random.choice(np.where(self.is_demo == 1)[0], size=demo_batch_size)
        policy_ind = np.random.choice(np.where(self.is_demo == 0)[0], size=policy_batch_size)
        ind = np.concatenate([demo_ind, policy_ind])
        
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device),
            torch.FloatTensor(self.is_demo[ind]).to(device)
        )
        
    def add_trajectory(self, states, actions, rewards, dones, is_demo=0):
        for i in range(len(states)-1):
            self.add(states[i], actions[i], states[i+1], rewards[i], dones[i], is_demo)

# Actor network for AWAC
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a).clamp(-20, 2)  # Constrain for numerical stability
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        # Add extra exploration noise during training
        if self.training:
            noise = torch.randn_like(x_t) * 0.1
            x_t = x_t + noise
            
        action = torch.tanh(x_t)
        return self.max_action * action
    
    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob

# Critic network for AWAC
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for stability)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

# AWAC Agent
class AWAC:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
        self.max_action = max_action
        self.awac_lambda = AWAC_LAMBDA
    
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            if evaluate:
                # For evaluation, use mean action instead of sampled action
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.max_action
            else:
                action = self.actor.sample(state_tensor)
        return action.cpu().numpy().flatten()
    
    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        # Sample from replay buffer
        state, action, next_state, reward, done, is_demo = replay_buffer.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            next_action = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * GAMMA * target_Q
            
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # AWAC actor update
        # Get Q-values for actions
        with torch.no_grad():
            current_Q1, current_Q2 = self.critic(state, action)
            current_Q = torch.min(current_Q1, current_Q2)
            
            # Sample new actions for advantage calculation
            new_actions = self.actor.sample(state)
            new_Q1, new_Q2 = self.critic(state, new_actions)
            new_Q = torch.min(new_Q1, new_Q2)
            
            # Calculate advantage
            advantage = (current_Q - new_Q).clamp(-20, 20)
            
            # Calculate weights with exponential advantage
            weights = torch.exp(advantage / self.awac_lambda).clamp(0, 10)
            
            # Extra weight for demonstrations
            demo_weight = 5.0
            weights = weights * (1.0 + is_demo * (demo_weight - 1.0))
        
        # Actor loss
        actor_loss = -self.actor.log_prob(state, action) * weights
        actor_loss = actor_loss.mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
    
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))

# Prepare demonstrations for buffer
def process_demos(reward_calculator):
    # Get expert demonstrations
    trajectories, _ = generate_trajectories_from_files()
    
    # Process demos
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for traj in trajectories:
        states = []
        actions = []
        rewards = []
        dones = []
        
        for i in range(len(traj)):
            state = traj[i][0]
            action = traj[i][1] if i < len(traj) - 1 else np.zeros_like(traj[0][1])
            
            states.append(state)
            actions.append(action)
            
            # Last state has no next state
            if i < len(traj) - 1:
                next_state = traj[i+1][0]
                reward = reward_calculator.calculate_reward(state, next_state)
                done = 0.0
            else:
                reward = 0.0
                done = 1.0
            rewards.append(reward)
            dones.append(done)
            
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)
    
    return all_states, all_actions, all_rewards, all_dones

# Evaluation function
def evaluate_policy(policy, reward_calculator, n_episodes=10):
    env = init_env(render=False)
    successes = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        state = np.concatenate((state['observation'], state['achieved_goal']))
        done = False
        
        while not done:
            action = policy.select_action(state, evaluate=True)
            next_state, _, done, info = env.step(action)
            next_state = np.concatenate((next_state['observation'], next_state['achieved_goal']))
            state = next_state
            
            # Check if episode was successful
            if info.get('is_success', False):
                successes += 1
                break
        
    env.close()
    return successes / n_episodes

# Add this class near the top of your file
class TrainingStats:
    def __init__(self):
        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []
        self.success_rates = []
        self.avg_rewards_history = []
        self.demo_ratios = []
        
    def add_training_stats(self, critic_loss, actor_loss, episode_reward, success_rate, avg_reward, demo_ratio):
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.episode_rewards.append(episode_reward)
        self.success_rates.append(success_rate)
        self.avg_rewards_history.append(avg_reward)
        self.demo_ratios.append(demo_ratio)
    
    def plot_stats(self, save_dir):
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Losses
        plt.subplot(2, 2, 1)
        plt.plot(self.critic_losses[-1000:], label='Critic Loss')
        plt.plot(self.actor_losses[-1000:], label='Actor Loss')
        plt.title('Recent Training Losses')
        plt.legend()
        
        # Plot 2: Episode Rewards
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_rewards[-100:], label='Episode Reward')
        plt.plot(self.avg_rewards_history[-100:], label='Average Reward')
        plt.title('Recent Rewards')
        plt.legend()
        
        # Plot 3: Success Rate
        plt.subplot(2, 2, 3)
        plt.plot(self.success_rates, label='Success Rate')
        plt.title('Success Rate Over Time')
        plt.legend()
        
        # Plot 4: Demo Ratio
        plt.subplot(2, 2, 4)
        plt.plot(self.demo_ratios, label='Demo Ratio')
        plt.title('Demonstration Ratio')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_stats.png")
        plt.close()
        
        # Print current statistics
        print("\nCurrent Training Statistics:")
        print(f"Recent Critic Loss: {np.mean(self.critic_losses[-100:]):.3f}")
        print(f"Recent Actor Loss: {np.mean(self.actor_losses[-100:]):.3f}")
        print(f"Recent Average Reward: {np.mean(self.episode_rewards[-10:]):.3f}")
        print(f"Recent Success Rate: {self.success_rates[-1]:.3f}")
        print(f"Current Demo Ratio: {self.demo_ratios[-1]:.3f}")

# Main training loop
def train():
    env = init_env(render=False)
    print(env.observation_space)
    print(env.action_space)
    
    # Determine state_dim from environment
    # The observation space includes both 'observation' and 'achieved_goal'
    state_dim = env.observation_space['observation'].shape[0] + env.observation_space['achieved_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize reward calculator
    reward_calculator = RewardCalculator(feature_weights)
    
    # Initialize policy
    policy = AWAC(state_dim, action_dim, max_action)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Process and add demonstrations
    print("Processing demonstrations...")
    demo_states, demo_actions, demo_rewards, demo_dones = process_demos(reward_calculator)
    
    # Add demonstrations to buffer
    print("Adding demonstrations to buffer...")
    for states, actions, rewards, dones in zip(demo_states, demo_actions, demo_rewards, demo_dones):
        replay_buffer.add_trajectory(states, actions, rewards, dones, is_demo=1)
    
    print(f"Demonstration buffer size: {replay_buffer.size}")
    
    # Training metrics
    evaluations = []
    episode_rewards = []
    total_timesteps = 0
    
    # Create plot for learning curve
    plt.figure(figsize=(10, 5))
    
    # Add these lists for monitoring
    critic_losses = []
    actor_losses = []
    advantages = []
    
    # Initialize stats tracker
    stats = TrainingStats()
    
    # Start training
    print("Starting training...")
    while total_timesteps < MAX_ENV_STEPS:
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        state = np.concatenate((state['observation'], state['achieved_goal']))
            
        # Collect experience with current policy
        states = [state]
        actions = []
        rewards = []
        dones = []
        
        while not done and episode_steps < MAX_EPISODE_STEPS:
            # Select action (passing the dictionary state)
            action = policy.select_action(state)
            actions.append(action)
            
            # Take step in environment
            next_state, _, done, _ = env.step(action)
            next_state = np.concatenate((next_state['observation'], next_state['achieved_goal']))
            # Calculate reward using learned reward function
            reward = reward_calculator.calculate_reward(state, next_state)
            rewards.append(reward)
            
            # Track done
            dones.append(float(done))
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Save next state
            states.append(next_state)
            state = next_state
            
            # Update policy and collect stats
            if replay_buffer.size > BATCH_SIZE:
                train_info = policy.train(replay_buffer)
                critic_losses.append(train_info['critic_loss'])
                actor_losses.append(train_info['actor_loss'])
                
                if total_timesteps % 1000 == 0:
                    success_rate = evaluate_policy(policy, reward_calculator)
                    demo_ratio = np.mean(replay_buffer.is_demo[:replay_buffer.size])
                    avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                    
                    # Add stats
                    stats.add_training_stats(
                        np.mean(critic_losses[-100:]),
                        np.mean(actor_losses[-100:]),
                        episode_reward,
                        success_rate,
                        avg_reward,
                        demo_ratio
                    )
                    
                    # Plot and save stats
                    stats.plot_stats(RESULTS_DIR)
                    
                    # Save stats to file
                    np.savez(
                        f"{RESULTS_DIR}/training_stats.npz",
                        critic_losses=stats.critic_losses,
                        actor_losses=stats.actor_losses,
                        episode_rewards=stats.episode_rewards,
                        success_rates=stats.success_rates,
                        avg_rewards=stats.avg_rewards_history,
                        demo_ratios=stats.demo_ratios
                    )
            
            # Save model periodically
            if total_timesteps % SAVE_INTERVAL == 0:
                policy.save(f"{MODEL_DIR}/awac_{total_timesteps}")
            
            # Evaluate policy periodically
            if total_timesteps % EVAL_INTERVAL == 0:
                print(f"Timesteps: {total_timesteps}, Evaluating policy...")
                success_rate = evaluate_policy(policy, reward_calculator)
                evaluations.append((total_timesteps, success_rate))
                print(f"Success rate: {success_rate}")
                
                # Update plot
                timesteps, success_rates = zip(*evaluations)
                plt.clf()
                plt.plot(timesteps, success_rates)
                plt.xlabel('Environment Steps')
                plt.ylabel('Success Rate')
                plt.title('AWAC Policy Learning Curve')
                plt.savefig(f"{RESULTS_DIR}/learning_curve.png")
                
                # Save data
                np.save(f"{RESULTS_DIR}/evaluations.npy", evaluations)
        
        # Add episode to buffer
        replay_buffer.add_trajectory(states[:-1], actions, rewards, dones)
        
        # Track episode rewards
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode Steps: {episode_steps}, Total Steps: {total_timesteps}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
    
    # Final evaluation and save
    success_rate = evaluate_policy(policy, reward_calculator, n_episodes=20)
    print(f"Final success rate: {success_rate}")
    policy.save(f"{MODEL_DIR}/awac_final")
    
    # Final plot
    timesteps, success_rates = zip(*evaluations)
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, success_rates)
    plt.xlabel('Environment Steps')
    plt.ylabel('Success Rate')
    plt.title('AWAC Policy Learning Curve')
    plt.savefig(f"{RESULTS_DIR}/final_learning_curve.png")
    
    return policy, evaluations

if __name__ == "__main__":
    train()
