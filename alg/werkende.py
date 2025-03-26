#############################################
# policy_learn.py
#############################################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # for plotting at the end

# Adjust these imports to match your local file names/structure
from feature_func import feature_function
from trajectory_utils import generate_trajectories_from_files
from init_env import init_env

################################################################################
# 1) Load Learned Weights
################################################################################
def load_learned_weights(path="final_feature_weights.csv"):
    """
    Loads the learned weights from CSV, which is assumed to contain a
    single line of the form: [w0 w1 w2 ... ].
    Returns a 1D NumPy array.
    """
    with open(path, 'r') as f:
        line = f.read().strip()  # e.g. "[-0.71209425 0.26310887 -0.26951163 -0.58270747 0.10728925]"
        line = line.replace('[', '').replace(']', '')
        weights = np.array([float(x) for x in line.split()])
    return weights


def compute_learned_reward(traj_step, weights):
    """
    Given a single transition (obs, act),
    compute the per-step reward using your learned reward function.

    We wrap the single step as a pseudo-trajectory of length 1
    to reuse your existing 'feature_function' in feature_func.py.
    """
    pseudo_traj = [traj_step]  # e.g. [(state, action)]
    phi = feature_function(pseudo_traj)   # shape = (num_features,)
    return float(phi @ weights)


################################################################################
# 2) Minimal Replay Buffer
################################################################################
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.storage = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            # Overwrite if full
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=64):
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idx]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, s_next, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(d)

        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32).reshape(-1, 1))


################################################################################
# 3) DDPG Networks
################################################################################
def mlp(input_dim, output_dim, hidden_dims=[256,256], activation=nn.ReLU):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = mlp(state_dim, action_dim, [256,256], activation=nn.ReLU)
        self.max_action = max_action
    
    def forward(self, x):
        # DDPG typically uses tanh on the last layer
        return self.max_action * torch.tanh(self.net(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, [256,256], activation=nn.ReLU)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


################################################################################
# 4) DDPG Agent
################################################################################
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state, noise_scale=0.1):
        """
        Selects action for a single state (np array).
        Adds some exploration noise if noise_scale > 0.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_t).detach().cpu().numpy()[0]
        # Add exploration noise
        action += noise_scale * np.random.randn(len(action))
        # Clip if needed
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update(self, replay_buffer, batch_size=64):
        # Sample from replay
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_Q = self.critic_target(next_states_t, next_actions)
            target_Q = rewards_t + self.gamma * (1 - dones_t) * target_Q

        current_Q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor loss (maximize Q => minimize -Q)
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # (Optional) If you want to print losses:
        # print(f"Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")


################################################################################
# 5) Evaluation
################################################################################
def evaluate_policy(agent, env, num_rollouts=10):
    """
    Rolls out the current policy for 'num_rollouts' episodes
    and returns the average success rate.
    """
    successes = 0
    for _ in range(num_rollouts):
        obs_dict = env.reset()  # { "observation", "achieved_goal", "desired_goal" }
        obs = np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]])
        done = False
        steps = 0
        info = {}
        while not done and steps < 150:
            action = agent.select_action(obs, noise_scale=0.0)  # no noise at test
            next_obs_dict, _, done, info = env.step(action)
            obs = np.concatenate([next_obs_dict["observation"], next_obs_dict["achieved_goal"]])
            steps += 1

        # If your environment sets "info['is_success']" or some other success indicator, check it:
        if "is_success" in info and info["is_success"]:
            successes += 1

    return successes / num_rollouts


################################################################################
# 6) Main Training Loop + Plot
################################################################################
if __name__ == "__main__":

    ############################
    # (a) Load Learned Weights
    ############################
    w = load_learned_weights("final_feature_weights.csv")
    print("Loaded learned weights:", w)

    ############################
    # (b) Init environment
    ############################
    env = init_env(render=False)
    obs_dict = env.reset()
    # Flatten (observation + achieved_goal)
    obs_dim = len(obs_dict["observation"]) + len(obs_dict["achieved_goal"])
    act_dim = env.action_space.shape[0]   # should be 4
    max_action = 1.0  # Because ActionNormalizer scales your actions to [-1, 1]

    ############################
    # (c) Create Agent
    ############################
    agent = DDPGAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        max_action=max_action,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
    )

    ############################
    # (d) Replay Buffer & Demonstrations
    ############################
    replay_buffer = ReplayBuffer()

    # Load the 20 expert demonstrations
    demos, starting_states = generate_trajectories_from_files()
    print(f"Loaded {len(demos)} demonstration trajectories.")

    # Fill replay buffer with demo transitions (using learned reward)
    for traj in demos:
        for i in range(len(traj) - 1):
            state_vec = np.array(traj[i][0], dtype=np.float32)
            action_vec = np.array(traj[i][1], dtype=np.float32)
            next_state_vec = np.array(traj[i+1][0], dtype=np.float32)

            # Mark done if it's the final step in the trajectory
            done = (i == (len(traj) - 2))

            # Convert to single-step input for compute_learned_reward
            s_a = (state_vec, action_vec)
            r = compute_learned_reward(s_a, w)

            replay_buffer.add(state_vec, action_vec, r, next_state_vec, done)

    print(f"Replay buffer size after adding demos: {len(replay_buffer.storage)}")


    ############################
    # (f) Online Training
    ############################
    max_env_steps = 50000     # *** up to 500k steps ***
    episode_length = 150
    steps_so_far = 0
    episode_count = 0
    updates_per_episode = 50

    # We'll evaluate success & save policy every 1k steps
    evaluate_interval = 1000

    # We'll store (env_steps, success_rate) to plot later
    success_rates = []
    episode_rewards = []

    while steps_so_far < max_env_steps:
        episode_count += 1

        # Reset environment & prepare for rollout
        obs_dict = env.reset()
        obs = np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]])
        done = False
        ep_steps = 0
        ep_reward = 0.0  # sum of learned rewards this episode

        
        while not done and ep_steps < episode_length:
            # 1) Select action
            action = agent.select_action(obs, noise_scale=0.1)

            # 2) Step environment
            next_obs_dict, _, env_done, info = env.step(action)
            next_obs = np.concatenate([next_obs_dict["observation"], next_obs_dict["achieved_goal"]])

            # 3) Compute learned reward
            s_a = (obs, action)
            learned_r = compute_learned_reward(s_a, w)

            # 4) Add to replay
            done = env_done
            replay_buffer.add(obs, action, learned_r, next_obs, done)

            # Accumulate reward for debugging
            ep_reward += learned_r

            # Move on
            obs = next_obs
            ep_steps += 1
            steps_so_far += 1

        episode_rewards.append(ep_reward)
        # Print episode info
        print(f"[Episode {episode_count}] steps_in_ep: {ep_steps}, total_env_steps: {steps_so_far}, "
              f"episode_learned_reward: {ep_reward:.2f}")

        # 5) Update policy after the episode
        for _ in range(updates_per_episode):
            agent.update(replay_buffer, batch_size=64)

        # 6) Evaluate & Save every 1k steps
        if steps_so_far % 1000 == 0:
            print(steps_so_far)
            # Evaluate policy on 10 test runs
            success_rate = evaluate_policy(agent, env, num_rollouts=10)
            success_rates.append((steps_so_far, success_rate))
            print(f"*** EVAL *** Env Steps: {steps_so_far} | Success Rate: {success_rate:.2f}")

            # Save model checkpoints
            torch.save(agent.actor.state_dict(), f"checkpoint_actor_{steps_so_far}.pth")
            torch.save(agent.critic.state_dict(), f"checkpoint_critic_{steps_so_far}.pth")

    print("Training complete!")
    if success_rates:
        print("Final success rate at last eval:", success_rates[-1])
    else:
        print("No evaluations were performed.")

    ############################
    # (g) Plot Learning Curve
    ############################
    if success_rates:
        x_vals = [t[0] for t in success_rates]  # environment steps
        y_vals = [t[1] for t in success_rates]  # success rates
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Success Rate (out of 10 test runs)")
        plt.title("Policy Learning Curve")
        plt.grid(True)
        plt.show()
    else:
        print("No success rate data to plot!")

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, label='Episode Learned Reward')
    plt.xlabel('Episode')
    plt.ylabel('Learned Reward')
    plt.title('Per-Episode Learned Reward Over Training')
    plt.grid(True)
    plt.legend()
    plt.show()