import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from feature_func import feature_function
from trajectory_utils import generate_trajectories_from_files
from init_env import init_env


def load_weights(filepath="final_feature_weights.csv"):
    """
    Loads the learned weights from a CSV file.
    Returns a 1D NumPy array of the weights.
    """
    with open(filepath, 'r') as f:
        line = f.read().strip()
        line = line.replace('[', '').replace(']', '')
        weights = np.array([float(x) for x in line.split()])
    return weights

def calc_reward(traj_step, weights):
    """
    Calculates the learned reward for a given trajectory using the learned weights.
    """
    single_traj = [traj_step]
    feat = feature_function(single_traj)
    return float(feat @ weights)

def make_mlp(in_dim, out_dim, hidden=[256,256], act=nn.ReLU):
    """
    A simple MLP with specified dimensions and activation function.
    """
    layers = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class actNet(nn.Module):
    """
    A simple actor network that maps states to actions.
    """
    def __init__(self, s_dim, a_dim, max_a):
        super().__init__()
        self.net = make_mlp(s_dim, a_dim, [256,256], act=nn.ReLU)
        self.max_a = max_a
    
    def forward(self, x):
        return self.max_a * torch.tanh(self.net(x))

class critNet(nn.Module):
    """
    A simple critic network that maps states and actions to a scalar value.
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = make_mlp(s_dim + a_dim, 1, [256,256], act=nn.ReLU)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class Agent:
    """
    A DDPG agent that uses a replay buffer to store transitions and updates the policy and critic networks.
    """
    def __init__(self, s_dim, a_dim, max_a=1.0, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = actNet(s_dim, a_dim, max_a)
        self.actor_target = actNet(s_dim, a_dim, max_a)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = critNet(s_dim, a_dim)
        self.critic_target = critNet(s_dim, a_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.max_a = max_a

    def select_action(self, state, noise_scale=0.1):
        """
        Selects an action based on the policy and state.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_t).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(len(action))
        action = np.clip(action, -self.max_a, self.max_a)
        return action

    def update(self, buffer, batch_size=64):
        """
        Updates the policy and critic networks using a batch of transitions from the replay buffer.
        """
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_Q = self.critic_target(next_states_t, next_actions)
            target_Q = rewards_t + self.gamma * (1 - dones_t) * target_Q

        current_Q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuf:
    """
    A replay buffer that stores transitions and samples batches of transitions.
    """
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.data = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        tup = (state, action, reward, next_state, done)
        if len(self.data) < self.max_size:
            self.data.append(tup)
        else:
            self.data[self.ptr] = tup
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=64):
        idx = np.random.randint(0, len(self.data), size=batch_size)
        batch = [self.data[i] for i in idx]
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

def eval_pol(agent, env, num_runs=10):
    """
    Evaluates the policy by running it in the environment for a specified number of runs and calculating the success rate.
    """
    succ = 0
    for _ in range(num_runs):
        o_dict = env.reset()
        o = np.concatenate([o_dict["observation"], o_dict["achieved_goal"]])
        done = False
        steps = 0
        info = {}
        while not done and steps < 150:
            act = agent.select_action(o, noise_scale=0.0)
            next_o_dict, _, done, info = env.step(act)
            o = np.concatenate([next_o_dict["observation"], next_o_dict["achieved_goal"]])
            steps += 1
        if "is_success" in info and info["is_success"]:
            succ += 1
    return succ / num_runs

if __name__ == "__main__":
    """
    Main function that initializes the environment, loads the learned weights, and trains the policy.
    """
    wts = load_weights("final_feature_weights.csv")
    print("Loaded learned weights:", wts)

    env = init_env(render=False)
    o_dict = env.reset()
    obsSize = len(o_dict["observation"]) + len(o_dict["achieved_goal"])
    actSize = env.action_space.shape[0]
    max_a = 1.0

    agent = Agent(
        s_dim=obsSize,
        a_dim=actSize,
        max_a=max_a,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
    )

    replay = ReplayBuf()

    demos, start_states = generate_trajectories_from_files()
    print(f"Loaded {len(demos)} demonstration trajectories.")

    for traj in demos:
        for i in range(len(traj) - 1):
            s_vec = np.array(traj[i][0], dtype=np.float32)
            a_vec = np.array(traj[i][1], dtype=np.float32)
            next_s_vec = np.array(traj[i+1][0], dtype=np.float32)
            done = (i == (len(traj) - 2))
            s_a = (s_vec, a_vec)
            r = calc_reward(s_a, wts)
            replay.add(s_vec, a_vec, r, next_s_vec, done)

    print(f"Replay buffer size after adding demos: {len(replay.data)}")

    max_steps = 50000
    ep_length = 150
    step_count = 0
    ep_count = 0
    updates_per_ep = 50

    eval_interval = 1000

    succ_rates = []
    ep_rewards = []

    while step_count < max_steps:
        ep_count += 1
        o_dict = env.reset()
        o = np.concatenate([o_dict["observation"], o_dict["achieved_goal"]])
        done = False
        ep_steps = 0
        ep_reward = 0.0

        while not done and ep_steps < ep_length:
            act = agent.select_action(o, noise_scale=0.1)
            next_o_dict, _, env_done, info = env.step(act)
            next_o = np.concatenate([next_o_dict["observation"], next_o_dict["achieved_goal"]])
            s_a = (o, act)
            learned_r = calc_reward(s_a, wts)
            done = env_done
            replay.add(o, act, learned_r, next_o, done)
            ep_reward += learned_r
            o = next_o
            ep_steps += 1
            step_count += 1

        ep_rewards.append(ep_reward)
        print(f"[Episode {ep_count}] steps: {ep_steps}, total_steps: {step_count}, ep_reward: {ep_reward:.2f}")

        for _ in range(updates_per_ep):
            agent.update(replay, batch_size=64)

        if step_count % 1000 == 0:
            print(step_count)
            sr = eval_pol(agent, env, num_runs=10)
            succ_rates.append((step_count, sr))
            print(f"*** EVAL *** Steps: {step_count} | Success: {sr:.2f}")
            torch.save(agent.actor.state_dict(), f"checkpoint_actor_{step_count}.pth")
            torch.save(agent.critic.state_dict(), f"checkpoint_critic_{step_count}.pth")

    print("Training complete!")
    if succ_rates:
        print("Final success rate:", succ_rates[-1])
    else:
        print("No evaluations were performed.")

    if succ_rates:
        x_vals = [t[0] for t in succ_rates]
        y_vals = [t[1] for t in succ_rates]
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("Env Steps")
        plt.ylabel("Avg Success (out of 10 runs)")
        plt.title("Learning Curve")
        plt.grid(True)
        plt.show()
    else:
        print("No success data to plot!")

    plt.figure(figsize=(8, 5))
    plt.plot(ep_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
