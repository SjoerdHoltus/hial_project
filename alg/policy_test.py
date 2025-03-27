import os
import numpy as np
import torch
from policy_learn import *
from init_env import init_env
from trajectory_utils import *


MAX_STEPS = 150

def load_final_policy(path_to_saved_policy):
    """
    Loads the final policy from a saved model file.
    """
    full_path = os.path.join("./models/", path_to_saved_policy)
    
    s_dim = 22
    a_dim = 4
    max_a = 1.0

    policy_model = actNet(s_dim, a_dim, max_a)
    state_dict = torch.load(full_path, map_location=torch.device('cpu'))
    policy_model.load_state_dict(state_dict)
    policy_model.eval()
    return policy_model

def get_policy_action(state, saved_policy_model):
    """
    Gets the action from the policy model for a given state.
    """
    observation = state["observation"]
    achieved_goal = state["achieved_goal"]
    flattened_state = np.concatenate([observation, achieved_goal])
    
    state_tensor = torch.tensor(flattened_state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action_tensor = saved_policy_model(state_tensor)
    
    action = action_tensor.cpu().numpy().squeeze(0)
    return action

path = 'checkpoint_actor_50100.pth'
video_filename = "final_policy_video.mp4"
agent = load_final_policy(path)
wts = load_weights("final_feature_weights.csv")

# Initialize environment
env = init_env(render=False)

# Initial state
state = env.reset()
o = np.concatenate([state["observation"], state["achieved_goal"]])

replay = ReplayBuf()

done = False
ep_steps = 0
step_count = 0
ep_reward = 0.0

# Create a trajectory that the final policy follows
traj = []

print('starting the trajectory')
while not done and ep_steps < MAX_STEPS:
    action = get_policy_action(state, agent)
    traj.append((state,action))
    next_state, reward, env_done, info = env.step(action)
    done = env_done
    state = next_state
    ep_steps += 1
print('finished trajectory')

traj.append((state, None))

env.close()

print("Trajectory generated with {} steps and total learned reward {:.2f}".format(ep_steps, ep_reward))

env_record = init_env(render=False)
starting_state = traj[0][0]['observation']
frames = record_trajectory(env_record, traj, starting_state=starting_state)
generate_clip(frames, video_filename, fps=30)

print('video saved!')