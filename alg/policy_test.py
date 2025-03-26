import os
import numpy as np
import torch
from policy_learn import actNet
from init_env import init_env

def load_final_policy(path_to_saved_policy):
    """
    Loads the final policy from a saved model file.
    """
    full_path = os.path.join("models", path_to_saved_policy)
    
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



