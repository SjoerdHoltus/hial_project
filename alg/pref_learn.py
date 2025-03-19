import numpy as np
import aprel
import sys
import os
from trajectory_utils import generate_trajectory, record_trajectory, generate_clip
from init_env import init_env


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper

ENV_NAME = 'PickingBananas'

def get_mean(states,index_min,index_max):
    '''
    Returns the mean of a part of an observation over all states in a trajectory
    '''
    ls = [state['observation'][index_min:index_max] for state in states]
    return np.average(ls)

def get_euclidean_distance(states):
    '''
    Returns the mean distance over all states in a trajectory
    '''
    achieved_goal = [state['achieved_goal'] for state in states]
    desired_goal = [state['desired_goal'] for state in states]
    return np.linalg.norm(np.array(np.array(achieved_goal)-np.array(desired_goal))).mean()

def get_min(states,index):
    '''
    Returns the minimum gripper width over all states
    '''
    ls = [state['observation'][index] for state in states]
    return np.min(ls)

def feature_function(traj):
    '''
    This function gives the following features of a trajectory: distance, 
    velocity, angle, rotation and finger width.
    '''

    states = np.array([state_action_pair[0] for state_action_pair in traj])

    distance = get_euclidean_distance(states)
    velocity = get_mean(states,16,19)
    rotation = get_mean(states,10,13)
    angle = get_mean(states,13,16)
    min_finger_width = get_min(states,6)

    return np.array([distance,velocity,rotation,angle,min_finger_width])

for i in range(10):
    env = init_env(render=False)

    trajectory_dir = 'random_trajectories'
    trajectory_name = f'trajectory_{i+1}'

    print(f"Generating trajectory {i+1} of 10")
    trajectory = generate_trajectory(env, max_episode_length=100,
                                     save_dir=trajectory_dir,
                                     save_name=trajectory_name,
                                     seed=int(trajectory_name.split('_')[-1]))
    
    env = init_env(render=True)
    frames = record_trajectory(env, trajectory)
    clip_name = f'{trajectory_dir}/{trajectory_name}/trajectory_{i+1}.mp4'
    generate_clip(frames, clip_name, fps=15)
