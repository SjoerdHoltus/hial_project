import numpy as np
import aprel
import sys
import os
import csv

from feature_func import feature_function
from trajectory_utils import transform_traj, generate_trajectories_from_files, generate_clip, generate_trajectory, record_trajectory
from init_env import init_env

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

ENV_NAME = 'PickingBananas'

# for i in range(10):
#     env = init_env(render=False)

#     trajectory_dir = 'random_trajectories'
#     trajectory_name = f'trajectory_{i+100}'

#     print(f"Generating trajectory {i+100} of 10")
#     trajectory = generate_trajectory(env, max_episode_length=100,
#                                      save_dir=trajectory_dir,
#                                      save_name=trajectory_name,
#                                      seed=int(trajectory_name.split('_')[-1]))
    
#     env = init_env(render=False)
#     frames = record_trajectory(env, trajectory)
#     clip_name = f'{trajectory_dir}/{trajectory_name}/trajectory_{i+100}.mp4'
#     generate_clip(frames, clip_name, fps=15)

file_path_random_traj = './random_trajectories/trajectory_'
actions = 'action_traj.csv'
states = 'state_traj.csv'
learned_weights  = 'final_feature_weights.csv'

env = init_env(render=False)
env = aprel.Environment(env, feature_function)

# Creating a trajectory set in the correct format
trajectory_set = []

# First adding the 10 random trajectories
for i in range(10):
    actions_file = f'{file_path_random_traj}{i+1}/{actions}'
    states_file = f'{file_path_random_traj}{i+1}/{states}'
    new_traj = transform_traj(actions_file,states_file)
    clip = f'{file_path_random_traj}{i+1}/trajectory_{i+1}.mp4'
    traj = aprel.Trajectory(env, new_traj, clip)
    trajectory_set.append(traj)

# Trajectories of the experts
trajectories, starting_states = generate_trajectories_from_files()

# Add them to the trajectory set
for i in range(len(trajectories)):
    traj = aprel.Trajectory(env, trajectories[i], f'./expert_trajectories/expert_trajectory_{i}.mp4')
    trajectory_set.append(traj)

trajectory_set = aprel.TrajectorySet(trajectory_set)

features_dim = len(trajectory_set[0].features)

# Initialize the optimizer
query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

true_user = aprel.HumanUser(delay=0.5)

params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
user_model = aprel.SoftmaxUser(params)
belief = aprel.SamplingBasedBelief(user_model, [], params)
print('Estimated user parameters: ' + str(belief.mean))
                                       
query = aprel.PreferenceQuery(trajectory_set[:2])
for query_no in range(10):
    queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
    print('Objective Value: ' + str(objective_values[0]))
    
    responses = true_user.respond(queries[0])
    belief.update(aprel.Preference(queries[0], responses[0]))
    print('Estimated user parameters: ' + str(belief.mean))
    
with open(learned_weights, 'w', newline='') as file:
    file.write(str(belief.mean['weights']))