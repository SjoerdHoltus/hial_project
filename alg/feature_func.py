import numpy as np

DESIRED_GOAL = np.array([ 0.  , -0.2 ,  0.02])

def get_mean(states, index_min, index_max):
    '''
    Returns the mean of a part of an observation over all states in a trajectory
    '''
    ls = [state[index_min:index_max] for state in states]
    return np.average(ls)

def get_euclidean_distance(states, desired_goal):
    '''
    Returns the mean distance over all states in a trajectory
    '''
    achieved_goal = np.array([state[-3:] for state in states])
    
    distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
    return distances.mean()

def get_min(states, index):
    '''
    Returns the minimum gripper width over all states
    '''
    ls = [state[index] for state in states]
    return np.min(ls)

def feature_function(traj):
    '''
    This function gives the following features of a trajectory: distance, 
    velocity, angle, rotation and finger width.
    '''
    states = np.array([state_action_pair[0] for state_action_pair in traj])
    hand_position = get_mean(states, 0, 3)
    banana_distance = get_euclidean_distance(states, DESIRED_GOAL)
    banana_rotation = get_mean(states, 10, 13)
    banana_velocity = get_mean(states, 13, 16)
    min_finger_width = get_min(states, 6)

    return np.array([hand_position, banana_distance, banana_rotation, banana_velocity, min_finger_width], dtype=np.float32)
