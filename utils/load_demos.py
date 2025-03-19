import numpy as np
import gym
import panda_gym
from time import sleep

import os
import sys

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import PnPNewRobotEnv

from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper



def prepare_demo_pool(demo_path):
    '''

    :param demo_path:
    :return: a list of dictionary, where each dictionary itself is one episode of demo
    '''

    state_traj = np.genfromtxt(demo_path + 'state_traj.csv', delimiter=' ')
    action_traj = np.genfromtxt(demo_path + 'action_traj.csv', delimiter=' ')
    next_state_traj = np.genfromtxt(demo_path + 'next_state_traj.csv', delimiter=' ')
    reward_traj = np.genfromtxt(demo_path + 'reward_traj.csv', delimiter=' ')
    done_traj = np.genfromtxt(demo_path + 'done_traj.csv', delimiter=' ')

    reward_traj = np.reshape(reward_traj, (-1, 1))
    done_traj = np.reshape(done_traj, (-1, 1))

    print("reward traj shape: {}".format(reward_traj.shape))
    print("done traj shape: {}".format(done_traj.shape))

    # first go through the loaded trajectory to get the index of episode starting signs
    starting_ids = []
    for i in range(state_traj.shape[0]):
        if state_traj[i][0] == np.inf:
            starting_ids.append(i)
    total_demo_num = len(starting_ids)

    demos = []
    for i in range(total_demo_num):
        if i < total_demo_num - 1:
            start_step_id = starting_ids[i]
            end_step_id = starting_ids[i + 1]
        else:
            start_step_id = starting_ids[i]
            end_step_id = state_traj.shape[0]

        states = state_traj[(start_step_id + 1):end_step_id, :]
        actions = action_traj[(start_step_id + 1):end_step_id, :]
        next_states = next_state_traj[(start_step_id + 1):end_step_id, :]
        rewards = reward_traj[(start_step_id + 1):end_step_id, :]
        dones = done_traj[(start_step_id + 1):end_step_id, :]
        demo = {'state_trajectory': states.copy(),
                'action_trajectory': actions.copy(),
                'next_state_trajectory': next_states.copy(),
                'reward_trajectory': rewards.copy(),
                'done_trajectory': dones.copy()}
        demos.append(demo)

    return demos



def main():
    # create the task environment
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    demo_path = PARENT_DIR + '/demo_data/PickAndPlace/'
    demos = prepare_demo_pool(demo_path)
    total_demo_num = len(demos)
    for i in range(total_demo_num):
        state_traj = demos[i]['state_trajectory']
        action_traj = demos[i]['action_trajectory']

        step = 0
        done = False

        env.reset(whether_random=False, object_pos=state_traj[step][7:10])
        while not done:
            action = action_traj[step]
            obs, reward, done, info = env.step(action)

            env.render(mode='human')
            sleep(0.01)

            step += 1
            if step >= state_traj.shape[0] - 1:
                step = state_traj.shape[0] - 1

        print("[Demo {}]: whether success: {}".format(i + 1, info['is_success']))
        print("***********************************")

    print('All finished')



if __name__ == '__main__':
    main()
