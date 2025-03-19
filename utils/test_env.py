import numpy as np
from time import sleep

import sys
import os

from test_env import max_steps

# Get current directory
CURRENT_DIR = os.getcwd()
# Get parent directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PARENT_DIR + '/envs/')
from task_envs import PnPNewRobotEnv

from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper


def main():
    # create the task environment
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    done = False
    obs = env.reset()

    # if you want to reset your environment to certain initial state (e.g., certain object position of the banana),
    # you can call the reset() function as follows:
    # obs = env.reset(whether_random=False, object_pos = [0, 0.2, 0.02])

    step = 0
    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)

        print("step: {}, success: {}".format(step + 1, info['is_success']))
        print('obs: {}'.format(obs))
        sleep(0.01)

        step += 1

    env.close()

    print("All finished")



if __name__ == '__main__':
    main()