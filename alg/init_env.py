import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper

def init_env(render=False):
    env = PnPNewRobotEnv(render=render)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)
    return env

