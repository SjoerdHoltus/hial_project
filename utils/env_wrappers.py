import gym
import numpy as np


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, info


class ResetWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset()

        whether_random = kwargs.get('whether_random', True)
        with self.env.sim.no_rendering():
            if whether_random:
                self.env.robot.reset()
                self.env.task.reset()
            else:
                self.env.robot.reset()
                self.env.task.sim.set_base_pose("target", self.env.task.goal, [0, 0, 0, 1])

                object_pos = kwargs.get('object_pos')  # 1d array of the form (x, y, z)
                self.env.task.sim.set_base_pose("object", object_pos, [0, 0, 0.7071, 0.7071])

        # get obs
        robot_obs = self.env.robot.get_obs()  # robot state
        task_obs = self.env.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.env.task.get_achieved_goal()

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
               }

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info['is_success']:
            done = True

        return obs, reward, done, info



def reconstruct_state(state):
    obs = state['observation'] # 1d np array and we exclude the last time feature
    goal = state['desired_goal'] # 1d np array in the form of (x, y, z)
    state = np.concatenate((obs, goal))

    return state