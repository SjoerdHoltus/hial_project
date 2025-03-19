import numpy as np
import pybullet as p
from time import sleep

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from tasks.ur_robot import UR5

from tasks.pick_and_place import PickAndPlaceTask



''' ************************************************** '''
''' Variation in robot (e.g., ur5 robot) '''
''' ************************************************** '''

class PnPNewRobotEnv(RobotTaskEnv):
    def __init__(self, render=False, reward_type='modified_sparse', control_type="ee"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = UR5(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceTask(sim, reward_type=reward_type)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.task.reward_type == 'modified_sparse':
            if info['is_success']:
                reward = 1000.0
                done = True
            else:
                reward = -1.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action





