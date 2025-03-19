from typing import Optional

import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

from collections import namedtuple
import math

import os


CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)


class UR5(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="ur5",
            file_name= PARENT_DIR + "/envs/tasks/ur5/urdf/ur5_robotiq_85.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([1, 2, 3, 4, 5, 6]),
            joint_forces=np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
        )

        self.fingers_indices = np.array([12, 17])
        self.neutral_joint_values = np.array([-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636])
        self.ee_link = 7

        self.load()

        """self.set_joint_neutral()

        robot_id = self.sim._bodies_idx['kinova']
        # attach the gripper to the robot
        gripper_id = self.sim.physics_client.loadURDF("/home/ullrich/catkin_ws/src/active_panda2/envs/tasks/robotiq_arg85_description/robots/robotiq_arg85_description.URDF")
        # gripper_id = self.sim.physics_client.loadSDF("gripper/wsg50_one_motor_gripper.sdf")[0]

        self.sim._bodies_idx['gripper'] = gripper_id
        gripper_position = self.get_ee_position()
        print("gripper position: {}".format(gripper_position))
        gripper_orientation = self.sim.physics_client.getQuaternionFromEuler([0, np.pi, np.pi/2.0])  # Adjust as needed

        # Position the gripper correctly relative to the end-effector link
        self.sim.physics_client.resetBasePositionAndOrientation(gripper_id, gripper_position, gripper_orientation)

        # Attach the gripper to the end-effector link
        self.sim.physics_client.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=gripper_id,
            childLinkIndex=-1,  # -1 means base link of the gripper
            jointType=self.sim.physics_client.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.035],  # Adjust relative to the end-effector
            childFramePosition=[0, 0, 0]
        )

        gripper_joint_indices = np.array([0, 2, 3, 4, 5, 6])
        gripper_angles = np.array([0.0, 0.5, 0.0, -0.5, -0.0, 0.0])
        # for joint_index in gripper_joint_indices:
        #     self.sim.physics_client.setJointMotorControl2(
        #         bodyUniqueId=gripper_id,
        #         jointIndex=joint_index,
        #         controlMode=self.sim.physics_client.POSITION_CONTROL,
        #         targetPosition=0.4,
        #         force=50  # Maximum force applied to the joint (tune this value if needed)
        #     )

        self.sim.set_joint_angles('gripper', joints=gripper_joint_indices, angles=gripper_angles)"""

        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)


    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def __init_robot__(self):
        self.eef_id = 8
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-0.10669020495539147, -1.3684361412151338, 1.6524135640831839, -1.854771397805994, -1.570735148190762, 3.0349002737915347]
        self.id = self.sim._bodies_idx['ur5']
        self.gripper_range = [0, 0.085]

    def __parse_joint_info__(self):
        numJoints = self.sim.physics_client.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self.sim.physics_client.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self.sim.physics_client.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                self.sim.physics_client.setJointMotorControl2(self.id, jointID, self.sim.physics_client.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.sim.physics_client.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=self.sim.physics_client.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            self.sim.physics_client.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])
        self.finger_width = self.gripper_range[1]

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])
        self.finger_width = self.gripper_range[0]

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        self.sim.physics_client.setJointMotorControl2(self.id, self.mimic_parent_id, self.sim.physics_client.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)


    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)


        for i, joint_id in enumerate(self.arm_controllable_joints):
            self.sim.physics_client.setJointMotorControl2(self.id, joint_id, self.sim.physics_client.POSITION_CONTROL, target_arm_angles[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

        if self.block_gripper:
            pass
        else:
            fingers_ctrl = action[-1]

            if fingers_ctrl > 0:
                self.close_gripper()
            else:
                self.open_gripper()

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        # target_arm_angles = self.inverse_kinematics(
        #     link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        # )

        joint_poses = self.sim.physics_client.calculateInverseKinematics(self.id, self.eef_id, target_ee_position, np.array([1.0, 0.0, 0.0, 0.0]),
                                                   self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges,
                                                   self.arm_rest_poses,
                                                   maxNumIterations=20)

        return joint_poses

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self) -> None:
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            self.sim.physics_client.resetJointState(self.id, joint_id, rest_pose)

    def reset_gripper(self):
        if self.block_gripper:
            self.close_gripper()
        else:
            self.open_gripper()


    def set_joint_neutral(self) -> None:
            """Set the robot to its neutral pose."""
            self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:

        return self.finger_width

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
