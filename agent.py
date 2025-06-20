"""
Evaluation Agent.
"""
import cv2
import time
import numpy as np
from PIL import Image
from device.robot import FlexivRobot, FlexivGripper
from device.camera import CameraD400
from utils.transformation import xyz_rot_transform


BLOCK_TIME = 0.1 - 0.03


class EvalAgent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    rot90 = False

    def __init__(
        self,
        robot_ip='192.168.2.100',
        camera_serials=['135122075425'],
        **kwargs
    ):

        print("Init robot, gripper, and camera.")
        self.robot = FlexivRobot(robot_ip)
        self.robot.send_tcp_pose(self.ready_pose, slow=True)
        time.sleep(2)
        self.gripper = FlexivGripper(self.robot)

        self.camera = [CameraD400(camera_serial, fps=30)
                       for camera_serial in camera_serials]
        self.camera_serials = camera_serials
        print('Init done')

    @property
    def intrinsics(self):
        raise DeprecationWarning()

    @property
    def ready_pose(self):
        return np.array([0.6, 0, 0.2, 0, 0.5**0.5, -0.5**0.5, 0])

    @property
    def ready_rot_6d(self):
        return xyz_rot_transform(
            self.ready_pose,
            from_rep='quaternion',
            to_rep="rotation_6d",
        )[3:]

    def _get_observation(self, i):
        colors, depths = self.camera[i].get_data()
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        return colors, depths / 1000.

    def get_observation(self):
        return [self._get_observation(i) for i in range(len(self.camera))]

    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        if self.rot90:
            tcp_pose = xyz_rot_transform(
                tcp_pose,
                from_rep="quaternion",
                to_rep="matrix",
            )
            tcp_pose[:3, :3] = tcp_pose[:3, :3] @ np.array([[0, -1, 0],
                                                            [1, 0, 0],
                                                            [0, 0, 1]])
            tcp_pose = xyz_rot_transform(
                tcp_pose,
                from_rep='matrix',
                to_rep="quaternion",
            )
        return tcp_pose

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False, slow=False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="matrix",
            from_convention=rotation_rep_convention
        )
        if self.rot90:
            tcp_pose[:3, :3] = tcp_pose[:3, :3] @ np.array([[0, 1, 0],
                                                            [-1, 0, 0],
                                                            [0, 0, 1]])
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep='matrix',
            to_rep="quaternion",
            from_convention=rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose, slow=slow)
        if blocking:
            time.sleep(BLOCK_TIME)

    def set_joint_pose(self, joint_pose, blocking=False):
        self.robot.send_joint_pose(joint_pose)
        if blocking:
            time.sleep(BLOCK_TIME)

    def set_gripper_width(self, width, force=30, blocking=False):
        # width = 1000 if width > 500 else 0
        print(width)
        self.gripper.move(width, force=force)
        if blocking:
            time.sleep(2)

    def stop(self):
        self.robot.stop()
