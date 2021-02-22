
import math
import logging

import numpy as np

from .realsense import RealsenseCam
from ..registry import VISION_SENSORS
from ...sim_environments.vrep import vrep_api, vrep_const, OPERATION_MODES
from ....utils import euler2rotm


@VISION_SENSORS.register_module
class RealsenseCamSim(RealsenseCam):
    """
    Python interface for Intel Realsense Family cameras in a simulated
    space.
    """
    logger = logging.getLogger(__name__)

    def __init__(self,
                 handle_name='Vision_sensor_persp',
                 color_res=(640, 480),
                 presp_angle=54.7,
                 clipping=(0.01, 10),
                 depth=1.,
                 mode='blocking'):

        self.handle_name = handle_name
        self.res_x, self.res_y = color_res
        self.presp_angle = presp_angle / 180 * math.pi
        self.near_clip, self.far_clip = clipping
        self.depth = depth
        self.mode = OPERATION_MODES[mode]

        # super().__init__()

    def connect(self, client_id):
        self.client_id = client_id
        self.init_pipeline()

    def init_pipeline(self):
        self.start()
        self.cam_setup()

    def cam_setup(self):
        sim_ret = vrep_api.simxSetObjectIntParameter(
            self.client_id, self.cam_handle,
            vrep_const.sim_visionintparam_resolution_x,
            self.res_x, self.mode)
        sim_ret = vrep_api.simxSetObjectIntParameter(
            self.client_id, self.cam_handle,
            vrep_const.sim_visionintparam_resolution_y,
            self.res_y, self.mode)
        sim_ret = vrep_api.simxSetObjectFloatParameter(
            self.client_id, self.cam_handle,
            vrep_const.sim_visionfloatparam_perspective_angle,
            self.presp_angle, self.mode)
        sim_ret = vrep_api.simxSetObjectFloatParameter(
            self.client_id, self.cam_handle,
            vrep_const.sim_visionfloatparam_near_clipping,
            self.near_clip, self.mode)
        sim_ret = vrep_api.simxSetObjectFloatParameter(
            self.client_id, self.cam_handle,
            vrep_const.sim_visionfloatparam_far_clipping,
            self.far_clip, self.mode)

    def get_property(self):
        # intrinsics
        ppx = self.res_x // 2
        ppy = self.res_y // 2
        fx = fy = (ppx if ppx > ppy else ppy) / math.tan(self.presp_angle / 2)
        self._intrinsics = np.array(
            [[fx, 0, ppx],
             [0, fy, ppy],
             [0, 0, 1]])

        # extrinsics
        cam_trans = np.eye(4, 4)
        cam_rotm = np.eye(4, 4)
        sim_ret, cam_pose = vrep_api.simxGetObjectPosition(
            self.client_id, self.cam_handle, -1, self.mode)
        sim_ret, cam_ori = vrep_api.simxGetObjectOrientation(
            self.client_id, self.cam_handle, -1, self.mode)
        cam_trans[0:3, 3] = np.array(cam_pose)
        cam_ori = [-cam_ori[0], -cam_ori[1], -cam_ori[2]]
        cam_rotm[0:3, 0:3] = np.linalg.inv(euler2rotm(cam_ori))
        self._extrinsics = np.dot(cam_trans, cam_rotm)

        # depth
        self._depth_scale = self.depth

    def capture(self):
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep_api.simxGetVisionSensorImage(
            self.client_id, self.cam_handle, 0, self.mode)
        color_img = np.array(raw_image).reshape(resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep_api.simxGetVisionSensorDepthBuffer(
            self.client_id, self.cam_handle, self.mode)
        depth_img = np.array(depth_buffer).reshape(resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        depth_img = depth_img * (self.far_clip - self.near_clip) + self.near_clip

        return color_img, depth_img

    def start(self):
        sim_ret, self.cam_handle = vrep_api.simxGetObjectHandle(
            self.client_id, self.handle_name, self.mode)
        self.get_property()

    def stop(self):
        self.cam_handle = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def extrinsics(self):
        return self._extrinsics

    @property
    def depth_scale(self):
        return self._depth_scale
