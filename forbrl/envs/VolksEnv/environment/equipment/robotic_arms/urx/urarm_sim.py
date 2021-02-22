
import logging

import numpy as np

from .urarm import URArm
from ..registry import ROBOTIC_ARMS
from ...sim_environments.vrep import vrep_api, OPERATION_MODES


@ROBOTIC_ARMS.register_module
class URArmSim(URArm):
    """
    Generic Python interface to an simulated industrial UR robotic arm.
    """
    logger = logging.getLogger(__name__)

    def __init__(self,
                 handle_name='UR5_target',
                 mode='blocking',
                 ):

        self.handle_name = handle_name
        self.mode = OPERATION_MODES[mode]
        # super().__init__()

    def connect(self, client_id):
        self.client_id = client_id
        sim_ret, self.arm_handle = vrep_api.simxGetObjectHandle(
            self.client_id, self.handle_name, self.mode)

    def stop(self):
        self.arm_handle = None

    def get_pos(self):
        sim_ret, arm_pos = vrep_api.simxGetObjectPosition(
            self.client_id, self.arm_handle, -1, self.mode)
        return np.array(arm_pos)

    def set_pos(self, pos):
        sim_ret = vrep_api.simxSetObjectPosition(
            self.client_id, self.arm_handle, -1,
            pos, self.mode)

    def get_orientation(self):
        sim_ret, arm_ori = vrep_api.simxGetObjectOrientation(
            self.client_id, self.arm_handle, -1, self.mode)
        return np.array(arm_ori)

    def set_orientation(self, ori):
        sim_ret = vrep_api.simxSetObjectOrientation(
            self.client_id, self.arm_handle, -1,
            ori, self.mode)

    def movel(self, pvector, lstep=0.02, rstep=0.3):
        tpos = pvector[:3]
        tori = pvector[3]

        pos = self.get_pos()
        ori = self.get_orientation()[1]

        move_direct = tpos - pos
        move_length = np.linalg.norm(move_direct)
        move_step = lstep * move_direct / move_length
        num_move_steps = int(np.floor(move_length / lstep))

        rotation_step = rstep if (tori - ori > 0) else -rstep
        num_rotation_steps = int(np.floor((tori - ori) / rotation_step))

        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            self.set_pos(pos + move_step * min(num_move_steps, step_iter))
            self.set_orientation(
                [np.pi / 2,
                 ori + rotation_step * min(num_rotation_steps, step_iter),
                 np.pi / 2])

        self.set_pos(tpos)
        self.set_orientation([np.pi / 2, tori, np.pi / 2])
