
import time
import logging

from .inspire import InspireGripper
from ..registry import END_EFFECTORS
from ...sim_environments.vrep import vrep_api, OPERATION_MODES


@END_EFFECTORS.register_module
class InspireGripperSim(InspireGripper):
    """A python interface for an Inspire gripper in a simulated space. """
    logger = logging.getLogger(__name__)

    def __init__(self,
                 handle_name='RG2_openCloseJoint',
                 tcp=None,
                 speed=0.5,
                 power=100,
                 openmax=0.0536,
                 openmin=-0.047,
                 time_delay=0.5,
                 mode='blocking',
                 ):

        self.handle_name = handle_name
        self.tcp = tcp
        self.speed = abs(speed)
        self.power = power
        self.openmax = openmax
        self.openmin = openmin
        self.time_delay = time_delay
        self.mode = OPERATION_MODES[mode]
        # super().__init__()

    def connect(self, client_id):
        self.client_id = client_id
        sim_ret, self.gripper_handle = vrep_api.simxGetObjectHandle(
            self.client_id, self.handle_name, self.mode)

    def stop(self):
        self.client_id = None

    def open(self, speed=None, power=None):
        if speed is None:
            speed = self.speed
        if power is None:
            power = self.power

        self.move(speed, power)

    def close(self, speed=None, power=None):
        if speed is None:
            speed = - self.speed
        if power is None:
            power = self.power

        self.move(speed, power)

        sim_ret, j_pos = vrep_api.simxGetJointPosition(
            self.client_id, self.gripper_handle, self.mode)

        closed = j_pos < self.openmin

        return closed


    def move(self, speed, power):

        vrep_api.simxSetJointForce(
            self.client_id, self.gripper_handle,
            power, self.mode)
        vrep_api.simxSetJointTargetVelocity(
            self.client_id, self.gripper_handle,
            speed, self.mode)

        time.sleep(self.time_delay)
