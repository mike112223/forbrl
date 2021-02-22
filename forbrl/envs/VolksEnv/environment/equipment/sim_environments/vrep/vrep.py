
from . import vrep_api
from .vrep_const import OPERATION_MODES
from ..registry import SIM_ENVIRONMENTS


@ SIM_ENVIRONMENTS.register_module
class Vrep(object):
    """A base class for a camera"""

    def __init__(self, address='127.0.0.1', port=19997, mode='blocking'):

        self.client_id = vrep_api.simxStart(address, port, True, True, 5000, 5)
        print(self.client_id)
        self.mode = OPERATION_MODES[mode]
        if self.client_id == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.start()


    def start(self):
        vrep_api.simxStartSimulation(
            self.client_id, self.mode)

    def stop(self):
        vrep_api.simxStopSimulation(
            self.client_id, self.mode)

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()
