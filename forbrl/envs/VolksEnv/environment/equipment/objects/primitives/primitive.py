
import os
import time

import numpy as np

from ..registry import OBJECTS
from ...sim_environments.vrep import vrep_api, vrep_const, OPERATION_MODES


@OBJECTS.register_module
class Primitive(object):

    def __init__(self,
                 sev_name='remoteApiCommandServer',
                 func_name='importShape',
                 num_obj=10,
                 obj_mesh_dir=None,
                 workspace=None,
                 drop_height=0.15,
                 drop_offset=0.1,
                 color_space=np.array([[255., 0., 0.]]),
                 mode='blocking'):
        self.sev_name = sev_name
        self.func_name = func_name
        self.num_obj = num_obj
        self.obj_mesh_dir = obj_mesh_dir
        self.workspace = workspace
        self.drop_height = drop_height
        self.drop_offset = drop_offset
        self.color_space = color_space
        self.mode = OPERATION_MODES[mode]

        # Read files in object mesh directory
        self.mesh_list = os.listdir(self.obj_mesh_dir)
        print(self.mesh_list)

        # Randomly choose objects to add to scene
        self.obj_mesh_idxs = np.random.randint(
            0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[
            np.array(range(self.num_obj)) % 10, :]

    def connect(self, client_id):
        self.client_id = client_id

    def stop(self):
        self.client_id = None

    def get_pos(self, obj_handle):
        sim_ret, obj_pos = vrep_api.simxGetObjectPosition(
            self.client_id, obj_handle, -1, self.mode)
        return np.array(obj_pos)

    def get_poss(self):
        obj_poss = []
        for obj_handle in self.object_handles:
            sim_ret, obj_pos = vrep_api.simxGetObjectPosition(
                self.client_id, obj_handle, -1, self.mode)
            obj_poss.append(obj_pos)
        return np.array(obj_poss)

    def set_pos(self, obj_handle, pos):
        sim_ret = vrep_api.simxSetObjectPosition(
            self.client_id, obj_handle, -1, pos, self.mode)

    def remove_obj(self, obj_handle):
        sim_ret = vrep_api.simxRemoveObject(
            self.client_id, obj_handle, self.mode)

    def add_objs(self):
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        for i, obj_mesh_idx in enumerate(self.obj_mesh_idxs):

            mesh_file = os.path.join(
                self.obj_mesh_dir, self.mesh_list[obj_mesh_idx])
            shape_name = 'shape_%02d' % i

            drop_xy = ((np.diff(self.workspace[:2], axis=1).reshape(-1) -
                        2 * self.drop_offset) * np.random.random_sample(2) +
                       self.workspace[:2, 0] + self.drop_offset)

            obj_pos = list(np.append(drop_xy, self.drop_height))
            obj_ori = list(2 * np.pi * np.random.random_sample(3))
            obj_color = list(self.obj_mesh_color[i])

            (ret_resp, ret_ints, ret_floats, ret_strings,
                ret_buffer) = vrep_api.simxCallScriptFunction(
                self.client_id, self.sev_name, vrep_const.sim_scripttype_childscript,
                self.func_name, [0, 0, 255, 0], obj_pos + obj_ori + obj_color,
                [mesh_file, shape_name], bytearray(), self.mode)

            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                self.stop()

            self.object_handles.append(ret_ints[0])
            time.sleep(2)
