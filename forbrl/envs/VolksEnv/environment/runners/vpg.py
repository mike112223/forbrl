
import time
import logging

import numpy as np

from .registry import RUNNERS
from ..utils import get_heightmap


@RUNNERS.register_module
class VPG(object):
    """
    A calibrator that does eye in hand calibration for a system that contains
    a robotic arm, a camera and a fiducial marker that serves as a
    calibration board.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, sim=None, arm=None, camera=None,
                 gripper=None, obj=None, work_dir=None,
                 workspace=None, resolution=None,
                 num_rotations=None,
                 grasp_reward=1., push_reward=0.5,
                 grasp_loc_margin=0.15, push_margin=0.1,
                 push_length=0.1,
                 pixel_thresh=300, depth_thresh=[0.01, 0.3],
                 no_change_thresh=10, empty_threshold=300):

        self.sim = sim
        self.arm = arm
        self.camera = camera
        self.gripper = gripper
        self.obj = obj
        self.work_dir = work_dir

        self.workspace = workspace
        self.resolution = resolution
        self.num_rotations = num_rotations

        self.grasp_reward = grasp_reward
        self.push_reward = push_reward
        self.grasp_loc_margin = grasp_loc_margin
        self.push_margin = push_margin
        self.push_length = push_length

        self.pixel_thresh = pixel_thresh
        self.depth_low_thresh, self.depth_high_thresh = depth_thresh
        self.no_change_thresh = no_change_thresh
        self.empty_threshold = empty_threshold

        # refresh after calling get-state()
        self.depth_heightmap = None
        self.no_change = [0, 0]

        print(self)

    def connect(self):
        self.arm.connect(self.sim.client_id)
        self.camera.connect(self.sim.client_id)
        self.gripper.connect(self.sim.client_id)
        self.obj.connect(self.sim.client_id)

    def close(self):
        self.arm.stop()
        self.camera.stop()
        self.gripper.stop()
        self.obj.stop()

    def new_episode(self):
        self.close()
        self.sim.stop()
        time.sleep(2)
        self.sim.start()
        self.connect()
        self.obj.add_objs()

    def get_state(self):
        color_img, depth_img = self.camera.capture()
        return self._get_heightmap(color_img, depth_img)

    def make_action(self, action):
        act = action['action']
        idx = action['best_idx']

        ori = np.deg2rad(idx[0] / self.num_rotations * 360.0)
        height = self.depth_heightmap[idx[1], idx[2]]

        y, x = idx[1:] * self.resolution
        pos = np.array([x, y, height]) + self.workspace[:, 0]

        grasp_res = False
        if act == 'grasp':
            grasp_res = self._grasp(pos, ori)
            changed = self._check_changed(grasp_res)
            reward = grasp_res * self.grasp_reward
        else:
            push_res = self._push(pos, ori)
            changed = self._check_changed()
            reward = push_res * self.push_reward

        if changed:
            if act == 'push':
                self.no_change[0] = 0
            elif act == 'grasp':
                self.no_change[1] = 0
        else:
            if act == 'push':
                self.no_change[0] += 1
            elif act == 'grasp':
                self.no_change[1] += 1

        return [reward, changed, grasp_res]

    def is_episode_finished(self):

        assert (self.depth_heightmap is not None), \
            'Should set depth_heightmap first!'

        stuff_count = np.sum(self.depth_heightmap > self.depth_low_thresh)
        done = stuff_count < self.empty_threshold or \
            (np.sum(self.no_change) > self.no_change_thresh)

        if done:
            self.no_change = [0, 0]

        return done

    def _get_heightmap(self, color_img, depth_img):
        color_heightmap, depth_heightmap = get_heightmap(
            color_img, depth_img,
            self.camera.intrinsics, self.camera.extrinsics,
            self.workspace, self.resolution)
        depth_heightmap[np.isnan(depth_heightmap)] = 0
        self.depth_heightmap = depth_heightmap

        return np.concatenate(
            [color_heightmap,
             depth_heightmap[:, :, None],
             depth_heightmap[:, :, None],
             depth_heightmap[:, :, None]], axis=2)

    def _move_to(self, pos, ori):
        pvector = np.append(pos, ori)
        self.arm.movel(pvector)

    def _open_gripper(self,):
        self.gripper.open()

    def _close_gripper(self):
        return self.gripper.close()

    def _check_gripper(self):
        pass

    def _check_changed(self, grasp_success=False):
        prev_depth_heightmap = self.depth_heightmap
        _ = self.get_state()
        depth_heightmap = self.depth_heightmap

        depth_diff = abs(depth_heightmap - prev_depth_heightmap)

        change_value = np.sum(
            (depth_diff > self.depth_low_thresh) *
            (depth_diff < self.depth_high_thresh))

        print('change_value: ', change_value)
        changed = change_value > self.pixel_thresh or grasp_success

        return changed

    def _grasp(self, pos, ori):

        # Compute tool ori from heightmap rotation angle
        ortho_ori = ori.copy()
        ortho_ori = (ortho_ori % np.pi) - np.pi / 2

        # Avoid collision with floor
        pos[2] = max(pos[2] - 0.04, self.workspace[2][0] + 0.02)

        # Move gripper to location above grasp target
        pos_above_target = pos.copy()
        pos_above_target[2] += self.grasp_loc_margin

        self._move_to(pos_above_target, ortho_ori)
        self._open_gripper()
        self._move_to(pos, ortho_ori)
        self._close_gripper()

        # Move gripper to location above grasp target
        self._move_to(pos_above_target, ortho_ori)

        # Check if grasp is successful
        closed = self._close_gripper()

        # Move the grasped object elsewhere
        if not closed:
            obj_pos_zs = self.obj.get_poss()[:, 2]
            grasped_obj_ind = np.argmax(obj_pos_zs)
            grasped_obj_handle = self.obj.object_handles[grasped_obj_ind]
            self.obj.remove_obj(grasped_obj_handle)

        return not closed

    def _push(self, pos, ori):

        # Compute tool orientation from heightmap rotation angle
        ortho_ori = ori.copy()
        ortho_ori = (ortho_ori % np.pi) - np.pi / 2

        # Adjust pushing point to be on tip of finger
        pos = pos + self.gripper.tcp

        # Compute pushing direction
        push_ori = [1.0, 0.0]
        push_dir = np.array(
            [push_ori[0] * np.cos(ori) - push_ori[1] * np.sin(ori),
             push_ori[0] * np.sin(ori) + push_ori[1] * np.cos(ori)])

        # Move gripper to location above pushing point
        pos_above_target = pos.copy()
        pos_above_target[2] += self.push_margin

        # Compute gripper pos and linear movement increments
        self._move_to(pos_above_target, ortho_ori)
        self._close_gripper()
        self._move_to(pos, ortho_ori)

        # Compute target location (push to the right)
        target_pos = np.minimum(
            np.maximum((pos[:2] + push_dir * self.push_length),
                       self.workspace[:2, 0]),
            self.workspace[:2, 1])

        # Move in pushing direction towards target location
        self._move_to(np.append(target_pos, pos[2]), ortho_ori)

        # Move gripper to location above grasp target
        self._move_to(np.append(target_pos, pos_above_target[2]), ortho_ori)

        return True
