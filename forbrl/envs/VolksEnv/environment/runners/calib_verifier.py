import glob
import logging
import pickle
from collections import defaultdict
from threading import Condition, Thread

import cv2
import math3d as m3d
import numpy as np

from .registry import RUNNERS

AVAILABLE_KEY = ['w', 's', 'a', 'd', 'e', 'r']


@RUNNERS.register_module
class CalibVerifier:
    """
    A verifier that goes through all poses that generated by path sampler
    without stopping while streaming. Used for verify that for all poses,
    the camera can see the target.
    """
    logger = logging.getLogger(__name__)
    key_map = defaultdict(None)
    base_map = defaultdict(int)
    pressed_key = None
    key_direction = {'w': (0, +1), 's': (0, -1),
                     'a': (1, +1), 'd': (1, -1),
                     'r': (2, +1), 'e': (2, -1)}
    point_idx = 0
    base_map[0] = 7
    base_map[1] = 1

    def __init__(self, arm=None, cam=None, calib_board=None, gripper=None,
                 acc=2, vel=2, session_dir=None, step=0.01,
                 verify_by_cam=True, base_pose=(-np.pi, 0, np.pi / 2)):
        self.arm = arm
        self.cam = cam
        self.calib_board = calib_board
        self.gripper = gripper
        self.move_controls = {'acc': acc, 'vel': vel}
        self.session_name = session_dir
        self.step = step
        self.verify_by_cam = verify_by_cam
        self.update_keyboard_mapping()
        tool_rot = m3d.Orientation.new_euler(base_pose, encoding='XYZ')
        self.tool_orientation = tool_rot.rotation_vector
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = (f"A calibration verifier with:\n"
               f"    arm:{self.arm}\n"
               f"    camera:{self.cam}\n"
               f"    calib_board:{self.calib_board}\n"
               f"on session: {self.session_name}")
        return msg

    def load_session(self):
        res = glob.glob(f"{self.session_name}/*.pkl")
        if len(res) > 4 or len(res) == 0:
            msg = f"Unexpected file number in {self.session_name}.\n" \
                  f"There are {len(res)} pickle files:\n    {res}"
            self.logger.error(msg)
        session = {'calib2base': None, 'cam2tool': None}
        for key in session:
            for res_file in res:
                if key in res_file:
                    session[key] = res_file
                    break
        for key, fn in session.items():
            if fn is None:
                raise FileExistsError(f"Didn't find {key} file in"
                                      f" {self.session_name}. Detected files "
                                      f"are: {res}")

            with open(fn, 'rb') as f:
                session[key] = pickle.load(f)
        self.logger.debug(f"Successfully load the session from: "
                          f"{self.session_name}")
        return session

    def reset_arm(self, use_gripper=False):
        session = self.load_session()
        if use_gripper:
            self.arm.set_tcp(self.gripper.tcp)
            self.logger.debug(f"Set TCP from gripper: {self.gripper.tcp}")
            self.gripper.close()
        else:
            self.arm.set_tcp(m3d.Transform(session['cam2tool']).inverse)
            self.logger.debug(f"Set TCP from inversed:\n"
                              f"{session['cam2tool']}")
        self.arm.set_csys(m3d.Transform(session['calib2base']).inverse)
        self.logger.info("Reset arm tcp and coordinates")

    def project_base_points(self):
        session = self.load_session()
        local_points = self.calib_board.local_points
        # calib2base = m3d.Transform(session['calib2base'])
        # base2tool = m3d.Transform(self.arm.get_pose().get_matrix())
        # tool2cam = m3d.Transform(session['cam2tool']).inverse
        # calib2cam = calib2base * base2tool * tool2cam
        # cam2calib = calib2cam.inverse

        base2calib = np.linalg.inv(session['calib2base'])
        tool2base = np.linalg.inv(self.arm.get_pose().get_matrix())
        cam2tool = session['cam2tool']
        cam2calib = cam2tool @ tool2base @ base2calib

        homo_row = np.ones((local_points.shape[0], 1))
        local_points_homo = np.hstack((local_points, homo_row))

        local_points_cam = cam2calib @ local_points_homo.T
        with np.printoptions(precision=3, suppress=True):
            print(np.max(local_points_cam[2, :]))
            print(np.min(local_points_cam[2, :]))
            print(np.max(local_points_cam[2, :]) -
                  np.min(local_points_cam[2, :]))
            print('*', '- -' * 10, '*')

    def update_keyboard_mapping(self):
        for key in AVAILABLE_KEY:
            self.key_map[ord(key)] = key

    def streaming(self, cond):
        self.cam.start()
        try:
            while True:
                color_img, _ = self.cam.capture()
                cv2.circle(color_img, (round(self.cam.intrinsics.ppx),
                                       round(self.cam.intrinsics.ppy)),
                           2, (0, 255, 0), 2)
                cv2.namedWindow("camera feed", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("camera feed", color_img)

                key = cv2.waitKey(1)
                self.project_base_points()
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    self.pressed_key = 'q'
                    break
                if key == -1:
                    continue
                if self.key_map[key]:
                    with cond:
                        self.logger.info(f"Pressed {self.key_map[key]}")
                        self.pressed_key = self.key_map[key]
                        cond.notifyAll()
        finally:
            self.logger.debug("Left camera feed loop")
            self.cam.stop()
            with cond:
                cond.notifyAll()
            self.cam.stop()

    def moving_mode_reset(self, use_gripper, initial_position):
        self.reset_arm(use_gripper=use_gripper)
        tool_position = initial_position
        self.arm.movelj([*tool_position, *self.tool_orientation],
                        **self.move_controls)

    def location_mode_get_point(self):
        axis, direction = self.key_direction[self.pressed_key]
        point = self.arm.getl()[:3]
        point[axis] += direction * self.step
        return point

    def board_mode_get_point(self):
        axis, direction = self.key_direction[self.pressed_key]
        self.point_idx += self.base_map[axis] * direction
        self.point_idx %= len(self.calib_board.local_points)
        self.logger.debug(f"Move to point #{self.point_idx} on board")
        point = self.calib_board.local_points[self.point_idx].copy()
        point[2] = self.arm.getl()[2]
        if axis == 2:
            point[2] += self.step * direction
        return point

    def move_on_command(self, cond, use_gripper, initial_position):
        self.moving_mode_reset(use_gripper=use_gripper,
                               initial_position=initial_position),
        try:
            with cond:
                while True:
                    cond.wait()
                    self.logger.debug("Moving thread resumed")
                    if self.pressed_key == 'q':
                        break
                    if self.pressed_key is None:
                        self.logger.warning("Pressed key is None but the "
                                            "thread is still waked up.")
                        continue
                    if use_gripper:
                        point = self.location_mode_get_point()
                    else:
                        point = self.board_mode_get_point()

                    self.logger.info(f"Move to point: "
                                     f"{[*point, *self.tool_orientation]}")
                    self.arm.movelj([*point, *self.tool_orientation],
                                    **self.move_controls)
                    self.pressed_key = None
        finally:
            self.logger.debug("Left arm feed loop")
            self.arm.close()

    def run(self):
        condition = Condition()
        thread_cam = Thread(target=self.streaming, args=(condition,))
        self.logger.debug(f"Cam thread created with: {self.cam}")
        if self.verify_by_cam:
            moving_args = (condition, False, [0, 0, 0.45])
        else:
            moving_args = (condition, True, [0.1, 0.1, 0.1])
        thread_arm = Thread(target=self.move_on_command, args=moving_args)

        self.logger.debug(f"Arm thread created with: {self.arm}")
        thread_cam.start()
        thread_arm.start()
        self.logger.info("Threads created... Started running.")
        thread_cam.join()
        thread_arm.join()
        self.logger.info("Finished.")
