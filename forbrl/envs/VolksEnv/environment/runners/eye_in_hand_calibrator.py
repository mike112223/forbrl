import glob
import logging
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from .registry import RUNNERS
from ..utils import get_time_iso, save_collected


@RUNNERS.register_module
class EyeInHandCalibrator:
    """
    A calibrator that does eye in hand calibration for a system that contains
    a robotic arm, a camera and a fiducial marker that serves as a
    calibration board.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, arm=None, cam=None, calhcam=None,
                 path_generator=None, axyb_solver=None, work_dir=None,
                 calculate_from_past_session=None):
        self.arm = arm
        self.cam = cam
        self.calhcam = calhcam
        self.path_generator = path_generator
        self.axyb_solver = axyb_solver
        self.work_dir = work_dir
        self.calculate_session = calculate_from_past_session
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = (f"Eye in hand calibrator with :\n"
               f"    arm:{self.arm}\n"
               f"    camera:{self.cam}\n"
               f"    calhcam:{self.calhcam}\n"
               f"    path_generator:{self.path_generator}\n"
               f"    axyb_solver:{self.axyb_solver}\n"
               f"    work_dir:{self.work_dir}\n"
               f"    calculate_only:{self.calculate_session}\n")
        return msg

    def run(self):
        if not self.calculate_session:
            session_dir = os.path.join(self.work_dir,
                                       f"Session_{get_time_iso()}")
            self.capture(session_dir)
        else:
            session_dir = os.path.join(self.work_dir, self.calculate_session)
        self.calculate(session_dir)

    def calculate(self, session_dir):
        stored = os.listdir(session_dir)
        for required in ['cal2cam_mat', 'base2tool_mat']:
            assert required in stored, f"404 {required} not found"

        samples = glob.glob(os.path.join(session_dir, 'cal2cam_mat', '*.pkl'))

        cc_mats, bt_mats = [], []
        for idx, cc_dir in enumerate(samples):
            bt_dir = cc_dir.replace('cal2cam_mat', 'base2tool_mat')
            with open(cc_dir, 'rb') as mat:
                cc_mats.append(pickle.load(mat))
            with open(bt_dir, 'rb') as mat:
                bt_mats.append(pickle.load(mat))
        cc_mats = np.stack(cc_mats, axis=-1)
        bt_mats = np.stack(bt_mats, axis=-1)

        estimation = self.axyb_solver.by_kronecker_product(cc_mats, bt_mats)

        calibrate_res = {'cam2tool': estimation[0],
                         'calib2base': estimation[1],
                         'calib2base_check': estimation[2],
                         'Error(mean, std)': estimation[3]}

        calibration_dir = os.path.join(session_dir, 'results')
        save_collected(calibration_dir, flat=True, **calibrate_res)

        msg = f"Calibration results:"
        for field, value in calibrate_res.items():
            msg += f"\n{' ' * 4}{field} : \n{np.array2string(value)}"
        self.logger.info(msg)

    def capture(self, session_dir):
        self.cam.start()
        for idx, point in enumerate(
                tqdm(self.path_generator.path_by_grid(),
                     total=self.path_generator.path_step ** 3,
                     dynamic_ncols=True,
                     desc="Calibrating Eye in hand",
                     unit='pose',
                     unit_scale=True)):
            try:
                self.arm.movej(point, 0.6, 3)  # 0.8acc + 0.5sleep not working
                time.sleep(0.8)
                color_img, depth_img = self.cam.capture()
                cam2cal = self.calhcam(color_img)
                if cam2cal is not None:
                    base2tool = np.array(self.arm.get_pose().get_matrix())
                    cal2cam = np.linalg.inv(cam2cal)
                    data = {
                        'color_img': color_img,
                        # 'depth_img': depth_img,
                        # 'movej_cmd': (idx, point),
                        # 'getl_cmd': (idx, ur5.getl()),
                        'cal2cam_mat': cal2cam,
                        'base2tool_mat': base2tool,
                        # 'tool2base_mat': tool2base
                    }
                    save_collected(session_dir, **data)
                    self.logger.debug(f"Successfully collected a data point "
                                      f"with joint positions: {point}.")
                else:
                    tqdm.write(f"Oops, didn't find the calibration board at "
                               f"frame: {idx} with joint positions: {point}.")

            except RuntimeError as e:
                self.logger.info(f"Unexpected event happened, {e}")

        self.cam.stop()
        self.arm.close()
