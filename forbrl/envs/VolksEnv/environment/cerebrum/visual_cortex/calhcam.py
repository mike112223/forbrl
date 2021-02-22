import logging

import cv2
import numpy as np
import pyrealsense2 as rs

from .registry import VISUAL_CORTEX

CV2_PNP_MODE = {'default': cv2.solvePnP,
                'ransac': cv2.solvePnPRansac}


@VISUAL_CORTEX.register_module
class CalHCam:
    """
    Class of a transform calculator between a fiducial marker and a camera or
    vice versa.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, calib_board, camera=None, cam_instrinsics=None,
                 cam_distortion=None, thres=90, pnp_method='default'):
        if camera is None:
            self.cam_intrinsic = cam_instrinsics
            self.cam_distortion = cam_distortion
        else:
            self.camera = camera
            self._cam_intrinsic = None
            self._cam_distortion = None
        self.calib_board = calib_board
        self.rvec = None
        self.tvec = None
        self.thres = thres
        self.pnp_method = pnp_method
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = (f"A translation estimator between {self.camera} and "
               f"{self.calib_board} with:\n    pnp_method: {self.pnp_method}\n")
        return msg

    def update_cam_parameters(self):
        self.cam_intrinsic = self.camera.intrinsics
        self.cam_distortion = self.camera.intrinsics

    def get_rotation_time(self, image, rvec, tvec):
        feat_projected, _ = cv2.projectPoints(self.calib_board.corner_points,
                                              rvec,
                                              tvec,
                                              self.cam_intrinsic,
                                              self.cam_distortion)
        feat_projected = np.rint(np.squeeze(feat_projected)).astype(int)
        gray_value, invalid = np.ones(len(feat_projected)) * 256, []
        w, h = image.shape

        for idx, cor in enumerate(feat_projected):
            if 0 <= cor[1] < w and 0 <= cor[0] < h:
                gray_value[idx] = image[cor[1], cor[0]]
            else:
                invalid.append(idx)

        min_idx = np.argmin(gray_value)
        if len(invalid) == 0:
            return min_idx
        if len(invalid) == 1:
            if gray_value.min() < self.thres:
                return min_idx
            else:
                return invalid[0]
        return None

    def project_axis(self, image, axis=None):
        for _ in [self.rvec, self.tvec]:
            assert _ is not None, "rotation not calculated yet"
        if axis is None:
            axis = self.calib_board.axis
        projected_axis, _ = cv2.projectPoints(axis,
                                              self.rvec,
                                              self.tvec,
                                              self.cam_intrinsic,
                                              self.cam_distortion)

        origin = tuple(projected_axis[0].ravel())
        for idx, vector in enumerate(projected_axis[1:]):
            color = np.zeros(3)
            color[idx] = 255
            cv2.line(image, origin, tuple(vector.ravel()), color, 5)

    def project_polygon(self, image, points=None, color=(0, 255, 0)):
        for _ in [self.rvec, self.tvec]:
            assert _ is not None, "rotation not calculated yet"
        projected_axis, _ = cv2.projectPoints(points,
                                              self.rvec,
                                              self.tvec,
                                              self.cam_intrinsic,
                                              self.cam_distortion)
        n = len(projected_axis)
        for idx, vector in enumerate(projected_axis):
            p_0 = tuple(vector.ravel())
            p_1 = tuple(projected_axis[(idx + 1) % n].ravel())
            cv2.line(image, p_0, p_1, color, 3)

    def project_corners(self, image):
        for _ in [self.rvec, self.tvec]:
            assert _ is not None, "rotation not calculated yet"

        for _, point in enumerate(self.calib_board.corner_points):
            self.project_point(image, point)

    def project_point(self, image, point):
        for _ in [self.rvec, self.tvec]:
            assert _ is not None, "rotation not calculated yet"
        projected, _ = cv2.projectPoints(np.array([point], np.float64),
                                         self.rvec,
                                         self.tvec,
                                         self.cam_intrinsic,
                                         self.cam_distortion)
        projected = np.rint(np.squeeze(projected)).astype(int)
        cv2.circle(image,
                   tuple(projected),
                   radius=2,
                   color=(0, 255, 0))

    def self_validate(self, depth, calib_point=None):
        w, h = depth.shape
        cam = {'cameraMatrix': self.cam_intrinsic,
               'distCoeffs': self.cam_distortion}
        if calib_point is None:
            calib_point = self.calib_board.local_points
        feature_img, _ = cv2.projectPoints(calib_point,
                                           self.rvec,
                                           self.tvec,
                                           **cam)
        feat_projected = np.rint(np.squeeze(feature_img)).astype(int)
        d_depth, d_mat = [], []

        for idx, cor in enumerate(feat_projected):
            if 0 <= cor[1] < w and 0 <= cor[0] < h:
                depth_read = depth[cor[1], cor[0]]
                if not depth_read:
                    continue
                d_depth.append(depth_read)
                cord = self.to_cam_coord(calib_point[idx])
                d_mat.append(np.linalg.norm(cord))

        d_depth, d_mat = np.array(d_depth), np.array(d_mat)
        diff = d_depth - d_mat
        with np.printoptions(precision=5):
            print(f"error: {diff.mean() * 100}cm +- {diff.std() * 100}cm")

    def to_cam_coord(self, point):
        mat = self.rt_matrix
        assert len(point) == 3, f"unsupported length {len(point)}"
        homo = np.ones((4, 1))
        for i in range(3):
            homo[i, 0] = point[i]
        res = (mat @ homo).squeeze()
        return res[:3]

    def __call__(self, image, as_matrix=True):
        if self.cam_intrinsic is None:
            self.update_cam_parameters()
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findCirclesGrid(img_gray, self.calib_board.shape)
        refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                           30,
                           0.001)
        cam = {'cameraMatrix': self.cam_intrinsic,
               'distCoeffs': self.cam_distortion}
        pnp_solver = CV2_PNP_MODE[self.pnp_method]

        if found:
            corners_refined = cv2.cornerSubPix(img_gray,
                                               corners,
                                               (3, 3),
                                               (-1, -1),
                                               refine_criteria)

            res = pnp_solver(self.calib_board.local_points,
                             corners_refined,
                             **cam)

            idx = self.get_rotation_time(img_gray, res[1], res[2])
            if idx is None:
                return None

            if idx != 0:
                lpoints = self.calib_board.rotate_lpoints(idx)
                res = pnp_solver(lpoints, corners_refined, **cam)

            self.rvec, self.tvec = res[1], res[2]

            if as_matrix:
                return self.rt_matrix
            else:
                return self.rvec, self.tvec

        return None

    @property
    def rt_matrix(self):
        for _ in [self.rvec, self.tvec]:
            assert _ is not None, "rotation not calculated yet"
        rotation_mat, _ = cv2.Rodrigues(self.rvec)
        pose_mat = np.hstack((rotation_mat, self.tvec))
        pose_mat = np.vstack((pose_mat, np.array([[0, 0, 0, 1]])))
        return pose_mat

    @property
    def cam_intrinsic(self):
        return self._cam_intrinsic

    @property
    def cam_distortion(self):
        return self._cam_distortion

    @cam_intrinsic.setter
    def cam_intrinsic(self, intrinsic):
        # TODO: remove realsense dependency in the future
        if isinstance(intrinsic, rs.intrinsics):
            res = np.zeros((3, 3))
            res[0, 0] = intrinsic.fx
            res[1, 1] = intrinsic.fy
            res[0, 2] = intrinsic.ppx
            res[1, 2] = intrinsic.ppy
            res[2, 2] = 1
            self._cam_intrinsic = res
            self._cam_distortion = np.array(intrinsic.coeffs)
        elif isinstance(intrinsic, np.ndarray):
            if intrinsic.shape == (3, 3):
                self._cam_intrinsic = intrinsic
            else:
                raise ValueError(f"intrinsic matrix should be of shape (3,3), "
                                 f"while shape {intrinsic.shape} provided")
        else:
            raise TypeError(f"supported types are 'pyrealsense2.intrinsics' | "
                            f"'np.ndarray', while {type(intrinsic)} provided")

    @cam_distortion.setter
    def cam_distortion(self, distortion):
        if distortion is None:
            self._cam_distortion = np.zeros(4)
        elif isinstance(distortion, rs.intrinsics):
            self._cam_distortion = np.array(distortion.coeffs)
        elif isinstance(distortion, np.ndarray):
            if 4 <= len(distortion) <= 8:
                self._cam_distortion = distortion
            else:
                raise ValueError(f"number of distortion parameters should be "
                                 f"in range(4, 9), while shape "
                                 f"{distortion.shape} provided")

        else:
            raise TypeError(f"supported types are 'pyrealsense2.intrinsics' | "
                            f"'np.ndarray', while {type(distortion)} provided")
