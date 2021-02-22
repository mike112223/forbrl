import logging
import numbers

import numpy as np


class GridBase:
    """Base class of a grid-based calibrate board."""

    logger = logging.getLogger(__name__)

    def __init__(self, shape=(7, 7), scale=0.015, corner=0.013, offset=0):
        self.shape = shape
        self.scale = scale
        self.corner_offset = corner

        self._local_points = None
        self._corner_points = None
        self._offset = None
        self._axis = None

        self.offset = offset
        self.gen_ref_points()
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def gen_ref_points(self):  # TODO: test the case of w != h
        w, h = self.shape

        local_points = np.zeros((w * h, 3), np.float32)
        local_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        local_points = local_points[:, (1, 0, 2)]
        self._local_points = local_points * self.scale  # covert unit to m
        self._local_points += self.offset
        self._local_points = np.around(self._local_points, decimals=4)

        delta = self.corner_offset / np.sqrt(2)  # calculate corner offset
        corner_base = np.float32([[0, 0, 0],
                                  [w - 1, 0, 0],
                                  [w - 1, h - 1, 0],
                                  [0, h - 1, 0]]).reshape(-1, 3) * self.scale
        corner_base += np.float32([[-delta, -delta, 0],
                                   [+delta, -delta, 0],
                                   [+delta, +delta, 0],
                                   [-delta, +delta, 0], ]).reshape(-1, 3)
        self._corner_points = corner_base + self.offset

        self._axis = np.float32([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]]).reshape(-1, 3)
        self._axis *= 3 * self.scale

    @property
    def local_points(self):
        return self._local_points

    @property
    def corner_points(self):
        return self._corner_points

    @property
    def offset(self):
        return self._offset

    @property
    def axis(self):
        return self._axis

    @offset.setter
    def offset(self, offset_):
        if isinstance(offset_, numbers.Number):
            self._offset = np.array([1, 1, 0]) * offset_
        elif isinstance(offset_, (list, np.ndarray, tuple)):
            if len(offset_) > 3:
                raise ValueError(f"offset should be of length 3, while "
                                 f"{len(offset_)} provided")
            self._offset = offset_
        else:
            raise TypeError(f"offset should be of type Number | Sequence, "
                            f"while {type(offset_)} provided")

    def rotate_lpoints(self, times):
        # TODO: test the case of w != h
        if int(times) != times:
            # TODO: need a better way to identify int
            raise TypeError(f"time of rotation should be integer while "
                            f"{type(times)} provided with value {times}")

        times %= 4
        res = self.local_points

        if times:
            # TODO: need a better way to do rotation
            w, h = self.shape
            res = res.reshape(w, h, -1)
            res = np.moveaxis(res, 0, 1)
            res = np.rot90(res, 4 - times)
            res = np.moveaxis(res, 0, 1)
            res = res.reshape(-1, 3)
        return res
