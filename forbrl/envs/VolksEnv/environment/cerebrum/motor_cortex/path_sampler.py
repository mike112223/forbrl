import logging
from itertools import chain, combinations, islice

import numpy as np

from .registry import MOTOR_CORTEX


@MOTOR_CORTEX.register_module
class PathSampler:
    """
    A sampler to sample points in space (might be higher dimensional space
    such as joint space for a 6 DoF robotic arm).
    """
    logger = logging.getLogger(__name__)

    # TODO: remove pattern -> ( if step is None: step = self.step)
    def __init__(self, points, max_path=100, path_step=4):
        if isinstance(points, list):
            self.points = np.array(points)
        else:
            self.points = points
        self.max_path = max_path
        self.path_step = path_step
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = (f"A {self.__class__.__name__} instance with points:\n"
               f"{self.points}\npath step : {self.path_step}\n")
        return msg

    def points_in_between(self, point_a, point_b, step=None):
        if step is None:
            step = self.path_step
        assert point_a.shape == point_b.shape, "points should be in same space"
        points = np.linspace(point_a, point_b, step)
        gen = (_ for _ in points)
        return gen

    def path_by_combo(self, limit=None):
        if limit is None:
            limit = self.max_path
        combo = combinations(self.points, 2)
        paths = islice(combo, limit)
        res = chain()
        for path in paths:
            res = chain(res, self.points_in_between(*path))
        return res

    def path_by_grid(self, step=None):
        if step is None:
            step = self.path_step
        assert len(self.points) in [4, 8], "currently only support 2D/3D grid"

        res = self.path_in_plane(self.points[:4], step=step)
        if len(self.points) == 8:
            res2 = self.path_in_plane(self.points[4:], step=step)
            res = self.sample_in_turn(res, res2)
        return res

    def path_in_plane(self, points, step=None):
        if step is None:
            step = self.path_step
        line1 = self.points_in_between(points[0], points[1], step)  # x1y1, x2y1
        line2 = self.points_in_between(points[2], points[3], step)  # x1y2, x2y2
        return self.sample_in_turn(line1, line2)

    def sample_in_turn(self, path1, path2):
        res = chain()
        for idx, (p1, p2) in enumerate(zip(path1, path2)):
            if idx % 2:  # alter this order to prevent useless movements
                res = chain(res, self.points_in_between(p1, p2))
            else:
                res = chain(res, self.points_in_between(p2, p1))
        return res
