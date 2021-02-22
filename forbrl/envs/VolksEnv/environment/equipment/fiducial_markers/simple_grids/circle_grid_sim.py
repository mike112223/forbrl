import logging

from .circle_grid import CircleGridBoard
from ..registry import FIDUCIAL_MARKERS


@FIDUCIAL_MARKERS.register_module
class CircleGridBoardSim(CircleGridBoard):
    """Class of circle grid calibrate board in a simulated space."""
    logger = logging.getLogger(__name__)

    def __init__(self, shape=(7, 7), scale=0.015, corner=0.013, offset=0):
        super().__init__(shape=shape, scale=scale, corner=corner, offset=offset)

    def __repr__(self):
        msg = (f"A simulated circle grid calibrate board with: \n"
               f"    shape={self.shape}\n    scale={self.scale}\n"
               f"    corner={self.corner_offset}\n    offset={self.offset}\n")
        return msg
