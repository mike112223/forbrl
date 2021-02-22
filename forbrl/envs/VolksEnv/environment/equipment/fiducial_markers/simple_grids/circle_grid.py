import logging

from .base import GridBase
from ..registry import FIDUCIAL_MARKERS


@FIDUCIAL_MARKERS.register_module
class CircleGridBoard(GridBase):
    """Class of circle grid calibrate board."""
    logger = logging.getLogger(__name__)

    def __init__(self, shape=(7, 7), scale=0.015, corner=0.013, offset=0):
        super().__init__(shape=shape, scale=scale, corner=corner, offset=offset)

    def __repr__(self):
        msg = (f"A circle grid calibrate board with: \n"
               f"    shape={self.shape}\n    scale={self.scale}\n"
               f"    corner={self.corner_offset}\n    offset={self.offset}\n")
        return msg
