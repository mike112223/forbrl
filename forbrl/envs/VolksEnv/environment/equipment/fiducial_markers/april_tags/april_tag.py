import logging

from ..registry import FIDUCIAL_MARKERS


@FIDUCIAL_MARKERS.register_module
class AprilTags:
    """Class of april tag -- not implemented yet."""

    logger = logging.getLogger(__name__)

    def __init__(self, family='tag6h8'):
        self.family = family
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = f"An April Tag of family {self.family} \n"
        return msg
