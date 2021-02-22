import logging

from .april_tag import AprilTags
from ..registry import FIDUCIAL_MARKERS


@FIDUCIAL_MARKERS.register_module
class AprilTagsSim(AprilTags):
    """Class of april tag in a simulated space -- not implemented yet."""
    logger = logging.getLogger(__name__)

    def __init__(self, family='tag6h8'):
        super().__init__(family=family)

    def __repr__(self):
        msg = f"A simulated April Tag of family {self.family}: \n"
        return msg
