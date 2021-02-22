import logging
from abc import abstractmethod


class GripperBase:
    """A base class for a gripper"""
    logger = logging.getLogger(__name__)
    tcp = [0, 0, 0, 0, 0, 0]

    def __init__(self, ):
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def grip(self):
        pass

    @abstractmethod
    def move_to(self, target):
        pass
