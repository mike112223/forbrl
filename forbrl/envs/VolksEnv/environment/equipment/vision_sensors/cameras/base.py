import logging
from abc import abstractmethod


class CamBase:
    """A base class for a camera"""
    logger = logging.getLogger(__name__)
    _intrinsics = None

    def __init__(self, ):
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def capture(self):
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()
