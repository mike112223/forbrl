

from abc import ABC, abstractmethod

class Algorithm(ABC):
    '''Abstract Algorithm class to define the API methods'''

    def __init__(self, agent):
        pass

    def act(self, state):
        '''Standard act method.'''
        pass

    @abstractmethod
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        raise NotImplementedError

    def save(self, ckpt=None):
        '''Save net models for algorithm given the required property self.net_names'''
        pass

    def load(self):
        '''Load net models for algorithm given the required property self.net_names'''
        pass
